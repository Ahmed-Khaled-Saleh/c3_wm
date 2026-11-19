import os
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from c3_wm.evaluators.deploy import MESSAGE_TOKENS
from mappo_baseline import FEATURES, EmbeddedPool, embed, load_jepa, make_world
from stable_marl.wm.tarmac.core import targeted_attention


class ProgressReward(gym.Wrapper):

    def __init__(self, env: gym.Env, coeff: float):
        super().__init__(env)
        self.coeff = float(coeff)

    def _distances(self) -> np.ndarray:
        from stable_marl.wm.mamba.core import _shaping_distances
        return _shaping_distances(self.env.unwrapped)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._distance = self._distances()
        return out

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        distance = self._distances()
        bonus, self._distance = self.coeff * (self._distance - distance), distance
        if isinstance(rewards, dict):
            rewards = {a: r + float(bonus[i]) for i, (a, r) in enumerate(rewards.items())}
        else:
            rewards = np.asarray(rewards, np.float32) + bonus
        return obs, rewards, terms, truncs, infos


def placeholder_tokens(pixels) -> np.ndarray:
    return np.zeros((len(pixels), MESSAGE_TOKENS), np.int64)


class DeliveredLinks:

    def __init__(self):
        self.current = None

    def reset(self, envs, seeds=None):
        pass

    def masks(self, position, alive, rounds: int, message_dim: int, message_bits=None) -> torch.Tensor:
        return torch.as_tensor(self.current, dtype=torch.bool)[:, None].expand(-1, rounds, -1, -1).clone()


class TarMACActor:

    def __init__(self, model, jepa, links: DeliveredLinks | None = None, cic: int = 0, seed: int = 0):
        from c3_wm.evaluators.deploy import MessageInfluence
        self.model, self.jepa, self.links = model, jepa, links
        self.cic, self._rng, self._sent = cic, np.random.default_rng(seed), None
        self.influence = MessageInfluence() if cic else None
        if cic and model.rounds != 1:
            raise ValueError("TarMAC's cic: one round of messages a step")

    def eval(self):
        return self

    def parameters(self):
        return self.model.parameters()

    def get_action(self, info: dict):
        info = dict(info)
        as_np = lambda v: v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
        for key, source in FEATURES:
            info[key] = embed(self.jepa, as_np(info[source])[:, -1])[:, None]
        if self.links is not None:
            arrived = as_np(info['msg_received'])[:, -1].astype(bool) & (as_np(info['msg_age'])[:, -1] == 0)
            self.links.current = arrived | np.eye(arrived.shape[-1], dtype=bool)
        if not self.cic:
            return self.model.get_action(info)
        before = self.model._state
        actions = self.model.get_action(info)
        self._influence(info, before)
        return actions

    @torch.no_grad()
    def _influence(self, info: dict, before):
        m = self.model
        columns = {k: torch.as_tensor(info[k])[:, -1].cpu().numpy() for k in m._info_keys()}
        obs, alive, hears = m.observe(columns)
        device = next(m.parameters()).device
        obs, alive_t, hears = obs.to(device), alive.to(device), hears.to(device)
        flush = info.get('_needs_flush')
        E, N = alive.shape
        fresh = np.ones(E, bool) if flush is None else torch.as_tensor(flush).cpu().numpy().astype(bool)
        sent, self._sent = self._sent, (m._state[0], hears[:, 0], alive_t)
        if before is None or sent is None or len(before[0]) != E or E < 2:
            return
        h, c = before
        h_sent, heard, alive_sent = sent
        eye = torch.eye(N, dtype=torch.bool, device=device)
        query = m.query(h_sent)
        attend = lambda states, mask: targeted_attention(query, m._send(m.signature(states)), m._send(m.value(states)),
                                                          mask & alive_sent.unsqueeze(1))[0]
        for i in range(N):
            got = ((heard[:, i] & alive_sent & ~eye[i]).any(-1).cpu().numpy() & alive[:, i].numpy() & ~fresh)
            senders = [k for k in range(N) if k != i]
            ok = alive_sent[:, senders].all(-1).cpu().numpy()
            donors = np.flatnonzero(ok)
            rows = [e for e in np.flatnonzero(got) if len(donors[donors != e])]
            if not rows:
                continue

            def act_with(read):
                cc = c.clone()
                cc[:, i] = read[:, i]
                return m.step(obs, alive_t, hears, h, cc)[0][:, i].softmax(-1)
            mute = heard.clone()
            mute[:, i] = eye[i]
            silent = act_with(attend(h_sent, mute))
            others = []
            for _ in range(self.cic):
                d = torch.as_tensor([self._rng.choice(donors[donors != e]) if e in rows else e for e in range(E)], device=device)
                swapped = h_sent.clone()
                swapped[:, senders] = h_sent[d][:, senders]
                others.append(act_with(attend(swapped, heard)))
            probs = torch.stack([act_with(attend(h_sent, heard)), *others], 1)
            for e in rows:
                self.influence.note(probs[e], silent[e])


@hydra.main(version_base=None, config_path="../configs", config_name="baselines/tarmac")
def main(cfg: DictConfig):
    from stable_marl.wm.commnet import commnet_features
    from stable_marl.wm.tarmac import TarMAC, train_tarmac
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    if cfg.device == 'cuda' and device != 'cuda':
        raise RuntimeError("device=cuda, but no GPU is visible")
    jepa = load_jepa(cfg, device)
    shaping = [lambda e: ProgressReward(e, cfg.near_coeff)] if cfg.near_coeff else []
    world = make_world(cfg, list(range(*cfg.layouts)), wrappers=shaping)
    world.envs = EmbeddedPool(world.envs, jepa)
    world.reset(seed=cfg.seed)
    env = world.envs.envs[0].unwrapped
    columns = {k: np.asarray(v)[:, -1] for k, v in world.infos.items() if not k.startswith('_')}
    obs_dim = commnet_features(columns, cfg.obs_keys, env.width, goal=False).shape[-1]
    model = TarMAC(obs_dim, world.num_agents, num_actions=cfg.actions, obs_keys=list(cfg.obs_keys), goal=False,
                   grid_size=env.width, **OmegaConf.to_container(cfg.model)).to(device)
    logger.info(f"TarMAC ({model.rounds} round(s), {model.message_bits} bits a message, progress reward {cfg.near_coeff}) on layouts {list(cfg.layouts)}: "
                f"{world.num_agents} agents, {obs_dim} features, {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters, on {device}")
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"tarmac-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    extra = dict(env_name=cfg.env_id, layouts=list(cfg.layouts), jepa=str(cfg.checkpoints.jepa), obs_keys=list(cfg.obs_keys))
    saved = [0]

    def log(row):
        logger.info({k: round(v, 4) for k, v in row.items()})
        if run is not None:
            run.log(row, step=int(row['episodes']))
        if row['episodes'] - saved[0] >= cfg.save_every * cfg.num_envs:
            model.save(out / 'tarmac.pt', episodes=int(row['episodes']), **extra)
            saved[0] = row['episodes']
    model = train_tarmac(model, world, seed=cfg.seed, log=log, **OmegaConf.to_container(cfg.train))
    model.save(out / 'tarmac.pt', episodes=int(cfg.train.episodes), **extra)
    logger.info(f"saved {out / 'tarmac.pt'}")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
