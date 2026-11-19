import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from mappo_baseline import FEATURES, EmbeddedPool, embed, load_jepa, make_world
from tarmac_baseline import DeliveredLinks, ProgressReward

KINDS = ('dreamer', 'codreamer')


def observations(cfg) -> dict:
    if cfg.features == 'vit':
        return dict(obs_keys=['view_emb', 'goal_emb', 'direction', 'position'], image_key=None)
    if cfg.features == 'cnn':
        return dict(obs_keys=['direction', 'position'], image_key=['pixels', 'goal'])
    raise ValueError(f"features: vit or cnn, not {cfg.features!r}")


class DreamerActor:

    def __init__(self, model, jepa=None, links: DeliveredLinks | None = None, cic: int = 0, seed: int = 0):
        from c3_wm.evaluators.deploy import MessageInfluence
        self.model, self.jepa, self.links = model, jepa, links
        self.cic, self._rng, self._seed, self._step = cic, np.random.default_rng(seed), seed, None
        self.influence = MessageInfluence() if cic else None
        if cic:
            act = model.act
            def recorded(trans, state, eval=False, _act=act):
                self._step = (trans, state)
                return _act(trans, state, eval)
            model.act = recorded

    def eval(self):
        return self

    def parameters(self):
        return self.model.parameters()

    def get_action(self, info: dict):
        info = dict(info)
        as_np = lambda v: v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
        if self.jepa is not None:
            for key, source in FEATURES:
                info[key] = embed(self.jepa, as_np(info[source])[:, -1])[:, None]
        if self.links is not None:
            arrived = as_np(info['msg_received'])[:, -1].astype(bool) & (as_np(info['msg_age'])[:, -1] == 0)
            self.links.current = arrived | np.eye(arrived.shape[-1], dtype=bool)
        actions = self.model.get_action(info)
        if self.cic and self._step is not None:
            self._influence(*self._step)
        return actions

    @torch.no_grad()
    def _influence(self, trans: dict, state):
        agent = self.model.agent
        E, N = trans['mask'].shape[:2]
        if E < 2:
            return
        obs = {k: v.flatten(0, 1).to(agent.device) for k, v in trans.items()}
        state = agent.get_initial_state(E * N) if state is None else state
        alive = trans['mask'][..., 0] > 0
        hears = trans['hears'][:, :, 0]
        seed = int(self._rng.integers(2 ** 31))

        def probs(**intervention):
            with torch.random.fork_rng(devices=[agent.device] if agent.device.type == 'cuda' else []):
                torch.manual_seed(seed)
                return agent.action_probs(obs, state, **intervention).view(E, N, -1)
        for i in range(N):
            senders = [k for k in range(N) if k != i]
            got = (hears[:, i, senders].any(-1) & alive[:, i]).numpy()
            donors = np.flatnonzero(alive[:, senders].all(-1).numpy())
            rows = [e for e in np.flatnonzero(got) if len(donors[donors != e])]
            if not rows:
                continue
            silent = probs(listener=i, silent=True)[:, i]
            others = []
            for _ in range(self.cic):
                d = torch.as_tensor([self._rng.choice(donors[donors != e]) if e in rows else e for e in range(E)],
                                    device=agent.device)
                others.append(probs(listener=i, swap=d)[:, i])
            received = probs(listener=i)[:, i]
            stacked = torch.stack([received, *others], 1)
            for e in rows:
                self.influence.note(stacked[e], silent[e])


def dreamer_actor(kind: str, path, jepa, jepa_path: str, device, sample: bool = False, cic: int = 0, seed: int = 0):
    from stable_marl.wm.codreamer import CoDreamerAgents
    from stable_marl.wm.dreamer import DreamerAgents
    model = (CoDreamerAgents if kind == 'codreamer' else DreamerAgents).load(path, map_location=device)
    model.agent.eval()
    model.temperature = 1.0 if sample else 0.0
    vit = model.checkpoint.get('features') == 'vit'
    trained_on = model.checkpoint.get('jepa')
    if vit and trained_on and Path(trained_on).name != Path(jepa_path).name:
        logger.warning(f"{kind} was trained on the features of {trained_on}, deployed with {jepa_path}")
    links = None
    if kind == 'codreamer':
        model.drop_prob = 0.0
        model.comm_range = None
        model.links = links = DeliveredLinks()
    return DreamerActor(model, jepa if vit else None, links, cic=cic if kind == 'codreamer' else 0, seed=seed)


@hydra.main(version_base=None, config_path="../configs", config_name="baselines/dreamer")
def main(cfg: DictConfig):
    from stable_marl.wm.codreamer import CoDreamerAgents
    from stable_marl.wm.commnet import commnet_features
    from stable_marl.wm.dreamer import DreamerAgents, dreamer_config, train_dreamer
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.kind not in KINDS:
        raise ValueError(f"kind: dreamer or codreamer, not {cfg.kind!r}")
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    if cfg.device == 'cuda' and device != 'cuda':
        raise RuntimeError("device=cuda, but no GPU is visible")
    seen = observations(cfg)
    shaping = [lambda e: ProgressReward(e, cfg.near_coeff)] if cfg.near_coeff else []
    world = make_world(cfg, list(range(*cfg.layouts)), wrappers=shaping)
    if cfg.features == 'vit':
        world.envs = EmbeddedPool(world.envs, load_jepa(cfg, device))
    world.reset(seed=cfg.seed)
    env = world.envs.envs[0].unwrapped
    columns = {k: np.asarray(v)[:, -1] for k, v in world.infos.items() if not k.startswith('_')}
    obs_dim = commnet_features(columns, seen['obs_keys'], env.width, goal=False).shape[-1]
    overrides = {k: v for k, v in OmegaConf.to_container(cfg.model.overrides).items() if v is not None}
    config = dreamer_config(cfg.model.size, cfg.model.rep_loss, device, **overrides)
    common = dict(num_actions=cfg.actions, goal=False, grid_size=env.width, team_reward=cfg.team_reward,
                  image_size=cfg.image_size, **seen)
    if cfg.kind == 'codreamer':
        model = CoDreamerAgents(config, obs_dim, world.num_agents, **common, **OmegaConf.to_container(cfg.comm))
        bits = f", {model.message_bits} bits a round"
    else:
        model = DreamerAgents(config, obs_dim, world.num_agents, **common)
        bits = ''
    logger.info(f"{cfg.kind} ({cfg.features} features{bits}, progress reward {cfg.near_coeff}) on layouts "
                f"{list(cfg.layouts)}: {world.num_agents} agents, {obs_dim} vector features, "
                f"{sum(p.numel() for p in model.agent._named_params.values()) / 1e6:.2f}M parameters, on {device}")
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{cfg.kind}.pt'
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project,
                         name=cfg.wandb.name or f"{cfg.kind}-{cfg.features}-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    extra = dict(env_name=cfg.env_id, layouts=list(cfg.layouts), features=cfg.features, kind=cfg.kind,
                 jepa=str(cfg.checkpoints.jepa) if cfg.features == 'vit' else None)
    saved = [0]

    def log(row):
        logger.info({k: round(v, 4) for k, v in row.items()})
        if run is not None:
            run.log(row, step=int(row['env_steps']))
        if row['env_steps'] - saved[0] >= cfg.save_every:
            model.save(path, env_steps=int(row['env_steps']), **extra)
            saved[0] = row['env_steps']
    train_dreamer(model, world, seed=cfg.seed, log=log, **OmegaConf.to_container(cfg.train))
    model.save(path, env_steps=int(cfg.train.steps), **extra)
    logger.info(f"saved {path}")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
