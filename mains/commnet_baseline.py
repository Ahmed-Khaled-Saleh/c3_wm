import json
import os
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from stable_marl.wm.commnet import CommNet, comm_mask, commnet_features, mean_of_others
from stable_marl.wm.omnivla.core import IGNORE_INDEX, shortest_path_actions, walkable_cells

OBS_KEYS = ('view_emb', 'goal_emb', 'direction', 'position')


class ViTCommNet(CommNet):

    def __init__(self, obs_dim: int, num_actions: int = 3, hidden_dim: int = 128, comm_steps: int = 2,
                 skip: bool = True, share_weights: bool = False, obs_keys=OBS_KEYS, grid_size: int = 15,
                 comm_range=None, drop_prob: float = 0.0, latent_bits: int | None = 3):
        super().__init__(obs_dim, num_actions, hidden_dim, comm_steps, skip, share_weights, obs_keys, goal=False,
                         grid_size=grid_size, comm_range=comm_range, drop_prob=drop_prob)
        self.config.pop('goal')
        self.config['latent_bits'] = self.latent_bits = latent_bits
        self._quantizer = None

    @property
    def message_bits(self) -> int:
        return self.config['hidden_dim'] * (self.latent_bits or 32)

    def quantize(self, h: torch.Tensor) -> torch.Tensor:
        if self.latent_bits is None:
            return h
        if self._quantizer is None:
            from comm_core import UniformQuantizer
            self._quantizer = UniformQuantizer(self.latent_bits, -1.0, 1.0)
        q = self._quantizer.decode(self._quantizer.encode(h.detach())).to(h.dtype)
        return h + (q - h).detach()

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        h0 = torch.tanh(self.encoder(obs))
        h = h0
        for j in range(self.comm_steps):
            m = 0 if self.share_weights else j
            out = self.H[m](h)
            if j > 0:
                heard = mask[:, j - 1] if mask is not None and mask.dim() == 4 else mask
                out = out + self.C[0 if self.share_weights else j - 1](mean_of_others(self.quantize(h), heard))
            if self.S is not None:
                out = out + self.S[m](h0)
            h = torch.tanh(out)
        return self.policy_head(h), self.baseline_head(h).squeeze(-1)


def features(view_emb, goal_emb, direction, position, grid_size: int = 15) -> np.ndarray:
    cols = dict(view_emb=view_emb, goal_emb=goal_emb, direction=direction, position=position)
    return commnet_features(cols, OBS_KEYS, grid_size, goal=False)


class Samples(Dataset):

    def __init__(self, h5, frames, goal_emb: dict, layouts, hindsight_prob: float = 0.5, max_goal_steps: int = 20,
                 grid_size: int = 15, labels: str = 'shortest_path', actions: int = 3):
        if labels not in ('shortest_path', 'collected'):
            raise ValueError(f"labels: shortest_path or collected, not {labels!r}")
        lo, hi = layouts
        self.frames, self.goal_emb, self.grid_size = frames, goal_emb, grid_size
        self.labels, self.actions = labels, actions
        self.hindsight_prob = 1.0 if labels == 'collected' else hindsight_prob
        self.max_goal_steps = max_goal_steps
        with h5py.File(h5, 'r') as f:
            offsets, lens, layout = f['ep_offset'][:], f['ep_len'][:], f['layout']
            keep = [(int(o), int(n)) for o, n in zip(offsets, lens) if lo <= int(layout[int(o), 0]) < hi]
            rows = np.concatenate([np.arange(o, o + n) for o, n in keep])
            span = slice(int(rows.min()), int(rows.max()) + 1)
            self.cols = {k: f[k][span] for k in ('position', 'direction', 'terminated', 'goal_position', 'action')}
            self.cols['state'] = np.concatenate([f['state'][a:min(a + 20000, span.stop)].astype(np.int8)
                                                 for a in range(span.start, span.stop, 20000)])
            self.base = int(rows.min())
        alive = ~self.cols['terminated'].astype(bool)
        self.rows = np.array([(t, o + n - 1, o) for o, n in keep for t in range(o, o + n - 1) if alive[t - self.base].any()],
                             np.int64)
        self.episodes = len(keep)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t, last, start = (int(v) for v in self.rows[i])
        c, r = self.cols, t - self.base
        N = c['position'].shape[1]
        alive = ~c['terminated'][r].astype(bool)
        hindsight = np.random.random() < self.hindsight_prob
        g = t + 1 + np.random.randint(min(self.max_goal_steps, last - t)) if hindsight else None
        walkable = walkable_cells(c['state'][r])
        goals, labels = [], np.full(N, IGNORE_INDEX, np.int64)
        for a in range(N):
            if g is None:
                target, goal = c['goal_position'][r, a], self.goal_emb[start][a]
            else:
                target, goal = c['position'][g - self.base, a], np.asarray(self.frames[g, a], np.float32)
            goals.append(goal)
            if alive[a] and self.labels == 'collected':
                if c['action'][r, a] < self.actions:
                    labels[a] = int(c['action'][r, a])
            elif alive[a]:
                path = shortest_path_actions(walkable, c['position'][r, a], c['direction'][r, a], target)
                if path is not None and len(path):
                    labels[a] = int(path[0])
        obs = features(np.asarray(self.frames[t], np.float32)[None], np.stack(goals)[None], c['direction'][r][None],
                       c['position'][r][None], self.grid_size)[0] * alive[:, None]
        return {'obs': torch.from_numpy(obs), 'alive': torch.from_numpy(alive), 'labels': torch.from_numpy(labels)}


class CommNetActor:

    def __init__(self, model, jepa, sample: bool = False, cic: int = 0, seed: int = 0):
        from c3_wm.evaluators.deploy import MessageInfluence
        self.model, self.jepa, self.sample = model, jepa, sample
        self.cic, self._rng = cic, np.random.default_rng(seed)
        self.influence = MessageInfluence() if cic else None

    def eval(self):
        return self

    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def get_action(self, info: dict):
        from mappo_baseline import embed
        as_np = lambda v: (v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v))[:, -1]
        alive = ~as_np(info['terminated']).astype(bool)
        obs = features(embed(self.jepa, as_np(info['pixels'])), embed(self.jepa, as_np(info['goal'])),
                       as_np(info['direction']), as_np(info['position']), self.model.grid_size) * alive[..., None]
        heard = as_np(info['msg_received']).astype(bool) & (as_np(info['msg_age']) == 0) & alive[:, None, :]
        device = next(self.model.parameters()).device
        logits, _ = self.model(torch.from_numpy(obs).to(device), torch.from_numpy(heard).to(device))
        if self.cic:
            self._influence(torch.from_numpy(obs).to(device), torch.from_numpy(heard).to(device), alive, logits)
        return torch.distributions.Categorical(logits=logits).sample() if self.sample else logits.argmax(-1)

    @torch.no_grad()
    def _influence(self, obs: torch.Tensor, heard: torch.Tensor, alive: np.ndarray, logits: torch.Tensor):
        E, N = alive.shape
        if E < 2:
            return
        eye = torch.eye(N, dtype=torch.bool, device=heard.device)
        for i in range(N):
            got = (heard[:, i] & ~eye[i]).any(-1).cpu().numpy() & alive[:, i]
            senders = [k for k in range(N) if k != i]
            donors = np.flatnonzero(alive[:, senders].all(-1))
            rows = [e for e in np.flatnonzero(got) if len(donors[donors != e])]
            if not rows:
                continue
            mute = heard.clone()
            mute[:, i] = mute[:, i] & eye[i]
            silent = self.model(obs, mute)[0][:, i].softmax(-1)
            others = []
            for _ in range(self.cic):
                d = torch.as_tensor([self._rng.choice(donors[donors != e]) if e in rows else e for e in range(E)],
                                    device=obs.device)
                swapped = obs.clone()
                swapped[:, senders] = obs[d][:, senders]
                others.append(self.model(swapped, heard)[0][:, i].softmax(-1))
            probs = torch.stack([logits[:, i].softmax(-1), *others], 1)
            for e in rows:
                self.influence.note(probs[e], silent[e])


def goal_embeddings(jepa, h5, episodes) -> dict:
    from mappo_baseline import embed
    out = {}
    with h5py.File(h5, 'r') as f:
        firsts = sorted(int(o) for o, _ in episodes)
        for k in range(0, len(firsts), 256):
            chunk = firsts[k:k + 256]
            images = np.stack([f['goal'][o] for o in chunk])
            for o, e in zip(chunk, embed(jepa, images)):
                out[o] = e
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="baselines/commnet")
def main(cfg: DictConfig):
    from gcrl_baseline import cache_frames
    from mappo_baseline import load_jepa
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    if cfg.device == 'cuda' and device != 'cuda':
        raise RuntimeError("device=cuda, but no GPU is visible")
    frames, meta = cache_frames(to_absolute_path(cfg.cache))
    made_with = (meta.get('checkpoints') or {}).get('jepa')
    if made_with and Path(made_with).name != Path(cfg.checkpoints.jepa).name:
        raise ValueError(f"the cache's features are {made_with}'s, not checkpoints.jepa's ({cfg.checkpoints.jepa})")
    jepa = load_jepa(cfg, device)
    h5 = Path(to_absolute_path(cfg.data_dir or meta['data_dir'])) / 'dataset.h5'
    frames = np.asarray(frames)
    with h5py.File(h5, 'r') as f:
        offsets, lens, layout = f['ep_offset'][:], f['ep_len'][:], f['layout']
        episodes = [(int(o), int(n)) for o, n in zip(offsets, lens) if cfg.layouts[0] <= int(layout[int(o), 0]) < cfg.layouts[1]]
    goals = goal_embeddings(jepa, h5, episodes)
    data = Samples(h5, frames, goals, tuple(cfg.layouts), cfg.hindsight_prob, cfg.max_goal_steps, labels=cfg.labels,
                   actions=cfg.actions)
    obs_dim = data[0]['obs'].shape[-1]
    model = ViTCommNet(obs_dim, num_actions=cfg.actions, **OmegaConf.to_container(cfg.model)).to(device)
    logger.info(f"CommNet ({model.message_bits} bits a message, {cfg.labels} labels) on {data.episodes} episodes of layouts {list(cfg.layouts)}: "
                f"{len(data)} samples, {obs_dim} features, {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters")
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"commnet-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    extra = dict(layouts=list(cfg.layouts), jepa=str(cfg.checkpoints.jepa), cache=str(cfg.cache))
    sampler = torch.utils.data.RandomSampler(data, replacement=True, num_samples=cfg.steps * cfg.batch_size)
    loader = DataLoader(data, cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, drop_last=True,
                        worker_init_fn=lambda w: np.random.seed(cfg.seed * 1000 + w))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    totals = {}
    model.train()
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        mask = comm_mask(batch['alive'], None, None, model.drop_prob)
        logits, _ = model(batch['obs'], mask)
        labels = batch['labels']
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        valid = labels != IGNORE_INDEX
        for k, v in (('loss', float(loss)), ('accuracy', float((logits.argmax(-1) == labels)[valid].float().mean()))):
            totals[k] = totals.get(k, 0.0) + v
        if step % cfg.log_every == 0 or step == cfg.steps:
            n = step % cfg.log_every or cfg.log_every
            row = {'step': step, **{k: v / n for k, v in totals.items()}}
            totals = {}
            logger.info({k: round(v, 4) for k, v in row.items()})
            if run is not None:
                run.log(row, step=step)
        if step % cfg.save_every == 0 or step == cfg.steps:
            model.save(out / 'commnet.pt', steps=step, **extra)
    logger.info(f"saved {out / 'commnet.pt'}")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
