import os
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from gcrl_baseline import resize
from stable_marl.wm.omnivla import OmniVLAEdge

COLUMNS = ('position', 'direction', 'goal_position', 'action', 'terminated')
FROZEN = {'all': (), 'after_vision': ('obs_encoder', 'goal_encoder_img', 'compress_obs_enc', 'compress_goal_enc_img'),
          'head': None}


def freeze(model, trainable: str) -> int:
    if trainable not in FROZEN:
        raise ValueError(f"trainable: one of {list(FROZEN)}, not {trainable!r}")
    frozen = FROZEN[trainable]
    is_frozen = (lambda n: not n.startswith('action_head')) if frozen is None else (lambda n: n.startswith(frozen))
    for name, p in model.named_parameters():
        p.requires_grad_(not is_frozen(name))
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and is_frozen(name + '.'):
            m.momentum = 0.0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GoalImages:

    def __init__(self, images: np.ndarray, episode_of_row: np.ndarray):
        self.images, self.episode_of_row = images, episode_of_row

    def __getitem__(self, index):
        row, agent = index
        return self.images[self.episode_of_row[row], agent]


class Episodes:

    def __init__(self, path, layouts, size: int, actions: int | None = None):
        lo, hi = layouts
        cols, goals, lengths = {k: [] for k in (*COLUMNS, 'pixels', 'state')}, [], []
        with h5py.File(path, 'r') as f:
            offsets, lens, layout = f['ep_offset'][:], f['ep_len'][:], f['layout']
            keep = [(int(o), int(n)) for o, n in zip(offsets, lens) if lo <= int(layout[int(o), 0]) < hi]
            for k, (o, n) in enumerate(keep):
                for c in COLUMNS:
                    cols[c].append(f[c][o:o + n])
                cols['pixels'].append(resize(f['pixels'][o:o + n], size))
                cols['state'].append(f['state'][o:o + n].astype(np.int8))
                goals.append(resize(f['goal'][o], size))
                lengths.append(n)
                if (k + 1) % 500 == 0:
                    logger.info(f"read {k + 1}/{len(keep)} episodes")
        self.cols = {c: np.concatenate(v) for c, v in cols.items()}
        if actions is not None:
            from stable_marl.wm.omnivla.core import IGNORE_INDEX
            action = self.cols['action']
            self.cols['action'] = np.where(action < actions, action, IGNORE_INDEX).astype(np.int64)
        self.lengths = np.array(lengths)
        self.offsets = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.cols['goal'] = GoalImages(np.stack(goals), np.repeat(np.arange(len(lengths)), self.lengths))
        self.episodes, self.skipped = len(keep), len(offsets) - len(keep)

    def get_col_data(self, col: str):
        return self.cols[col]


class ViTOmniVLA(OmniVLAEdge):

    def __init__(self, num_actions: int = 3, context_size: int = 5, horizon: int = 8, encoding_size: int = 1024,
                 num_heads: int = 4, num_layers: int = 4, ff_factor: int = 4, image_size: int = 96, dropout: float = 0.2,
                 emb_dim: int = 192):
        super().__init__(num_actions, context_size, horizon, encoding_size, num_heads, num_layers, ff_factor, image_size,
                         dropout)
        del self.obs_encoder, self.goal_encoder_img
        self.compress_obs_enc = torch.nn.Linear(emb_dim, encoding_size)
        self.compress_goal_enc_img = torch.nn.Linear(2 * emb_dim, encoding_size)
        self.config['emb_dim'] = emb_dim

    def forward(self, obs, goal_pose, goal_image, goal_mask):
        B, T = obs.shape[:2]
        obs, goal = obs.float(), goal_image.float()
        obs_tokens = self.compress_obs_enc(self.feature_dropout(obs))
        image_token = self.compress_goal_enc_img(self.feature_dropout(torch.cat([obs[:, -1], goal], -1)))
        tokens = torch.cat([obs_tokens, self.local_goal(goal_pose.float()).unsqueeze(1), image_token.unsqueeze(1)], 1)
        goal_mask = goal_mask.bool()
        padding = torch.cat([torch.zeros(B, T, dtype=torch.bool, device=obs.device), ~goal_mask], 1)
        modality_id = torch.where(goal_mask[:, 0], torch.where(goal_mask[:, 1], 5, 4), 6)
        x = self.decoder(tokens, self.token_positions, padding, modality_id)
        return self.action_head(x).view(B, self.horizon, self.num_actions), self.dist_predictor(x).squeeze(-1)

    @torch.no_grad()
    def get_action(self, info: dict) -> torch.Tensor:
        emb = torch.as_tensor(info['pixels'])[:, -1].float()
        E, A = emb.shape[:2]
        flush = info.get('_needs_flush')
        flush = torch.ones(E, dtype=torch.bool) if flush is None else torch.as_tensor(flush, dtype=torch.bool)
        if self._history is None or self._history.shape[:2] != (E, A) or self._history.device != emb.device:
            self._history, flush = emb.unsqueeze(2).repeat(1, 1, self.context_size + 1, 1), torch.ones(E, dtype=torch.bool)
        self._history = torch.cat([self._history[:, :, 1:], emb.unsqueeze(2)], 2)
        flush = flush.to(emb.device)
        self._history[flush] = emb[flush].unsqueeze(2).expand_as(self._history[flush])
        goal = torch.as_tensor(info['goal'])[:, -1].float().to(emb.device)
        mask = torch.tensor([False, True], device=emb.device).expand(E * A, 2)
        logits, _ = self(self._history.flatten(0, 1), torch.zeros(E * A, 4, device=emb.device), goal.flatten(0, 1), mask)
        logits = logits[:, 0]
        if self.temperature > 0:
            action = torch.distributions.Categorical(logits=logits / self.temperature).sample()
        else:
            action = logits.argmax(-1)
        return action.view(E, A)


class ViTEpisodes(Episodes):

    def __init__(self, path, layouts, frames, goal_emb: dict, actions: int | None = None):
        lo, hi = layouts
        cols, goals, lengths = {k: [] for k in (*COLUMNS, 'pixels', 'state')}, [], []
        with h5py.File(path, 'r') as f:
            offsets, lens, layout = f['ep_offset'][:], f['ep_len'][:], f['layout']
            keep = [(int(o), int(n)) for o, n in zip(offsets, lens) if lo <= int(layout[int(o), 0]) < hi]
            for o, n in keep:
                for c in COLUMNS:
                    cols[c].append(f[c][o:o + n])
                cols['pixels'].append(np.asarray(frames[o:o + n]))
                cols['state'].append(f['state'][o:o + n].astype(np.int8))
                goals.append(np.asarray(goal_emb[o], np.float32))
                lengths.append(n)
        self.cols = {c: np.concatenate(v) for c, v in cols.items()}
        if actions is not None:
            from stable_marl.wm.omnivla.core import IGNORE_INDEX
            action = self.cols['action']
            self.cols['action'] = np.where(action < actions, action, IGNORE_INDEX).astype(np.int64)
        self.lengths = np.array(lengths)
        self.offsets = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.cols['goal'] = GoalImages(np.stack(goals), np.repeat(np.arange(len(lengths)), self.lengths))
        self.episodes, self.skipped = len(keep), len(offsets) - len(keep)


def omnivla_from(path, map_location='cpu'):
    config = torch.load(path, map_location='cpu', weights_only=True)['config']
    return (ViTOmniVLA if 'emb_dim' in config else OmniVLAEdge).load(path, map_location=map_location)


class OmniVLAActor:

    def __init__(self, model, jepa=None):
        self.model, self.jepa = model, jepa
        model.goal_modalities = ('image',)

    def eval(self):
        return self

    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def get_action(self, info: dict):
        info = dict(info)
        as_np = lambda v: v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
        device = next(self.model.parameters()).device
        for key in ('pixels', 'goal'):
            if self.jepa is not None:
                from mappo_baseline import embed
                info[key] = torch.from_numpy(embed(self.jepa, as_np(info[key])[:, -1])[:, None]).to(device)
            else:
                info[key] = torch.from_numpy(resize(as_np(info[key])[:, -1:], self.model.image_size)).to(device)
        return self.model.get_action(info)


@hydra.main(version_base=None, config_path="../configs", config_name="baselines/omnivla")
def main(cfg: DictConfig):
    from stable_marl.wm.omnivla import OmniVLADataset, train_omnivla
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    if cfg.device == 'cuda' and device != 'cuda':
        raise RuntimeError("device=cuda, but no GPU is visible")
    if cfg.features not in ('pixels', 'vit'):
        raise ValueError(f"features: pixels or vit, not {cfg.features!r}")
    vit = cfg.features == 'vit'
    if vit and cfg.trainable != 'all':
        raise ValueError("features=vit: trainable=all (every part but our frozen ViT, which is outside the model, learns)")
    model = ViTOmniVLA(num_actions=cfg.actions, emb_dim=cfg.pipeline.embed_dim, **OmegaConf.to_container(cfg.model)) if vit \
        else OmniVLAEdge(num_actions=cfg.actions, **OmegaConf.to_container(cfg.model))
    if cfg.pretrained:
        loaded, fresh = model.load_upstream(to_absolute_path(cfg.pretrained))
        logger.info(f"from the released OmniVLA-edge: {len(loaded)} tensors loaded; left as initialized: {fresh}")
    learning = freeze(model, cfg.trainable)
    logger.info(f"trainable={cfg.trainable}: {learning / 1e6:.2f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters learn")
    h5 = Path(to_absolute_path(cfg.data_dir)) / 'dataset.h5'
    if vit:
        from commnet_baseline import goal_embeddings
        from gcrl_baseline import cache_frames
        from mappo_baseline import load_jepa
        if not cfg.cache:
            raise ValueError("features=vit reads embed_cache.py's frames: cache=<its run folder>")
        frames, meta = cache_frames(to_absolute_path(cfg.cache))
        made_with = (meta.get('checkpoints') or {}).get('jepa')
        if made_with and Path(made_with).name != Path(cfg.checkpoints.jepa).name:
            raise ValueError(f"the cache's features are {made_with}'s, not checkpoints.jepa's ({cfg.checkpoints.jepa})")
        if meta.get('data_dir') and Path(meta['data_dir']).resolve() != h5.parent.resolve():
            raise ValueError(f"the cache was made from {meta['data_dir']}, not data_dir ({cfg.data_dir}): data_dir=<that>")
        with h5py.File(h5, 'r') as f:
            offsets, lens, layout = f['ep_offset'][:], f['ep_len'][:], f['layout']
            lo, hi = cfg.layouts
            kept = [(int(o), int(n)) for o, n in zip(offsets, lens) if lo <= int(layout[int(o), 0]) < hi]
        goal_emb = goal_embeddings(load_jepa(cfg, device), h5, kept)
        episodes = ViTEpisodes(h5, tuple(cfg.layouts), frames, goal_emb, cfg.actions)
    else:
        episodes = Episodes(h5, tuple(cfg.layouts), cfg.model.image_size, cfg.actions)
    data = OmniVLADataset(episodes, cfg.model.context_size, cfg.model.horizon, labels=cfg.labels,
                          hindsight_prob=cfg.hindsight_prob, max_goal_steps=cfg.max_goal_steps, modalities=['image'])
    logger.info(f"OmniVLA-edge ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, {cfg.labels} labels) on {episodes.episodes} "
                f"episodes of layouts {list(cfg.layouts)} ({episodes.skipped} others left out): {len(data)} samples")
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"omnivla-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))

    def log(row):
        logger.info({k: round(v, 4) for k, v in row.items()})
        if run is not None:
            run.log(row, step=int(row['step']))
        if row['step'] % cfg.save_every == 0:
            model.save(out / 'omnivla.pt')
    train = OmegaConf.to_container(cfg.train)
    model = train_omnivla(model, data, device=device, log=log, **train)
    model.save(out / 'omnivla.pt')
    logger.info(f"saved {out / 'omnivla.pt'}")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
