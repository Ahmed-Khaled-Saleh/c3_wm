import json
import os
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def resize(images, size: int) -> np.ndarray:
    x = torch.as_tensor(np.asarray(images))
    lead, (h, w, c) = x.shape[:-3], x.shape[-3:]
    if (h, w) == (size, size):
        return x.numpy()
    x = x.reshape(-1, h, w, c).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode='area')
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).reshape(*lead, size, size, c).numpy()


class EpisodeColumns:

    def __init__(self, path, layouts, size: int, frames: np.ndarray | None = None):
        lo, hi = layouts
        columns = ('pixels' if frames is None else 'view_emb', 'action', 'terminated')
        cols, lengths = {k: [] for k in columns}, []
        with h5py.File(path, 'r') as f:
            offsets, lens, layout = f['ep_offset'][:], f['ep_len'][:], f['layout']
            keep = [(int(o), int(n)) for o, n in zip(offsets, lens) if lo <= int(layout[int(o), 0]) < hi]
            for k, (o, n) in enumerate(keep):
                for c in columns:
                    if c == 'view_emb':
                        cols[c].append(np.asarray(frames[o:o + n], np.float32))
                        continue
                    v = f[c][o:o + n]
                    cols[c].append(resize(v, size) if c == 'pixels' else v)
                lengths.append(n)
                if (k + 1) % 500 == 0:
                    logger.info(f"read {k + 1}/{len(keep)} episodes")
        self.cols = {c: np.concatenate(v) for c, v in cols.items()}
        self.lengths = np.array(lengths)
        self.offsets = np.r_[0, np.cumsum(self.lengths)[:-1]]
        self.episodes, self.skipped = len(keep), len(offsets) - len(keep)

    def get_col_data(self, col: str) -> np.ndarray:
        return self.cols[col]


def cache_frames(cache) -> tuple[np.ndarray, dict]:
    parts = sorted(Path(cache).glob('part_*/frames.npy'), key=lambda p: int(np.load(p.parent / 'rows.npy')[0]))
    if not parts:
        raise FileNotFoundError(f"no part_*/frames.npy in {cache}: embed_cache.py with frames=true")
    spans = [np.load(p.parent / 'rows.npy') for p in parts]
    if any(a[1] != b[0] for a, b in zip(spans, spans[1:])) or spans[0][0] != 0:
        raise ValueError(f"{cache}: the parts' rows {[list(s) for s in spans]} do not join from 0")
    meta = json.loads((parts[0].parent / 'meta.json').read_text())
    return np.concatenate([np.load(p, mmap_mode='r') for p in parts]), meta


class GCRLActor:

    def __init__(self, model, size: int | None = None, jepa=None):
        self.model, self.size, self.jepa = model, size, jepa

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def get_action(self, info: dict) -> np.ndarray:
        last = lambda v: (v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v))[:, -1:]
        if self.jepa is not None:
            from mappo_baseline import embed
            return self.model.get_action({'view_emb': embed(self.jepa, last(info['pixels'])),
                                          'goal_view_emb': embed(self.jepa, last(info['goal']))})
        return self.model.get_action({k: resize(last(info[k]), self.size) for k in ('pixels', 'goal')})


@hydra.main(version_base=None, config_path="../configs", config_name="baselines/gcrl")
def main(cfg: DictConfig):
    from stable_marl.wm.gcrl import IndependentGCRL, agent_datasets
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    frames, meta, key = None, {}, 'pixels'
    if cfg.features == 'vit':
        if not cfg.cache:
            raise ValueError("features=vit reads embed_cache.py's frames: cache=<its run folder>")
        frames, meta = cache_frames(to_absolute_path(cfg.cache))
        key = 'view_emb'
    elif cfg.features != 'pixels':
        raise ValueError(f"features: pixels or vit, not {cfg.features!r}")
    data_dir = meta.get('data_dir') if frames is not None and cfg.data_dir is None else cfg.data_dir
    h5 = Path(to_absolute_path(data_dir)) / 'dataset.h5'
    data = EpisodeColumns(h5, tuple(cfg.layouts), cfg.image_size, frames)
    per_agent = agent_datasets(data, key)
    logger.info(f"{data.episodes} episodes of layouts {list(cfg.layouts)} ({data.skipped} others left out), "
                f"transitions per agent {[len(d['actions']) for d in per_agent]}, views {per_agent[0]['observations'].shape[1:]}")
    model = IndependentGCRL.create(cfg.agent, per_agent, cfg.num_actions, obs_keys=key, seed=cfg.seed,
                                   temperature=cfg.temperature, **OmegaConf.to_container(cfg.overrides))
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"gcrl-{cfg.agent}-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))

    def log(row):
        logger.info(f"step {row['step']}: " + ', '.join(f"{k} {v:.4f}" for k, v in row.items() if k.endswith('loss')))
        if run is not None:
            run.log(row, step=row['step'])

    history = model.train(per_agent, cfg.steps, log_every=cfg.log_every, seed=cfg.seed, log=log)
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    path = model.save(out / 'gcrl.pkl')
    (out / 'history.json').write_text(json.dumps({'history': history, 'image_size': cfg.image_size, 'agent': cfg.agent,
                                                  'layouts': list(cfg.layouts), 'episodes': data.episodes,
                                                  'features': cfg.features, 'cache': cfg.cache,
                                                  'jepa': (meta.get('checkpoints') or {}).get('jepa')}, indent=1))
    if run is not None:
        run.finish()
    logger.info(f"saved {path}")


if __name__ == '__main__':
    main()
