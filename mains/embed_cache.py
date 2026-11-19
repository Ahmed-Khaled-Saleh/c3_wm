import json
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from numpy.lib.format import open_memmap
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from c3_wm.data.datasets import FindGoalWorldModelDataset
from c3_wm.data.transforms import get_transforms
from c3_wm.trainers.wm_tune import encode_tokens
from goal_head import Rollouts, embeddings, imagine
from message_gain import load_models


def picks_of(n: int, windows, seed: int, part: int, parts: int) -> np.ndarray:
    picks = np.arange(n) if windows is None else np.sort(np.random.default_rng(seed).choice(n, min(windows, n), replace=False))
    return picks[part * len(picks) // parts:(part + 1) * len(picks) // parts]


@torch.no_grad()
def write(jepa, ds: Rollouts, picks: np.ndarray, out: Path, history_size: int, batch_size: int, num_workers: int, device):
    out.mkdir(parents=True, exist_ok=True)
    W, arrays = len(picks), {}

    def array(name, shape, dtype):
        if name not in arrays:
            arrays[name] = open_memmap(out / f'{name}.npy', mode='w+', dtype=dtype, shape=(W, *shape))
        return arrays[name]

    np.save(out / 'window.npy', picks)
    np.save(out / 'index.npy', ds.ds.index[picks])
    loader = DataLoader(Subset(ds, picks), batch_size=batch_size, num_workers=num_workers)
    at = 0
    for k, batch in enumerate(loader):
        batch = {key: v.to(device) for key, v in batch.items()}
        seen, imagined, goal = embeddings(jepa, batch, history_size)
        imagined_random = imagine(jepa, batch, history_size, batch['random_actions'])
        B = len(seen)
        for name, v, dtype in (('seen', seen, np.float16), ('imagined', imagined, np.float16), ('goal', goal[:, 0], np.float16),
                               ('label', batch['label'], np.int16), ('moves', batch['moves'], np.int16),
                               ('tiles', batch['tiles'], np.uint8),
                               ('valid', batch['valid'], np.bool_), ('actions', batch['actions'], np.int8),
                               ('imagined_random', imagined_random, np.float16),
                               ('random_actions', batch['random_actions'], np.int8),
                               ('random_moves', batch['random_moves'], np.int16), ('random_valid', batch['random_valid'], np.bool_)):
            array(name, v.shape[1:], dtype)[at:at + B] = v.cpu().numpy().astype(dtype)
        at += B
        if k % 50 == 0:
            logger.info(f"{out.name}: {at}/{W} windows")
    for a in arrays.values():
        a.flush()
    return W


class Frames(Dataset):

    def __init__(self, h5: str, lo: int, hi: int, transform):
        self.h5, self.lo, self.hi, self.transform, self.file = h5, lo, hi, transform, None

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, i):
        if self.file is None:
            self.file = h5py.File(self.h5, 'r')
        return torch.stack([self.transform(p) for p in self.file['pixels'][self.lo + i]])


@torch.no_grad()
def write_frames(jepa, h5: str, lo: int, hi: int, out: Path, transform, batch_size: int, num_workers: int, device):
    out.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(Frames(h5, lo, hi, transform), batch_size=batch_size, num_workers=num_workers)
    frames, at = None, 0
    for k, pixels in enumerate(loader):
        emb = jepa.encode({'pixels': pixels.to(device)})['emb']
        if frames is None:
            frames = open_memmap(out / 'frames.npy', mode='w+', dtype=np.float16, shape=(hi - lo, *emb.shape[1:]))
        frames[at:at + len(emb)] = emb.cpu().numpy().astype(np.float16)
        at += len(emb)
        if k % 50 == 0:
            logger.info(f"frames: {lo + at}/{hi} rows")
    frames.flush()
    np.save(out / 'rows.npy', np.array([lo, hi]))


@hydra.main(version_base=None, config_path="../configs", config_name="analysis/embed_cache")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    if not 0 <= cfg.part < cfg.parts:
        raise ValueError(f"part {cfg.part} of {cfg.parts}")
    jepa, vqvae = load_models(cfg, device)
    h5 = str(Path(to_absolute_path(cfg.data_dir)) / 'dataset.h5')
    tokens = encode_tokens(h5, vqvae, device=device, num_workers=cfg.num_workers)
    T = cfg.pipeline.history_size
    out = Path(to_absolute_path(cfg.out_dir)) / f'part_{cfg.part}'
    counts = {}
    for split, windows, seed in (('train', cfg.train_windows, cfg.seed), ('val', cfg.val_windows, cfg.seed + 1)):
        ds = Rollouts(FindGoalWorldModelDataset(
            h5, split, tuple(cfg.val_layouts), history_size=T, num_preds=cfg.rollout_steps, tokens=tokens,
            max_age=jepa.predictor.max_age, transform=get_transforms(img_size=cfg.pipeline.img_size, model='JEPA')[1]),
            seed=seed)
        counts[split] = write(jepa, ds, picks_of(len(ds), windows, seed, cfg.part, cfg.parts), out / split, T,
                              cfg.batch_size, cfg.num_workers, device)
    rows = None
    if cfg.frames:
        with h5py.File(h5, 'r') as f:
            n = len(f['pixels'])
        rows = [cfg.part * n // cfg.parts, (cfg.part + 1) * n // cfg.parts]
        write_frames(jepa, h5, *rows, out, get_transforms(img_size=cfg.pipeline.img_size, model='JEPA')[1],
                     cfg.batch_size, cfg.num_workers, device)
    meta = {'checkpoints': OmegaConf.to_container(cfg.checkpoints), 'data_dir': str(to_absolute_path(cfg.data_dir)),
            'val_layouts': list(cfg.val_layouts), 'history_size': T, 'rollout_steps': cfg.rollout_steps,
            'embed_dim': cfg.pipeline.embed_dim, 'train_windows': cfg.train_windows, 'val_windows': cfg.val_windows,
            'seed': cfg.seed, 'part': cfg.part, 'parts': cfg.parts, 'windows': counts, 'frames': bool(cfg.frames), 'rows': rows}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    logger.info(f"cached {counts} windows in {out}")


if __name__ == '__main__':
    main()
