import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from c3_wm.data.datasets import TILE_NAMES, FindGoalWorldModelDataset
from c3_wm.data.transforms import get_transforms
from c3_wm.models.jepa import latest_messages
from c3_wm.trainers.wm_tune import encode_tokens

SITUATIONS = ('no message', 'goal in own view', 'goal only in a message', 'partner in own view', 'neither')
GOAL, PARTNER = TILE_NAMES.index('goal'), TILE_NAMES.index('partner')


def situation(ds: FindGoalWorldModelDataset, i: int) -> int:
    f, (r, a) = ds.file, ds.index[i]
    T, span, max_age = ds.history_size, ds.history_size + ds.num_preds, ds.max_age
    own = f['image'][r:r + span, a, :, :, 0].astype(np.int64)
    own[:, own.shape[1] // 2, -1] = -1
    senders = [j for j in range(ds.agents) if j != a]
    rows = np.arange(r - max_age, r + T)
    inside = rows >= ds.episode_start[r]
    rows = np.maximum(rows, ds.episode_start[r])
    sent = inside[:, None] & ~ds.done[rows][:, senders]
    read = set()
    for t in range(max_age, max_age + T):
        for s in range(len(senders)):
            k = np.flatnonzero(sent[:t + 1, s])
            if len(k) and t - k[-1] <= max_age:
                read.add((int(rows[k[-1]]), senders[s]))
    if not read:
        return SITUATIONS.index('no message')
    if (own == GOAL).any():
        return SITUATIONS.index('goal in own view')
    if any((f['image'][row, s, :, :, 0] == GOAL).any() for row, s in read):
        return SITUATIONS.index('goal only in a message')
    if (own == PARTNER).any():
        return SITUATIONS.index('partner in own view')
    return SITUATIONS.index('neither')


class Situated(Dataset):

    def __init__(self, ds, picks):
        self.ds, self.picks = ds, picks

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, j):
        i = int(self.picks[j])
        item = self.ds[i]
        return {**item, 'index': i, 'situation': situation(self.ds, i)}


def load_models(cfg, device):
    jepa = hydra.utils.instantiate({k: v for k, v in OmegaConf.to_container(cfg.jepa, resolve=True).items()
                                    if k != 'checkpoint_path'})
    jepa.load_state_dict(torch.load(to_absolute_path(cfg.checkpoints.jepa), map_location='cpu')['wm_model_state_dict'])
    vqvae = hydra.utils.instantiate({k: v for k, v in OmegaConf.to_container(cfg.vqvae, resolve=True).items()
                                     if k != 'checkpoint_path'})
    vqvae.load_state_dict(torch.load(to_absolute_path(cfg.checkpoints.vqvae), map_location='cpu')['model_state_dict'])
    return jepa.to(device).eval(), vqvae.to(device).eval()


@torch.no_grad()
def losses(jepa, loader, history_size: int, num_preds: int, device) -> dict:
    T, out = history_size, {k: [] for k in ('index', 'situation', 'loss_messages', 'loss_alone')}
    for n, batch in enumerate(loader):
        out['index'].append(batch.pop('index').numpy())
        out['situation'].append(batch.pop('situation').numpy())
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch['pixels'].shape[0]
        enc = jepa.encode(batch)
        emb, act = enc['emb'], enc['act_emb'].reshape(B, T, -1)
        target = emb[:, num_preds:]
        for key, received in (('loss_messages', batch['msg_sent']), ('loss_alone', torch.zeros_like(batch['msg_sent']))):
            messages = latest_messages(batch['msg_tokens'], received, batch['msg_sender'], batch['msg_receiver'],
                                       steps=T, max_age=jepa.predictor.max_age)
            pred = jepa.predict(emb[:, :T], act[:, :T], messages)
            out[key].append((pred - target).pow(2).mean((1, 2)).cpu().numpy())
        if n % 20 == 0:
            logger.info(f"batch {n + 1}/{len(loader)}")
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(w: dict) -> dict:
    def stats(m):
        if not m.any():
            return {'windows': 0}
        gain = w['loss_alone'][m] - w['loss_messages'][m]
        return {'windows': int(m.sum()), 'loss_messages': float(w['loss_messages'][m].mean()),
                'loss_alone': float(w['loss_alone'][m].mean()), 'gain': float(gain.mean()),
                'gain_se': float(gain.std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else None,
                'relative_gain': float(gain.mean() / w['loss_alone'][m].mean())}
    return {'all': stats(np.ones(len(w['situation']), bool)),
            **{name: stats(w['situation'] == k) for k, name in enumerate(SITUATIONS)}}


@hydra.main(version_base=None, config_path="../configs", config_name="analysis/message_gain")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    jepa, vqvae = load_models(cfg, device)
    h5 = str(Path(to_absolute_path(cfg.data_dir)) / 'dataset.h5')
    tokens = encode_tokens(h5, vqvae, device=device, num_workers=cfg.num_workers)
    ds = FindGoalWorldModelDataset(h5, 'val', tuple(cfg.val_layouts), history_size=cfg.pipeline.history_size,
                                   num_preds=cfg.pipeline.num_preds, tokens=tokens, max_age=jepa.predictor.max_age,
                                   transform=get_transforms(img_size=cfg.pipeline.img_size, model='JEPA')[1])
    n = len(ds) if cfg.windows is None else min(cfg.windows, len(ds))
    picks = np.sort(np.random.default_rng(cfg.seed).choice(len(ds), n, replace=False))
    loader = DataLoader(Situated(ds, picks), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    w = losses(jepa, loader, cfg.pipeline.history_size, cfg.pipeline.num_preds, device)
    results = summarize(w)
    for name, s in results.items():
        logger.info(f"{name:24s} " + (f"windows {s['windows']:6d}  messages {s['loss_messages']:.4f}  alone {s['loss_alone']:.4f}  "
                                      f"gain {s['gain']:.4f} ± {s['gain_se'] or 0:.4f} ({100 * s['relative_gain']:.0f}%)"
                                      if s['windows'] else 'windows      0'))
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps({'situations': results, 'checkpoints': OmegaConf.to_container(cfg.checkpoints)},
                                                 indent=1))
    np.savez_compressed(out / 'windows.npz', **w, situations=np.array(SITUATIONS))
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"message-gain-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, job_type='analysis', dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
        columns = ['situation', 'windows', 'loss_messages', 'loss_alone', 'gain', 'gain_se', 'relative_gain']
        run.log({'message_gain/situations': wandb.Table(columns=columns, data=[[name] + [s.get(c) for c in columns[1:]]
                                                                                for name, s in results.items()])})
        run.summary.update({f'message_gain/{name}/{k}': v for name, s in results.items() for k, v in s.items()})
        run.finish()
    logger.info(f"results in {out / 'results.json'}")


if __name__ == '__main__':
    main()
