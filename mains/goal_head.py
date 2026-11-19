import json
import os
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, RandomSampler

from c3_wm.data.datasets import TILE_NAMES, FindGoalWorldModelDataset
from c3_wm.data.transforms import get_transforms
from c3_wm.evaluators.deploy import CostToGoHead, GoalHead, TileHead
from c3_wm.models.jepa import latest_messages
from c3_wm.trainers.wm_tune import encode_tokens
from message_gain import load_models

GOAL, WALL, ABSENT = TILE_NAMES.index('goal'), TILE_NAMES.index('wall'), GoalHead.VIEW ** 2
NEAR = 10


def fewest_moves(walls: np.ndarray, goal) -> np.ndarray:
    moves = np.full(walls.shape, -1, np.int64)
    moves[tuple(goal)] = 0
    queue = deque([tuple(int(v) for v in goal)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < walls.shape[0] and 0 <= ny < walls.shape[1] and not walls[nx, ny] and moves[nx, ny] < 0:
                moves[nx, ny] = moves[x, y] + 1
                queue.append((nx, ny))
    return moves


HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def walk(walls: np.ndarray, field: np.ndarray, cell, heading: int, actions) -> tuple:
    (x, y), moves, used, arrived = (int(v) for v in cell), [], [], field[tuple(cell)] == 0
    for a in actions:
        heading = (heading - 1) % 4 if a == 0 else (heading + 1) % 4 if a == 1 else heading
        if a == 2:
            nx, ny = x + HEADINGS[heading][0], y + HEADINGS[heading][1]
            if 0 <= nx < walls.shape[0] and 0 <= ny < walls.shape[1] and not walls[nx, ny]:
                x, y = nx, ny
        moves.append(field[x, y])
        used.append(not arrived and field[x, y] >= 0)
        arrived |= field[x, y] == 0
    return np.array(moves), np.array(used)


class Rollouts(Dataset):

    def __init__(self, ds: FindGoalWorldModelDataset, seed: int = 0):
        self.ds, self.seed = ds, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        f, (r, a) = self.ds.file, self.ds.index[i]
        n = self.ds.history_size + self.ds.num_preds
        act = f['action'][r:r + n - 1, a]
        acting = np.isfinite(act) & ~f['terminated'][r:r + n - 1, a]
        valid = np.concatenate([[True], np.cumprod(acting).astype(bool)]) & ~f['terminated'][r:r + n, a]
        kinds = f['image'][r:r + n, a, :, :, 0]
        goal = kinds == GOAL
        label = np.where(goal.reshape(n, -1).any(-1), goal.reshape(n, -1).argmax(-1), ABSENT)
        field = fewest_moves(f['state'][r, :, :, 0] == WALL, f['goal_position'][r, a])
        cells = f['position'][r:r + n, a]
        image = f['goal'][r, a]
        T = self.ds.history_size
        plan = np.random.default_rng([self.seed, int(i)]).integers(0, 3, self.ds.num_preds)
        random_moves, random_valid = walk(f['state'][r, :, :, 0] == WALL, field, cells[T - 1],
                                          int(f['direction'][r + T - 1, a]), plan)
        return {**item, 'actions': torch.as_tensor(np.nan_to_num(act), dtype=torch.long),
                'random_actions': torch.as_tensor(np.r_[np.nan_to_num(act[:T - 1]), plan], dtype=torch.long),
                'random_moves': torch.as_tensor(random_moves, dtype=torch.float32),
                'random_valid': torch.as_tensor(random_valid & valid[T - 1]),
                'label': torch.as_tensor(label), 'valid': torch.as_tensor(valid),
                'tiles': torch.as_tensor(np.swapaxes(kinds, -1, -2).reshape(n, -1), dtype=torch.long),
                'moves': torch.as_tensor(field[cells[:, 0], cells[:, 1]], dtype=torch.float32),
                'goal': self.ds.transform(image) if self.ds.transform else torch.from_numpy(image).permute(2, 0, 1).float() / 255}


@torch.no_grad()
def embeddings(jepa, batch, history_size: int):
    T = history_size
    seen = jepa.encode({'pixels': batch['pixels']})['emb']
    imagined = imagine(jepa, batch, T, batch['actions'])
    return seen, imagined, jepa.encode({'pixels': batch['goal'][:, None]})['emb']


@torch.no_grad()
def imagine(jepa, batch, history_size: int, actions):
    T = history_size
    messages = latest_messages(batch['msg_tokens'], batch['msg_sent'], batch['msg_sender'], batch['msg_receiver'],
                               steps=T, max_age=jepa.predictor.max_age)
    info = {'pixels': batch['pixels'][:, None, :T], 'messages': {k: v[:, None] for k, v in messages.items()}}
    return jepa.rollout(info, actions[:, None], history_size=T)['predicted_emb'][:, 0, T:]


def targets(batch, kind: str):
    if kind == 'goal_location':
        return batch['label'], batch['valid']
    if kind == 'tiles':
        return batch['tiles'], batch['valid']
    return batch['moves'], batch['valid'] & (batch['moves'] >= 0)


def outputs(head, emb, goal_emb, kind: str):
    return head(emb, goal_emb) if kind == 'cost_to_go' else head(emb)


def loss_of(out, y, kind: str):
    if not len(y):
        return out.sum() * 0
    if kind == 'tiles':
        return F.cross_entropy(out.flatten(0, -2), y.flatten())
    return F.cross_entropy(out, y) if kind == 'goal_location' else F.smooth_l1_loss(out, y)


def scores(out, y, kind: str) -> dict:
    if kind == 'tiles':
        pred, out_ = out.argmax(-1), {'frames': int(len(y))}
        if not len(y):
            return out_
        out_['accuracy'] = float((pred == y).float().mean())
        for name, p, t in (('wall', pred == WALL, y == WALL), ('ahead', pred[:, TileHead.AHEAD] == WALL, y[:, TileHead.AHEAD] == WALL)):
            out_[f'{name}_recall'] = float(p[t].float().mean()) if t.any() else None
            out_[f'{name}_precision'] = float(t[p].float().mean()) if p.any() else None
            out_[f'{name}_balanced'] = float(((p[t].float().mean() if t.any() else 0) + ((~p)[~t].float().mean() if (~t).any() else 0)) / 2)
        return out_
    if kind == 'goal_location':
        pred, there = out.argmax(-1), y != ABSENT
        return {'frames': int(len(y)), 'in_view': float(((pred != ABSENT) == there).float().mean()) if len(y) else None,
                'where': float((pred[there] == y[there]).float().mean()) if there.any() else None, 'goal_frames': int(there.sum())}
    err, near = (out - y).abs(), y <= NEAR
    return {'frames': int(len(y)), 'mae': float(err.mean()) if len(y) else None,
            'mae_near': float(err[near].mean()) if near.any() else None, 'mae_far': float(err[~near].mean()) if (~near).any() else None,
            'corr': float(torch.corrcoef(torch.stack([out, y]))[0, 1]) if len(y) > 1 else None}


def summary(s: dict, kind: str) -> str:
    if kind == 'tiles':
        return (f"tiles {s['accuracy']:.3f}, walls recall {s['wall_recall'] or 0:.3f} precision {s['wall_precision'] or 0:.3f}, "
                f"ahead recall {s['ahead_recall'] or 0:.3f} precision {s['ahead_precision'] or 0:.3f}")
    if kind == 'goal_location':
        return f"in view {s['in_view']:.3f} where {s['where'] or 0:.3f}"
    return f"error {s['mae']:.2f} moves (near {s['mae_near'] or 0:.2f}, far {s['mae_far'] or 0:.2f}) corr {s['corr'] or 0:.3f}"


@torch.no_grad()
def evaluate(head, batches, history_size: int, kind: str) -> dict:
    head.eval()
    T, seen, imag = history_size, ([], []), ([], [], [])
    for s, im, g, batch in batches:
        y, use = targets(batch, kind)
        o = outputs(head, s, g, kind)
        seen[0].append(o[use].cpu()), seen[1].append(y[use].cpu())
        imag[0].append(outputs(head, im, g, kind).cpu()), imag[1].append(y[:, T:].cpu()), imag[2].append(use[:, T:].cpu())
    head.train()
    out = {'seen': scores(torch.cat(seen[0]), torch.cat(seen[1]), kind)}
    o, y, use = (torch.cat(x) for x in imag)
    for k in range(o.shape[1]):
        out[f'imagined_{k + 1}'] = scores(o[:, k][use[:, k]], y[:, k][use[:, k]], kind)
    return out


CACHED = ('seen', 'imagined', 'goal', 'label', 'moves', 'valid', 'tiles')


def load_cache(root, split: str, device, keys=CACHED) -> tuple:
    parts = sorted(Path(root).glob('part_*/meta.json'))
    metas = [json.loads(m.read_text()) for m in parts]
    if not metas or len(metas) != metas[0]['parts'] or \
            any({k: v for k, v in m.items() if k not in ('part', 'windows', 'rows')} !=
                {k: v for k, v in metas[0].items() if k not in ('part', 'windows', 'rows')}
                for m in metas):
        raise SystemExit(f"{root}: {len(metas)} complete parts of {metas[0]['parts'] if metas else '?'}, or parts of other caches")
    dtype = {'label': torch.long, 'moves': torch.float32, 'valid': torch.bool, 'tiles': torch.long, 'actions': torch.long,
             'random_actions': torch.long, 'random_moves': torch.float32, 'random_valid': torch.bool, 'index': torch.long}
    missing = [k for k in keys if not (parts[0].parent / split / f'{k}.npy').exists()]
    if missing:
        raise SystemExit(f"{root}: no {missing} (a cache made before they were cached: make it again)")
    data = {k: torch.from_numpy(np.concatenate([np.load(m.parent / split / f'{k}.npy') for m in parts])).to(device, dtype.get(k))
            for k in keys}
    return data, metas[0]


def cached_batch(data: dict, idx):
    emb = {k: data[k][idx].float() for k in ('seen', 'imagined', 'goal')}
    return emb['seen'], emb['imagined'], emb['goal'][:, None], {k: data[k][idx] for k in ('label', 'moves', 'valid', 'tiles')}


def selection(m: dict, kind: str) -> float:
    steps = [s for k, s in m.items() if k.startswith('imagined') and s['frames']]
    if kind == 'tiles':
        return float(np.mean([s['ahead_balanced'] for s in steps]))
    return float(np.mean([s['where'] or 0 for s in steps])) if kind == 'goal_location' else -float(np.mean([s['mae'] for s in steps]))


@hydra.main(version_base=None, config_path="../configs", config_name="analysis/goal_head")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    kind = cfg.head.kind
    if kind not in ('goal_location', 'cost_to_go', 'tiles'):
        raise ValueError(f"head.kind: goal_location, cost_to_go or tiles, not {kind!r}")
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    torch.manual_seed(cfg.seed)
    if cfg.cache:
        data, meta = load_cache(to_absolute_path(cfg.cache), 'train', device)
        val, _ = load_cache(to_absolute_path(cfg.cache), 'val', device)
        T, dim, source = meta['history_size'], meta['embed_dim'], {'cache': str(cfg.cache), **meta['checkpoints']}
        logger.info(f"cache {cfg.cache}: {len(data['label'])} training windows, {len(val['label'])} validation")
        g = torch.Generator().manual_seed(cfg.seed)
        train_batches = (cached_batch(data, torch.randint(len(data['label']), (cfg.batch_size,), generator=g).to(device))
                         for _ in range(cfg.steps))
        val_batches = lambda: (cached_batch(val, torch.arange(i, min(i + cfg.batch_size, len(val['label'])), device=device))
                               for i in range(0, len(val['label']), cfg.batch_size))
    else:
        jepa, vqvae = load_models(cfg, device)
        for p in jepa.parameters():
            p.requires_grad_(False)
        h5 = str(Path(to_absolute_path(cfg.data_dir)) / 'dataset.h5')
        tokens = encode_tokens(h5, vqvae, device=device, num_workers=cfg.num_workers)
        T, dim, source = cfg.pipeline.history_size, cfg.pipeline.embed_dim, OmegaConf.to_container(cfg.checkpoints)
        split = lambda s: Rollouts(FindGoalWorldModelDataset(
            h5, s, tuple(cfg.val_layouts), history_size=T, num_preds=cfg.rollout_steps, tokens=tokens,
            max_age=jepa.predictor.max_age, transform=get_transforms(img_size=cfg.pipeline.img_size, model='JEPA')[1]))
        train, val = split('train'), split('val')
        picks = np.random.default_rng(cfg.seed).choice(len(val), min(cfg.val_windows, len(val)), replace=False)
        val_loader = DataLoader(torch.utils.data.Subset(val, np.sort(picks)), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        loader = DataLoader(train, batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=True,
                            sampler=RandomSampler(train, replacement=True, num_samples=cfg.steps * cfg.batch_size))
        on_the_fly = lambda batches: (embeddings(jepa, b, T) + (b,) for b in ({k: v.to(device) for k, v in b.items()}
                                                                              for b in batches))
        train_batches, val_batches = on_the_fly(loader), lambda: on_the_fly(val_loader)
    head = (GoalHead(dim, cfg.head.hidden, cfg.head.absent) if kind == 'goal_location'
            else TileHead(dim, cfg.head.hidden) if kind == 'tiles' else CostToGoHead(dim, cfg.head.hidden)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"{kind.replace('_', '-')}-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, job_type=kind, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    best, metrics = None, {}
    for step, (seen, imagined, goal, batch) in enumerate(train_batches, 1):
        y, use = targets(batch, kind)
        loss = loss_of(outputs(head, seen, goal, kind)[use], y[use], kind) + \
            loss_of(outputs(head, imagined, goal, kind)[use[:, T:]], y[:, T:][use[:, T:]], kind)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.log_every == 0:
            logger.info(f"step {step}/{cfg.steps}  loss {loss.item():.4f}")
            if run:
                run.log({'train/loss': loss.item(), 'train/lr': sched.get_last_lr()[0]}, step=step)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            m = evaluate(head, val_batches(), T, kind)
            logger.info(f"step {step} val: " + ', '.join(f"{k} {summary(s, kind)}" for k, s in m.items() if s['frames']))
            if run:
                run.log({f'val/{k}/{n}': x for k, s in m.items() for n, x in s.items() if x is not None}, step=step)
            score = selection(m, kind)
            if best is None or score > best:
                best, metrics = score, {'step': step, **m}
                torch.save({'kind': kind, 'config': head.config, 'state_dict': head.state_dict(), 'metrics': metrics,
                            'checkpoints': source}, out / 'goal_head.pt')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    if run:
        run.summary.update({f'best/{k}/{n}': x for k, s in metrics.items() if isinstance(s, dict) for n, x in s.items()})
        run.finish()
    logger.info(f"the {kind} head (step {metrics.get('step')}) in {out / 'goal_head.pt'}")


if __name__ == '__main__':
    main()
