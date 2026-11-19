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

from c3_wm.data.datasets import TILE_NAMES
from c3_wm.evaluators.deploy import MemoryCostHead
from c3_wm.models.jepa import relative_pose
from c3_wm.trainers.wm_tune import encode_tokens
from goal_head import fewest_moves, load_cache

GOAL, WALL = TILE_NAMES.index('goal'), TILE_NAMES.index('wall')
KEYS = ('index', 'imagined', 'actions', 'moves', 'valid', 'imagined_random', 'random_actions', 'random_moves', 'random_valid')
GROUPS = ('in view', 'remembered', 'heard only', 'unknown')


def episode_table(h5: str, device) -> dict:
    with h5py.File(h5, 'r') as f:
        action, position, direction = f['action'][:], f['position'][:], f['direction'][:]
        terminated, goal = f['terminated'][:], f['goal_position'][:]
        offset, length = f['ep_offset'][:], f['ep_len'][:]
        walls = f['state'][np.sort(offset), :, :, 0] == WALL
        walls = dict(zip(np.sort(offset), walls))
        sees = np.zeros(terminated.shape, bool)
        for at in range(0, len(sees), 4096):
            sees[at:at + 4096] = (f['image'][at:at + 4096, ..., 0] == GOAL).any((-1, -2))
    rows, A = terminated.shape
    start = np.repeat(offset, length)
    acted = np.where(np.isfinite(action), action, MemoryCostHead.NO_ACTION).astype(np.int64)
    prev = np.full((rows, A), MemoryCostHead.NO_ACTION, np.int64)
    prev[1:] = acted[:-1]
    prev[np.arange(rows) == start] = MemoryCostHead.NO_ACTION
    moves = np.full((rows, A), -1, np.int64)
    for o, n in zip(offset, length):
        for a in range(A):
            field = fewest_moves(walls[o], goal[o, a])
            cells = position[o:o + n, a]
            on = (cells >= 0).all(-1)
            moves[o:o + n, a][on] = field[cells[on, 0], cells[on, 1]]
    t = lambda v, dtype=None: torch.as_tensor(v, device=device, dtype=dtype)
    pos, dirs = t(position, torch.long), t(direction, torch.long)
    return dict(prev=t(prev), position=pos, direction=dirs, terminated=t(terminated), start=t(start, torch.long),
                moves=t(moves, torch.float32), sees=t(sees),
                offset=relative_pose(t(goal, torch.long), torch.zeros_like(dirs), pos, dirs)[..., :2].float())


def load_frames(root, device) -> torch.Tensor:
    parts = sorted(Path(root).glob('part_*/frames.npy'), key=lambda p: int(np.load(p.parent / 'rows.npy')[0]))
    if not parts:
        raise SystemExit(f"{root}: no frames (embed_cache.py frames=true: make the cache again)")
    spans = [np.load(p.parent / 'rows.npy') for p in parts]
    if spans[0][0] != 0 or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise SystemExit(f"{root}: the parts' frames do not cover the rows: {[s.tolist() for s in spans]}")
    return torch.from_numpy(np.concatenate([np.load(p) for p in parts])).to(device)


def sequences(tab: dict, frames, tokens, index, history_size: int, max_prefix: int, drop):
    r, a = index[:, 0], index[:, 1]
    t = r + history_size - 1
    first = torch.maximum(tab['start'][r], t - max_prefix + 1)
    lengths = t - first + 1
    steps = torch.arange(int(lengths.max()), device=r.device)
    pad = steps[None] >= lengths[:, None]
    rows = torch.minimum(first[:, None] + steps[None], t[:, None])
    A = tab['terminated'].shape[1]
    me = a[:, None].expand_as(rows)
    others = torch.arange(A, device=r.device)[None, None] != a[:, None, None]
    sent = others & ~tab['terminated'][rows] & ~pad[..., None]
    new = sent & (torch.rand(sent.shape, device=r.device) >= drop[:, None, None])
    pose = relative_pose(tab['position'][rows], tab['direction'][rows], tab['position'][rows, me][:, :, None],
                         tab['direction'][rows, me][:, :, None])
    moves = tab['moves'][rows, me]
    acting = ~pad & ~tab['terminated'][rows, me]
    sees_now = tab['sees'][t, a]
    seen = (tab['sees'][rows, me] & ~pad).any(1)
    heard = (tab['sees'][rows] & sent).any((1, 2))
    groups = {'in view': sees_now, 'remembered': seen & ~sees_now, 'heard only': heard & ~seen, 'unknown': ~seen & ~heard}
    return dict(emb=frames[rows, me].float(), prev=tab['prev'][rows, me], tokens=tokens[rows].long(), pose=pose, new=new,
                own=tokens[rows, me].long(),
                lengths=lengths, moves=moves, use=acting & (moves >= 0), aux=acting, offset=tab['offset'][rows, me] / 15,
                groups=groups, rows=rows, me=me,
                known=torch.cummax(((tab['sees'][rows, me] | (tab['sees'][rows] & new).any(-1)) & ~pad).int(), 1).values.bool())


def noisy(x: torch.Tensor, noise: float) -> torch.Tensor:
    return x + noise * x.detach().std(dim=tuple(range(x.dim() - 1)), keepdim=True) * torch.randn_like(x) if noise else x


def forward(head, seq: dict, win: dict, T: int, noise: float = 0.0) -> dict:
    emb = noisy(seq['emb'], noise)
    x = head.inputs(emb, seq['prev'], seq['tokens'], seq['pose'], seq['new'], own=seq['own'])
    out, h = head.observe(x, None, seq['lengths'])
    return dict(seen=head.cost(emb, out), goal=head.goal(out),
                imagined=head.imagine(h, noisy(win['imagined'].float(), noise), win['actions'][:, T - 1:]),
                random=head.imagine(h, noisy(win['imagined_random'].float(), noise), win['random_actions'][:, T - 1:]))


def labels(seq: dict, win: dict, T: int) -> dict:
    return dict(seen=(seq['moves'], seq['use']), imagined=(win['moves'][:, T:], win['valid'][:, T:] & (win['moves'][:, T:] >= 0)),
                random=(win['random_moves'], win['random_valid'] & (win['random_moves'] >= 0)))


def rank_loss(out: dict, seq: dict, win: dict, T: int) -> torch.Tensor:
    lab = labels(seq, win, T)
    (yi, ui), (yr, ur) = lab['imagined'], lab['random']
    true_d, true_r = _fewest(yi, ui), _fewest(yr, ur)
    pred_d, pred_r = _fewest(out['imagined'], ui), _fewest(out['random'], ur)
    differ = torch.isfinite(true_d) & torch.isfinite(true_r) & (true_d != true_r)
    if not differ.any():
        return out['imagined'].sum() * 0
    sign = torch.sign(true_r - true_d)[differ]
    return F.softplus(-sign * (pred_r - pred_d)[differ]).mean()


def loss_of(out: dict, seq: dict, win: dict, T: int, aux_weight: float, rank_weight: float = 0.0):
    loss = 0
    for k, (y, use) in labels(seq, win, T).items():
        loss = loss + (F.smooth_l1_loss(out[k][use], y[use]) if use.any() else out[k].sum() * 0)
    if rank_weight:
        loss = loss + rank_weight * rank_loss(out, seq, win, T)
    return loss + aux_weight * F.mse_loss(out['goal'][seq['aux']], seq['offset'][seq['aux']])


def _scores(o, y) -> dict:
    err = (o - y).abs()
    return {'steps': int(len(y)), 'mae': float(err.mean()) if len(y) else None,
            'corr': float(torch.corrcoef(torch.stack([o, y]))[0, 1]) if len(y) > 1 else None}


def _fewest(moves, use):
    return torch.where(use, moves, torch.full_like(moves, float('inf'))).min(-1).values


@torch.no_grad()
def evaluate(head, batches, T: int) -> dict:
    head.eval()
    got = {k: ([], []) for k in ('seen', 'imagined', 'random')}
    by_group, order, cells = {g: ([], []) for g in GROUPS}, [], {g: [] for g in GROUPS}
    for seq, win in batches:
        out, lab = forward(head, seq, win, T), labels(seq, win, T)
        last = torch.arange(len(seq['lengths']), device=seq['lengths'].device), seq['lengths'] - 1
        miss = 15 * (out['goal'][last] - seq['offset'][last]).norm(dim=-1)
        for g in GROUPS:
            cells[g].append(miss[seq['groups'][g] & seq['aux'][last]].cpu())
        for k, (y, use) in lab.items():
            got[k][0].append(out[k][use].cpu()), got[k][1].append(y[use].cpu())
        (yr, ur) = lab['random']
        for g in GROUPS:
            m = ur & seq['groups'][g][:, None]
            by_group[g][0].append(out['random'][m].cpu()), by_group[g][1].append(yr[m].cpu())
        (yi, ui) = lab['imagined']
        true_d, true_r = _fewest(yi, ui), _fewest(yr, ur)
        pred_d, pred_r = _fewest(out['imagined'], ui), _fewest(out['random'], ur)
        differ = torch.isfinite(true_d) & torch.isfinite(true_r) & (true_d != true_r)
        order.append(((pred_d < pred_r) == (true_d < true_r))[differ].cpu())
    head.train()
    m = {k: _scores(torch.cat(o), torch.cat(y)) for k, (o, y) in got.items()}
    m['random_by_group'] = {g: _scores(torch.cat(o), torch.cat(y)) for g, (o, y) in by_group.items()}
    order = torch.cat(order)
    m['order'] = float(order.float().mean()) if len(order) else None
    m['order_windows'] = int(len(order))
    m['goal_cells'] = {g: float(torch.cat(c).mean()) if len(torch.cat(c)) else None for g, c in cells.items()}
    return m


def flat(d: dict, prefix: str) -> dict:
    out = {}
    for k, v in d.items():
        out.update(flat(v, f'{prefix}/{k}') if isinstance(v, dict) else {} if v is None else {f'{prefix}/{k}': v})
    return out


def summary(m: dict) -> str:
    f = lambda v: f"{v:.2f}" if v is not None else '-'
    return (f"order {f(m['order'])}; " + ', '.join(f"{k} mae {f(m[k]['mae'])} corr {f(m[k]['corr'])}" for k in ('seen', 'imagined', 'random'))
            + '; random by group: ' + ', '.join(f"{g} {f(s['mae'])}" for g, s in m['random_by_group'].items())
            + '; goal placed off by (cells): ' + ', '.join(f"{g} {f(v)}" for g, v in m['goal_cells'].items()))


def setup(cfg: DictConfig, device):
    root = to_absolute_path(cfg.cache)
    data, meta = load_cache(root, 'train', device, KEYS)
    val, _ = load_cache(root, 'val', device, KEYS)
    if not meta.get('frames'):
        raise SystemExit(f"{root}: no frames (embed_cache.py frames=true: make the cache again)")
    T = meta['history_size']
    h5 = str(Path(to_absolute_path(cfg.data_dir or meta['data_dir'])) / 'dataset.h5')
    vq_path = to_absolute_path(cfg.checkpoints.vqvae or meta['checkpoints']['vqvae'])
    vqvae = hydra.utils.instantiate({k: v for k, v in OmegaConf.to_container(cfg.vqvae, resolve=True).items()
                                     if k != 'checkpoint_path'})
    vqvae.load_state_dict(torch.load(vq_path, map_location='cpu')['model_state_dict'])
    tokens = torch.from_numpy(np.load(encode_tokens(h5, vqvae, device=device, num_workers=cfg.num_workers)).astype(np.int16)).to(device)
    tab, frames = episode_table(h5, device), load_frames(root, device)
    logger.info(f"cache {cfg.cache}: {len(data['index'])} training windows, {len(val['index'])} validation; "
                f"{len(frames)} rows of frames")
    win = lambda d, idx: {k: d[k][idx] for k in KEYS if k != 'index'}

    def batch(d, idx, drop):
        return sequences(tab, frames, tokens, d['index'][idx], T, cfg.max_prefix, drop), win(d, idx)
    return data, val, meta, vqvae.num_embeddings, batch


@hydra.main(version_base=None, config_path="../configs", config_name="analysis/memory_cost")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    torch.manual_seed(cfg.seed)
    data, val, meta, codes, batch = setup(cfg, device)
    T, dim = meta['history_size'], meta['embed_dim']
    g = torch.Generator(device=device).manual_seed(cfg.seed)

    def train_batches():
        for _ in range(cfg.steps):
            idx = torch.randint(len(data['index']), (cfg.batch_size,), device=device, generator=g)
            drop = torch.rand(cfg.batch_size, device=device, generator=g) if cfg.drop is None else \
                torch.full((cfg.batch_size,), float(cfg.drop), device=device)
            yield batch(data, idx, drop)

    def val_batches(drop: float, d=val, windows=None):
        torch.manual_seed(cfg.seed)
        n = len(d['index']) if windows is None else min(windows, len(d['index']))
        for i in range(0, n, cfg.batch_size):
            idx = torch.arange(i, min(i + cfg.batch_size, n), device=device)
            yield batch(d, idx, torch.full((len(idx),), drop, device=device))

    head = MemoryCostHead(dim, cfg.head.hidden, codes, cfg.head.token_dim, reads=cfg.head.reads,
                          queries=cfg.head.queries, dropout=cfg.head.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"memory-cost-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, job_type='memory_cost', dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    best, metrics = None, {}
    for step, (seq, w) in enumerate(train_batches(), 1):
        loss = loss_of(forward(head, seq, w, T, cfg.noise), seq, w, T, cfg.aux_weight, cfg.rank_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0:
            logger.info(f"step {step}/{cfg.steps}  loss {loss.item():.4f}")
            if run:
                run.log({'train/loss': loss.item(), 'train/lr': sched.get_last_lr()[0]}, step=step)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            m = {'messages': evaluate(head, val_batches(0.0), T), 'no_messages': evaluate(head, val_batches(1.0), T),
                 'train_messages': evaluate(head, val_batches(0.0, data, cfg.train_eval_windows), T)}
            for k, s in m.items():
                logger.info(f"step {step} val, {k.replace('_', ' ')}: {summary(s)}")
            if run:
                run.log(flat(m, 'val'), step=step)
            score = m['messages']['order'] or 0
            if best is None or score > best:
                best, metrics = score, {'step': step, **m}
                torch.save({'kind': MemoryCostHead.kind, 'config': head.config, 'state_dict': head.state_dict(),
                            'metrics': metrics, 'checkpoints': {'cache': str(cfg.cache), **meta['checkpoints']}},
                           out / 'memory_cost.pt')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    if run:
        run.summary.update({'best/order': metrics['messages']['order'], 'best/order_no_messages': metrics['no_messages']['order'],
                            'best/step': metrics['step']})
        run.finish()
    logger.info(f"the memory cost head (step {metrics.get('step')}) in {out / 'memory_cost.pt'}")


if __name__ == '__main__':
    main()
