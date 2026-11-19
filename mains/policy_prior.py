import json
import os
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import torch.nn.functional as F

from c3_wm.evaluators.deploy import PolicyPriorHead, RecurrentPriorHead, load_head
from memory_cost import GROUPS, WALL, flat, noisy, setup

HEADINGS = ((1, 0), (0, 1), (-1, 0), (0, -1))
LEFT, RIGHT, FORWARD = 0, 1, 2


def pose_moves(walls: np.ndarray, goal) -> np.ndarray:
    W, H = walls.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')
    ahead = []
    for dx, dy in HEADINGS:
        nx, ny = xs + dx, ys + dy
        ok = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        ok[ok] = ~walls[nx[ok], ny[ok]]
        ahead.append((np.where(ok, nx, xs), np.where(ok, ny, ys)))
    moves = np.full((W, H, 4), np.inf)
    moves[tuple(goal)] = 0
    while True:
        best = np.stack([np.minimum(np.minimum(moves[:, :, (d - 1) % 4], moves[:, :, (d + 1) % 4]),
                                    moves[ahead[d][0], ahead[d][1], d]) for d in range(4)], -1) + 1
        new = np.minimum(moves, best)
        new[tuple(goal)] = 0
        if np.array_equal(new, moves):
            return moves
        moves = new


def optimal_actions(moves: np.ndarray, walls: np.ndarray, cells: np.ndarray, headings: np.ndarray) -> np.ndarray:
    W, H = walls.shape
    x, y, d = cells[:, 0], cells[:, 1], headings
    now = moves[x, y, d]
    dx, dy = np.array(HEADINGS)[d].T
    nx, ny = x + dx, y + dy
    ok = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
    ok[ok] = ~walls[nx[ok], ny[ok]]
    after = np.stack([moves[x, y, (d - 1) % 4], moves[x, y, (d + 1) % 4],
                      moves[np.where(ok, nx, x), np.where(ok, ny, y), d]], -1)
    return np.isfinite(now)[:, None] & (now[:, None] > 0) & (after == now[:, None] - 1)


def action_table(h5: str, device) -> torch.Tensor:
    with h5py.File(h5, 'r') as f:
        position, direction, goal = f['position'][:], f['direction'][:], f['goal_position'][:]
        offset, length = f['ep_offset'][:], f['ep_len'][:]
        walls = dict(zip(np.sort(offset), f['state'][np.sort(offset), :, :, 0] == WALL))
    optimal = np.zeros((*position.shape[:2], 3), bool)
    for o, n in zip(offset, length):
        for a in range(position.shape[1]):
            cells, heads = position[o:o + n, a], direction[o:o + n, a]
            on = (cells >= 0).all(-1)
            if on.any():
                moves = pose_moves(walls[o], goal[o, a])
                optimal[o:o + n, a][on] = optimal_actions(moves, walls[o], cells[on], heads[on])
    return torch.as_tensor(optimal, device=device)


def memory_of(head, seq: dict) -> torch.Tensor:
    x = head.inputs(seq['emb'], seq['prev'], seq['tokens'], seq['pose'], seq['new'], own=seq['own'])
    return head.observe(x, None, seq['lengths'])[0]


def forward(prior, memory_head, seq: dict, noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None]:
    if memory_head is None:
        out = memory_of(prior, seq)
        return prior(noisy(seq['emb'], noise), noisy(out, noise)), prior.goal(out)
    with torch.no_grad():
        out = memory_of(memory_head, seq)
    return prior(noisy(seq['emb'], noise), noisy(out, noise)), None


def targets(optimal: torch.Tensor, seq: dict, known_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    good = optimal[seq['rows'], seq['me']]
    use = seq['aux'] & good.any(-1)
    return good, use & seq['known'] if known_only else use


def loss_of(logp: torch.Tensor, good: torch.Tensor, use: torch.Tensor) -> torch.Tensor:
    target = good.float() / good.sum(-1, keepdim=True).clamp_min(1)
    return -(target * logp).sum(-1)[use].mean()


@torch.no_grad()
def evaluate(prior, memory_head, optimal, batches) -> dict:
    prior.eval()
    hit, losses, by_group = [], [], {g: [] for g in (*GROUPS, 'detour', 'known')}
    for seq, _ in batches:
        logp, _ = forward(prior, memory_head, seq)
        good, use = targets(optimal, seq)
        right = good.gather(-1, logp.argmax(-1, keepdim=True)).squeeze(-1)
        hit.append(right[use].cpu()), losses.append(float(loss_of(logp, good, use)) if use.any() else None)
        last = torch.arange(len(seq['lengths']), device=right.device), seq['lengths'] - 1
        for g in GROUPS:
            m = seq['groups'][g] & use[last]
            by_group[g].append(right[last][m].cpu())
        by_group['detour'].append(right[last][seq['groups']['in view'] & (seq['moves'][last] >= 10) & use[last]].cpu())
        by_group['known'].append(right[use & seq['known']].cpu())
    prior.train()
    acc = lambda x: float(torch.cat(x).float().mean()) if len(torch.cat(x)) else None
    return {'accuracy': acc(hit), 'steps': int(len(torch.cat(hit))),
            'loss': float(np.mean([v for v in losses if v is not None])),
            'by_group': {g: acc(v) for g, v in by_group.items()}}


def summary(m: dict) -> str:
    f = lambda v: f"{v:.3f}" if v is not None else '-'
    return (f"accuracy {f(m['accuracy'])} loss {f(m['loss'])}; at the last step: "
            + ', '.join(f"{g} {f(v)}" for g, v in m['by_group'].items()))


@hydra.main(version_base=None, config_path="../configs", config_name="analysis/policy_prior")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("device=cuda, but no GPU is visible")
    torch.manual_seed(cfg.seed)
    data, val, meta, codes, batch = setup(cfg, device)
    memory_head = None
    if not cfg.head.recurrent:
        if not cfg.checkpoints.memory_cost:
            raise ValueError("checkpoints.memory_cost: the memory the prior reads (or head.recurrent=true: its own)")
        memory_head = load_head(to_absolute_path(cfg.checkpoints.memory_cost)).to(device)
        if memory_head.kind != 'memory_cost':
            raise ValueError(f"checkpoints.memory_cost: memory_cost.py's head, not a {memory_head.kind} one")
        if memory_head.config['codes'] != codes:
            raise ValueError(f"the memory head reads {memory_head.config['codes']} codes, the VQ-VAE has {codes}")
        memory_head.requires_grad_(False)
    h5 = str(Path(to_absolute_path(cfg.data_dir or meta['data_dir'])) / 'dataset.h5')
    optimal = action_table(h5, device)
    logger.info(f"optimal actions: {float(optimal.any(-1).float().mean()):.3f} of the rows' agents have one, "
                f"{float(optimal.sum(-1)[optimal.any(-1)].float().mean()):.2f} on average")
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

    if cfg.head.recurrent:
        prior = RecurrentPriorHead(meta['embed_dim'], cfg.head.hidden, codes, cfg.head.token_dim, reads=cfg.head.reads,
                                   queries=cfg.head.queries, width=cfg.head.width, dropout=cfg.head.dropout).to(device)
    else:
        prior = PolicyPriorHead(meta['embed_dim'], memory_head.config['hidden'], cfg.head.width,
                                dropout=cfg.head.dropout).to(device)
    if cfg.labels not in ('all', 'known'):
        raise ValueError(f"labels: all or known, not {cfg.labels!r}")
    logger.info(f"{type(prior).__name__}: {sum(p.numel() for p in prior.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(prior.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"policy-prior-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, job_type='policy_prior', dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    best, metrics = None, {}
    for step, (seq, _) in enumerate(train_batches(), 1):
        good, use = targets(optimal, seq, cfg.labels == 'known')
        logp, goal = forward(prior, memory_head, seq, cfg.noise)
        loss = loss_of(logp, good, use)
        if goal is not None and cfg.aux_weight:
            loss = loss + cfg.aux_weight * F.mse_loss(goal[seq['aux']], seq['offset'][seq['aux']])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0:
            logger.info(f"step {step}/{cfg.steps}  loss {loss.item():.4f}")
            if run:
                run.log({'train/loss': loss.item(), 'train/lr': sched.get_last_lr()[0]}, step=step)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            m = {'messages': evaluate(prior, memory_head, optimal, val_batches(0.0)),
                 'no_messages': evaluate(prior, memory_head, optimal, val_batches(1.0)),
                 'train_messages': evaluate(prior, memory_head, optimal, val_batches(0.0, data, cfg.train_eval_windows))}
            for k, s in m.items():
                logger.info(f"step {step} val, {k.replace('_', ' ')}: {summary(s)}")
            if run:
                run.log(flat(m, 'val'), step=step)
            score = m['messages']['by_group']['known'] or 0
            if best is None or score > best:
                best, metrics = score, {'step': step, **m}
                read = None if cfg.head.recurrent else str(cfg.checkpoints.memory_cost)
                torch.save({'kind': prior.kind, 'config': prior.config, 'state_dict': prior.state_dict(), 'metrics': metrics,
                            'checkpoints': {'cache': str(cfg.cache), 'memory_cost': read, **meta['checkpoints']}},
                           out / 'policy_prior.pt')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    if run:
        best_of = metrics['messages']['by_group']
        run.summary.update({'best/accuracy': metrics['messages']['accuracy'], 'best/known': best_of['known'],
                            'best/detour': best_of['detour'],
                            'best/accuracy_no_messages': metrics['no_messages']['accuracy'], 'best/step': metrics['step']})
        run.finish()
    logger.info(f"the policy prior (step {metrics.get('step')}) in {out / 'policy_prior.pt'}")


if __name__ == '__main__':
    main()
