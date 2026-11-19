from __future__ import annotations

import copy
import math

import numpy as np

from .channel_data import STEP_FEATURES
from .power_control import fit_predictor

GEOMETRY, MOVES = 3, 2

__all__ = ['GEOMETRY', 'MOVES', 'KINDS', 'SequencePredictor', 'gain_loss', 'fit_sequence_predictor', 'predictor_name',
           'fit_model']

def _net(history: int, cell: str, steps: bool, hidden: int, layers: int, head: int, dropout: float):
    import torch
    from torch import nn

    class GainSequenceNet(nn.Module):

        def __init__(self):
            super().__init__()
            rnn = {'lstm': nn.LSTM, 'gru': nn.GRU}[cell]
            self.rnn = rnn(STEP_FEATURES if steps else 1, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden + GEOMETRY + MOVES + 1, head), nn.ReLU(),
                                      nn.Linear(head, head), nn.ReLU(), nn.Linear(head, 1))

        def forward(self, x, s=None):
            last = x[:, GEOMETRY]
            past = s if steps else x[:, GEOMETRY:GEOMETRY + history].unsqueeze(-1)
            out, _ = self.rnn(past.flip(1))
            static = torch.cat([x[:, :GEOMETRY], x[:, -MOVES:], last[:, None]], 1)
            return last + self.head(torch.cat([out[:, -1], static], 1)).squeeze(1)

    return GainSequenceNet()


class SequencePredictor:

    def __init__(self, config: dict, scale: dict, state: dict, curve: list, quantile: float | None = None,
                 importance: dict | None = None):
        self.config, self.scale, self.curve, self.quantile = config, scale, curve, quantile
        self.importance = importance
        self._build(state)

    @property
    def steps(self) -> bool:
        return self.config['steps']

    def _build(self, state):
        import torch
        self._torch, self._gpu = torch, None
        self.net = _net(**self.config)
        self.net.load_state_dict(state)
        self.net.eval()

    def scaled(self, X: np.ndarray, S: np.ndarray | None = None) -> tuple:
        c = self.scale
        Z = (np.asarray(X, np.float32) - c['mean']) / c['std']
        Zs = (np.asarray(S, np.float32)[:, :self.config['history']] - c['step_mean']) / c['step_std'] if self.steps else None
        return Z, Zs

    def target(self, gain: np.ndarray) -> np.ndarray:
        c = self.scale
        return ((10 * np.log10(np.maximum(np.asarray(gain, float), 1e-30)) - c['mean'][GEOMETRY]) / c['std'][GEOMETRY]).astype(np.float32)

    def loaded(self, state: dict):
        self.net.load_state_dict({k: v.detach().cpu() for k, v in state.items()})
        self.net.eval()
        self._gpu = None

    def predict(
        self,
        X: np.ndarray,
        S: np.ndarray | None = None,
        batch_size: int = 65536,
    ) -> np.ndarray:
        torch, c = self._torch, self.scale
        if self.steps and S is None:
            raise ValueError('a per-step predictor needs the past steps (S): build_dataset(..., steps=True)')
        net = self.net
        if len(X) >= 4096 and torch.cuda.is_available():
            if self._gpu is None:
                self._gpu = copy.deepcopy(self.net).cuda()
            net = self._gpu
        device = next(net.parameters()).device
        Z, Zs = self.scaled(X, S)
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), batch_size):
                s = torch.from_numpy(Zs[i:i + batch_size]).to(device) if self.steps else None
                out.append(net(torch.from_numpy(Z[i:i + batch_size]).to(device), s).cpu().numpy())
        y = np.concatenate(out) if out else np.zeros(0, np.float32)
        return (y * c['std'][GEOMETRY] + c['mean'][GEOMETRY]).astype(float)

    def __getstate__(self):
        return dict(config=self.config, scale=self.scale, curve=self.curve, quantile=self.quantile, importance=self.importance,
                    state={k: v.detach().cpu() for k, v in self.net.state_dict().items()})

    def __setstate__(self, s):
        self.config, self.scale, self.curve = s['config'], s['scale'], s['curve']
        self.quantile, self.importance = s.get('quantile'), s.get('importance')
        self._build(s['state'])


def gain_loss(quantile: float | None):
    if quantile is None:
        return lambda p, y: ((p - y) ** 2).mean()
    import torch
    return lambda p, y: torch.maximum(quantile * (y - p), (quantile - 1) * (y - p)).mean()

def fit_sequence_predictor(
    X: np.ndarray,
    gain: np.ndarray,
    quantile: float | None = 0.05,
    cell: str = 'lstm',
    S: np.ndarray | None = None,
    hidden: int = 64,
    layers: int = 1,
    head: int = 64,
    dropout: float = 0.0,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    groups: np.ndarray | None = None,
    val_fraction: float = 0.15,
    device: str = 'auto',
    seed: int = 0,
    verbose: bool = False,
    si_xi: float = 0.1,
) -> SequencePredictor:
    import torch
    if cell not in ('lstm', 'gru'):
        raise ValueError(f"cell: 'lstm' or 'gru', not {cell!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    X = np.asarray(X, np.float32)
    y = (10 * np.log10(np.maximum(np.asarray(gain, float), 1e-30))).astype(np.float32)
    history, steps = X.shape[1] - GEOMETRY - MOVES, S is not None
    if steps:
        S = np.asarray(S, np.float32)
        if S.shape[:2] != (len(X), history) or S.shape[2] != STEP_FEATURES:
            raise ValueError(f'S: ({len(X)}, {history}, {STEP_FEATURES}) for these features, not {S.shape}')
    if groups is not None:
        ids = np.unique(groups)
        val = np.isin(groups, rng.choice(ids, max(1, int(round(val_fraction * len(ids)))), replace=False))
    else:
        val = rng.random(len(y)) < val_fraction
    mean, std = X[~val].mean(0), X[~val].std(0) + 1e-6
    past = slice(GEOMETRY, GEOMETRY + history)
    mean[past], std[past] = mean[GEOMETRY], std[GEOMETRY]
    scale = dict(mean=mean, std=std)
    if steps:
        flat = S[~val].reshape(-1, STEP_FEATURES)
        step_mean, step_std = flat.mean(0), flat.std(0) + 1e-6
        step_mean[0], step_std[0] = mean[GEOMETRY], std[GEOMETRY]
        scale.update(step_mean=step_mean, step_std=step_std)
    Z = torch.from_numpy((X - mean) / std)
    Zs = torch.from_numpy((S - scale['step_mean']) / scale['step_std']) if steps else None
    t = torch.from_numpy((y - mean[GEOMETRY]) / std[GEOMETRY])
    config = dict(history=history, cell=cell, steps=steps, hidden=hidden, layers=layers, head=head, dropout=dropout)
    net = _net(**config).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, patience // 3))
    loss_fn = gain_loss(quantile)
    start = {k: p.detach().clone() for k, p in net.named_parameters()}
    path = {k: torch.zeros_like(p) for k, p in net.named_parameters()}
    split = lambda a, m: a[torch.from_numpy(m)].to(device) if a is not None else None
    (Ztr, Zstr, ttr), (Zva, Zsva, tva) = ([split(a, m) for a in (Z, Zs, t)] for m in (~val, val))
    best, best_state, waited, curve = math.inf, None, 0, []
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(ttr), device=device)
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Ztr[b], Zstr[b] if steps else None), ttr[b])
            loss.backward()
            eta = opt.param_groups[0]['lr']
            for k, p in net.named_parameters():
                if p.grad is not None:
                    path[k] += p.grad.detach() ** 2 * eta
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Zva, Zsva), tva)) if len(tva) else float(loss)
        curve.append(v)
        sched.step(v)
        if verbose:
            print(f'epoch {epoch}: validation loss {v:.4f}')
        if v < best - 1e-5:
            best, best_state, waited = v, {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    importance = {k: (path[k] / ((best_state[k].to(device) - start[k]) ** 2 + si_xi)).cpu() for k in path}
    return SequencePredictor(config, scale, best_state, curve, quantile, importance)

KINDS = ('trees', 'lstm', 'gru')

def predictor_name(kind: str, quantile: float | None, steps: bool = False) -> str:
    what = 'quantile' if quantile is not None else 'mean'
    if kind == 'trees':
        return 'geometry + history mean' if quantile is None else 'geometry + history quantile (trees)'
    return f"{kind.upper()}, {'per-step geometry' if steps else 'geometry'} + history {what}"

def fit_model(
    train: dict,
    kind: str = 'trees',
    quantile: float | None = 0.05,
    steps: bool = False,
    trees: dict | None = None,
    sequence: dict | None = None,
):
    if kind not in KINDS:
        raise ValueError(f'kind: one of {KINDS}, not {kind!r}')
    if kind == 'trees':
        if steps:
            raise ValueError('the trees read the features only (steps=False)')
        return fit_predictor(train['X'], train['gain'], quantile, **(trees or {}))
    if steps and 'S' not in train:
        raise ValueError('a per-step model needs the past steps: build_dataset(..., steps=True)')
    return fit_sequence_predictor(train['X'], train['gain'], quantile, kind, train['S'] if steps else None,
                                  groups=train.get('layout'), **(sequence or {}))
