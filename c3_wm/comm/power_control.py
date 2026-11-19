from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .channel_data import DIRS, db, features, step_features

__all__ = ['MARGINS_DB', 'TAUS_DB', 'OOV', 'Budget', 'fit_predictor', 'predict_gain', 'predict_on', 'outcome', 'calibrate',
           'evaluate', 'layouts_allowed_below', 'layout_outcome', 'calibrate_layouts', 'save_power_control',
           'load_power_control', 'LearnedChannelInversion', 'Schedule', 'Tokens', 'c3wm_link']

@dataclass
class Budget:
    snr_target_db: float = 10.0
    noise_dbm: float = -94.0
    max_power_dbm: float = 20.0

    @property
    def g_min(self) -> float:
        return 10 ** ((self.snr_target_db + self.noise_dbm - self.max_power_dbm) / 10)

    @property
    def noise_w(self) -> float: return 10 ** ((self.noise_dbm - 30) / 10)

    @property
    def max_power_w(self) -> float: return 10 ** ((self.max_power_dbm - 30) / 10)

def fit_predictor(
    X: np.ndarray,
    gain: np.ndarray,
    quantile: float | None = 0.05,
    max_iter: int = 400,
    learning_rate: float = 0.05,
    **kwargs,
):
    from sklearn.ensemble import HistGradientBoostingRegressor
    loss = dict(loss='quantile', quantile=quantile) if quantile is not None else dict(loss='squared_error')
    model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, early_stopping=False,
                                          random_state=0, **loss, **kwargs)
    return model.fit(X, db(gain))

def _predict(model, X: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
    return model.predict(X, S) if getattr(model, 'steps', False) else model.predict(X)

def predict_gain(model, X_possible: np.ndarray, possible_mask: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
    n, k, F = X_possible.shape
    S = None if S is None else np.repeat(S, k, axis=0)
    v = _predict(model, X_possible.reshape(n * k, F), S).reshape(n, k)
    return 10 ** (np.where(possible_mask, v, np.inf).min(1) / 10)

def predict_on(model, data: dict) -> np.ndarray:
    return predict_gain(model, data['X_possible'], data['possible_mask'], data.get('S'))

MARGINS_DB = np.arange(-10, 40.01, 0.25)
TAUS_DB = (0.0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, math.inf)

def outcome(estimate, gain, g_min: float, tau_db: float = math.inf, margins_db=MARGINS_DB) -> dict:
    margins_db = np.atleast_1d(np.asarray(margins_db, float))
    servable, need = gain >= g_min, 1 / gain
    rho = 10 ** (margins_db[:, None] / 10) / np.maximum(estimate, 1e-30)[None, :]
    sent = rho <= 10 ** (tau_db / 10) / g_min if np.isfinite(tau_db) else np.ones_like(rho, bool)
    power = np.where(sent, np.minimum(rho, 1 / g_min), 0)
    reached = sent & (power >= need * (1 - 1e-9))
    return dict(reliability=reached[:, servable].mean(1), power=power.mean(1) / np.where(servable, need, 0).mean(),
                withheld=1 - sent.mean(1))

def calibrate(estimate, gain, g_min: float, target: float, margins_db=MARGINS_DB, taus_db=TAUS_DB) -> dict:
    best = None
    for tau in taus_db:
        o = outcome(estimate, gain, g_min, tau, margins_db)
        for k, margin in enumerate(np.atleast_1d(margins_db)):
            met = bool(o['reliability'][k] >= target)
            key = (met, -o['power'][k] if met else o['reliability'][k])
            if best is None or key > best[0]:
                best = (key, dict(margin_db=float(margin), tau_db=float(tau), met=met, reliability=float(o['reliability'][k]),
                                  power=float(o['power'][k]), withheld=float(o['withheld'][k])))
    return best[1]

def evaluate(estimate, gain, g_min: float, margin_db: float, tau_db: float) -> dict:
    o = outcome(estimate, gain, g_min, tau_db, [margin_db])
    return dict(reliability=float(o['reliability'][0]), power=float(o['power'][0]), withheld=float(o['withheld'][0]),
                servable=float((gain >= g_min).mean()))

def layouts_allowed_below(n: int, coverage: float, confidence: float | None = None) -> int:
    if confidence is None:
        return int(math.floor((1 - coverage) * (n + 1) + 1e-9)) - 1
    from scipy.stats import binom
    m = -1
    while m + 1 < n and binom.cdf(m + 1, n, 1 - coverage) <= 1 - confidence:
        m += 1
    return m

def layout_outcome(estimate, gain, layout, g_min: float, tau_db: float = math.inf, margins_db=MARGINS_DB,
                   block: int = 20000) -> dict:
    margins_db = np.atleast_1d(np.asarray(margins_db, float))
    layouts, column = np.unique(layout, return_inverse=True)
    reached = np.zeros((len(margins_db), len(layouts)))
    power, sent_n = np.zeros(len(margins_db)), np.zeros(len(margins_db))
    servable_all = gain >= g_min
    for s in range(0, len(gain), block):
        e, g = np.maximum(estimate[s:s + block], 1e-30), gain[s:s + block]
        rho = 10 ** (margins_db[:, None] / 10) / e[None, :]
        sent = rho <= 10 ** (tau_db / 10) / g_min if math.isfinite(tau_db) else np.ones_like(rho, bool)
        p = np.where(sent, np.minimum(rho, 1 / g_min), 0)
        ok = (sent & (p >= (1 / g) * (1 - 1e-9))) & servable_all[s:s + block]
        onehot = np.zeros((len(g), len(layouts)), np.float32)
        onehot[np.arange(len(g)), column[s:s + block]] = 1
        reached += ok.astype(np.float32) @ onehot
        power += p.sum(1)
        sent_n += sent.sum(1)
    servable = np.bincount(column, weights=servable_all, minlength=len(layouts))
    need = np.where(servable_all, 1 / gain, 0).mean()
    return dict(reliability=reached.sum(1) / max(servable_all.sum(), 1), power=power / len(gain) / need,
                withheld=1 - sent_n / len(gain), layouts=layouts, servable=servable,
                per_layout=reached / np.maximum(servable, 1))

def calibrate_layouts(
    estimate, gain, layout,
    g_min: float, target: float,
    coverage: float = 0.9,
    confidence: float | None = None,
    margins_db=MARGINS_DB, taus_db=TAUS_DB,
    min_servable: int = 20,
) -> dict:
    best = None
    for tau in taus_db:
        o = layout_outcome(estimate, gain, layout, g_min, tau, margins_db)
        keep = o['servable'] >= min_servable
        n = int(keep.sum())
        allowed = layouts_allowed_below(n, coverage, confidence)
        below = (o['per_layout'][:, keep] < target - 1e-12).sum(1)
        for k, margin in enumerate(np.atleast_1d(margins_db)):
            met = bool(allowed >= 0 and below[k] <= allowed)
            key = (met, -o['power'][k]) if met else (False, -below[k], -o['power'][k])
            if best is None or key > best[0]:
                best = (key, dict(margin_db=float(margin), tau_db=float(tau), met=met, below=int(below[k]),
                                  allowed=int(allowed), layouts=n, reliability=float(o['reliability'][k]),
                                  power=float(o['power'][k]), withheld=float(o['withheld'][k]),
                                  worst=float(o['per_layout'][k, keep].min()) if n else math.nan))
    return best[1]

def save_power_control(path, model, history: int, calibration: dict, budget: Budget, **info):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(dict(model=model, history=history, calibration=calibration, budget=asdict(budget), info=info), f)

def load_power_control(path) -> dict:
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    saved['budget'] = Budget(**saved['budget'])
    return saved

from comm_core import BPSK, Airtime, Block, IndexCodec, PowerControl, UncodedChain

class LearnedChannelInversion(PowerControl):

    def __init__(
        self,
        model,
        walls: np.ndarray,
        positions,
        headings,
        history: int = 4,
        target_snr_db: float = 10.0,
        margin_db: float = 0.0,
        fallback_power: float = math.inf,
        geometry: bool = True,
    ):
        self.model, self.walls, self.positions, self.headings = model, walls, positions, headings
        self.history, self.target_snr_db, self.margin_db = history, target_snr_db, margin_db
        self.fallback_power, self.geometry = fallback_power, geometry
        self.reset()

    @classmethod
    def from_saved(cls, saved: dict, walls, positions, headings):
        b, c = saved['budget'], saved['calibration']
        return cls(saved['model'], walls, positions, headings, saved['history'], b.snr_target_db, c['margin_db'],
                   b.max_power_w)

    def children(self):
        return []

    def reset(self, seed=None):
        self._past, self._rounds, self._round = {}, [], 0
        self.predicted = None

    def update(self, time):
        self._round += 1
        self._rounds = (self._rounds + [({k: tuple(int(v) for v in p) for k, p in self.positions().items()},
                                          dict(self.headings()))])[-max(3, self.history + 2):]

    def _cells(self, r):
        k = r - self._round + len(self._rounds) - 1
        return self._rounds[k][0] if 0 <= k < len(self._rounds) else None

    def _steps(self, past, link) -> np.ndarray:
        rows = []
        for g, r in past[-self.history:][::-1]:
            at, before = self._cells(r), self._cells(r - 1)
            moved = lambda node: before is not None and node in before and before[node] != at[node]
            rows.append(step_features(self.walls, at[link.sender], at[link.receiver], g, moved(link.sender), moved(link.receiver)))
        return np.array(rows, np.float32)

    def _rows(self, link, past) -> np.ndarray:
        (earlier, _), (before, heads), (now, _) = self._rounds[-3:]
        gains = [g for g, _ in past[-self.history:][::-1]]
        tx, rx = now[link.sender], before[link.receiver]
        moved = (tx != before[link.sender], rx != earlier[link.receiver])
        if not self.geometry:
            return np.array([[*db(gains), float(moved[0]), float(moved[1])]])
        d = DIRS[int(heads[link.receiver])]
        free = lambda c: 0 <= c[0] < self.walls.shape[0] and 0 <= c[1] < self.walls.shape[1] and not self.walls[c]
        possible = [c for c in (rx, (rx[0] + d[0], rx[1] + d[1])) if free(c) and c != tx] or [rx]
        return np.array([features(self.walls, tx, c, gains, *moved) for c in possible])

    def allocate(self, gain, noise_power, link=None, message=None):
        past = self._past.setdefault((link.sender, link.receiver), [])
        if len(past) < self.history or len(self._rounds) < 3:
            self.predicted = None
            power = self.fallback_power
        else:
            X = self._rows(link, past)
            S = np.repeat(self._steps(past, link)[None], len(X), 0) if getattr(self.model, 'steps', False) else None
            self.predicted = 10 ** (_predict(self.model, X, S).min() / 10)
            power = 10 ** ((self.target_snr_db + self.margin_db) / 10) * noise_power / self.predicted
        past.append((float(gain), self._round))
        return power, True


class Schedule(Block):
    outputs = ('power', 'active')

    def __init__(
        self,
        max_power: float,
        tau_db: float = 0.0,
    ):
        self.max_power, self.tau_db = max_power, tau_db

    def call(self, power, active):
        power = np.asarray(power, float)
        return np.minimum(power, self.max_power), np.asarray(active) & (power <= self.max_power * 10 ** (self.tau_db / 10))


OOV = -1

class Tokens(Block):
    outputs = ('payload', 'info')

    def call(self, transmission, success, payload, info):
        tokens, extra = [], []
        for t, ok, p, i in zip(transmission, success, payload, info):
            sent = np.asarray(t.message.payload)
            got = np.asarray(p) if ok else np.full(sent.shape, OOV)
            tokens.append(got)
            extra.append({**(i or {}), 'wrong_tokens': int((got != sent).sum()) if ok else None})
        return tokens, extra


def c3wm_link(
    fading,
    power_control,
    max_power: float,
    tau_db: float = 0.0,
    bits_per_token: int = 8,
    crc: bool = False,
) -> list:
    return [fading, power_control, Schedule(max_power, tau_db),
            UncodedChain(IndexCodec(2 ** bits_per_token), BPSK(), drop_corrupted=crc), Tokens(), Airtime()]
