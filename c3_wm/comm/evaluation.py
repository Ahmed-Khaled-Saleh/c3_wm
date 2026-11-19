from __future__ import annotations

import math
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .channel_data import ChannelTable, db, episode_fading, table_path
from .power_control import (Budget, LearnedChannelInversion, TAUS_DB, c3wm_link, calibrate, calibrate_layouts,
                                          evaluate, fit_predictor, outcome, predict_gain, predict_on)
from .sequence import fit_model, predictor_name

__all__ = ['LEARNED', 'REFERENCES', 'DECOMPOSED', 'LEVELS', 'QUANTILE_LEVELS', 'DISTRIBUTIONAL', 'PSI_FLOOR', 'OPTIMAL',
           'OUTAGES', 'past_gains', 'rician_quantile', 'rician_tail', 'DecomposedPredictor', 'fit_decomposed',
           'MeanSpread', 'QuantileSet', 'fit_distributional', 'fit_baselines', 'baselines', 'prediction_errors',
           'operating_points', 'frontier', 'layout_intervals', 'layout_spread', 'layout_summary', 'plot_layout_spread',
           'transmissions', 'received', 'received_at', 'TableFading', 'network_methods',
           'DistributionalChannelInversion', 'save_lambda_policies', 'load_lambda_policies', 'network_metrics',
           'run_network', 'analysis', 'compare_network', 'network_summary', 'estimates', 'snr_sweep',
           'lagrangian_policy', 'choose_lambda', 'lambda_frontier', 'optimal_reference', 'cross_fitted',
           'distributional_policies', 'evaluate_power_control', 'motion_summary', 'summary', 'guarantee_summary',
           'plot_frontiers', 'snr_summary', 'plot_snr']

LEARNED = 'geometry + history quantile (learned)'

def past_gains(data: dict, history: int) -> np.ndarray:
    return 10 ** (data['X'][:, 3:3 + history] / 10)

REFERENCES = ('large-scale CSI (oracle)', 'conditional quantile (oracle)')
DECOMPOSED = 'learned large-scale + fading statistics'

def rician_quantile(q, nu2, var) -> np.ndarray:
    from scipy.stats import ncx2
    q, nu2, var = (np.array(a, float) for a in np.broadcast_arrays(q, nu2, var))
    out = nu2.copy()
    on = var > 1e-12
    out[on] = var[on] / 2 * ncx2.ppf(q[on], 2, np.maximum(2 * nu2[on] / var[on], 1e-12))
    return out

def rician_tail(g, nu2, var) -> np.ndarray:
    from scipy.stats import ncx2
    g, nu2, var = (np.array(a, float) for a in np.broadcast_arrays(g, nu2, var))
    out = (nu2 >= g).astype(float)
    on = var > 1e-12
    out[on] = ncx2.sf(2 * g[on] / var[on], 2, np.maximum(2 * nu2[on] / var[on], 1e-12))
    return out

class DecomposedPredictor:

    def __init__(self, large, direct, quantile: float = 0.05):
        self.large, self.direct, self.quantile = large, direct, quantile

    def predict_on(self, data: dict) -> np.ndarray:
        n, k, F = data['X_possible'].shape
        X = data['X_possible'].reshape(n * k, F)
        L = self.large.predict(X)
        d = np.clip(self.direct.predict(X), 0.0, 0.999)
        rho = np.repeat(data['rho'], k)
        faded, r = np.isfinite(rho), np.nan_to_num(rho)
        still = data['possible_still'].reshape(-1) & faded
        last = 10 ** ((X[:, 3] - L) / 10)
        nu2 = np.where(still, ((1 - r) * np.sqrt(d) + r * np.sqrt(last)) ** 2, d)
        var = np.where(still, (1 - d) * (1 - r ** 2), 1 - d)
        fade = np.where(faded, rician_quantile(self.quantile, nu2, var), 1.0)
        v = db(10 ** (L / 10) * fade).reshape(n, k)
        return 10 ** (np.where(data['possible_mask'], v, np.inf).min(1) / 10)

def fit_decomposed(train: dict, quantile: float = 0.05, max_iter: int = 400) -> DecomposedPredictor:
    from sklearn.ensemble import HistGradientBoostingRegressor
    large = fit_predictor(train['X'], train['large_scale'], None, max_iter)
    direct = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.05, early_stopping=False,
                                           random_state=0).fit(train['X'], train['direct'])
    return DecomposedPredictor(large, direct, quantile)

LEVELS = np.array([0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
QUANTILE_LEVELS = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])

def _candidates(data: dict, predict) -> np.ndarray:
    n, k, F = data['X_possible'].shape
    v = np.asarray(predict(data['X_possible'].reshape(n * k, F)), float).reshape(n, k)
    return np.where(data['possible_mask'], v, np.nan)

def _row_cdf(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    K, rows = xp.shape[1], np.arange(len(x))
    k = (xp <= x[:, None]).sum(1)
    lo, hi = np.clip(k - 1, 0, K - 1), np.clip(k, 0, K - 1)
    x0, x1 = xp[rows, lo], xp[rows, hi]
    w = np.where(x1 > x0, (x - x0) / np.where(x1 > x0, x1 - x0, 1), 0.0)
    f = fp[lo] + w * (fp[hi] - fp[lo])
    return np.where(k == 0, fp[0], np.where(k == K, 1.0, f))

class MeanSpread:

    def __init__(self, mean, spread, levels=LEVELS):
        self.mean, self.spread, self.levels, self.errors = mean, spread, np.asarray(levels, float), None

    def _pessimistic(self, data: dict) -> tuple:
        mu = _candidates(data, self.mean.predict)
        sd = _candidates(data, lambda X: np.maximum(self.spread.predict(X), 0.1))
        i = np.nanargmin(mu, axis=1)
        rows = np.arange(len(i))
        return mu[rows, i], sd[rows, i]

    def calibrate(self, cal: dict):
        mu, sd = self._pessimistic(cal)
        self.errors = np.sort((db(cal['gain']) - mu) / sd)
        return self

    def quantiles(self, data: dict) -> np.ndarray:
        mu, sd = self._pessimistic(data)
        return mu[:, None] + sd[:, None] * np.quantile(self.errors, self.levels)[None, :]

    def cdf(self, data: dict, g_db: float) -> np.ndarray:
        mu, sd = self._pessimistic(data)
        return np.searchsorted(self.errors, (g_db - mu) / sd, side='right') / len(self.errors)

class QuantileSet:

    def __init__(self, models: dict):
        self.models, self.levels, self.shift = models, np.array(sorted(models)), None

    def _raw(self, data: dict) -> np.ndarray:
        return np.stack([np.nanmin(_candidates(data, self.models[q].predict), axis=1) for q in self.levels], axis=1)

    def calibrate(self, cal: dict):
        err = self._raw(cal) - db(cal['gain'])[:, None]
        self.shift = np.array([np.quantile(err[:, k], 1 - q) for k, q in enumerate(self.levels)])
        return self

    def quantiles(self, data: dict) -> np.ndarray:
        return np.sort(self._raw(data) - self.shift[None, :], axis=1)

    def cdf(self, data: dict, g_db: float) -> np.ndarray:
        q = self.quantiles(data)
        return _row_cdf(np.full(len(q), g_db), q, self.levels)

def fit_distributional(train: dict, mean_model=None, max_iter: int = 400, quantile_levels=QUANTILE_LEVELS) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    mean = mean_model if mean_model is not None else fit_predictor(train['X'], train['gain'], None, max_iter)
    spread = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.05, early_stopping=False, random_state=0).fit(
        train['X'], np.abs(db(train['gain']) - mean.predict(train['X'])))
    return {DISTRIBUTIONAL[0]: MeanSpread(mean, spread),
            DISTRIBUTIONAL[1]: QuantileSet({float(q): fit_predictor(train['X'], train['gain'], float(q), max_iter)
                                            for q in quantile_levels})}

DISTRIBUTIONAL = ('mean + spread, λ-policy', 'quantiles, λ-policy')

def fit_baselines(train: dict, history: int, quantile: float = 0.05, max_iter: int = 400, extra=(),
                  distributional: bool = True) -> dict:
    from comm_core import fit_gain_predictor
    sequences = [np.r_[p[::-1], g] for p, g in zip(past_gains(train, history), train['gain'])]
    out = dict(psi=fit_gain_predictor(sequences, history=history),
                psi_db=fit_gain_predictor(sequences, history=history, domain='db'),
                history_model=fit_predictor(train['X'][:, 3:], train['gain'], quantile, max_iter),
                mean_model=fit_predictor(train['X'], train['gain'], None, max_iter),
                extra={predictor_name(s.get('kind', 'trees'), s.get('quantile', 0.05), s.get('steps', False)): fit_model(train, **s)
                       for s in extra},
                **({'decomposed': fit_decomposed(train, quantile, max_iter)} if 'large_scale' in train else {}))
    if distributional:
        out['distributional'] = fit_distributional(train, out['mean_model'], max_iter)
    return out

PSI_FLOOR = 1e-30

def baselines(train: dict, history: int, quantile: float = 0.05, max_iter: int = 400, fitted: dict | None = None) -> dict:
    f = fitted if fitted is not None else fit_baselines(train, history, quantile, max_iter)
    return {
        "C3-WM's (linear in |h|²)": lambda d: np.array([max(f['psi'](p), PSI_FLOOR) for p in past_gains(d, history)]),
        "C3-WM's, in dB": lambda d: np.array([f['psi_db'](p) for p in past_gains(d, history)]),
        'history quantile': lambda d: 10 ** (f['history_model'].predict(d['X'][:, 3:]) / 10),
        'geometry + history mean': lambda d: predict_on(f['mean_model'], d),
        **{name: (lambda d, m=m: predict_on(m, d)) for name, m in f.get('extra', {}).items()},
        **({DECOMPOSED: lambda d: f['decomposed'].predict_on(d)} if 'decomposed' in f else {}),
        **({REFERENCES[0]: lambda d: d['large_scale'].copy(),
            REFERENCES[1]: lambda d: d['large_scale'] * rician_quantile(quantile, d['fading_nu2'], d['fading_var'])}
           if 'large_scale' in train else {}),
        'perfect CSI': lambda d: d['gain'].copy(),
    }

def prediction_errors(estimate: np.ndarray, gain: np.ndarray) -> dict:
    error = db(estimate) - db(gain)
    return dict(bias_db=float(error.mean()), spread_db=float(error.std()), coverage=float((gain >= estimate * (1 - 1e-9)).mean()))

def operating_points(cal: tuple, test: tuple, g_min: float, targets=(0.9, 0.95, 0.99), taus_db=TAUS_DB, slack: float = 0.0) -> dict:
    out = {}
    for target in targets:
        chosen = calibrate(*cal, g_min, target + slack, taus_db=taus_db)
        out[target] = dict(evaluate(*test, g_min, chosen['margin_db'], chosen['tau_db']), margin_db=chosen['margin_db'],
                           tau_db=chosen['tau_db'], met_on_calibration=chosen['met'])
    return out

def frontier(estimate: np.ndarray, gain: np.ndarray, g_min: float, taus_db=TAUS_DB) -> tuple:
    points = np.concatenate([np.c_[o['power'], o['reliability']] for o in (outcome(estimate, gain, g_min, t) for t in taus_db)])
    points = points[np.argsort(points[:, 0])]
    return points[:, 0], np.maximum.accumulate(points[:, 1])

def layout_intervals(reached, servable, layout, episode, confidence: float = 0.95, boots: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for k in np.unique(layout):
        on = (layout == k) & servable
        if not on.any():
            continue
        eps, idx = np.unique(episode[on], return_inverse=True)
        r = np.bincount(idx, weights=reached[on].astype(float), minlength=len(eps))
        n = np.bincount(idx, minlength=len(eps)).astype(float)
        draw = rng.integers(len(eps), size=(boots, len(eps)))
        ratio = r[draw].sum(1) / np.maximum(n[draw].sum(1), 1)
        a = (1 - confidence) / 2
        out[int(k)] = (float(np.quantile(ratio, a)), float(np.quantile(ratio, 1 - a)), float(ratio.std()))
    return out

def layout_spread(estimate, gain, layout, g_min: float, margin_db: float, tau_db: float, target: float | None = None,
                  min_servable: int = 20, episode=None, confidence: float = 0.95) -> dict:
    layout = np.asarray(layout)
    per, count = {}, {}
    for k in np.unique(layout):
        on = layout == k
        n = int((gain[on] >= g_min).sum())
        if n:
            per[int(k)] = float(outcome(estimate[on], gain[on], g_min, tau_db, [margin_db])['reliability'][0])
            count[int(k)] = n
    r = np.array([per[k] for k in per if count[k] >= min_servable])
    out = dict(per_layout=per, servable=count, layouts=len(r))
    if len(r):
        out.update(mean=float(r.mean()), std=float(r.std()), min=float(r.min()), p10=float(np.percentile(r, 10)),
                   median=float(np.median(r)))
        if target is not None:
            out.update(below=float((r < target).mean()), shortfall=float(max(target - r.min(), 0.0)))
        if episode is not None:
            power, sent = transmissions(estimate, g_min, margin_db, tau_db)
            servable = gain >= g_min
            iv = layout_intervals(sent & (power >= (1 / gain) * (1 - 1e-9)), servable, layout, np.asarray(episode), confidence)
            keep = [k for k in per if count[k] >= min_servable]
            se = np.array([iv[k][2] for k in keep])
            out.update(interval={k: iv[k][:2] for k in keep},
                       between_std=float(math.sqrt(max(r.var() - (se ** 2).mean(), 0.0))))
            if target is not None:
                out['significantly_below'] = float(np.mean([iv[k][1] < target for k in keep]))
    return out

def layout_summary(results: dict) -> str:
    names = [n for n, r in results['predictors'].items() if 'layouts' in r]
    targets = list(results['predictors'][names[0]]['layouts'])
    w = max(map(len, names)) + 2
    lines = [f"reliability per test layout, κ and τ chosen on the calibration layouts "
             f"({results['predictors'][names[0]]['layouts'][targets[0]]['layouts']} layouts)",
             f"{'':{w}s}" + ''.join(f"{f'{100 * t:g}%: pooled / mean ± std (between) / p10 / min / below (significantly)':>72s}"
                                    for t in targets)]
    for name in names:
        row = f"{name:{w}s}"
        for t in targets:
            s, pooled = results['predictors'][name]['layouts'][t], results['predictors'][name]['targets'][t]['reliability']
            row += (f"{100 * pooled:22.1f}% {100 * s['mean']:6.1f} ± {100 * s['std']:4.1f} ({100 * s.get('between_std', math.nan):4.1f})"
                    f" {100 * s['p10']:6.1f} {100 * s['min']:6.1f} {100 * s['below']:5.0f}% ({100 * s.get('significantly_below', math.nan):3.0f}%)  ")
        lines.append(row)
    lines.append('below: the share of the test layouts under the target (significantly: their interval over episodes entirely under it); '
                 'between: the std between layouts without the sampling noise')
    return '\n'.join(lines)

def plot_layout_spread(results: dict, target: float, path=None, ax=None):
    import matplotlib.pyplot as plt
    names = [n for n, r in results['predictors'].items() if 'layouts' in r and n != 'perfect CSI']
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        s = results['predictors'][name]['layouts'][target]
        r = 100 * np.array([s['per_layout'][k] for k in s['per_layout'] if s['servable'][k] >= 1])
        ax.plot(i + rng.uniform(-0.15, 0.15, len(r)), r, 'o', ms=5, alpha=0.7, color='#1f77b4' if name == LEARNED else '#7f7f7f')
        ax.plot([i - 0.3, i + 0.3], [100 * results['predictors'][name]['targets'][target]['reliability']] * 2, 'k-', lw=2)
    ax.axhline(100 * target, color='#d62728', lw=1, ls='--', label=f'target {100 * target:g}%')
    ax.set_xticks(range(len(names)), [n.replace(' (learned)', '\n(learned)').replace(', ', ',\n') for n in names], fontsize=8)
    ax.set(ylabel='servable transmissions reaching Γ (%)', title=f'Per test layout (dots) and pooled (bars), κ and τ for {100 * target:g}%')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    if path is not None:
        ax.figure.savefig(path, dpi=120, bbox_inches='tight')
    return ax

def transmissions(estimate: np.ndarray, g_min: float, margin_db: float, tau_db: float) -> tuple:
    rho = 10 ** (margin_db / 10) / np.maximum(estimate, 1e-30)
    sent = rho <= 10 ** (tau_db / 10) / g_min if math.isfinite(tau_db) else np.ones(len(rho), bool)
    return np.where(sent, np.minimum(rho, 1 / g_min), 0.0), sent

def received(estimate, gain, g_min: float, margin_db: float, tau_db: float, snr_target_db: float,
             message_bits: int = 392, bits_per_token: int = 8) -> dict:
    power, sent = transmissions(estimate, g_min, margin_db, tau_db)
    return received_at(np.where(sent, power, 0.0), gain, g_min, snr_target_db, message_bits, bits_per_token)

def received_at(power, gain, g_min: float, snr_target_db: float, message_bits: int = 392, bits_per_token: int = 8) -> dict:
    sent = power > 0
    snr = 10 ** (snr_target_db / 10) * power * gain
    ber = 0.5 * np.vectorize(math.erfc)(np.sqrt(snr))
    servable = gain >= g_min
    clean = np.where(sent, (1 - ber) ** message_bits, 0.0)[servable].mean()
    wrong = (1 - (1 - ber[sent & servable]) ** bits_per_token).mean() if (sent & servable).any() else float('nan')
    return dict(error_free=float(clean), wrong_tokens=float(wrong))

from comm_core import Fading

class TableFading(Fading):

    def __init__(self, table: ChannelTable, channel: str = 'coherent'):
        self.table, self.channel, self.node_positions, self.fades = table, channel, {}, {}

    def set_positions(self, positions):
        self.node_positions.update({k: tuple(int(v) for v in p) for k, p in positions.items()})

    def coefficient(self, transmission, link) -> complex:
        a, b = self.node_positions[transmission.sender], self.node_positions[transmission.receiver]
        if a == b:
            return math.inf
        fade = self.fades.get(tuple(sorted((transmission.sender, transmission.receiver))), 1.0)
        return complex(math.sqrt(self.table.gain(a, b, self.channel) * fade))

def network_methods(
    saved: dict,
    fitted: dict,
    comparison: dict | None = None,
    target: float = 0.99,
) -> dict:
    from comm_core import ChannelInversion, PredictedChannelInversion
    b, L = saved['budget'], saved['history']
    snr, rho = b.snr_target_db, b.max_power_w
    def predicted(psi, margin):
        floored = lambda past: max(psi(past), PSI_FLOOR)
        return lambda walls, positions, headings: PredictedChannelInversion(floored, L, snr, fallback_power=rho, margin_db=margin)
    def learned(model, margin, geometry=True):
        return lambda walls, positions, headings: LearnedChannelInversion(model, walls, positions, headings, L, snr, margin,
                                                                          rho, geometry)
    makers = {LEARNED: lambda m: learned(saved['model'], m),
              "C3-WM's (linear in |h|²)": lambda m: predicted(fitted['psi'], m),
              "C3-WM's, in dB": lambda m: predicted(fitted['psi_db'], m),
              'history quantile': lambda m: learned(fitted['history_model'], m, geometry=False),
              'geometry + history mean': lambda m: learned(fitted['mean_model'], m),
              **{name: (lambda m, model=model: learned(model, m)) for name, model in fitted.get('extra', {}).items()}}
    method = lambda make, margin, tau, predictor, crc=False: dict(make=make, margin_db=float(margin), tau_db=float(tau),
                                                                  crc=crc, predictor=predictor)
    out = {'perfect CSI': method(lambda walls, positions, headings: ChannelInversion(snr), 0.0, 0.0, 'perfect CSI'),
           'C3-WM as written (Ψ, κ = 0, δ)': method(predicted(fitted['psi'], 0.0), 0.0, 0.0, "C3-WM's (linear in |h|²)")}
    points = ({name: r['targets'][target] for name, r in comparison['predictors'].items()} if comparison is not None
              else {LEARNED: saved['calibration']})
    for name, make in makers.items():
        if name in points:
            p = points[name]
            out[f'{name}, κ, τ'] = method(make(p['margin_db']), p['margin_db'], p['tau_db'], name)
    p = points[LEARNED]
    out[f'{LEARNED}, κ, τ, CRC'] = method(makers[LEARNED](p['margin_db']), p['margin_db'], p['tau_db'], LEARNED, crc=True)
    for name, dist in fitted.get('distributional', {}).items():
        if comparison is None or target not in comparison.get('distributional', {}).get(name, {}):
            continue
        lam = comparison['distributional'][name][target]['lam']
        make = lambda walls, positions, headings, dist=dist, lam=lam: DistributionalChannelInversion(
            dist, lam, b.g_min, walls, positions, headings, L, snr, rho)
        out[f'{name}, λ'] = dict(method(make, math.nan, math.inf, name),
                                 powers=lambda d, dist=dist, lam=lam: _optimal_choice(_distribution_options(dist, d, b.g_min), lam))
    return out

class DistributionalChannelInversion(LearnedChannelInversion):

    def __init__(self, dist, lam: float, g_min: float, walls, positions, headings, history: int = 4,
                 target_snr_db: float = 10.0, fallback_power: float = math.inf):
        super().__init__(None, walls, positions, headings, history, target_snr_db, 0.0, fallback_power)
        self.dist, self.lam, self.g_min = dist, lam, g_min

    @classmethod
    def from_saved(cls, policies: dict, name: str, target: float, walls, positions, headings):
        p, b = policies['policies'][name], policies['budget']
        return cls(p['dist'], p['lam'][float(target)], b.g_min, walls, positions, headings, policies['history'],
                   b.snr_target_db, b.max_power_w)

    def allocate(self, gain, noise_power, link=None, message=None):
        past = self._past.setdefault((link.sender, link.receiver), [])
        if len(past) < self.history or len(self._rounds) < 3:
            power, send = self.fallback_power, True
        else:
            X = self._rows(link, past)
            data = {'X_possible': X[None], 'possible_mask': np.ones((1, len(X)), bool)}
            units = float(_optimal_choice(_distribution_options(self.dist, data, self.g_min), self.lam)[0])
            power, send = units * 10 ** (self.target_snr_db / 10) * noise_power, units > 0
        past.append((float(gain), self._round))
        return power, send

def save_lambda_policies(path, fitted: dict, comparison: dict, saved: dict):
    policies = {name: dict(dist=fitted['distributional'][name], lam={float(t): float(o['lam']) for t, o in by_target.items()})
                for name, by_target in comparison.get('distributional', {}).items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(dict(policies=policies, history=saved['history'], budget=asdict(saved['budget'])), f)

def load_lambda_policies(path) -> dict:
    with open(path, 'rb') as f:
        out = pickle.load(f)
    out['budget'] = Budget(**out['budget'])
    return out

def network_metrics(rows: list, budget: Budget, tokens: int, cost: list) -> dict:
    g, p, sent, ok = (np.array([r[k] for r in rows], float if k < 2 else bool) for k in range(4))
    wrong = np.array([r[4] if r[4] is not None else 0 for r in rows])
    servable, snr = g >= budget.g_min, p * g / budget.noise_w
    gamma = 10 ** (budget.snr_target_db / 10)
    perfect = np.where(servable, gamma * budget.noise_w / g, 0.0).mean()
    return dict(reached=float((sent & (snr >= gamma * (1 - 1e-9)))[servable].mean()), power=float(p.mean() / perfect),
                withheld=float(1 - sent.mean()), intact=float((ok & (wrong == 0))[servable].mean()),
                wrong=float((wrong[ok] / tokens).mean()) if ok.any() else 0.0,
                garbled=float((ok & (wrong > 0))[~servable].mean()) if (~servable).any() else 0.0,
                cost=float(np.mean(cost)), transmissions=len(rows))

def run_network(
    method: dict,
    saved: dict,
    layout_dir,
    trajectory_dir,
    seeds,
    fading: str | None = None,
    fading_correlation: float = 0.9,
    symmetric: bool = True,
    message_bits: int = 392,
    bits_per_token: int = 8,
    seed: int = 0,
    channel: str = 'coherent',
    skip: int | None = None,
) -> dict:
    with threadpool_limits(1):
        return _run_network(method, saved, layout_dir, trajectory_dir, seeds, fading, fading_correlation, symmetric,
                            message_bits, bits_per_token, seed, channel, skip)

def _run_network(method, saved, layout_dir, trajectory_dir, seeds, fading, fading_correlation, symmetric, message_bits,
                 bits_per_token, seed, channel, skip) -> dict:
    from comm_core import FullMeshTopology, IdealProtocol, Message, Network, Node, PipelineBackend
    budget, history = saved['budget'], saved['history'] if skip is None else skip
    rng = np.random.default_rng([seed, 1])
    tokens = message_bits // bits_per_token
    rows, cost = [], []
    for layout in seeds:
        table = ChannelTable(table_path(layout_dir, layout), symmetric=symmetric)
        z = np.load(Path(trajectory_dir) / f'trajectories_{layout}.npz')
        agents = [f'agent_{i}' for i in range(z['cells'].shape[2])]
        fading_channel = TableFading(table, channel)
        fading_rng = np.random.default_rng([seed, int(layout)])
        heads = {}
        control = method['make'](table.walls, lambda: fading_channel.node_positions, lambda: heads)
        backend = PipelineBackend(c3wm_link(fading_channel, control, budget.max_power_w, method['tau_db'], bits_per_token, method['crc']),
                                  noise_power=budget.noise_w)
        net = Network([Node(a) for a in agents], FullMeshTopology(agents), IdealProtocol(), backend)
        for cells, dirs in zip(z['cells'], z['dirs']):
            net.reset(seed=int(rng.integers(2 ** 31)))
            fading_channel.node_positions.clear()
            fades = episode_fading(cells, fading_rng, fading_correlation, table) if fading == 'rician' else {}
            for t in range(len(cells)):
                if (cells[t, :, 0] < 0).any():
                    break
                backend.set_positions({a: tuple(int(v) for v in c) for a, c in zip(agents, cells[t])})
                fading_channel.fades = {(agents[i], agents[j]): f[t] for (i, j), f in fades.items()}
                heads.update({a: int(d) for a, d in zip(agents, dirs[t])})
                for a in agents:
                    for b in agents:
                        if a != b:
                            net.send(Message(a, b, payload=rng.integers(2 ** bits_per_token, size=tokens)))
                for r in net.step():
                    if t >= max(history, 2):
                        rows.append((10 ** (r.metadata['channel_gain_db'] / 10), r.metadata['tx_power'],
                                     not r.metadata.get('outage', False), r.success, r.metadata.get('wrong_tokens')))
            cost.append(net.metrics.summary().bits_sent / 8 / 1000)
    return network_metrics(rows, budget, tokens, cost)

def analysis(estimate, gain, budget: Budget, margin_db: float, tau_db: float, crc: bool = False, message_bits: int = 392,
             power=None) -> dict:
    if power is None:
        r = evaluate(estimate, gain, budget.g_min, margin_db, tau_db)
        power, sent = transmissions(estimate, budget.g_min, margin_db, tau_db)
    else:
        power, sent = np.asarray(power, float), np.asarray(power) > 0
        r = _judged(power, {'gain': gain}, budget.g_min)
    snr = 10 ** (budget.snr_target_db / 10) * power * gain
    clean = (1 - 0.5 * np.vectorize(math.erfc)(np.sqrt(snr))) ** message_bits
    servable = gain >= budget.g_min
    garbled = 0.0 if crc or not (~servable).any() else float(np.where(sent, 1 - clean, 0.0)[~servable].mean())
    return dict(reached=r['reliability'], power=r['power'], withheld=r['withheld'],
                intact=float(np.where(sent, clean, 0.0)[servable].mean()), garbled=garbled)

def compare_network(
    saved: dict,
    data: dict,
    layout_dir,
    trajectory_dir,
    seeds,
    comparison: dict | None = None,
    target: float = 0.99,
    fitted: dict | None = None,
    quantile: float = 0.05,
    max_iter: int = 400,
    **network,
) -> dict:
    fitted = fitted if fitted is not None else fit_baselines(data['train'], saved['history'], quantile, max_iter)
    methods = network_methods(saved, fitted, comparison, target)
    predict = {**baselines(data['train'], saved['history'], fitted=fitted),
               LEARNED: lambda d: predict_on(saved['model'], d)}
    on = np.isin(data['test']['layout'], list(seeds))
    test = {k: v[on] for k, v in data['test'].items()}
    out = {}
    for name, m in methods.items():
        out[name] = dict(margin_db=m['margin_db'], tau_db=m['tau_db'], crc=m['crc'],
                         network=run_network(m, saved, layout_dir, trajectory_dir, seeds, **network),
                         analysis=analysis(None if 'powers' in m else predict[m['predictor']](test), test['gain'], saved['budget'],
                                           m['margin_db'], m['tau_db'], m['crc'], network.get('message_bits', 392),
                                           m['powers'](test) if 'powers' in m else None))
    return out

def network_summary(results: dict) -> str:
    label = lambda tau: '∞' if math.isinf(tau) else f'{tau:g}'
    w = max(map(len, results)) + 2
    head = (f"{'':{w}s}{'κ (dB)':>7s}{'τ':>5s}{'reached':>9s}{'power':>9s}{'withheld':>9s}{'intact':>8s}{'wrong':>8s}"
            f"{'garbled':>9s}{'cost':>12s}")
    lines = [head]
    for name, r in results.items():
        n, a = r['network'], r['analysis']
        lines.append(f"{name:{w}s}{r['margin_db']:7.2f}{label(r['tau_db']):>5s}{100 * n['reached']:8.1f}%{n['power']:8.3g}×"
                     f"{100 * n['withheld']:8.1f}%{100 * n['intact']:7.1f}%{100 * n['wrong']:7.2f}%{100 * n['garbled']:8.1f}%"
                     f"{n['cost']:7.2f} KB/ep")
        lines.append(f"{'  (analysis)':{w + 12}s}{100 * a['reached']:8.1f}%{a['power']:8.3g}×{100 * a['withheld']:8.1f}%"
                     f"{100 * a['intact']:7.1f}%{'':8s}{100 * a['garbled']:8.1f}%")
    lines.append("reached: servable transmissions reaching Γ; power: × perfect CSI's; intact: servable ones with every token "
                 "right; wrong: wrong tokens in what arrived; garbled: unservable ones arriving with wrong tokens")
    return '\n'.join(lines)

def estimates(
    saved: dict,
    data: dict,
    compare: bool = True,
    quantile: float = 0.05,
    max_iter: int = 400,
    fitted: dict | None = None,
) -> dict:
    out = {LEARNED: lambda d: predict_on(saved['model'], d)}
    if compare:
        out.update(baselines(data['train'], saved['history'], quantile, max_iter, fitted))
    out['perfect CSI'] = lambda d: d['gain'].copy()
    return {name: (predict(data['calibration']), predict(data['test'])) for name, predict in out.items()}

def _operating(name, cal, test, budget: Budget, targets, taus_db, slack, message_bits, bits_per_token) -> dict:
    if name == 'perfect CSI':
        points = {t: dict(evaluate(*test, budget.g_min, 0.0, 0.0), margin_db=0.0, tau_db=0.0, met_on_calibration=True)
                  for t in targets}
    else:
        points = operating_points(cal, test, budget.g_min, targets, taus_db, slack)
    return {t: dict(p, **received(*test, budget.g_min, p['margin_db'], p['tau_db'], budget.snr_target_db,
                                  message_bits, bits_per_token)) for t, p in points.items()}

def snr_sweep(
    estimated: dict,
    data: dict,
    budget: Budget,
    snr_targets_db=(4, 7, 10, 13, 16),
    targets=(0.9, 0.95, 0.99),
    taus_db=TAUS_DB,
    slack: float = 0.0,
    message_bits: int = 392,
    bits_per_token: int = 8,
    fitted: dict | None = None,
    optimal: bool = False,
) -> dict:
    cal, test = data['calibration'], data['test']
    out = {}
    for snr in snr_targets_db:
        b = replace(budget, snr_target_db=float(snr))
        policy = dict(snr_target_db=float(snr), message_bits=message_bits, bits_per_token=bits_per_token, per_layout=False)
        policies = {OPTIMAL: optimal_reference(cal, test, b.g_min, targets, slack, **policy)} \
            if optimal and 'large_scale' in test else {}
        if fitted is not None and fitted.get('distributional'):
            policies.update(distributional_policies(fitted, cal, test, b.g_min, targets, slack, **policy))
        out[float(snr)] = dict(
            g_min_db=float(db(b.g_min)), servable=float((test['gain'] >= b.g_min).mean()),
            predictors={name: _operating(name, (c, cal['gain']), (t, test['gain']), b, targets, taus_db, slack, message_bits,
                                         bits_per_token) for name, (c, t) in estimated.items()},
            policies=policies)
    return out

OPTIMAL = 'optimal policy (oracle)'
OUTAGES = np.geomspace(1e-5, 0.6, 64)

def _optimal_options(data: dict, g_min: float, outages=OUTAGES) -> tuple:
    L, nu2, var = data['large_scale'][:, None], data['fading_nu2'][:, None], data['fading_var'][:, None]
    power = 1 / (L * rician_quantile(outages[None, :], nu2, var))
    success = np.broadcast_to(1 - outages, power.shape)
    power = np.where(power <= 1 / g_min, power, np.inf)
    cap = rician_tail(g_min / L, nu2, var)
    return np.c_[power, np.full(len(L), 1 / g_min)], np.c_[success, cap]

def _optimal_choice(options: tuple, lam: float) -> np.ndarray:
    power, success = options
    cost = power - lam * success
    k = np.argmin(cost, axis=1)
    best = cost[np.arange(len(k)), k]
    return np.where(best < 0, power[np.arange(len(k)), k], 0.0)

def _judged(power: np.ndarray, data: dict, g_min: float, keep=None) -> dict:
    keep = np.ones(len(power), bool) if keep is None else keep
    g, p = data['gain'][keep], power[keep]
    servable = g >= g_min
    reached = (p > 0) & (p * g >= 1 - 1e-9)
    return dict(reliability=float(reached[servable].mean()), power=float(p.mean() / np.where(servable, 1 / g, 0).mean()),
                withheld=float((p == 0).mean()))

def lagrangian_policy(
    cal_options: tuple, test_options: tuple,
    cal: dict, test: dict,
    g_min: float,
    targets=(0.9, 0.95, 0.99),
    slack: float = 0.0,
    snr_target_db: float | None = None,
    message_bits: int = 392, bits_per_token: int = 8,
    per_layout: bool = True,
) -> dict:
    from c3_wm.comm.deployment import judge
    parts = (('moved', test['fresh']), ('still', ~test['fresh'])) if 'fresh' in test else ()
    out = {}
    for target in targets:
        lam = choose_lambda(cal_options, cal, g_min, target + slack)
        power = _optimal_choice(test_options, lam)
        layouts = judge(power, power > 0, test['gain'], test['layout'], g_min, target, episode=test.get('episode')) \
            if per_layout else {}
        out[target] = dict(lam=lam, **_judged(power, test, g_min),
                           **(received_at(power, test['gain'], g_min, snr_target_db, message_bits, bits_per_token)
                              if snr_target_db is not None else {}),
                           **{k: v for k, v in layouts.items() if k.startswith('layout_')},
                           motion={part: dict(share=float(m.mean()), **_judged(power, test, g_min, m)) for part, m in parts if m.any()})
    return out

def choose_lambda(options: tuple, data: dict, g_min: float, reliability: float) -> float:
    scale = float(np.mean(1 / data['gain'][data['gain'] >= g_min]))
    lo, hi = -4.0, 8.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if _judged(_optimal_choice(options, scale * 10 ** mid), data, g_min)['reliability'] >= reliability:
            hi = mid
        else:
            lo = mid
    return scale * 10 ** hi

def lambda_frontier(options: tuple, data: dict, g_min: float, points: int = 121) -> tuple:
    scale = float(np.mean(1 / data['gain'][data['gain'] >= g_min]))
    judged = [_judged(_optimal_choice(options, scale * 10 ** e), data, g_min) for e in np.linspace(-4, 8, points)]
    points = np.array(sorted((j['power'], j['reliability']) for j in judged))
    return points[:, 0], np.maximum.accumulate(points[:, 1])

def optimal_reference(
    cal: dict, test: dict,
    g_min: float,
    targets=(0.9, 0.95, 0.99),
    slack: float = 0.0,
    outages=OUTAGES,
    **policy,
) -> dict:
    return lagrangian_policy(_optimal_options(cal, g_min, outages), _optimal_options(test, g_min, outages), cal, test,
                             g_min, targets, slack, **policy)

def _distribution_options(dist, data: dict, g_min: float) -> tuple:
    power = 10 ** (-dist.quantiles(data) / 10)
    success = np.broadcast_to(1 - dist.levels, power.shape)
    power = np.where(power <= 1 / g_min, power, np.inf)
    cap = 1 - dist.cdf(data, float(db(g_min)))
    return np.c_[power, np.full(len(power), 1 / g_min)], np.c_[success, cap]

def _subset(data: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in data.items()}

def cross_fitted(dist, cal: dict, g_min: float) -> tuple:
    ids = np.unique(cal['layout'])
    if len(ids) < 2:
        dist.calibrate(cal)
        return _distribution_options(dist, cal, g_min), cal
    half = np.isin(cal['layout'], ids[::2])
    options, parts = [], []
    for fit_on, judged_on in ((half, ~half), (~half, half)):
        dist.calibrate(_subset(cal, fit_on))
        options.append(_distribution_options(dist, _subset(cal, judged_on), g_min))
        parts.append(_subset(cal, judged_on))
    dist.calibrate(cal)
    return (tuple(np.concatenate([o[i] for o in options]) for i in range(2)),
            {k: np.concatenate([p[k] for p in parts]) for k in cal})

def distributional_policies(
    fitted: dict,
    cal: dict, test: dict,
    g_min: float, targets=(0.9, 0.95, 0.99), slack: float = 0.0,
    **policy,
) -> dict:
    out = {}
    for name, dist in fitted.get('distributional', {}).items():
        cal_options, cal_order = cross_fitted(dist, cal, g_min)
        out[name] = lagrangian_policy(cal_options, _distribution_options(dist, test, g_min), cal_order, test, g_min,
                                      targets, slack, **policy)
    return out

def evaluate_power_control(
    saved: dict,
    data: dict,
    targets=(0.9, 0.95, 0.99),
    taus_db=TAUS_DB,
    slack: float = 0.0,
    quantile: float = 0.05,
    max_iter: int = 400,
    message_bits: int = 392,
    bits_per_token: int = 8,
    compare: bool = True,
    snr_targets_db=(),
    fitted: dict | None = None,
    layout_coverage: float | None = None,
    layout_confidence: float | None = None,
    min_servable: int = 20,
) -> dict:
    budget = saved['budget']
    estimated = estimates(saved, data, compare, quantile, max_iter, fitted)
    test_gain = data['test']['gain']
    results = {name: dict(prediction=prediction_errors(t, test_gain),
                          targets=_operating(name, (c, data['calibration']['gain']), (t, test_gain), budget, targets,
                                             taus_db, slack, message_bits, bits_per_token),
                          frontier=[a.tolist() for a in frontier(t, test_gain, budget.g_min, taus_db)])
               for name, (c, t) in estimated.items()}
    for name, (c, t) in estimated.items():
        results[name]['layouts'] = {target: layout_spread(t, test_gain, data['test']['layout'], budget.g_min, p['margin_db'],
                                                          p['tau_db'], target, min_servable, data['test'].get('episode'))
                                    for target, p in results[name]['targets'].items()}
    if 'fresh' in data['test']:
        parts = {'moved': data['test']['fresh'], 'still': ~data['test']['fresh']}
        for name, (c, t) in estimated.items():
            results[name]['motion'] = {
                part: dict(share=float(m.mean()), prediction=prediction_errors(t[m], test_gain[m]),
                           targets={target: evaluate(t[m], test_gain[m], budget.g_min, p['margin_db'], p['tau_db'])
                                    for target, p in results[name]['targets'].items()})
                for part, m in parts.items() if m.any()}
    if layout_coverage is not None:
        cal = data['calibration']
        for name, (c, t) in estimated.items():
            if name == 'perfect CSI':
                continue
            results[name]['layout_targets'] = {}
            for target in targets:
                ch = calibrate_layouts(c, cal['gain'], cal['layout'], budget.g_min, target, layout_coverage, layout_confidence,
                                       taus_db=taus_db, min_servable=min_servable)
                results[name]['layout_targets'][target] = dict(
                    evaluate(t, test_gain, budget.g_min, ch['margin_db'], ch['tau_db']), margin_db=ch['margin_db'],
                    tau_db=ch['tau_db'], met_on_calibration=ch['met'], calibration_below=ch['below'], allowed=ch['allowed'],
                    calibration_layouts=ch['layouts'],
                    layouts=layout_spread(t, test_gain, data['test']['layout'], budget.g_min, ch['margin_db'], ch['tau_db'],
                                          target, min_servable, data['test'].get('episode')))
    out = dict(servable=float((test_gain >= budget.g_min).mean()), transmissions=len(test_gain), predictors=results,
               layout_coverage=layout_coverage, layout_confidence=layout_confidence)
    if compare and 'large_scale' in data['test']:
        out['optimal'] = optimal_reference(data['calibration'], data['test'], budget.g_min, targets, slack,
                                           snr_target_db=budget.snr_target_db, message_bits=message_bits,
                                           bits_per_token=bits_per_token)
    if compare and fitted is not None and fitted.get('distributional'):
        out['distributional'] = distributional_policies(fitted, data['calibration'], data['test'], budget.g_min, targets, slack,
                                                        snr_target_db=budget.snr_target_db, message_bits=message_bits,
                                                        bits_per_token=bits_per_token)
    frontiers = {}
    if 'optimal' in out:
        frontiers[OPTIMAL] = lambda_frontier(_optimal_options(data['test'], budget.g_min), data['test'], budget.g_min)
    for name in out.get('distributional', {}):
        frontiers[name] = lambda_frontier(_distribution_options(fitted['distributional'][name], data['test'], budget.g_min),
                                          data['test'], budget.g_min)
    if frontiers:
        out['frontiers'] = {name: [a.tolist() for a in f] for name, f in frontiers.items()}
    if len(snr_targets_db):
        out['snr'] = snr_sweep(estimated, data, budget, snr_targets_db, targets, taus_db, slack, message_bits, bits_per_token,
                               fitted if compare else None, optimal=compare)
    return out

def motion_summary(results: dict, target: float) -> str:
    names = [n for n, r in results['predictors'].items() if 'motion' in r]
    if not names:
        return ''
    parts = list(results['predictors'][names[0]]['motion'])
    w = max(map(len, names)) + 2
    shares = ', '.join(f"{p} {100 * results['predictors'][names[0]]['motion'][p]['share']:.0f}%" for p in parts)
    lines = [f"{100 * target:g}% (κ and τ of all the transmissions), {shares} of the test transmissions",
             f"{'':{w}s}" + ''.join(f"{p + ': spread / reached / power':>34s}" for p in parts)]
    for name in names:
        m = results['predictors'][name]['motion']
        lines.append(f"{name:{w}s}" + ''.join(f"{m[p]['prediction']['spread_db']:13.1f} dB {100 * m[p]['targets'][target]['reliability']:6.1f}%"
                                               f"{m[p]['targets'][target]['power']:7.2f}×   " for p in parts))
    policies = {**({OPTIMAL: results['optimal']} if 'optimal' in results else {}), **results.get('distributional', {})}
    for name, by_target in policies.items():
        if target not in by_target:
            continue
        o = by_target[target]
        lines.append(f"{name:{w}s}" + ''.join(f"{'':16s}{100 * o['motion'][p]['reliability']:6.1f}%{o['motion'][p]['power']:7.2f}×   "
                                              for p in parts if p in o['motion']))
        lines.append(f"{'':4s}all: {100 * o['reliability']:.1f}% reached, {o['power']:.2f}× perfect CSI's power, "
                     f"{100 * o['withheld']:.1f}% withheld")
    return '\n'.join(lines)

def summary(results: dict) -> str:
    names = list(results['predictors'])
    targets = list(results['predictors'][names[0]]['targets'])
    w = max(map(len, names)) + 2
    head = f"{'':{w}s}{'bias':>7s}{'spread':>8s}{'cover':>7s}" + ''.join(f'{f"{100 * t:g}%: reached / power / error-free":>36s}' for t in targets)
    lines = [f"{results['transmissions']} test transmissions, {100 * results['servable']:.1f}% servable", head]
    for name in names:
        r = results['predictors'][name]
        p = r['prediction']
        row = f"{name:{w}s}{p['bias_db']:+6.1f} {p['spread_db']:6.1f} {100 * p['coverage']:6.1f}%"
        for t in targets:
            o = r['targets'][t]
            flag = ' ' if o['met_on_calibration'] else '✗'
            row += f"{100 * o['reliability']:13.1f}% {o['power']:6.1f}× {100 * o['error_free']:6.1f}%{flag}  "
        lines.append(row)
    lines.append('✗: no margin and threshold reach the target on the calibration layouts (the most reliable is used)')
    return '\n'.join(lines)

def guarantee_summary(results: dict) -> str:
    names = [n for n, r in results['predictors'].items() if 'layout_targets' in r]
    if not names:
        return ''
    targets = list(results['predictors'][names[0]]['layout_targets'])
    w = max(map(len, names)) + 2
    cov, conf = results['layout_coverage'], results['layout_confidence']
    lines = [f"κ and τ chosen pooled, and over layouts for {100 * cov:g}% of them"
             + (f" with {100 * conf:g}% confidence" if conf else " (in expectation)")
             + ": reached (pooled) / power / test layouts reaching the target / significantly below",
             f"{'':{w}s}" + ''.join(f"{f'{100 * t:g}%  pooled':>36s}{'over layouts':>36s}" for t in targets)]
    for name in names:
        r, row = results['predictors'][name], f"{name:{w}s}"
        for t in targets:
            for o, s in ((r['targets'][t], r['layouts'][t]), (r['layout_targets'][t], r['layout_targets'][t]['layouts'])):
                flag = ' ' if o['met_on_calibration'] else '✗'
                row += (f"{100 * o['reliability']:13.1f}% {o['power']:6.2f}× {100 * (1 - s.get('below', math.nan)):5.0f}% "
                        f"{100 * s.get('significantly_below', math.nan):5.0f}%{flag}")
        lines.append(row)
    lines.append('✗: not met on the calibration layouts (over layouts: more of them miss than allowed, at any κ and τ)')
    return '\n'.join(lines)

def plot_frontiers(results: dict, path=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in results['predictors'].items():
        if name == 'perfect CSI':
            continue
        power, reliability = r['frontier']
        line, = ax.plot(power, 100 * np.asarray(reliability), lw=2.2 if name == LEARNED else 1.3, label=name)
        for o in r['targets'].values():
            ax.plot(o['power'], 100 * o['reliability'], 'o', color=line.get_color(), ms=5)
    policies = {**({OPTIMAL: results['optimal']} if 'optimal' in results else {}), **results.get('distributional', {})}
    for name, (power, reliability) in results.get('frontiers', {}).items():
        line, = ax.plot(power, 100 * np.asarray(reliability), '--', lw=1.8, label=f'{name} (λ)')
        for o in policies.get(name, {}).values():
            ax.plot(o['power'], 100 * o['reliability'], 's', color=line.get_color(), ms=5)
    ax.plot(1, 100, 'k*', ms=12, label='perfect CSI')
    ax.set(xscale='log', xlim=(0.5, 50), ylim=(80, 100.5), xlabel='mean power, × perfect CSI',
           ylabel='servable transmissions reaching Γ (%)',
           title='Test layouts: frontiers (lines; dashed: λ swept) and calibrated operating points')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if path is not None:
        ax.figure.savefig(path, dpi=120, bbox_inches='tight')
    return ax

def snr_summary(sweep: dict) -> str:
    first = next(iter(sweep.values()))
    targets = list(next(iter(first['predictors'].values())))
    w = max(len(n) for r in sweep.values() for n in {**r['predictors'], **r.get('policies', {})}) + 2
    head = f"{'Γ (dB)':>7s}{'g_min':>8s}{'servable':>10s}  {'':{w}s}" + ''.join(f'{f"{100 * t:g}%: reached / power / error-free":>36s}' for t in targets)
    lines = [head]
    for snr, r in sweep.items():
        for i, (name, points) in enumerate({**r['predictors'], **{f'{n} (λ)': p for n, p in r.get('policies', {}).items()}}.items()):
            row = (f"{snr:7g}{r['g_min_db']:7.0f} {100 * r['servable']:8.1f}%  " if i == 0 else ' ' * 27) + f"{name:{w}s}"
            for t in targets:
                o = points[t]
                flag = ' ' if o.get('met_on_calibration', True) else '✗'
                row += f"{100 * o['reliability']:13.1f}% {o['power']:6.1f}× {100 * o['error_free']:6.1f}%{flag}  "
            lines.append(row)
    lines.append('g_min: the weakest servable gain (dB). ✗: no margin and threshold reach the target on the calibration layouts')
    return '\n'.join(lines)

def plot_snr(sweep: dict, path=None):
    import matplotlib.pyplot as plt
    snrs = list(sweep)
    names = list(sweep[snrs[0]]['predictors'])
    policies = list(sweep[snrs[0]].get('policies', {}))
    targets = list(sweep[snrs[0]]['predictors'][names[0]])
    fig, axes = plt.subplots(len(targets), 3, figsize=(13, 3.2 * len(targets)), squeeze=False)
    for row, t in zip(axes, targets):
        for name in names:
            points = [sweep[s]['predictors'][name][t] for s in snrs]
            style = dict(lw=2.2 if name == LEARNED else 1.3, marker='o', ms=3, label=name,
                         **(dict(color='k', ls='--') if name == 'perfect CSI' else {}))
            row[0].plot(snrs, [100 * p['reliability'] for p in points], **style)
            row[1].plot(snrs, [p['power'] for p in points], **style)
            row[2].plot(snrs, [100 * p['error_free'] for p in points], **style)
        for name in policies:
            points = [sweep[s]['policies'][name][t] for s in snrs]
            style = dict(lw=1.8, ls='--', marker='s', ms=3, label=f'{name} (λ)')
            row[0].plot(snrs, [100 * p['reliability'] for p in points], **style)
            row[1].plot(snrs, [p['power'] for p in points], **style)
            row[2].plot(snrs, [100 * p['error_free'] for p in points], **style)
        row[0].axhline(100 * t, color='grey', lw=0.8, ls=':')
        row[0].set(ylabel=f'target {100 * t:g}%\nservable reaching Γ (%)')
        row[1].set(yscale='log', ylabel='mean power, × perfect CSI')
        row[2].set(ylabel='servable messages error-free (%)')
        for ax in row:
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set(xlabel='target SNR Γ (dB)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Test layouts, κ and τ (and λ, dashed) chosen again for each Γ', y=1.0)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches='tight')
    return axes
