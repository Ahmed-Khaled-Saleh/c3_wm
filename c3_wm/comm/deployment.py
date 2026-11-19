from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .channel_data import interior_walls, layout_edits, make_env, nearest_layout, reset_layout
from .evaluation import (LEARNED, _distribution_options, _optimal_choice, baselines, choose_lambda, cross_fitted,
                                       layout_intervals, prediction_errors, transmissions)
from .power_control import MARGINS_DB, TAUS_DB, calibrate, calibrate_layouts, predict_gain, predict_on

__all__ = ['GROUPS', 'EDITED', 'PER_LAYOUT', 'ONLINE', 'OVER_LAYOUTS', 'deployment_layouts', 'layout_edits_of', 'save_deployment',
           'load_deployment', 'deployment_summary', 'sequences', 'margin_for', 'per_layout_margins', 'online_margins',
           'judge', 'choose_online', 'evaluate_deployment', 'deployment_table', 'plot_deployment']

GROUPS = {'in-distribution': (0.0, 0.15), 'slightly OOD': (0.2, 0.4), 'OOD': (0.55, 1.0)}
EDITED = ('in-distribution', 'slightly OOD')

def _edits_for(distance: float, walls: int) -> int:
    return max(1, int(round(2 * distance * walls / (2 - distance))))

def deployment_layouts(
    env_cfg,
    train,
    pool,
    groups: dict = GROUPS,
    edited=EDITED,
    per_group: int = 30,
    seed: int = 0,
    first_id: int = 1_000_000,
    tries: int = 50,
) -> dict:
    env = make_env(env_cfg)
    train = [int(s) for s in train]
    reference = np.stack([interior_walls(reset_layout(env, s)) for s in train])
    train_walls = reference.reshape(len(train), -1).sum(1)
    rng = np.random.default_rng(seed)
    layouts, members = {}, {g: [] for g in groups}
    fresh = (int(s) for s in pool if int(s) not in set(train))
    for g in (g for g in groups if g not in edited):
        lo, hi = groups[g]
        while len(members[g]) < per_group:
            s = next(fresh, None)
            if s is None:
                raise ValueError(f"the pool ran out with {len(members[g])} of {per_group} layouts in {g!r} {groups[g]}")
            w = interior_walls(reset_layout(env, s))
            i, e, d = nearest_layout(w, reference)
            if lo <= d <= hi:
                layouts[s] = dict(group=g, base=None, cells=[], distance=d, edits=e, nearest=train[i], walls=int(w.sum()))
                members[g].append(s)
    match = [layouts[s]['walls'] for g in members if g not in edited for s in members[g]] or train_walls.tolist()
    next_id = int(first_id)
    for g in (g for g in groups if g in edited):
        lo, hi = groups[g]
        for n in range(per_group):
            near = np.argsort(np.abs(train_walls - match[n % len(match)]) + rng.random(len(train)))[:5]
            for _ in range(tries):
                b = int(rng.choice(near))
                reset_layout(env, train[b])
                try:
                    cells = layout_edits(env, _edits_for(rng.uniform(lo, hi), int(train_walls[b])), rng)
                except RuntimeError:
                    continue
                w = interior_walls(reset_layout(env, next_id, {next_id: dict(base=train[b], cells=cells)}))
                i, e, d = nearest_layout(w, reference)
                if lo <= d <= hi:
                    break
            else:
                raise RuntimeError(f"no edits of the training layouts land in {g!r} {groups[g]} in {tries} tries")
            layouts[next_id] = dict(group=g, base=train[b], cells=cells, distance=d, edits=e, nearest=train[i],
                                    walls=int(w.sum()))
            members[g].append(next_id)
            next_id += 1
    return dict(layouts=layouts, groups=members, train=train, bands={g: list(b) for g, b in groups.items()})

def layout_edits_of(deployment: dict) -> dict:
    return {k: dict(base=v['base'], cells=v['cells']) for k, v in deployment['layouts'].items() if v['base'] is not None}

def save_deployment(path, deployment: dict):
    path = Path(path)
    tmp = path.with_suffix(f'.{np.random.default_rng().integers(1 << 30)}.tmp')
    tmp.write_text(json.dumps(deployment, indent=1))
    tmp.replace(path)

def load_deployment(path) -> dict:
    d = json.loads(Path(path).read_text())
    d['layouts'] = {int(k): v for k, v in d['layouts'].items()}
    return d

def deployment_summary(deployment: dict) -> str:
    lines = [f"{'group':18s}{'band':>14s}{'layouts':>9s}{'distance':>20s}{'edits':>16s}{'walls':>16s}"]
    for g, ids in deployment['groups'].items():
        v = {k: np.array([deployment['layouts'][i][k] for i in ids], float) for k in ('distance', 'edits', 'walls')}
        lo, hi = deployment['bands'][g]
        lines.append(f"{g:18s}{f'[{lo:g}, {hi:g}]':>14s}{len(ids):9d}" + ''.join(
            f"{v[k].mean():9.2f} ({v[k].min():.2f}-{v[k].max():.2f})" if k == 'distance' else
            f"{v[k].mean():7.1f} ({v[k].min():.0f}-{v[k].max():.0f})" for k in ('distance', 'edits', 'walls')))
    lines.append('distance: the edits to the nearest training layout over the walls in either (0: the same walls, 1: none in common)')
    return '\n'.join(lines)

def sequences(data: dict, layouts=None) -> tuple[list, np.ndarray]:
    order = np.lexsort((data['link'][:, 0], data['link'][:, 2], data['episode'], data['layout']))
    layout = data['layout'][order]
    layouts = list(np.unique(layout)) if layouts is None else list(layouts)
    rows = [order[layout == k] for k in layouts]
    index = np.full((len(rows), max(map(len, rows), default=0)), -1)
    for r, idx in enumerate(rows):
        index[r, :len(idx)] = idx
    return [int(k) for k in layouts], index

def margin_for(estimate, gain, sent, g_min: float, target: float, margins_db=MARGINS_DB,
               confidence: float | None = None) -> tuple[float, bool]:
    servable = gain >= g_min
    m = np.asarray(margins_db, float)[:, None]
    power = np.where(sent, np.minimum(10 ** (m / 10) / np.maximum(estimate, 1e-30), 1 / g_min), 0.0)
    reached = (power >= 1 / gain * (1 - 1e-9))[:, servable]
    reliability = reached.mean(1)
    if confidence is None:
        met = reliability >= target
    else:
        from scipy.stats import beta
        x, n = reached.sum(1), servable.sum()
        met = np.where(x > 0, beta.ppf(1 - confidence, np.maximum(x, 1), n - x + 1), 0.0) >= target
    if met.any():
        return float(m[met, 0][np.argmin(power[met].mean(1))]), True
    return float(m[np.argmax(reliability), 0]), False

def per_layout_margins(
    estimate, gain, layout,
    pilot: np.ndarray,
    g_min: float, target: float, tau_db: float,
    fallback_db: float,
    min_servable: int = 50,
    margins_db=MARGINS_DB,
    confidence: float | None = 0.9,
) -> dict:
    _, sent = transmissions(estimate, g_min, fallback_db, tau_db)
    out = {}
    for k in np.unique(layout):
        on = (layout == k) & pilot
        if (gain[on] >= g_min).sum() < min_servable:
            out[int(k)] = float(fallback_db)
            continue
        kappa, met = margin_for(estimate[on], gain[on], sent[on], g_min, target, margins_db, confidence)
        out[int(k)] = kappa if met else max(kappa, float(fallback_db))
    return out

def online_margins(
    estimate, gain,
    index: np.ndarray,
    g_min: float,
    margin_db: float, tau_db: float,
    steps_db, success,
    bounds_db=(-10.0, 30.0),
) -> np.ndarray:
    steps, success = np.broadcast_arrays(np.atleast_1d(np.asarray(steps_db, float)), np.atleast_1d(np.asarray(success, float)))
    down = steps * (1 - success) / success
    runs, (L, T) = len(steps), index.shape
    kappa = np.full((runs, L), float(margin_db))
    lo, hi = margin_db + bounds_db[0], margin_db + bounds_db[1]
    out = np.full((runs, len(gain)), np.nan)
    cap = 1 / g_min
    for t in range(T):
        idx = index[:, t]
        on = idx >= 0
        if not on.any():
            break
        i = idx[on]
        est, g = np.maximum(estimate[i], 1e-30), gain[i]
        out[:, i] = kappa[:, on]
        sent = 10 ** (margin_db / 10) / est <= (10 ** (tau_db / 10) * cap if math.isfinite(tau_db) else np.inf)
        rho = 10 ** (kappa[:, on] / 10) / est
        power = np.minimum(rho, cap)
        ok = power * g >= 1 - 1e-9
        counted = sent & (ok | (rho < cap))
        kappa[:, on] = np.clip(kappa[:, on] + np.where(counted, np.where(ok, -down[:, None], steps[:, None]), 0.0), lo, hi)
    return out

def _sent_with(estimate, g_min: float, margin_db, tau_db: float, schedule_db: float) -> tuple:
    _, sent = transmissions(estimate, g_min, schedule_db, tau_db)
    power = np.where(sent, np.minimum(10 ** (np.asarray(margin_db) / 10) / np.maximum(estimate, 1e-30), 1 / g_min), 0.0)
    return power, sent

def judge(power, sent, gain, layout, g_min: float, target: float | None = None, min_servable: int = 20, episode=None) -> dict:
    servable, need = gain >= g_min, 1 / np.maximum(gain, 1e-300)
    reached = sent & (power >= need * (1 - 1e-9))
    out = dict(reliability=float(reached[servable].mean()) if servable.any() else math.nan,
               power=float(power.mean() / np.where(servable, need, 0).mean()) if servable.any() else math.nan,
               withheld=float(1 - sent.mean()), servable=float(servable.mean()), transmissions=int(len(gain)))
    per = np.array([reached[(layout == k) & servable].mean() for k in np.unique(layout)
                    if ((layout == k) & servable).sum() >= min_servable])
    out['layouts'] = int(len(per))
    if len(per):
        out.update(layout_mean=float(per.mean()), layout_std=float(per.std()), layout_p10=float(np.percentile(per, 10)),
                   layout_min=float(per.min()))
        if target is not None:
            out['layout_below'] = float((per < target).mean())
        if episode is not None:
            keep = [int(k) for k in np.unique(layout) if ((layout == k) & servable).sum() >= min_servable]
            iv = layout_intervals(reached, servable, layout, np.asarray(episode))
            se = np.array([iv[k][2] for k in keep])
            out['layout_between_std'] = float(math.sqrt(max(per.var() - (se ** 2).mean(), 0.0)))
            if target is not None:
                out['layout_significantly_below'] = float(np.mean([iv[k][1] < target for k in keep]))
    return out

def choose_online(
    estimate, gain, data: dict,
    g_min: float, target: float, margin_db: float, tau_db: float,
    pilot: int = 2,
    steps_db=(0.25, 0.5, 1.0, 2.0),
    failure_scales=(1.0, 0.75, 0.5, 0.25),
    bounds_db=(-10.0, 30.0),
) -> dict:
    steps, scales = np.meshgrid(np.asarray(steps_db, float), np.asarray(failure_scales, float), indexing='ij')
    steps, success = steps.ravel(), 1 - (1 - target) * scales.ravel()
    _, index = sequences(data)
    margins = online_margins(estimate, gain, index, g_min, margin_db, tau_db, steps, success, bounds_db)
    judged = data['episode'] >= pilot
    best = None
    for r in range(len(steps)):
        power, sent = _sent_with(estimate[judged], g_min, margins[r, judged], tau_db, margin_db)
        o = judge(power, sent, gain[judged], data['layout'][judged], g_min)
        met = o['reliability'] >= target
        key = (met, -o['power'] if met else o['reliability'])
        if best is None or key > best[0]:
            best = (key, dict(step_db=float(steps[r]), success=float(success[r]), met=bool(met),
                              reliability=o['reliability'], power=o['power']))
    return best[1]

PER_LAYOUT, ONLINE, OVER_LAYOUTS = ' + per-layout κ', ' + online κ', ' + κ, τ over layouts'

def evaluate_deployment(
    saved: dict,
    data: dict,
    deployment: dict,
    fitted: dict | None = None,
    targets=(0.9, 0.95, 0.99),
    taus_db=TAUS_DB,
    slack: float = 0.0,
    pilot: int = 2,
    confidence: float | None = 0.9,
    adapt=('geometry + history mean', LEARNED),
    min_servable: int = 50,
    steps_db=(0.25, 0.5, 1.0, 2.0),
    failure_scales=(1.0, 0.75, 0.5, 0.25),
    bounds_db=(-10.0, 30.0),
    quantile: float = 0.05, max_iter: int = 400,
    layout_coverage: float | None = None,
    layout_confidence: float | None = None,
) -> dict:
    budget, g_min = saved['budget'], saved['budget'].g_min
    cal, dep = data['calibration'], data['deployment']
    predictors = {LEARNED: lambda d: predict_on(saved['model'], d),
                  **baselines(data['train'], saved['history'], quantile, max_iter, fitted)}
    meta = deployment['layouts']
    groups = {g: [k for k in ids if k in set(dep['layout'].tolist())] for g, ids in deployment['groups'].items()}
    judged = dep['episode'] >= pilot
    member = {'all': judged, **{g: judged & np.isin(dep['layout'], ids) for g, ids in groups.items()}}
    out = dict(pilot=pilot, transmissions=int(judged.sum()),
               groups={g: dict(layouts=len(ids), **{k: float(np.mean([meta[i][k] for i in ids])) for k in ('distance', 'edits', 'walls')})
                       for g, ids in groups.items()},
               methods={})

    def metrics(power, sent, target):
        return {g: judge(power[m], sent[m], dep['gain'][m], dep['layout'][m], g_min, target, episode=dep['episode'][m])
                for g, m in member.items()}

    for name, predict in predictors.items():
        c_est, d_est = predict(cal), predict(dep)
        errors = {g: prediction_errors(d_est[m], dep['gain'][m]) for g, m in member.items()}
        kinds = {name: {}}
        if name in adapt and name != 'perfect CSI':
            kinds[name + PER_LAYOUT], kinds[name + ONLINE] = {}, {}
            if layout_coverage is not None:
                kinds[name + OVER_LAYOUTS] = {}
        for target in targets:
            if name == 'perfect CSI':
                offline = dict(margin_db=0.0, tau_db=0.0, met=True)
            else:
                offline = calibrate(c_est, cal['gain'], g_min, target + slack, taus_db=taus_db)
            kappa, tau = offline['margin_db'], offline['tau_db']
            power, sent = transmissions(d_est, g_min, kappa, tau)
            kinds[name][target] = dict(margin_db=kappa, tau_db=tau, met_on_calibration=offline['met'],
                                       groups=metrics(power, sent, target))
            if name + PER_LAYOUT in kinds:
                margins = per_layout_margins(d_est, dep['gain'], dep['layout'], ~judged, g_min, target, tau, kappa,
                                             min_servable, confidence=confidence)
                power, sent = _sent_with(d_est, g_min, np.array([margins[int(k)] for k in dep['layout']]), tau, kappa)
                m = np.array(list(margins.values()))
                kinds[name + PER_LAYOUT][target] = dict(margin_db=kappa, tau_db=tau, layout_margin_mean_db=float(m.mean()),
                                                        layout_margin_std_db=float(m.std()), per_layout_db=margins,
                                                        groups=metrics(power, sent, target))
                chosen = choose_online(c_est, cal['gain'], cal, g_min, target + slack, kappa, tau, pilot, steps_db,
                                       failure_scales, bounds_db)
                _, index = sequences(dep)
                margin = online_margins(d_est, dep['gain'], index, g_min, kappa, tau, chosen['step_db'], chosen['success'],
                                        bounds_db)[0]
                power, sent = _sent_with(d_est, g_min, margin, tau, kappa)
                kinds[name + ONLINE][target] = dict(margin_db=kappa, tau_db=tau, step_db=chosen['step_db'],
                                                    success=chosen['success'], met_on_calibration=chosen['met'],
                                                    final_margin_mean_db=float(np.nanmean([margin[i[i >= 0][-1]] for i in index])),
                                                    groups=metrics(power, sent, target))
            if name + OVER_LAYOUTS in kinds:
                ch = calibrate_layouts(c_est, cal['gain'], cal['layout'], g_min, target, layout_coverage, layout_confidence,
                                       taus_db=taus_db)
                power, sent = transmissions(d_est, g_min, ch['margin_db'], ch['tau_db'])
                kinds[name + OVER_LAYOUTS][target] = dict(margin_db=ch['margin_db'], tau_db=ch['tau_db'], met_on_calibration=ch['met'],
                                                          calibration_below=ch['below'], allowed=ch['allowed'],
                                                          groups=metrics(power, sent, target))
        for method, per_target in kinds.items():
            out['methods'][method] = dict(predictor=name, prediction=errors, targets=per_target)
    for name, dist in (fitted or {}).get('distributional', {}).items():
        cal_options, cal_order = cross_fitted(dist, cal, g_min)
        dep_options = _distribution_options(dist, dep, g_min)
        per_target = {}
        for target in targets:
            lam = choose_lambda(cal_options, cal_order, g_min, target + slack)
            power = _optimal_choice(dep_options, lam)
            per_target[target] = dict(margin_db=math.nan, tau_db=math.nan, lam=lam, met_on_calibration=True,
                                      groups=metrics(power, power > 0, target))
        out['methods'][name] = dict(predictor=name, prediction={}, targets=per_target)
    return out

def deployment_table(results: dict, target: float, keys=('reliability', 'power', 'layout_p10', 'layout_below')) -> str:
    groups = ['all', *results['groups']]
    names = list(results['methods'])
    w = max(map(len, names)) + 2
    head = f"{'':{w}s}" + ''.join(f"{g:>44s} " for g in groups) + '\n' + f"{'':{w}s}" + ''.join(
        f"{'reached':>21s}{'power':>8s}{'p10':>8s}{'below':>7s} " for _ in groups)
    lines = [f"target {100 * target:g}%, judged after a pilot of {results['pilot']} episodes ({results['transmissions']} transmissions)",
             *(f"  {g}: {v['layouts']} layouts, distance {v['distance']:.2f}, {v['edits']:.1f} edits, {v['walls']:.0f} walls"
               for g, v in results['groups'].items()), head]
    for n in names:
        row = f"{n:{w}s}"
        for g in groups:
            o = results['methods'][n]['targets'][target]['groups'][g]
            row += (f"{100 * o['reliability']:20.1f}% {o['power']:6.2f}× {100 * o.get('layout_p10', math.nan):6.1f}% "
                    f"{100 * o.get('layout_below', math.nan):5.0f}% ")
        lines.append(row)
    lines.append('p10: the 10th-percentile layout\'s reliability; below: the share of layouts under the target')
    return '\n'.join(lines)

def plot_deployment(results: dict, target: float, path=None, methods=None):
    import matplotlib.pyplot as plt
    groups = list(results['groups'])
    methods = methods or [n for n in results['methods'] if n != 'perfect CSI']
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(groups))
    predictors = list(dict.fromkeys(results['methods'][n]['predictor'] for n in methods))
    colors = {p: plt.cm.tab10(i % 10) for i, p in enumerate(predictors)}
    for k, n in enumerate(methods):
        o = [results['methods'][n]['targets'][target]['groups'][g] for g in groups]
        kind = 'per-layout' if n.endswith(PER_LAYOUT) else 'online' if n.endswith(ONLINE) else 'offline'
        style = dict(marker={'offline': '.', 'per-layout': 'o', 'online': 's'}[kind], ms=6, lw=1.5, label=n,
                     ls={'offline': ':', 'per-layout': '-', 'online': '--'}[kind], color=colors[results['methods'][n]['predictor']])
        dx = 0.03 * (k - len(methods) / 2)
        a.plot(x + dx, [100 * v['reliability'] for v in o], **style)
        a.plot(x + dx, [100 * v.get('layout_p10', math.nan) for v in o], marker='v', ms=4, ls='none', color=style['color'], alpha=0.5)
        b.plot(x, [v['power'] for v in o], **style)
    a.axhline(100 * target, color='#d62728', lw=1, ls='--')
    a.set(ylabel='servable transmissions reaching Γ (%)', title=f'Reliability, target {100 * target:g}% (▼: 10th-percentile layout)')
    b.set(ylabel='power (× perfect CSI)', yscale='log', title='Power')
    for ax in (a, b):
        ax.set_xticks(x, groups)
        ax.spines[['top', 'right']].set_visible(False)
    b.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches='tight')
    return fig
