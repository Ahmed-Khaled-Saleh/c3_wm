import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

DESCRIPTION = "Compare deployment methods episode by episode."


def load(paths, episodes: int | None = None) -> dict:
    out = {}
    for p in paths:
        for method, r in json.loads(Path(p).read_text()).items():
            if method in out:
                raise SystemExit(f"{method!r} is in two files")
            out[method] = dict(seeds=np.asarray(r['seeds'], int), success=np.asarray(r['episode_successes'], bool),
                               agents=np.asarray(r['episode_returns'], float) > 0,
                               length=np.asarray(r['episode_lengths'], float), link=r.get('link'), file=str(p),
                               starts={k: np.asarray(v) for k, v in (r.get('starts') or {}).items()},
                               energy=energy(r), age=r.get('age'), decision=r.get('decision'))
            if episodes is not None:
                keep = out[method]['seeds'] < episodes
                for k in ('seeds', 'success', 'agents', 'length'):
                    out[method][k] = out[method][k][keep]
                out[method]['starts'] = {k: v[keep] for k, v in out[method]['starts'].items()}
    return out


def energy(r: dict) -> dict | None:
    rows = r.get('link_rows')
    if not rows:
        return None
    p, sent = np.array([row[1] for row in rows], float), np.array([row[2] for row in rows], bool)
    n = len(r['seeds'])
    return dict(transmissions=len(rows) / n, energy=1e3 * p.sum() / n, power=1e3 * p[sent].mean() if sent.any() else 0.0)


def paired(a: dict, b: dict):
    common = np.intersect1d(a['seeds'], b['seeds'])
    ia = {s: i for i, s in enumerate(a['seeds'])}
    ib = {s: i for i, s in enumerate(b['seeds'])}
    return np.array([ia[s] for s in common]), np.array([ib[s] for s in common])


def interval(x: np.ndarray, rng, n: int = 10000) -> tuple:
    return tuple(np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5]))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple:
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n, k = n01 + n10, min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p


def sight(r: dict) -> dict | None:
    seen = r['starts'].get('seen_at')
    if seen is None:
        return None
    first = np.where(seen < 0, np.iinfo(int).max, seen)
    A = first.shape[1]
    partner = np.stack([np.delete(first, a, axis=1).min(1) for a in range(A)], 1) if A > 1 else np.full_like(first, np.iinfo(int).max)
    return {'saw it first': (seen >= 0) & (first <= partner),
            'partner saw it first': (partner < first),
            'nobody saw it': (seen < 0) & (partner == np.iinfo(int).max)}


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--reference', default='no communication')
    ap.add_argument('--budgets', type=int, nargs='+', default=[50, 75, 100, 125, 150])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=None, help='use only seeds below this value')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    runs = load(args.files, args.episodes)
    seeds = {m: tuple(sorted(r['seeds'])) for m, r in runs.items()}
    if len(set(seeds.values())) > 1:
        print("note: the methods played different episodes; the comparisons use those they share:",
              {m: len(s) for m, s in seeds.items()})

    print(f"\n{'method':18s} {'episodes':>8s} {'success (95% CI)':>22s} {'agents':>8s} {'length':>7s}   "
          + '  '.join(f"<={b:<4d}" for b in args.budgets))
    for m, r in runs.items():
        lo, hi = interval(r['success'].astype(float), rng)
        within = [100 * (r['success'] & (r['length'] <= b)).mean() for b in args.budgets]
        print(f"{m:18s} {len(r['seeds']):8d} {100 * r['success'].mean():8.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}] "
              f"{100 * r['agents'].mean():7.1f}% {r['length'].mean():7.1f}   " + '  '.join(f"{w:5.1f}%" for w in within))

    if args.reference not in runs:
        print(f"\nno {args.reference!r} among {list(runs)}: no paired comparison (--reference)")
    else:
        ref = runs[args.reference]
        print(f"\nagainst {args.reference!r}, on the episodes both played (paired; 95% bootstrap intervals):")
        print(f"{'method':18s} {'pairs':>5s} {'success diff':>24s} {'only it / only ref':>19s} {'McNemar p':>9s} "
              f"{'agents diff':>12s} {'length diff (steps)':>27s}")
        for m, r in runs.items():
            if m == args.reference:
                continue
            i, j = paired(r, ref)
            d = r['success'][i].astype(float) - ref['success'][j]
            lo, hi = interval(d, rng)
            only_it, only_ref, p = mcnemar(r['success'][i], ref['success'][j])
            da = r['agents'][i].astype(float) - ref['agents'][j]
            dl = r['length'][i] - ref['length'][j]
            llo, lhi = interval(dl, rng)
            print(f"{m:18s} {len(i):5d} {100 * d.mean():+7.1f} pts [{100 * lo:+5.1f}, {100 * hi:+5.1f}] "
                  f"{only_it:9d} / {only_ref:<7d} {p:9.3f} {100 * da.mean():+10.1f} pts "
                  f"{dl.mean():+8.1f} [{llo:+5.1f}, {lhi:+5.1f}]")

    if all(sight(r) is not None for r in runs.values()):
        print(f"\nthe agents by who saw the goal first (agents reaching it; steps to reach it, of those that did):")
        for m, r in runs.items():
            reached_at = r['starts']['reached_at']
            print(f"{m:18s} " + '   '.join(
                f"{g}: {mask.sum():4d}, {100 * r['agents'][mask].mean():5.1f}%, "
                f"{reached_at[mask & (reached_at >= 0)].mean() if (mask & (reached_at >= 0)).any() else float('nan'):5.1f}"
                for g, mask in sight(r).items()))

    links = {m: r['link'] for m, r in runs.items() if r['link']}
    if links:
        print(f"\nthe link: {'method':18s} {'reached':>8s} {'power':>7s} {'withheld':>9s} {'intact':>7s} {'transmissions':>13s}")
        for m, l in links.items():
            print(f"          {m:18s} {100 * l['reached']:7.1f}% {l['power']:6.2f}x {100 * l['withheld']:8.1f}% "
                  f"{100 * l['intact']:6.1f}% {l['transmissions']:13d}")

    energies = {m: r['energy'] for m, r in runs.items() if r['energy']}
    if energies:
        base = next((e['energy'] for m, e in energies.items() if m == 'perfect CSI' or m.startswith('perfect CSI (')),
                    None)
        print(f"\nthe energy (over each run's episodes; one transmission a step):")
        print(f"          {'method':18s} {'per episode':>15s} {'x perfect CSI':>13s} {'transmissions':>13s} {'mean power':>11s}")
        for m, e in energies.items():
            rel = f"{e['energy'] / base:12.2f}x" if base else f"{'—':>13s}"
            print(f"          {m:18s} {e['energy']:7.2f} mW·step {rel} {e['transmissions']:13.1f} {e['power']:8.3f} mW")

    ages = {m: r['age'] for m, r in runs.items() if r['age']}
    if ages:
        print(f"\nthe age of information (steps since the receiver last heard its partner, both still playing; the run's episodes):")
        print(f"          {'method':18s} {'mean':>6s} {'median':>7s} {'p95':>5s} {'nothing yet':>12s}")
        for m, a in ages.items():
            if a['mean'] is None:
                print(f"          {m:18s} {'—':>6s} {'—':>7s} {'—':>5s} {100 * a['never']:11.1f}%")
            else:
                print(f"          {m:18s} {a['mean']:6.2f} {a['median']:7.1f} {a['p95']:5.1f} {100 * a['never']:11.1f}%")

    decisions = {m: r['decision'] for m, r in runs.items() if r['decision']}
    if decisions:
        print(f"\nthe decision time (each planner call, the GPU synchronized; a call decides for its run's envs at once):")
        print(f"          {'method':18s} {'call median':>12s} {'call p95':>9s} {'envs':>5s} {'per env':>9s}")
        for m, d in decisions.items():
            print(f"          {m:18s} {d['call_median']:10.3f} s {d['call_p95']:7.3f} s {d['num_envs']:5d} {d['per_env_median']:7.3f} s")


if __name__ == '__main__':
    main()
