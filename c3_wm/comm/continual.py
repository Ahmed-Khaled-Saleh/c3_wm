from __future__ import annotations

import copy
import math

import numpy as np

from .deployment import judge
from .evaluation import prediction_errors, transmissions
from .power_control import TAUS_DB, calibrate, fit_predictor, predict_on
from .sequence import GEOMETRY, SequencePredictor, fit_model, gain_loss, predictor_name

METHODS = ('frozen', 'finetune', 'replay', 'lars', 'ewc', 'si', 'lwf', 'joint')
SAMPLES = ('X', 'S', 'gain')


def subset(data: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in data.items()}

__all__ = ['METHODS', 'SAMPLES', 'KINDS', 'METHOD_COLUMNS', 'subset', 'ReplayBuffer', 'ContinualLearner', 'TreesRefit',
           'layout_stream', 'stream_order', 'run_stream', 'evaluate_continual', 'continual_summary', 'continual_rows',
           'task_columns', 'plot_continual']

class ReplayBuffer:

    def __init__(self, capacity: int | None = 5000, lars: bool = False, seed: int = 0, eps: float = 1e-8):
        self.capacity, self.lars, self.eps = capacity, lars, eps
        self.rng = np.random.default_rng(seed)
        self.seen, self.n, self.arrays = 0, 0, None

    def __len__(self):
        return self.n

    def _grow(self, rows: dict, need: int):
        size = need if self.capacity is not None else max(need, 2 * (len(self.arrays['gain']) if self.arrays else 0), 1024)
        new = {k: np.zeros((size, *v.shape[1:]), v.dtype) for k, v in rows.items()}
        if self.arrays is not None:
            for k in new:
                new[k][:self.n] = self.arrays[k][:self.n]
        self.arrays = new

    def add(self, X: np.ndarray, gain: np.ndarray, loss: np.ndarray, S: np.ndarray | None = None):
        rows = dict(X=np.asarray(X, np.float32), gain=np.asarray(gain, np.float64), loss=np.asarray(loss, np.float64))
        if S is not None:
            rows['S'] = np.asarray(S, np.float32)
        if self.arrays is None or (self.capacity is None and self.n + len(gain) > len(self.arrays['gain'])):
            self._grow(rows, self.capacity if self.capacity is not None else self.n + len(gain))
        for i in range(len(gain)):
            self.seen += 1
            if self.capacity is None or self.n < self.capacity:
                slot = self.n
                self.n += 1
            else:
                j = int(self.rng.integers(self.seen))
                if j >= self.capacity:
                    continue
                if self.lars:
                    w = 1 / (self.arrays['loss'][:self.n] + self.eps)
                    j = int(self.rng.choice(self.n, p=w / w.sum()))
                slot = j
            for k, v in rows.items():
                self.arrays[k][slot] = v[i]

    def sample(self, n: int) -> dict:
        idx = self.rng.integers(self.n, size=n)
        return {k: v[idx] for k, v in self.arrays.items() if k != 'loss'}

class ContinualLearner:

    def __init__(
        self,
        model: SequencePredictor,
        method: str = 'replay',
        train: dict | None = None,
        capacity: int = 5000,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 5e-4,
        replay_weight: float = 0.5,
        ewc: float = 0.4,
        si: float = 0.4,
        si_xi: float = 0.1,
        lwf: float = 0.5,
        clip: float = 1.0,
        fisher_samples: int = 20000,
        device: str = 'auto',
        seed: int = 0,
    ):
        import torch
        if method not in METHODS:
            raise ValueError(f'method: one of {METHODS}, not {method!r}')
        if not isinstance(model, SequencePredictor):
            raise ValueError('a ContinualLearner learns a sequence predictor (comm.sequence); the trees: TreesRefit')
        if method in ('replay', 'lars', 'joint', 'ewc') and train is None:
            raise ValueError(f'{method} needs the training samples (train)')
        if method == 'si' and model.importance is None:
            raise ValueError("si needs the predictor's importance over its training (fit_sequence_predictor keeps it)")
        self.model, self.method, self._torch = copy.deepcopy(model), method, torch
        self.epochs, self.batch_size, self.lr, self.replay_weight = epochs, batch_size, lr, replay_weight
        self.alpha, self.beta, self.xi, self.lwf, self.clip = ewc, si, si_xi, lwf, clip
        self.fisher_samples = fisher_samples
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.loss = gain_loss(model.quantile)
        self.tasks = 0
        if method in ('replay', 'lars', 'joint'):
            self.buffer = ReplayBuffer(None if method == 'joint' else capacity, method == 'lars', seed)
            self.buffer.add(train['X'], train['gain'], self.losses(train), train.get('S') if model.steps else None)
        if method == 'ewc':
            self.anchors = [(self._weights(), self._fisher(train))]
        if method == 'si':
            self.omega = {k: v.clone() for k, v in model.importance.items()}
            self.anchor = self._weights()

    @property
    def steps(self) -> bool:
        return self.model.steps

    def predict(self, X: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
        return self.model.predict(X, S) if self.steps else self.model.predict(X)

    def _weights(self) -> dict:
        return {k: p.detach().clone() for k, p in self.model.net.named_parameters()}

    def _tensors(self, data: dict) -> tuple:
        torch = self._torch
        Z, Zs = self.model.scaled(data['X'], data.get('S'))
        t = self.model.target(data['gain'])
        to = lambda a: torch.from_numpy(a).to(self.device) if a is not None else None
        return to(Z), to(Zs), to(t)

    def losses(self, data: dict) -> np.ndarray:
        c, q = self.model.scale, self.model.quantile
        p = (self.predict(data['X'], data.get('S')) - c['mean'][GEOMETRY]) / c['std'][GEOMETRY]
        e = self.model.target(data['gain']) - p
        return (e ** 2 if q is None else np.maximum(q * e, (q - 1) * e)).astype(float)

    def _fisher(self, data: dict) -> dict:
        torch = self._torch
        n = len(data['gain'])
        keep = self.rng.choice(n, min(n, self.fisher_samples), replace=False)
        Z, Zs, t = self._tensors(subset({k: v for k, v in data.items() if k in SAMPLES}, keep))
        net = copy.deepcopy(self.model.net).to(self.device)
        net.train()
        F = {k: torch.zeros_like(p) for k, p in net.named_parameters()}
        batches = 0
        for i in range(0, len(t), self.batch_size):
            net.zero_grad()
            self.loss(net(Z[i:i + self.batch_size], Zs[i:i + self.batch_size] if Zs is not None else None),
                      t[i:i + self.batch_size]).backward()
            for k, p in net.named_parameters():
                if p.grad is not None:
                    F[k] += p.grad.detach() ** 2
            batches += 1
        return {k: (v / max(batches, 1)).cpu() for k, v in F.items()}

    def learn(self, task: dict):
        self.tasks += 1
        if self.method == 'frozen' or not len(task['gain']):
            return
        torch, dev = self._torch, self.device
        task = {k: v for k, v in task.items() if k in SAMPLES and (k != 'S' or self.steps)}
        net = copy.deepcopy(self.model.net).to(dev)
        teacher = copy.deepcopy(net).eval() if self.method == 'lwf' else None
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        Z, Zs, t = self._tensors(task)
        anchors = [({k: v.to(dev) for k, v in a.items()}, {k: v.to(dev) for k, v in F.items()})
                   for a, F in getattr(self, 'anchors', [])]
        if self.method == 'si':
            omega = {k: v.to(dev) for k, v in self.omega.items()}
            anchor = {k: v.to(dev) for k, v in self.anchor.items()}
            path = {k: torch.zeros_like(p) for k, p in net.named_parameters()}
        replay = self.method in ('replay', 'lars', 'joint') and len(self.buffer) > 0
        n = len(t)
        for _ in range(self.epochs):
            order = torch.randperm(n, device=dev)
            for i in range(0, n, self.batch_size):
                b = order[i:i + self.batch_size]
                out = net(Z[b], Zs[b] if Zs is not None else None)
                loss = self.loss(out, t[b])
                if replay:
                    Rz, Rzs, Rt = self._tensors(self.buffer.sample(len(b)))
                    loss = self.replay_weight * loss + (1 - self.replay_weight) * self.loss(net(Rz, Rzs), Rt)
                if teacher is not None:
                    with torch.no_grad():
                        target = teacher(Z[b], Zs[b] if Zs is not None else None)
                    loss = loss + self.lwf * ((out - target) ** 2).mean()
                params = dict(net.named_parameters())
                if anchors:
                    loss = loss + self.alpha * sum((F[k] * (params[k] - a[k]) ** 2).sum() for a, F in anchors for k in F)
                if self.method == 'si':
                    loss = loss + self.beta * sum((omega[k] * (params[k] - anchor[k]) ** 2).sum() for k in omega)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.clip)
                if self.method == 'si':
                    for k, p in params.items():
                        if p.grad is not None:
                            path[k] += p.grad.detach() ** 2 * self.lr
                opt.step()
        self.model.loaded(net.state_dict())
        if self.method in ('replay', 'lars', 'joint'):
            self.buffer.add(task['X'], task['gain'], self.losses(task), task.get('S'))
        if self.method == 'ewc':
            self.anchors.append((self._weights(), self._fisher(task)))
        if self.method == 'si':
            now = self._weights()
            self.omega = {k: (omega[k] + path[k] / ((now[k].to(dev) - anchor[k]) ** 2 + self.xi)).cpu() for k in omega}
            self.anchor = now

class TreesRefit:
    steps = False

    def __init__(self, model, train: dict, quantile: float | None, every: int | None = 10, **trees):
        self.model, self.quantile, self.every, self.trees = model, quantile, every, trees
        self.seen = [{k: train[k] for k in ('X', 'gain')}]
        self.tasks = 0

    def predict(self, X: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
        return self.model.predict(X)

    def learn(self, task: dict):
        self.tasks += 1
        if not self.every:
            return
        self.seen.append({k: task[k] for k in ('X', 'gain')})
        if self.tasks % self.every == 0:
            X, gain = (np.concatenate([s[k] for s in self.seen]) for k in ('X', 'gain'))
            self.model = fit_predictor(X, gain, self.quantile, **self.trees)

def layout_stream(data: dict, order, update_episodes: int = 2) -> list[dict]:
    out = []
    for k in order:
        on = data['layout'] == k
        out.append(dict(layout=int(k), update=subset(data, on & (data['episode'] < update_episodes)),
                        judged=subset(data, on & (data['episode'] >= update_episodes))))
    return out


def stream_order(groups: dict, order: str = 'groups', seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    if order == 'groups':
        return [int(k) for ids in groups.values() for k in rng.permutation(list(ids))]
    if order == 'random':
        return [int(k) for k in rng.permutation([k for ids in groups.values() for k in ids])]
    raise ValueError(f"order: groups or random, not {order!r}")


def _params(learner, calibration: dict, g_min: float, targets, taus_db, slack) -> dict:
    estimate = predict_on(learner, calibration)
    return {t: calibrate(estimate, calibration['gain'], g_min, t + slack, taus_db=taus_db) for t in targets}


def _judged(estimates, gains, layouts, episodes, params, g_min, target, members) -> dict:
    power, sent = map(np.concatenate, zip(*[transmissions(e, g_min, p['margin_db'], p['tau_db']) for e, p in zip(estimates, params)]))
    gain, layout, episode = (np.concatenate(a) for a in (gains, layouts, episodes))
    return {g: judge(power[m], sent[m], gain[m], layout[m], g_min, target, episode=episode[m])
            for g, ids in members.items() for m in [np.isin(layout, ids)] if m.any()}


def run_stream(
    learners: dict,
    tasks: list,
    calibration: dict,
    test: dict,
    g_min: float,
    targets=(0.9, 0.95, 0.99),
    taus_db=TAUS_DB, slack: float = 0.0,
    groups: dict | None = None,
    recalibrate_every: int = 1,
) -> dict:
    members = {'all': [t['layout'] for t in tasks], **(groups or {})}
    group_of = {k: g for g, ids in (groups or {}).items() for k in ids}
    out = {}
    for name, learner in learners.items():
        start = _params(learner, calibration, g_min, targets, taus_db, slack)
        params, rows, est = start, [], []
        for i, task in enumerate(tasks):
            j = task['judged']
            before = predict_on(learner, j)
            learner.learn(task['update'])
            if (i + 1) % recalibrate_every == 0:
                params = _params(learner, calibration, g_min, targets, taus_db, slack)
            after = predict_on(learner, j)
            est.append((after, dict(params)))
            row = dict(task=i, layout=task['layout'], group=group_of.get(task['layout'], 'all'), transmissions=len(j['gain']),
                       before=prediction_errors(before, j['gain']), after=prediction_errors(after, j['gain']))
            for t in targets:
                power, sent = transmissions(after, g_min, params[t]['margin_db'], params[t]['tau_db'])
                row[t] = {k: v for k, v in judge(power, sent, j['gain'], j['layout'], g_min, t).items()
                          if k in ('reliability', 'power', 'withheld')}
            rows.append(row)
        final = _params(learner, calibration, g_min, targets, taus_db, slack)
        final_est = [predict_on(learner, t['judged']) for t in tasks]
        test_est = predict_on(learner, test)
        js = [t['judged'] for t in tasks]
        g, lay, ep = ([d[k] for d in js] for k in ('gain', 'layout', 'episode'))
        result = {}
        for t in targets:
            result[t] = dict(
                adapted=_judged([e for e, _ in est], g, lay, ep, [p[t] for _, p in est], g_min, t, members),
                fixed=_judged([e for e, _ in est], g, lay, ep, [start[t]] * len(est), g_min, t, members),
                final=_judged(final_est, g, lay, ep, [final[t]] * len(js), g_min, t, members),
                test=judge(*transmissions(test_est, g_min, final[t]['margin_db'], final[t]['tau_db']), test['gain'],
                           test['layout'], g_min, t, episode=test['episode']),
                margin_db=final[t]['margin_db'], tau_db=final[t]['tau_db'])
        out[name] = dict(tasks=rows, targets=result,
                         prediction={g: prediction_errors(np.concatenate([e for (e, _), t in zip(est, tasks) if t['layout'] in ids]),
                                                          np.concatenate([t['judged']['gain'] for t in tasks if t['layout'] in ids]))
                                     for g, ids in members.items() if any(t['layout'] in ids for t in tasks)})
    return out


def evaluate_continual(
    data: dict,
    stream: dict,
    groups: dict,
    g_min: float,
    model: dict | None = None,
    methods=METHODS,
    trees: dict | None = None,
    trees_every: int | None = 10,
    order: str = 'groups',
    runs: int = 1,
    update_episodes: int = 2,
    targets=(0.9, 0.95, 0.99), taus_db=TAUS_DB, slack: float = 0.0,
    recalibrate_every: int = 1,
    seed: int = 0,
    **learning,
) -> dict:
    spec = {**dict(kind='lstm', quantile=0.05, steps=False), **(model or {})}
    if spec['kind'] == 'trees':
        raise ValueError('the learning predictor is a sequence predictor (lstm or gru); the trees: trees=...')
    base = fit_model(data['train'], **spec)
    name = predictor_name(spec['kind'], spec['quantile'], spec['steps'])
    tree_spec = None if trees is None else dict(kind='trees', quantile=spec['quantile'], **trees)
    tree = fit_model(data['train'], **tree_spec) if tree_spec else None
    out = dict(predictor=name, trees=predictor_name('trees', spec['quantile']) if tree else None, runs=[])
    for run in range(runs):
        order_r = stream_order(groups, order, seed + run)
        tasks = layout_stream(stream, order_r, update_episodes)
        learners = {m: ContinualLearner(base, m, data['train'], seed=seed + run, **learning) for m in methods}
        if tree is not None:
            learners['trees, frozen'] = TreesRefit(tree, data['train'], spec['quantile'], None)
            if trees_every:
                learners[f'trees, refit every {trees_every}'] = TreesRefit(copy.deepcopy(tree), data['train'], spec['quantile'],
                                                                         trees_every, **(trees.get('trees') or {}))
        out['runs'].append(dict(order=order_r, results=run_stream(learners, tasks, data['calibration'], data['test'], g_min,
                                                                  targets, taus_db, slack, groups, recalibrate_every)))
    return out


def continual_summary(results: dict, target: float, kinds=('adapted', 'fixed', 'final', 'test')) -> str:
    runs = results['runs']
    methods = list(runs[0]['results'])
    w = max(map(len, methods)) + 2
    lines = [f"{results['predictor']} learning over the stream (trees: {results['trees']}), target {100 * target:g}%, "
             f"mean over {len(runs)} orders: reached / power (× perfect CSI)",
             f"{'':{w}s}" + ''.join(f'{k:>22s}' for k in kinds)]
    for m in methods:
        row = f'{m:{w}s}'
        for k in kinds:
            get = lambda r: r['results'][m]['targets'][target][k]
            o = [get(r) if k == 'test' else get(r)['all'] for r in runs]
            row += f"{100 * np.mean([x['reliability'] for x in o]):13.1f}% {np.mean([x['power'] for x in o]):6.2f}×"
        lines.append(row)
    lines.append('adapted: each layout sent after learning from its feedback, κ and τ chosen again; fixed: the start\'s κ and τ; '
                 'final: the last predictor on every layout of the stream; test: on the test layouts')
    return '\n'.join(lines)

KINDS = ('adapted', 'fixed', 'final', 'test')

def continual_rows(results: dict) -> tuple[list, list]:
    tasks, methods = [], []
    for r, run in enumerate(results['runs']):
        for m, res in run['results'].items():
            for row in res['tasks']:
                tasks.append([r, m, row['task'], row['layout'], row['group'], row['transmissions'],
                              *(row[w][k] for w in ('before', 'after') for k in ('bias_db', 'spread_db', 'coverage')),
                              *(row[t][k] for t in res['targets'] for k in ('reliability', 'power', 'withheld'))])
            for t, o in res['targets'].items():
                for kind in KINDS:
                    for g, v in ({'test': o[kind]} if kind == 'test' else o[kind]).items():
                        methods.append([r, m, t, kind, g, *(v.get(k) for k in ('reliability', 'power', 'withheld', 'layout_p10', 'layout_below'))])
    return tasks, methods

def task_columns(results: dict) -> list[str]:
    targets = list(results['runs'][0]['results'][next(iter(results['runs'][0]['results']))]['targets'])
    return ['run', 'method', 'task', 'layout', 'group', 'transmissions',
            *(f'{w}_{k}' for w in ('before', 'after') for k in ('bias_db', 'spread_db', 'coverage')),
            *(f'{100 * t:g}%_{k}' for t in targets for k in ('reliability', 'power', 'withheld'))]

METHOD_COLUMNS = ['run', 'method', 'target', 'kind', 'group', 'reliability', 'power', 'withheld', 'layout_p10', 'layout_below']

def plot_continual(results: dict, target: float, path=None, window: int = 10, ax=None):
    import matplotlib.pyplot as plt
    runs = results['runs']
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in runs[0]['results']:
        p = np.nanmean([[row[target]['power'] for row in r['results'][m]['tasks']] for r in runs], 0)
        w = max(1, min(window, len(p) // 3))
        y = np.convolve(p, np.ones(w) / w, mode='valid')
        ax.plot(np.arange(len(y)) + (w - 1) / 2, y, label=m, marker='.' if len(y) < 20 else None,
                lw=1.5 if not m.startswith('trees') else 1.0, ls='-' if not m.startswith('trees') else '--')
    groups = [row['group'] for row in runs[0]['results'][next(iter(runs[0]['results']))]['tasks']]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axvline(i - 0.5, color='grey', ls=':', lw=1)
    ax.set(xlabel='layout of the stream', ylabel=f'power for {100 * target:g}% (× perfect CSI)',
           title=f"{results['predictor']} learning over the stream")
    ax.legend(fontsize=7, ncol=2)
    if fig is not None and path is not None:
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return ax
