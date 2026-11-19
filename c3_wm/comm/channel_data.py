from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np

__all__ = ['DIRS', 'CHANNELS', 'DIRECT_TOLERANCE', 'STEP_FEATURES', 'FADINGS', 'make_env', 'free_cells', 'reset_layout',
           'reset_in_layout', 'interior_walls', 'layout_edits', 'edit_distance', 'layout_distance', 'nearest_layout',
           'trace_layouts', 'wall_geometry', 'ChannelTable', 'table_path', 'collect_trajectories', 'file_lock',
           'replace_npz', 'PairCache', 'visited_pairs', 'trace_visited_pairs', 'CachedRayTracedFading', 'db',
           'features', 'step_features', 'fading_correlation', 'episode_fading', 'link_samples', 'build_dataset',
           'with_history']

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def _plain(cfg) -> dict:
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)

def make_env(env_cfg):
    kwargs = _plain(env_cfg)
    target = kwargs.pop('_target_')
    kwargs.pop('_convert_', None)
    variations = tuple(kwargs.pop('variations', None) or ())
    module, name = target.rsplit('.', 1)
    env = getattr(importlib.import_module(module), name)(**kwargs)
    if variations:
        known = set(_variation_names(env.unwrapped.variation_space))
        unknown = [v for v in variations if v not in known]
        if unknown:
            raise ValueError(f"unknown variations {unknown}: {sorted(known)}")
        env.unwrapped.DEFAULT_VARIATIONS = variations
    return env

def _variation_names(space, prefix='') -> list[str]:
    subspaces = getattr(space, 'spaces', None)
    if not isinstance(subspaces, dict):
        return [prefix[:-1]]
    return [n for k, s in subspaces.items() for n in _variation_names(s, f'{prefix}{k}.')]

def free_cells(env) -> list[tuple[int, int]]:
    g = env.unwrapped.grid
    return [(x, y) for x in range(g.width) for y in range(g.height) if (o := g.get(x, y)) is None or o.can_overlap()]

def reset_layout(env, layout: int, edits: dict | None = None, options: dict | None = None):
    from stable_marl.envs.multigrid.core.constants import Color
    from stable_marl.envs.multigrid.core.world_object import Wall
    spec = (edits or {}).get(int(layout))
    env.reset(seed=int(spec['base'] if spec else layout), options=options)
    if spec:
        g = env.unwrapped.grid
        color = Color.from_index(int(env.unwrapped._variation('wall.color')))
        for x, y, wall in spec['cells']:
            g.set(int(x), int(y), Wall(color) if wall else None)
    return env

def reset_in_layout(env, layout_seed: int, rng: np.random.Generator, edits: dict | None = None):
    from stable_marl.envs.multigrid.core.world_object import Goal
    reset_layout(env, layout_seed, edits)
    u = env.unwrapped
    goal = getattr(u, 'goal_pos', None)
    if goal is not None:
        u.grid.set(int(goal[0]), int(goal[1]), None)
    cells = free_cells(env)
    starts = cells
    if goal is not None:
        g = cells[rng.integers(len(cells))]
        u.goal_pos = np.asarray(g)
        u.grid.set(*g, Goal())
        far = lambda c: abs(c[0] - g[0]) + abs(c[1] - g[1])
        cells = [c for c in cells if c != g]
        try:
            wanted = int(u._variation('goal.min_spawn_distance'))
        except (AttributeError, KeyError):
            wanted = 0
        reach = min(wanted, max(map(far, cells)))
        starts = [c for c in cells if far(c) >= reach]
        if len(starts) < len(u.agents):
            starts = sorted(cells, key=far)[-len(u.agents):]
    for agent, k in zip(u.agents, rng.choice(len(starts), len(u.agents), replace=False)):
        agent.state.pos = starts[k]
        agent.state.dir = int(rng.integers(4))
    return env

def interior_walls(env) -> np.ndarray:
    from stable_marl.envs.multigrid.core.constants import Type
    g = env.unwrapped.grid
    return np.array([[(o := g.get(x, y)) is not None and o.type == Type.wall for y in range(1, g.height - 1)]
                     for x in range(1, g.width - 1)])

def _connected(free: set) -> bool:
    if not free:
        return False
    start = next(iter(free))
    seen, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if c in free and c not in seen:
                seen.add(c)
                stack.append(c)
    return len(seen) == len(free)

def layout_edits(env, k: int, rng: np.random.Generator, tries: int = 100) -> list[list[int]]:
    from stable_marl.envs.multigrid.core.constants import Type
    g = env.unwrapped.grid
    inner = [(x, y) for x in range(1, g.width - 1) for y in range(1, g.height - 1)]
    walls = [c for c in inner if (o := g.get(*c)) is not None and o.type == Type.wall]
    empty = [c for c in inner if g.get(*c) is None]
    free = set(free_cells(env))
    n_remove = min(k // 2, len(walls))
    n_add = min(k - n_remove, len(empty) - 2)
    for _ in range(tries):
        removed = [walls[i] for i in rng.choice(len(walls), n_remove, replace=False)]
        added = [empty[i] for i in rng.choice(len(empty), n_add, replace=False)]
        if _connected((free | set(removed)) - set(added)):
            return [[int(x), int(y), 0] for x, y in removed] + [[int(x), int(y), 1] for x, y in added]
    raise RuntimeError(f"no {k} edits keep the layout connected in {tries} tries")

def edit_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, bool) != np.asarray(b, bool)).reshape(*np.broadcast_shapes(np.shape(a), np.shape(b))[:-2], -1).sum(-1)

def layout_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).reshape(*np.broadcast_shapes(a.shape, b.shape)[:-2], -1).sum(-1)
    return np.where(union > 0, edit_distance(a, b) / np.maximum(union, 1), 0.0)

def nearest_layout(walls: np.ndarray, reference: np.ndarray) -> tuple[int, int, float]:
    d = layout_distance(walls[None], reference)
    i = int(np.argmin(d))
    return i, int(edit_distance(walls, reference[i])), float(d[i])

def _scene_config(scene: dict | None):
    from stable_marl.envs.multigrid.sionna_export import SceneConfig
    config = SceneConfig()
    for key, value in (scene or {}).items():
        if isinstance(getattr(config, key), dict):
            getattr(config, key).update(value)
        else:
            setattr(config, key, value)
    return config

def trace_layouts(
    env_cfg,
    seeds,
    out_dir,
    frequency: float = 2.4e9,
    max_depth: int = 5,
    scene: dict | None = None,
    overwrite: bool = False,
    progress: bool = True,
    edits: dict | None = None,
    **solver_kwargs,
) -> list[Path]:
    import time
    from stable_marl.envs.multigrid.sionna_export import trace_channel_table
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in seeds:
        path = out_dir / f'layout_{seed}.npz'
        paths.append(path)
        if path.exists() and not overwrite:
            continue
        env = reset_layout(make_env(env_cfg), seed, edits)
        start = time.time()
        table = trace_channel_table(env, _scene_config(scene), frequency, max_depth, **solver_kwargs)
        source = dict(env=_plain(env_cfg), seed=int(seed), frequency=frequency, max_depth=max_depth, scene=scene or {},
                      solver=solver_kwargs, edit=(edits or {}).get(int(seed)))
        np.savez_compressed(path, H=table['H'].astype(np.complex64), cells=table['cells'].astype(np.int16),
                            walls=table['walls'].astype(np.uint8), source=json.dumps(source, default=str))
        if progress:
            print(f'layout {seed}: {len(table["cells"])} cells, traced in {time.time() - start:.0f} s', flush=True)
    return paths

def wall_geometry(walls: np.ndarray, a, b) -> tuple[float, int]:
    n = max(int(abs(a[0] - b[0]) + abs(a[1] - b[1])) * 4, 1)
    xs, ys = np.linspace(a[0], b[0], n + 1), np.linspace(a[1], b[1], n + 1)
    crossed = {(int(round(x)), int(round(y))) for x, y in zip(xs, ys)} - {tuple(a), tuple(b)}
    return math.hypot(a[0] - b[0], a[1] - b[1]), int(sum(walls[c] for c in crossed))

def _pair(a, b) -> tuple:
    a, b = tuple(int(v) for v in a), tuple(int(v) for v in b)
    return (a, b) if a <= b else (b, a)

CHANNELS = ('coherent', 'large_scale')

class ChannelTable:

    def __init__(
        self,
        path,
        symmetric: bool = True,
    ):
        z = np.load(path)
        self.cells = [tuple(int(v) for v in c) for c in z['cells']]
        self.idx = {c: i for i, c in enumerate(self.cells)}
        self.walls = z['walls']
        self.source = json.loads(str(z['source'])) if 'source' in z.files else {}
        self.power = self.direct = None
        if 'pairs' in z.files:
            self.G = None
            keys = [_pair(a, b) for a, b in z['pairs'].tolist()]
            self.pairs = {k: max(abs(complex(h)) ** 2, 1e-15) for k, h in zip(keys, z['H'])}
            if 'P' in z.files:
                self.power = {k: max(float(p), 1e-15) for k, p in zip(keys, z['P'])}
                self.direct = {k: float(d) for k, d in zip(keys, z['D'])}
        else:
            G = np.abs(z['H'].astype(np.complex128)) ** 2
            self.G = np.maximum((G + G.T) / 2 if symmetric else G, 1e-15)
            self.pairs = None

    def gain(self, tx, rx, channel: str = 'coherent') -> float:
        if channel == 'large_scale':
            if self.power is None:
                raise ValueError("this layout was traced without the paths' powers (a full table, or a pair cache from before "
                                 "they were kept): trace it again (into a new layout directory) for the large-scale channel")
            return self.power[_pair(tx, rx)]
        if channel != 'coherent':
            raise ValueError(f'channel: one of {CHANNELS}, not {channel!r}')
        if self.pairs is not None:
            return self.pairs[_pair(tx, rx)]
        return float(self.G[self.idx[tuple(rx)], self.idx[tuple(tx)]])

    def direct_fraction(self, tx, rx) -> float:
        if self.power is None:
            return 0.0
        k = _pair(tx, rx)
        return float(min(self.direct[k] / self.power[k], 1.0))

def table_path(layout_dir, seed) -> Path:
    full = Path(layout_dir) / f'layout_{seed}.npz'
    return full if full.exists() else Path(layout_dir) / f'pairs_{seed}.npz'

def collect_trajectories(
    env_cfg,
    seeds,
    out_dir,
    episodes: int = 5,
    max_steps: int = 150,
    policy: dict | None = None,
    epsilon: float = 0.3,
    seed: int = 0,
    overwrite: bool = False,
    edits: dict | None = None,
) -> list[Path]:
    from stable_marl.envs.multigrid.expert import MixedPolicy
    weights = _plain(policy) if policy is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for layout in seeds:
        path = out_dir / f'trajectories_{layout}.npz'
        paths.append(path)
        if path.exists() and not overwrite and len(np.load(path)['cells']) >= episodes:
            continue
        rng = np.random.default_rng([seed, int(layout)])
        agents_policy = MixedPolicy(weights, epsilon, seed=int(rng.integers(2 ** 63)))
        env = make_env(env_cfg)
        n = len(env.unwrapped.agents)
        cells = np.full((episodes, max_steps, n, 2), -1, np.int16)
        dirs = np.full((episodes, max_steps, n), -1, np.int8)
        used = []
        for e in range(episodes):
            reset_in_layout(env, layout, rng, edits)
            u = env.unwrapped
            for t in range(max_steps):
                for i, agent in enumerate(u.agents):
                    if not agent.state.terminated and tuple(agent.state.pos) != (-1, -1):
                        cells[e, t, i] = agent.state.pos
                        dirs[e, t, i] = int(agent.state.dir)
                actions = agents_policy.act(u)
                if t == 0:
                    used.append(agents_policy.behaviour(u))
                _, _, terminated, truncated, _ = env.step({i: int(a) for i, a in enumerate(actions)})
                if all(terminated.values()) or all(truncated.values()):
                    break
        np.savez_compressed(path, cells=cells, dirs=dirs, behaviour=np.array(used), layout=int(layout))
    return paths

DIRECT_TOLERANCE = 0.2e-9

from contextlib import contextmanager

@contextmanager
def file_lock(path, wait: float = 600.0):
    import time
    lock = Path(path).with_name(Path(path).name + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > wait:
                    lock.rmdir()
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.5 + 0.5 * np.random.rand())
    try:
        yield
    finally:
        lock.rmdir()

def replace_npz(path, **arrays):
    import os
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)

class PairCache:

    def __init__(
        self,
        path,
        env_cfg,
        layout_seed: int,
        frequency: float = 2.4e9,
        max_depth: int = 5,
        scene: dict | None = None,
        tracer=None,
        edits: dict | None = None,
        **solver_kwargs,
    ):
        self.path, self.env_cfg, self.seed = Path(path), env_cfg, int(layout_seed)
        self.frequency, self.max_depth, self.scene, self.solver_kwargs = frequency, max_depth, scene or {}, solver_kwargs
        self.tracer, self._scene, self.edit = tracer, None, (edits or {}).get(self.seed)
        self.env = reset_layout(make_env(env_cfg), self.seed, edits)
        self.cells = free_cells(self.env)
        g = self.env.unwrapped.grid
        self.walls = np.ones((g.width, g.height), np.uint8)
        for c in self.cells:
            self.walls[c] = 0
        self.H, self.P, self.D, self.unsaved, self.powers = {}, {}, {}, 0, True
        if self.path.exists():
            z = np.load(self.path)
            if [tuple(c) for c in z['cells'].tolist()] != self.cells:
                raise ValueError(f"{self.path} holds another layout than layout {self.seed}'s")
            if 'source' in z.files:
                was = json.loads(str(z['source']))
                then = json.dumps({k: was.get(k) for k in ('frequency', 'max_depth', 'scene', 'solver')}, sort_keys=True, default=str)
                if then != self._settings():
                    raise ValueError(f"{self.path} was traced with other settings ({then}) than these ({self._settings()}): "
                                     "trace into another layout directory")
            keys = [_pair(a, b) for a, b in z['pairs'].tolist()]
            self.H = {k: complex(h) for k, h in zip(keys, z['H'])}
            self.powers = 'P' in z.files
            if self.powers:
                self.P = {k: float(p) for k, p in zip(keys, z['P'])}
                self.D = {k: float(d) for k, d in zip(keys, z['D'])}

    def _settings(self) -> str:
        return json.dumps(dict(frequency=self.frequency, max_depth=self.max_depth, scene=self.scene, solver=self.solver_kwargs),
                          sort_keys=True, default=str)

    def missing(self, pairs) -> list[tuple]:
        return sorted({p for p in (_pair(a, b) for a, b in pairs) if p[0] != p[1] and p not in self.H})

    def trace(self, pairs, progress: bool = False) -> int:
        from collections import Counter, defaultdict
        todo = self.missing(pairs)
        if todo and not self.powers:
            raise ValueError(f"{self.path} was traced before the paths' powers were kept: trace into another layout directory")
        degree = Counter(c for p in todo for c in p)
        groups = defaultdict(list)
        for a, b in todo:
            tx, rx = (a, b) if (degree[a], a) >= (degree[b], b) else (b, a)
            groups[tx].append(rx)
        for k, (tx, rxs) in enumerate(groups.items()):
            for rx, r in zip(rxs, self._trace(tx, rxs)):
                h, p, d = r if isinstance(r, tuple) else (r, abs(complex(r)) ** 2, 0.0)
                k = _pair(tx, rx)
                self.H[k], self.P[k], self.D[k] = complex(h), float(p), float(d)
            if progress:
                print(f'{k + 1} / {len(groups)} transmitters', end='\r')
        self.unsaved += len(todo)
        return len(todo)

    def _trace(self, tx, rxs):
        if self.tracer is not None:
            return self.tracer(tx, rxs)
        import contextlib, io, os, tempfile
        import sionna.rt as rt
        from comm_core import trace_paths
        from stable_marl.envs.multigrid.sionna_export import export_scene, grid_to_world
        config = _scene_config(self.scene)
        if self._scene is None:
            self._dir = tempfile.TemporaryDirectory()
            with contextlib.redirect_stdout(io.StringIO()):
                export_scene(self.env, self._dir.name, config, frequency=self.frequency)
            self._scene = rt.load_scene(os.path.join(self._dir.name, 'scene.xml'))
            self._scene.frequency = self.frequency
        g = self.env.unwrapped.grid
        point = lambda c: grid_to_world(c[0], c[1], g.width, g.height, config, config.antenna_height)
        paths = trace_paths(self._scene, [point(tx)], [point(c) for c in rxs], max_depth=self.max_depth, **self.solver_kwargs)
        a, tau = paths['a'][0], paths['tau'][0]
        straight = np.linalg.norm(paths['rx_points'] - paths['tx_points'][0], axis=1) / 299792458.0
        direct = (np.abs(a) > 0) & (np.abs(tau - straight[:, None]) < DIRECT_TOLERANCE)
        power = (np.abs(a) ** 2).sum(-1)
        return [(complex(h), float(p), float(d)) for h, p, d in zip(a.sum(-1), power, (np.abs(a) ** 2 * direct).sum(-1))]

    def gain(self, a, b, channel: str = 'coherent') -> float:
        self.trace([(a, b)])
        k = _pair(a, b)
        return max(self.P[k] if channel == 'large_scale' else abs(self.H[k]) ** 2, 1e-15)

    def _absorb(self):
        if not self.path.exists():
            return
        z = np.load(self.path)
        if [tuple(c) for c in z['cells'].tolist()] != self.cells or (self.powers and 'P' not in z.files):
            return
        for i, k in enumerate(_pair(a, b) for a, b in z['pairs'].tolist()):
            if k not in self.H:
                self.H[k] = complex(z['H'][i])
                if self.powers:
                    self.P[k], self.D[k] = float(z['P'][i]), float(z['D'][i])

    def save(self, wait: float = 600.0):
        with file_lock(self.path, wait):
            self._absorb()
            pairs = sorted(self.H)
            source = dict(env=_plain(self.env_cfg), seed=self.seed, frequency=self.frequency, max_depth=self.max_depth,
                          scene=self.scene, solver=self.solver_kwargs, edit=self.edit)
            powers = dict(P=np.array([self.P[p] for p in pairs], np.float64), D=np.array([self.D[p] for p in pairs], np.float64)) \
                if self.powers else {}
            replace_npz(self.path, pairs=np.array(pairs, np.int16).reshape(-1, 2, 2),
                        H=np.array([self.H[p] for p in pairs], np.complex64), cells=np.array(self.cells, np.int16),
                        walls=self.walls, source=json.dumps(source, default=str), **powers)
        self.unsaved = 0

def visited_pairs(trajectories) -> set[tuple]:
    cells = (np.load(trajectories) if isinstance(trajectories, (str, Path)) else trajectories)['cells']
    on = cells[..., 0] >= 0
    out = set()
    for i in range(cells.shape[2]):
        for j in range(i + 1, cells.shape[2]):
            both = on[..., i] & on[..., j]
            for a, b in zip(cells[..., i, :][both].tolist(), cells[..., j, :][both].tolist()):
                if a != b:
                    out.add(_pair(a, b))
    return out

def trace_visited_pairs(
    env_cfg,
    trajectory_dir,
    seeds,
    out_dir,
    frequency: float = 2.4e9,
    max_depth: int = 5,
    scene: dict | None = None,
    progress: bool = True,
    tracer=None,
    edits: dict | None = None,
    **solver_kwargs,
) -> list[Path]:
    import time
    paths = []
    for seed in seeds:
        cache = PairCache(Path(out_dir) / f'pairs_{seed}.npz', env_cfg, seed, frequency, max_depth, scene, tracer, edits,
                          **solver_kwargs)
        start = time.time()
        pairs = visited_pairs(Path(trajectory_dir) / f'trajectories_{seed}.npz')
        traced = cache.trace(pairs)
        cache.save()
        paths.append(cache.path)
        if progress:
            print(f'layout {seed}: {len(pairs)} pairs visited, {traced} traced now, in {time.time() - start:.0f} s', flush=True)
    return paths

from comm_core import Fading

class CachedRayTracedFading(Fading):

    def __init__(
        self,
        cache: PairCache,
        save_every: int = 50,
        channel: str = 'coherent',
    ):
        self.cache, self.save_every, self.node_positions, self.channel = cache, save_every, {}, channel

    def set_positions(self, positions):
        self.node_positions.update({k: tuple(int(v) for v in p) for k, p in positions.items()})

    def gain_between(self, tx_position, rx_position) -> float:
        a, b = tuple(int(v) for v in tx_position), tuple(int(v) for v in rx_position)
        return math.inf if a == b else self.cache.gain(a, b, self.channel)

    def coefficient(self, transmission, link) -> complex:
        a, b = self.node_positions[transmission.sender], self.node_positions[transmission.receiver]
        if a == b:
            return math.inf
        if self.cache.trace([(a, b)]) and self.cache.unsaved >= self.save_every:
            self.cache.save()
        k = _pair(a, b)
        h = self.cache.H[k] if self.channel == 'coherent' else complex(math.sqrt(self.cache.P[k]))
        return h if abs(h) ** 2 >= 1e-15 else complex(math.sqrt(1e-15))

    def save(self):
        self.cache.save()

def db(g):
    return 10 * np.log10(np.maximum(np.asarray(g, float), 1e-30))

def features(walls: np.ndarray, tx, rx, past_gains, tx_moved: bool, rx_moved: bool) -> list[float]:
    distance, crossed = wall_geometry(walls, tx, rx)
    return [math.log10(max(distance, 0.5)), distance, crossed, *db(past_gains), float(tx_moved), float(rx_moved)]

STEP_FEATURES = 6

def step_features(walls: np.ndarray, tx, rx, gain: float, tx_moved: bool, rx_moved: bool) -> list[float]:
    distance, crossed = wall_geometry(walls, tx, rx)
    return [float(db(gain)), math.log10(max(distance, 0.5)), distance, crossed, float(tx_moved), float(rx_moved)]

FADINGS = (None, 'rician')

def fading_correlation(step_seconds: float = 1.0, doppler_hz: float = 0.1) -> float:
    from scipy.special import j0
    return float(np.clip(j0(2 * np.pi * doppler_hz * step_seconds), 0.0, 0.9999))

def _fading(cells: np.ndarray, rng: np.random.Generator, rho: float, direct=None, conditional: bool = False):
    direct = np.zeros(len(cells)) if direct is None else np.clip(np.asarray(direct, float), 0.0, 1.0)
    draw = lambda: (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    w, phase, out = draw(), rng.uniform(0, 2 * np.pi), np.ones(len(cells))
    nu2, var, fresh = np.ones(len(cells)), np.zeros(len(cells)), np.ones(len(cells), bool)
    for t in range(len(cells)):
        d = direct[t]
        if t and not np.array_equal(cells[t], cells[t - 1]):
            w, phase = draw(), rng.uniform(0, 2 * np.pi)
        elif t:
            nu2[t] = abs(np.sqrt(d) * np.exp(1j * phase) + np.sqrt(1 - d) * rho * w) ** 2
            var[t], fresh[t] = (1 - d) * (1 - rho ** 2), False
            w = rho * w + np.sqrt(1 - rho ** 2) * draw()
        if fresh[t]:
            nu2[t], var[t] = d, 1 - d
        out[t] = abs(np.sqrt(d) * np.exp(1j * phase) + np.sqrt(1 - d) * w) ** 2
    return (out, nu2, var, fresh) if conditional else out

def episode_fading(cells: np.ndarray, rng: np.random.Generator, rho: float = 0.9, table: ChannelTable | None = None,
                   conditional: bool = False) -> dict:
    n = cells.shape[1]
    on = cells[..., 0] >= 0
    def direct(i, j):
        if table is None:
            return None
        return [table.direct_fraction(cells[t, i], cells[t, j]) if on[t, i] and on[t, j] and tuple(cells[t, i]) != tuple(cells[t, j])
                else 0.0 for t in range(len(cells))]
    return {(i, j): _fading(cells[:, [i, j]], np.random.default_rng(rng.integers(2 ** 63)), rho, direct(i, j),
                            conditional)
            for i in range(n) for j in range(i + 1, n)}

def link_samples(
    table: ChannelTable,
    cells: np.ndarray,
    dirs: np.ndarray,
    history: int = 4,
    fading: str | None = None,
    fading_correlation: float = 0.9,
    rng: np.random.Generator | None = None,
    steps: bool = False,
    channel: str = 'coherent',
) -> dict:
    rng = rng if rng is not None else np.random.default_rng(0)
    if fading not in FADINGS:
        raise ValueError(f"fading: one of {FADINGS}, not {fading!r} (Rayleigh: 'rician' on a table without a direct path)")
    if fading is not None and channel != 'large_scale':
        raise ValueError("the coherent channel already holds a small-scale draw: fading on top of the large_scale channel only")
    T, n = cells.shape[:2]
    fades = episode_fading(cells, rng, fading_correlation, table, conditional=True) if fading == 'rician' else None
    on = (cells[..., 0] >= 0)
    large = channel == 'large_scale'
    X, Xp, mask, gain, link, S = [], [], [], [], [], []
    extra = {k: [] for k in ('large_scale', 'direct', 'possible_still', 'fading_nu2', 'fading_var', 'fresh')} if large else {}
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            both = on[:, i] & on[:, j]
            g = np.zeros(T)
            for t in np.flatnonzero(both):
                g[t] = table.gain(cells[t, j], cells[t, i], channel)
            L = g.copy()
            if fades is not None:
                fade, nu2, var, fresh = fades[min(i, j), max(i, j)]
                g = g * fade
            if steps:
                moved = lambda a, t: t > 0 and on[t - 1, a] and tuple(cells[t, a]) != tuple(cells[t - 1, a])
                step_rows = {t: step_features(table.walls, tuple(cells[t, j]), tuple(cells[t, i]), g[t], moved(j, t), moved(i, t))
                        for t in np.flatnonzero(both)}
            for t in range(max(history, 2), T):
                if not both[t - history:t + 1].all() or not on[t - 2, i]:
                    continue
                tx, rx, rx_before = tuple(cells[t, j]), tuple(cells[t, i]), tuple(cells[t - 1, i])
                past = g[t - history:t][::-1]
                moved = (tx != tuple(cells[t - 1, j]), rx_before != tuple(cells[t - 2, i]))
                d = DIRS[int(dirs[t - 1, i])]
                ahead = (rx_before[0] + d[0], rx_before[1] + d[1])
                possible = [c for c in (rx_before, ahead) if c in table.idx and c != tx] or [rx_before]
                X.append(features(table.walls, tx, rx, past, *moved))
                rows = [features(table.walls, tx, c, past, *moved) for c in possible]
                Xp.append(rows + [rows[0]] * (2 - len(rows)))
                mask.append([True] * len(rows) + [False] * (2 - len(rows)))
                gain.append(g[t])
                link.append((j, i, t))
                if large:
                    still = [c == rx_before and not moved[0] for c in possible]
                    extra['large_scale'].append(L[t])
                    extra['direct'].append(table.direct_fraction(tx, rx))
                    extra['possible_still'].append(still + [False] * (2 - len(still)))
                    pair_moved = tx != tuple(cells[t - 1, j]) or rx != rx_before
                    extra['fading_nu2'].append(nu2[t] if fades is not None else 1.0)
                    extra['fading_var'].append(var[t] if fades is not None else 0.0)
                    extra['fresh'].append(bool(fresh[t]) if fades is not None else pair_moved)
                if steps:
                    S.append([step_rows[s] for s in range(t - 1, t - history - 1, -1)])
    F = 3 + history + 2
    out = dict(X=np.array(X, float).reshape(-1, F), X_possible=np.array(Xp, float).reshape(-1, 2, F),
               possible_mask=np.array(mask, bool).reshape(-1, 2), gain=np.array(gain, float), link=np.array(link, int).reshape(-1, 3))
    if steps:
        out['S'] = np.array(S, np.float32).reshape(-1, history, STEP_FEATURES)
    if large:
        out.update({k: np.array(v, bool if k in ('possible_still', 'fresh') else float) for k, v in extra.items()})
        out['possible_still'] = out['possible_still'].reshape(-1, 2)
        out['rho'] = np.full(len(out['gain']), float(fading_correlation) if fades is not None else math.nan)
    return out

def build_dataset(
    layout_dir,
    trajectory_dir,
    seeds,
    history: int = 4,
    fading: str | None = None,
    fading_correlation: float = 0.9,
    symmetric: bool = True,
    seed: int = 0,
    steps: bool = False,
    channel: str = 'coherent',
) -> dict:
    parts = []
    for layout in seeds:
        table = ChannelTable(table_path(layout_dir, layout), symmetric=symmetric)
        z = np.load(Path(trajectory_dir) / f'trajectories_{layout}.npz')
        rng = np.random.default_rng([seed, int(layout)])
        for e, (cells, dirs) in enumerate(zip(z['cells'], z['dirs'])):
            s = link_samples(table, cells, dirs, history, fading, fading_correlation, rng, steps, channel)
            s['layout'] = np.full(len(s['gain']), int(layout))
            s['episode'] = np.full(len(s['gain']), e)
            parts.append(s)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

def with_history(data: dict, history: int) -> dict:
    F = data['X'].shape[-1]
    built = F - 5
    if not 1 <= history <= built:
        raise ValueError(f'history: 1 to {built} (the data has {built} past gains), not {history}')
    cols = [*range(3 + history), F - 2, F - 1]
    out = {**data, 'X': data['X'][..., cols], 'X_possible': data['X_possible'][..., cols]}
    if 'S' in data:
        out['S'] = data['S'][:, :history]
    return out
