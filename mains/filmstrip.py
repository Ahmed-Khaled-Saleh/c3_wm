import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_marl

from c3_wm.utils.env_utils import LayoutWrapper
from deploy_trace import group_layouts

DESCRIPTION = "Draw example episodes without and with communication."

ENV = dict(agents=2, num_obstacles=6, n_clutter=10, max_steps=150)
COLORS = ('#e8463c', '#3c5ae8')
NOT_REACHED = 10 ** 6


def episodes(record_dir) -> dict:
    out = {}
    for path in sorted(Path(record_dir).glob('trajectories_*.npz')):
        z = np.load(path)
        for seed, cells, dirs in zip(z['seeds'], z['cells'], z['dirs']):
            n = int(np.flatnonzero((cells >= 0).all(-1).any(-1)).max()) + 1
            out[int(seed)] = (cells[:n], dirs[:n])
    return out


def starts(results) -> dict:
    (r,) = json.loads(Path(results).read_text()).values()
    return {int(s): (np.asarray(seen), np.asarray(reached))
            for s, seen, reached in zip(r['seeds'], r['starts']['seen_at'], r['starts']['reached_at'])}


def pick(without: dict, with_: dict, n: int) -> list:
    found = []
    for seed in sorted(set(without) & set(with_)):
        (seen0, reached0), (seen1, reached1) = without[seed], with_[seed]
        for a in range(len(seen1)):
            p = 1 - a
            first = lambda seen: seen[p] >= 0 and (seen[a] < 0 or seen[p] < seen[a])
            if first(seen1) and first(seen0) and reached1[a] >= 0:
                late = reached0[a] if reached0[a] >= 0 else NOT_REACHED
                found.append((late - reached1[a], seed, a))
    return [(seed, a) for _, seed, a in sorted(found, reverse=True)[:n]]


def frame(env, cells, dirs, t: int) -> np.ndarray:
    from stable_marl.envs.multigrid.core.grid import Grid
    u = env.unwrapped
    t = min(t, len(cells) - 1)
    everyone, on = list(u.agents), []
    for a, agent in enumerate(everyone):
        if (cells[t, a] >= 0).all():
            agent.state.terminated = False
            agent.state.pos, agent.state.dir = tuple(int(c) for c in cells[t, a]), int(dirs[t, a])
            on.append(agent)
    Grid._tile_cache.clear()
    u.agents = on
    try:
        return u.render()
    finally:
        u.agents = everyone


def goal_cell(env) -> np.ndarray:
    u = env.unwrapped
    return np.array(next((x, y) for x in range(u.width) for y in range(u.height)
                         if (o := u.grid.get(x, y)) is not None and o.type == 'goal'))


def path(ax, cells, t: int, tile: float, reached, goal):
    for a in range(cells.shape[1]):
        c = cells[:min(t, len(cells) - 1) + 1, a]
        c = c[(c >= 0).all(-1)]
        if 0 <= reached[a] <= t:
            c = np.vstack([c, goal[None]])
        if len(c):
            ax.plot((c[:, 0] + 0.5) * tile, (c[:, 1] + 0.5) * tile, color=COLORS[a], lw=1.4, alpha=0.6, zorder=1)


def filmstrip(env, seed: int, agent: int, runs: dict, out: Path):
    import matplotlib.pyplot as plt
    (cells0, dirs0), (seen0, reached0) = runs['without']
    (cells1, dirs1), (seen1, reached1) = runs['with']
    partner = 1 - agent
    sight, arrive = int(seen1[partner]), int(reached1[agent])
    sees = f"{'red' if partner == 0 else 'blue'} sees the goal"
    if sight > 0:
        steps, labels = [0, sight, (sight + arrive) // 2, arrive], ['start', sees, 'halfway', 'arrival (with)']
    else:
        steps = [0, arrive // 3, 2 * arrive // 3, arrive]
        labels = [f'start: {sees}', 'a third of the way', 'two thirds', 'arrival (with)']
    fig, axes = plt.subplots(2, len(steps) + 1, figsize=(3.0 * (len(steps) + 1), 6.4))
    rows = (('without communication', cells0, dirs0, reached0), ('with communication', cells1, dirs1, reached1))
    for row, (name, cells, dirs, reached) in enumerate(rows):
        env.reset(seed=seed)
        tile, goal = env.unwrapped.render().shape[0] / env.unwrapped.height, goal_cell(env)
        for col, t in enumerate(steps + [len(cells) - 1]):
            ax = axes[row, col]
            ax.imshow(frame(env, cells, dirs, t))
            path(ax, cells, t, tile, reached, goal)
            ax.set_xticks([]), ax.set_yticks([])
            got = f"step {reached[agent]}" if reached[agent] >= 0 else 'never'
            title = f"step {t}: {labels[col]}" if col < len(steps) else \
                f"step {t}: the end\n{'red' if agent == 0 else 'blue'} at the goal: {got}"
            if row == 1 and t == sight and col < len(steps):
                title += '\nits message (goal in view) arrives'
            ax.set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(name, fontsize=10)
    fig.suptitle(f"episode {seed} (layout {env.layout}): the {'red' if agent == 0 else 'blue'} agent, its partner seeing "
                 f"the goal first", fontsize=11)
    fig.tight_layout(h_pad=2.5)
    path_ = out / f'filmstrip_{seed}.png'
    fig.savefig(path_, dpi=150)
    plt.close(fig)
    return path_


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('--without', nargs=2, required=True, metavar=('RECORD_DIR', 'RESULTS'), help='recording folder and results.json without communication')
    ap.add_argument('--with', dest='with_', nargs=2, required=True, metavar=('RECORD_DIR', 'RESULTS'), help='recording folder and results.json with communication')
    ap.add_argument('--layout-group', default='OOD', help='layout group of the runs')
    ap.add_argument('--layouts-file', default=str(Path(__file__).resolve().parents[1] / 'configs/deploy/layouts.json'))
    ap.add_argument('--seeds', type=int, nargs='*', help='episode seeds to draw')
    ap.add_argument('--n', type=int, default=4, help='number of episodes to pick')
    ap.add_argument('--out', default='filmstrips')
    args = ap.parse_args()
    traj = {k: episodes(d) for k, (d, _) in (('without', args.without), ('with', args.with_))}
    when = {k: starts(r) for k, (_, r) in (('without', args.without), ('with', args.with_))}
    layouts, edits = group_layouts(args.layouts_file, args.layout_group)
    env = LayoutWrapper(gym.make('MultiGrid-FindGoal-15x15-v0', render_mode='rgb_array', **ENV), layouts, edits=edits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = pick(when['without'], when['with'], args.n)
    if args.seeds:
        agent = dict(pick(when['without'], when['with'], 10 ** 6))
        chosen = [(s, agent.get(s, 0)) for s in args.seeds]
    for seed, a in chosen:
        if seed not in traj['without'] or seed not in traj['with']:
            print(f"episode {seed}: not in both recordings")
            continue
        runs = {k: (traj[k][seed], when[k][seed]) for k in traj}
        print(f"episode {seed}, agent {a}: {filmstrip(env, seed, a, runs, out)}")


if __name__ == '__main__':
    main()
