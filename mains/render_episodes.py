import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stable_marl
from c3_wm.utils.env_utils import LayoutWrapper
from deploy_trace import group_layouts
from filmstrip import ENV, COLORS, episodes, frame, goal_cell, path, pick, starts

NAMES = ('red', 'blue')


def render(env, seed, agent, runs, out: Path, fps: int, hold: float):
    (c0, d0), (seen0, reached0) = runs['without']
    (c1, d1), (seen1, reached1) = runs['with']
    partner = 1 - agent
    T = max(len(c0), len(c1))
    env.reset(seed=seed)
    tile, goal = env.unwrapped.render().shape[0] / env.unwrapped.height, goal_cell(env)
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(8, 5.0), dpi=110)
        for ax, (name, cells, dirs, seen, reached) in zip(
                axes, (('Without communication', c0, d0, seen0, reached0), ('With communication', c1, d1, seen1, reached1))):
            env.reset(seed=seed)
            ax.imshow(frame(env, cells, dirs, t))
            path(ax, cells, t, tile, reached, goal)
            ax.set_xticks([]), ax.set_yticks([])
            notes = []
            for a in range(2):
                if 0 <= reached[a] <= t:
                    notes.append(f"{NAMES[a]}: reached the goal at step {reached[a]}")
                elif 0 <= seen[a] <= t:
                    notes.append(f"{NAMES[a]}: saw the goal at step {seen[a]}")
                else:
                    notes.append(f"{NAMES[a]}: searching")
            ax.set_title(name, fontsize=11)
            ax.set_xlabel('\n'.join(notes), fontsize=9.5, linespacing=1.5)
        fig.suptitle(f'Step {t}', fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    frames += [frames[-1]] * int(hold * fps)
    mp4 = out / f'episode_{seed}.mp4'
    imageio.mimwrite(mp4, frames, fps=fps, codec='libx264', quality=8, macro_block_size=1,
                     ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    imageio.imwrite(out / f'episode_{seed}.jpg', frames[0])
    return mp4, dict(seed=seed, agent=NAMES[agent], partner=NAMES[partner],
                     partner_sees=int(seen1[partner]), reached_with=int(reached1[agent]),
                     reached_without=int(reached0[agent]), steps=T)


def draw_paths(img, cells, t: int, tile: float, reached, goal) -> np.ndarray:
    im = Image.fromarray(img)
    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for a in range(cells.shape[1]):
        c = cells[:min(t, len(cells) - 1) + 1, a]
        c = c[(c >= 0).all(-1)]
        if 0 <= reached[a] <= t:
            c = np.vstack([c, goal[None]])
        if len(c) > 1:
            rgb = tuple(int(COLORS[a][i:i + 2], 16) for i in (1, 3, 5))
            draw.line([((x + 0.5) * tile, (y + 0.5) * tile) for x, y in c], fill=rgb + (190,), width=max(3, int(tile / 6)))
    return np.asarray(Image.alpha_composite(im.convert('RGBA'), layer).convert('RGB'))


def montage(env, seeds, traj, when, out: Path, cols: int, rows: int, fps: int, hold: float, size: int = 240, gap: int = 6):
    clips = []
    for seed in seeds:
        cells, dirs = traj[seed]
        reached = when[seed][1]
        env.reset(seed=seed)
        tile, goal = env.unwrapped.render().shape[0] / env.unwrapped.height, goal_cell(env)
        clip = []
        for t in range(len(cells)):
            img = draw_paths(frame(env, cells, dirs, t), cells, t, tile, reached, goal)
            clip.append(np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS)))
        clips.append(clip)
    length = max(len(c) for c in clips) + int(hold * fps)
    width, height = cols * size + (cols + 1) * gap, rows * size + (rows + 1) * gap
    frames = []
    for t in range(length):
        canvas = np.full((height, width, 3), 255, np.uint8)
        for k, clip in enumerate(clips):
            r, c = divmod(k, cols)
            y, x = gap + r * (size + gap), gap + c * (size + gap)
            canvas[y:y + size, x:x + size] = clip[min(t, len(clip) - 1)]
        frames.append(canvas)
    mp4 = out / 'montage.mp4'
    imageio.mimwrite(mp4, frames, fps=fps, codec='libx264', quality=8, macro_block_size=1,
                     ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    imageio.imwrite(out / 'montage.jpg', frames[len(frames) // 2])
    return mp4


def main():
    ap = argparse.ArgumentParser(description='Render episodes without and with communication as videos.')
    ap.add_argument('--without', nargs=2, required=True)
    ap.add_argument('--with', dest='with_', nargs=2, required=True)
    ap.add_argument('--layouts-file', required=True)
    ap.add_argument('--layout-group', default='OOD')
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--fps', type=int, default=4)
    ap.add_argument('--hold', type=float, default=2.0)
    ap.add_argument('--out', default='videos')
    ap.add_argument('--montage', type=int, nargs=2, metavar=('COLS', 'ROWS'),
                    help='grid of episodes with communication instead of side-by-side videos')
    ap.add_argument('--max-steps', type=int, default=40, help='longest episode in the montage')
    args = ap.parse_args()
    traj = {k: episodes(d) for k, (d, _) in (('without', args.without), ('with', args.with_))}
    when = {k: starts(r) for k, (_, r) in (('without', args.without), ('with', args.with_))}
    layouts, edits = group_layouts(args.layouts_file, args.layout_group)
    env = LayoutWrapper(gym.make('MultiGrid-FindGoal-15x15-v0', render_mode='rgb_array', **ENV), layouts, edits=edits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.montage:
        cols, rows = args.montage
        ranked = [s for s, _ in pick(when['without'], when['with'], 10 ** 6)] + sorted(when['with'])
        seeds = []
        for s in args.seeds or ranked:
            seen, reached = when['with'][s]
            if s not in seeds and s in traj['with'] and (reached >= 0).all() and 12 <= reached.max() <= args.max_steps:
                seeds.append(s)
        seeds = seeds[:cols * rows]
        print(montage(env, seeds, traj['with'], when['with'], out, cols, rows, args.fps, args.hold), seeds)
        return
    agent = dict(pick(when['without'], when['with'], 10 ** 6))
    chosen = [(s, agent.get(s, 0)) for s in args.seeds] if args.seeds else pick(when['without'], when['with'], args.n)
    meta = []
    for seed, a in chosen:
        runs = {k: (traj[k][seed], when[k][seed]) for k in traj}
        mp4, info = render(env, seed, a, runs, out, args.fps, args.hold)
        meta.append(info)
        print(mp4, info)
    (out / 'episodes.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
