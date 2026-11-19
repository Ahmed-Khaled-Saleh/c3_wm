import argparse
from pathlib import Path

from c3_wm.comm.deployment import deployment_layouts, deployment_summary, save_deployment

DESCRIPTION = "Generate the deployment layout groups."

ENV = dict(_target_='stable_marl.envs.multigrid.FindGoalEnv', agents=2, size=15, num_obstacles=6, n_clutter=10,
           max_steps=150)


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'configs/deploy/layouts.json'))
    ap.add_argument('--train', type=int, nargs=2, default=[0, 200], help='training layouts [start, stop)')
    ap.add_argument('--pool', type=int, nargs=2, default=[260, 20000], help='seeds of new layouts [start, stop)')
    ap.add_argument('--per-group', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0, help='random seed for the layout edits')
    args = ap.parse_args()
    groups = deployment_layouts(ENV, range(*args.train), range(*args.pool), per_group=args.per_group, seed=args.seed)
    save_deployment(args.out, groups)
    print(deployment_summary(groups))
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
