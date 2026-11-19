import argparse
import json
import os
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from c3_wm.evaluators.deploy import merge_results
from deploy import report, save_results

DESCRIPTION = "Merge the parts of a deployment run split over several jobs."

SAME = ('seed', 'episodes', 'out_dir', 'paths', 'wandb')


def same(a, b) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        a, b = a or {}, b or {}
        return isinstance(a, dict) and isinstance(b, dict) and all(same(a.get(k), b.get(k)) for k in set(a) | set(b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('split', type=Path, help='folder containing part_<i>_job_<id>/ subfolders')
    parser.add_argument('--parts', type=int, default=None, help='expected number of parts')
    parser.add_argument('--wandb-mode', default=None, help='wandb mode for the merged run')
    args = parser.parse_args()
    dirs = sorted(d for d in args.split.glob('part_*') if (d / 'results.json').exists())
    if not dirs or (args.parts is not None and len(dirs) != args.parts):
        raise SystemExit(f"{args.split}: {len(dirs)} parts with results, {args.parts or 'some'} expected")
    configs = [OmegaConf.to_container(OmegaConf.load(d / 'config.yaml')) for d in dirs]
    differ = {k for c in configs[1:] for k in set(c) | set(configs[0]) if k not in SAME and not same(c.get(k), configs[0].get(k))}
    if differ:
        raise SystemExit(f"the parts' configs differ in {sorted(differ)}: not one run's parts")
    results = merge_results([json.loads((d / 'results.json').read_text()) for d in dirs])
    seeds = [int(s) for s in next(iter(results.values()))['seeds']]
    config = {**configs[0], 'seed': min(seeds), 'episodes': len(seeds), 'out_dir': str(args.split), 'parts': len(dirs),
              'paths': {**configs[0]['paths'], 'run': str(args.split), 'wandb': str(args.split / 'wandb')}}
    if args.wandb_mode:
        config['wandb'] = {**config['wandb'], 'mode': args.wandb_mode}
    if not config['wandb']['name']:
        config['wandb'] = {**config['wandb'], 'name': f"deploy-split-{os.getenv('SLURM_ARRAY_JOB_ID', args.split.name)}"}
    logger.info(f"{len(dirs)} parts, {len(seeds)} episodes (seeds {min(seeds)}-{max(seeds)}): {[d.name for d in dirs]}")
    save_results(results, config, args.split)
    report(results, config, args.split)


if __name__ == '__main__':
    main()
