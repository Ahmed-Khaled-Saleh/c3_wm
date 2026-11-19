import os
import time
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from c3_wm.comm.channel_data import PairCache
from c3_wm.comm.deployment import layout_edits_of, load_deployment

CHUNK = 1000


def group_layouts(file, name: str) -> tuple[list, dict]:
    groups = load_deployment(file)
    names = {g.replace(' ', '-'): g for g in groups['groups']} | {g.replace(' ', '_'): g for g in groups['groups']}
    name = name if name in groups['groups'] else names.get(name, name)
    if name not in groups['groups']:
        raise ValueError(f"layout_group.name: one of {list(groups['groups'])}, not {name!r}")
    return groups['groups'][name], layout_edits_of(groups)


@hydra.main(version_base=None, config_path="../configs", config_name="deploy/config")
def main(cfg: DictConfig):
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    layouts, edits = list(range(*cfg.layouts)), {}
    if cfg.layout_group.file:
        layouts, edits = group_layouts(to_absolute_path(cfg.layout_group.file), cfg.layout_group.name)
    index, count = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)), int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))
    mine = layouts[index::count]
    trace = OmegaConf.to_container(cfg.tracing, resolve=True)
    logger.info(f"task {index} of {count}: {len(mine)} of {len(layouts)} layouts, {mine}, into {cfg.data_dir}/layouts")
    for n, layout in enumerate(mine):
        cache = PairCache(Path(cfg.data_dir) / 'layouts' / f'pairs_{layout}.npz', env_cfg, layout, trace['frequency'],
                          trace['max_depth'], trace['scene'], edits=edits, **trace['solver'])
        cells = cache.cells
        todo = cache.missing([(a, b) for i, a in enumerate(cells) for b in cells[i + 1:]])
        began = time.time()
        for start in range(0, len(todo), CHUNK):
            cache.trace(todo[start:start + CHUNK])
            cache.save()
            done = min(start + CHUNK, len(todo))
            logger.info(f"layout {layout} ({n + 1}/{len(mine)}): {done}/{len(todo)} pairs, "
                        f"{(time.time() - began) / done * (len(todo) - done) / 60:.1f} min left")
        logger.info(f"layout {layout}: {len(cells)} cells, {len(todo)} pairs traced now "
                    f"({len(cells) * (len(cells) - 1) // 2} in all), {(time.time() - began) / 60:.1f} min")


if __name__ == "__main__":
    main()
