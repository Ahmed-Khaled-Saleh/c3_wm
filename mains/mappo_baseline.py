import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from c3_wm.evaluators.deploy import ActionSubset, jepa_transform
from c3_wm.utils.env_utils import LayoutWrapper

FEATURES = (('view_emb', 'pixels'), ('goal_emb', 'goal'))


def load_jepa(cfg, device):
    jepa = hydra.utils.instantiate({k: v for k, v in OmegaConf.to_container(cfg.jepa, resolve=True).items()
                                    if k != 'checkpoint_path'})
    jepa.load_state_dict(torch.load(to_absolute_path(cfg.checkpoints.jepa), map_location='cpu')['wm_model_state_dict'])
    return jepa.to(device).eval().requires_grad_(False)


@torch.no_grad()
def embed(jepa, images, chunk: int = 256) -> np.ndarray:
    x = torch.as_tensor(np.asarray(images))
    lead = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:])
    device = next(jepa.parameters()).device
    out = [jepa.encode({'pixels': jepa_transform(x[i:i + chunk].to(device).permute(0, 3, 1, 2))[:, None]})['emb'][:, 0]
           for i in range(0, len(x), chunk)]
    return torch.cat(out).float().cpu().numpy().reshape(*lead, -1)


class EmbeddedPool:

    def __init__(self, pool, jepa):
        self.pool, self.jepa = pool, jepa

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def _add(self, infos: dict) -> dict:
        for key, source in FEATURES:
            infos[key] = embed(self.jepa, np.asarray(infos[source])[:, -1])[:, None]
        return infos

    def reset(self, *args, **kwargs):
        obs, infos = self.pool.reset(*args, **kwargs)
        return obs, self._add(infos)

    def step(self, *args, **kwargs):
        *out, infos = self.pool.step(*args, **kwargs)
        return (*out, self._add(infos))


class MAPPOActor:

    def __init__(self, model, jepa):
        self.model, self.jepa = model, jepa

    def eval(self):
        return self

    def parameters(self):
        return self.model.parameters()

    def get_action(self, info: dict):
        info = dict(info)
        for key, source in FEATURES:
            images = info[source].cpu().numpy() if hasattr(info[source], 'cpu') else np.asarray(info[source])
            info[key] = embed(self.jepa, images[:, -1])[:, None]
        return self.model.get_action(info)


def make_world(cfg, layouts, edits=None, wrappers=()):
    import stable_marl as sm
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    kwargs = {k: v for k, v in env_cfg.items() if k not in ('_target_', 'variations')}
    return sm.World(cfg.env_id, num_envs=cfg.num_envs, goal_conditioned=True,
                    pre_wrappers=[lambda e: LayoutWrapper(e, layouts, edits=edits or {}), lambda e: ActionSubset(e, cfg.actions),
                                  *wrappers],
                    **kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="baselines/mappo")
def main(cfg: DictConfig):
    from stable_marl.wm.commnet import commnet_features
    from stable_marl.wm.mappo import MAPPO, train_mappo
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    if cfg.device == 'cuda' and device != 'cuda':
        raise RuntimeError("device=cuda, but no GPU is visible")
    jepa = load_jepa(cfg, device)
    world = make_world(cfg, list(range(*cfg.layouts)))
    world.envs = EmbeddedPool(world.envs, jepa)
    world.reset(seed=cfg.seed)
    env = world.envs.envs[0].unwrapped
    columns = {k: np.asarray(v)[:, -1] for k, v in world.infos.items() if not k.startswith('_')}
    obs_dim = commnet_features(columns, cfg.obs_keys, env.width, goal=False).shape[-1]
    model = MAPPO(obs_dim, world.num_agents, num_actions=cfg.actions, critic=cfg.critic, agent_specific=cfg.agent_specific,
                  obs_keys=list(cfg.obs_keys), goal=False, grid_size=env.width, **OmegaConf.to_container(cfg.model)).to(device)
    logger.info(f"MAPPO (critic={cfg.critic}) on layouts {list(cfg.layouts)}: {world.num_agents} agents, {obs_dim} features, "
                f"{sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters, on {device}")
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"mappo-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    extra = dict(env_name=cfg.env_id, layouts=list(cfg.layouts), jepa=str(cfg.checkpoints.jepa), obs_keys=list(cfg.obs_keys))
    saved = [0]

    def log(row):
        logger.info({k: round(v, 4) for k, v in row.items()})
        if run is not None:
            run.log(row, step=int(row['episodes']))
        if row['episodes'] - saved[0] >= cfg.save_every * cfg.num_envs:
            model.save(out / 'mappo.pt', episodes=int(row['episodes']), **extra)
            saved[0] = row['episodes']
    model = train_mappo(model, world, seed=cfg.seed, log=log, **OmegaConf.to_container(cfg.train))
    model.save(out / 'mappo.pt', episodes=int(cfg.train.episodes), **extra)
    logger.info(f"saved {out / 'mappo.pt'}")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
