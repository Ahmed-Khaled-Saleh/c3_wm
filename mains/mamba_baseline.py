import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from mappo_baseline import FEATURES, EmbeddedPool, embed, load_jepa, make_world


def mamba_policy(mamba, jepa, sample: bool = False):
    from stable_marl.wm.mamba import MambaPolicy, agent_features

    class MambaActor(MambaPolicy):
        def _step_masked(self, controller, obs: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
            mask = blocked.repeat_interleave(8, 0)
            state = controller.model(obs, controller.prev_actions, controller.prev_rnn_state, mask)
            _, logits = controller.actor(state.get_features())
            chosen = torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax(-1)
            action = torch.nn.functional.one_hot(chosen, logits.shape[-1]).float()
            controller.advance_rnns(state)
            controller.prev_actions = action.clone()
            return action.squeeze(0)

        @torch.no_grad()
        def get_action(self, obs: dict, **kwargs) -> np.ndarray:
            m = self.mamba
            as_np = lambda v: v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
            info = {k: as_np(v) for k, v in obs.items() if not k.startswith('_') or k == '_needs_flush'}
            for key, source in FEATURES:
                info[key] = embed(jepa, info[source][:, -1])[:, None]
            flush = info.get('_needs_flush')
            actions = []
            for i, controller in enumerate(self._controllers):
                if flush is not None and flush[i]:
                    controller.init_rnns()
                x = agent_features(info, i, m.obs_keys, m.grid_size)
                done = info['terminated'][i, -1].astype(bool)
                x[done] = 0.0
                heard = info['msg_received'][i, -1].astype(bool) & (info['msg_age'][i, -1] == 0)
                blocked = torch.from_numpy(~(heard | np.eye(len(done), dtype=bool)))[None]
                actions.append(self._step_masked(controller, torch.as_tensor(x)[None], blocked).argmax(-1).numpy())
            return np.stack(actions)

    return MambaActor(mamba, deterministic=not sample)


@hydra.main(version_base=None, config_path="../configs", config_name="baselines/mamba")
def main(cfg: DictConfig):
    from stable_marl.wm.mamba import Mamba
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    if cfg.device == 'cuda' and device != 'cuda':
        raise RuntimeError("device=cuda, but no GPU is visible")
    jepa = load_jepa(cfg, device)
    world = make_world(cfg, list(range(*cfg.layouts)))
    world.envs = EmbeddedPool(world.envs, jepa)
    settings = OmegaConf.to_container(cfg.settings)
    if device != 'cuda':
        settings['DEVICE'] = 'cpu'
    mamba = Mamba(world.num_agents, cfg.actions, obs_keys=list(cfg.obs_keys), grid_size=15, seed=cfg.seed,
                  **OmegaConf.to_container(cfg.mamba), **settings)
    out = Path(to_absolute_path(cfg.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if cfg.wandb.enabled and cfg.wandb.mode != 'disabled':
        import wandb
        Path(cfg.paths.wandb).mkdir(parents=True, exist_ok=True)
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name or f"mamba-{os.getenv('SLURM_JOB_ID', 'local')}",
                         mode=cfg.wandb.mode, dir=cfg.paths.wandb, config=OmegaConf.to_container(cfg, resolve=True))
    extra = dict(env_name=cfg.env_id, layouts=list(cfg.layouts), jepa=str(cfg.checkpoints.jepa))
    saved = [0]

    def log(row):
        logger.info({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        if run is not None:
            run.log(row, step=int(row['episode']))
        if row['episode'] - saved[0] >= cfg.save_every:
            mamba.save(out / 'mamba.pt', episodes=int(row['episode']), **extra)
            saved[0] = row['episode']
    logger.info(f"MAMBA on layouts {list(cfg.layouts)}: {cfg.episodes} episodes, settings {settings}")
    mamba.train(world, cfg.episodes, seed=cfg.seed, log_every=cfg.log_every, log=log)
    mamba.save(out / 'mamba.pt', episodes=int(cfg.episodes), **extra)
    logger.info(f"saved {out / 'mamba.pt'}")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
