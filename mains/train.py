import os
import random
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from dotenv import load_dotenv
from loguru import logger
from c3_wm.utils import init_data, init_model, init_trainer

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(version_base=None, config_path="../configs/train/", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    load_dotenv("../.env")
    seed_everything(cfg.exp_params.manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using runtime hardware device: {device}")

    slurm_jobid = os.getenv("SLURM_JOB_ID", "local_run")
    logger.info(f"SLURM_JOB_ID: {slurm_jobid}")

    os.makedirs(cfg.paths.wandb, exist_ok=True)
    wandb.init(
        name="wm",
        dir=cfg.paths.wandb,
        project= cfg.logging_params.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    logger.info("Weights & Biases Initialized Successfully!")
    data_module = init_data(cfg)
    logger.info("Data Module Initialized Successfully!")

    model = init_model(cfg)
    logger.info("Model Initialized Successfully!")

    trainer = init_trainer(cfg, data_module, model, device, slurm_jobid)
    logger.info("Trainer Initialized Successfully!")

    trainer.fit(cfg)


    wandb.finish()

if __name__ == "__main__":
    main()
