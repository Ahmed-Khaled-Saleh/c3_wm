__all__ = ['MESSAGE_PARAMETERS', 'ENCODER_PARAMETERS', 'init_data', 'load_blind', 'init_model', 'init_trainer']

import os

from omegaconf import OmegaConf, DictConfig
import hydra
import torch

def init_data(cfg: DictConfig):
    print(f"Initializing Datamodule: {cfg.pipeline.datamodule._target_}")
    return hydra.utils.instantiate(cfg.pipeline.datamodule)


def get_ckp_path(model_name):
    raise ValueError(f"pipeline.model.{model_name}.checkpoint_path is not set")


MESSAGE_PARAMETERS = ("predictor.msg_", "predictor.tile_", "predictor.no_msg")

ENCODER_PARAMETERS = ("encoder.", "projector.", "sigreg.")

def load_blind(jepa, state: dict, parts: str = "all"):
    if parts not in ("all", "encoder"):
        raise ValueError(f"the blind model's parts to fine-tune from: all or encoder, not {parts!r}")
    if parts == "encoder":
        state = {k: v for k, v in state.items() if k.startswith(ENCODER_PARAMETERS)}
        result = jepa.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(k.startswith(ENCODER_PARAMETERS) for k in result.missing_keys):
            raise RuntimeError(f"the blind checkpoint's encoder does not fit: {result.unexpected_keys + result.missing_keys}")
        print(f"The blind model's encoder and projector loaded; started fresh: {sorted({k.split('.')[0] for k in result.missing_keys})}")
        return
    result = jepa.load_state_dict(state, strict=False)
    other = [k for k in result.missing_keys + result.unexpected_keys if not k.startswith(MESSAGE_PARAMETERS)]
    if other:
        raise RuntimeError(f"the blind checkpoint does not fit the model beyond its message parameters: {other}")
    if result.missing_keys or result.unexpected_keys:
        print(f"Message parameters started fresh: {result.missing_keys}; not used: {result.unexpected_keys}")

def _given_path(model_cfg, name):
    return OmegaConf.select(model_cfg, f"{name}.checkpoint_path", default=None, throw_on_missing=False) or None

def init_model(cfg: DictConfig):
    stage = cfg.pipeline.stage
    model_cfg = cfg.pipeline.model

    if stage == 0 or stage == 1:
        return hydra.utils.instantiate(model_cfg)

    elif stage == 2:
        given = {k: _given_path(model_cfg, k) for k in ("jepa", "vqvae")}
        model_cfg.jepa.checkpoint_path = ""
        model_cfg.vqvae.checkpoint_path = ""

        models = {
            "jepa": hydra.utils.instantiate(model_cfg.jepa),
        }
        model_cfg.jepa.checkpoint_path = given["jepa"] or get_ckp_path("jepa")
        print(f"Loading pretrained JEPA weights from {model_cfg.jepa.checkpoint_path}")

        checkpoint = torch.load(model_cfg.jepa.checkpoint_path, map_location="cpu")
        load_blind(models["jepa"], checkpoint["wm_model_state_dict"], cfg.pipeline.get("blind", "all"))

        vqvae = hydra.utils.instantiate(model_cfg.vqvae)

        model_cfg.vqvae.checkpoint_path = given["vqvae"] or get_ckp_path("vqvae")
        print(f"Loading pretrained VQ-VAE weights from {model_cfg.vqvae.checkpoint_path}")

        checkpoint = torch.load(model_cfg.vqvae.checkpoint_path, map_location="cpu")
        vqvae.load_state_dict(checkpoint["model_state_dict"])

        vqvae.eval()
        for param in vqvae.parameters():
            param.requires_grad = False

        models["vqvae"] = vqvae
        return models


    else:
        raise ValueError(f"Unknown pipeline stage: {stage}")


def init_trainer(cfg: DictConfig, data_module, models, device, slurm_jobid):
    return hydra.utils.instantiate(
        cfg.pipeline.trainer,
        data_module=data_module,
        model=models,
        device=device,
        slurm_jobid=slurm_jobid
    )
