from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

try:
    import wandb
except ImportError:
    wandb = None


def apply_peft(
    model: PreTrainedModel,
    cfg: DictConfig,
    logger: logging.Logger = logging.getLogger(),
) -> any:
    if cfg["peft"]["use"]:
        peft_config = LoraConfig(**cfg["peft"]["configs"])
        model_peft = get_peft_model(model, peft_config)
        logger.info(f"LoRA: rank is {peft_config.r}")
        if cfg["trainer"]["do_train"] is False:
            logger.warning("LoRA: trainer.do_train is False, but peft used.")
    else:
        logger.info("LoRA not used")
        return model

    total_num_params = 0
    trainable_num_params = 0
    for _, param in model_peft.named_parameters():
        total_num_params += param.numel()
        if param.requires_grad:
            trainable_num_params += param.numel()
    logger.info(f"LoRA: total params:      {total_num_params:,d}")
    logger.info(f"LoRA: trainable params:  {trainable_num_params:,d}")
    logger.info(
        f"LoRA: percentage params: {trainable_num_params / total_num_params * 100:.3f}%"
    )

    return model_peft


def get_wandb_run(cfg: DictConfig):
    """get_wandb_run helper for getting Wandb run object or None if not required

    Parameters
    ----------
    cfg : DictConfig
        Omegaconf DictConfig, used with Hydra

    Returns
    -------
    Run | RunDisabled | None:
        Wandb run or None

    Raises
    ------
    RuntimeError
        If config wandb use is True and Wandb not installed, raised Error
    """
    if cfg["wandb"]["use"] is True:
        cfg["trainer"]["report_to"] = "wandb"
        if wandb is None:
            raise RuntimeError("Install `wandb' package: pip install wandb")
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_run = wandb.init(
            # entity=cfg["wandb"]["entity"],
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("name", None),
        )
    else:
        wandb_run = None

    return wandb_run


def is_dir_empty(output_dir_path: str | Path) -> bool:
    """is_dir_empty helper for checking directory

    Parameters
    ----------
    output_dir_path : str | Path
        path to directory

    Returns
    -------
    bool
        return True if output_dir_path not exist OR empty
    """
    if os.path.exists(output_dir_path):
        listdir = os.listdir(output_dir_path)
        if listdir == []:
            return True
        return False
    return True


def can_output_dir_be_used(cfg: DictConfig):
    """can_output_dir_be_used helper for test, is output_dir can be used
    trainer.output_dir must be empty OR overwrite_output_dir should be True

    Parameters
    ----------
    cfg : DictConfig
        Omegaconf DictConfig, used with Hydra

    Raises
    ------
    Exception
        If trainer.output_dir can not be used, raised Exception
    """
    if (
        cfg["trainer"].get("overwrite_output_dir", False) is False
        and is_dir_empty(cfg["trainer"].get("output_dir")) is False
    ):
        raise Exception(
            "The directory {trainer.output_dir} is not empty. "
            "Set trainer.overwrite_output_dir=True for overwrite trainer.output_dir directory "
            "or set trainer.output_dir to empty directory"
        )
