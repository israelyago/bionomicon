import pathlib

import torch
import logs

logger = logs.get_logger("model_loader")


def load_checkpoint(checkpoint_dir: str, experiment_name: str):
    sub_path = pathlib.Path(experiment_name, "last.pth")
    checkpoint_path = pathlib.Path(checkpoint_dir, sub_path)

    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        return torch.load(checkpoint_path)
    logger.info(f"No checkpoint found. Starting training from scratch")
    return None
