import pathlib

import torch
import logs

logger = logs.get_logger("model_loader")


def load_model_params(model_dir: str, experiment_name: str):
    sub_path = pathlib.Path(experiment_name, "last.pth")
    model_path = pathlib.Path(model_dir, sub_path)

    if model_path.exists():
        logger.info(f"Loading model params from {model_path}")
        return torch.load(model_path)
    logger.info(f"No checkpoint found. Starting training from scratch")
    return None
