import pathlib

import torch
import logs
from datetime import datetime
import json
import copy

logger = logs.get_logger("model_saver")


def save_checkpoint(
    checkpoint, save_dir: str, experiment_name: str, name_without_extension=None
):
    dir = pathlib.Path(save_dir, experiment_name)
    dir.mkdir(exist_ok=True)

    if name_without_extension is None:
        name_without_extension = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = f"{name_without_extension}.pth"
    file = pathlib.Path(dir, name)

    logger.info(
        f"Saving model as {experiment_name}/{name}",
        extra={"model_path": file, "model_name": name},
    )
    torch.save(checkpoint, file)
    json_file_path = pathlib.Path(dir, "configuration.json")
    checkpoint_as_config = copy.deepcopy(checkpoint)
    checkpoint_as_config["model"].pop("state_dict", None)
    with open(json_file_path, "w+") as f:
        json.dump(checkpoint_as_config, f)
