import pathlib
import sys
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src import arguments, logs
from src.h5dataset import H5Dataset

torch.use_deterministic_algorithms(True)

logger = logs.get_logger("main")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

args = arguments.get_args()

DATASET_FILE_PATH = args.dataset
OUTPUT_FOLER = args.output

if OUTPUT_FOLER.exists() and not OUTPUT_FOLER.is_dir():
    logger.error(f"Output folder expected to be a dir {OUTPUT_FOLER}")
    sys.exit("Fatal error. Provided output file path is not a dir")

OUTPUT_FOLER.mkdir(exist_ok=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using {device} device for torch")
logger.info(f"Seed used: {args.seed}")

start_time = datetime.now()
dataset = H5Dataset(DATASET_FILE_PATH)
end_time = datetime.now()

logger.info("It took: {} to open the dataset".format(end_time - start_time))

g = torch.Generator()
g.manual_seed(args.seed)

dataset = DataLoader(dataset, batch_size=32, shuffle=False, generator=g)

train_dataset, val_dataset = random_split(
    dataset=dataset, lengths=[0.7, 0.3], generator=g
)
