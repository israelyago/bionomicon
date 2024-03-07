import numpy as np
import arguments
import h5dataset
import logs

logger = logs.get_logger("calculate_data_weights")

args = arguments.get_args()
train_data = h5dataset.H5Dataset(args.dataset)

num_total_samples = len(train_data)
num_class_0 = 0
num_class_1 = 0

logger.info(
    "Counting the amount of data for each of the binary class. This may take a while..."
)

for i, label in enumerate(train_data._hdf5_file["default"]["is_enzyme"]):
    if label:
        num_class_1 += 1
    else:
        num_class_0 += 1

    if i % 1000000 == 0:
        percentage = (i / num_total_samples) * 100
        logger.info(
            f"Current iteration at {i} / {num_total_samples} ({percentage:.1f}%)"
        )

assert num_class_0 + num_class_1 == num_total_samples

logger.info(f"You have {num_class_0} class 0 (non-enzymes)")
logger.info(f"You have {num_class_1} class 1 (enzymes)")
logger.info(f"You have {num_total_samples} total samples")

logger.info(
    f"Put it into your training procedure: class_weights = torch.tensor([{num_total_samples} / {num_class_0}, {num_total_samples} / {num_class_1}], dtype=torch.float)"
)
logger.info(f"class_weights = 1.0 / (class_weights + 0.1) # Apply smoothing")
