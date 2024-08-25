import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import arguments
import h5dataset
import logs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import torchmetrics
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

logger = logs.get_logger("train_classical")
device = torch.device("cpu")

logger.info(f"Using {device} device for torch")

checkpoint = None
experiment_name = None

args = arguments.get_args()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

train_dataset_size, testing_dataset_size = [
    0.7,
    0.3,
]
# ignore_size = 1 - train_dataset_size - testing_dataset_size


def save_result(result, save_dir: str, experiment_name: str):
    dir = pathlib.Path(save_dir, experiment_name)
    dir.mkdir(exist_ok=True)

    logger.info(
        f"Saving model results as {experiment_name}/result.json",
        extra={"dir": dir, "experiment_name": experiment_name},
    )
    json_file_path = pathlib.Path(dir, "result.json")
    with open(json_file_path, "w+") as f:
        json.dump(result, f)


def test_classical(model, dataset, config):

    logger.info("Calculating metrics for current model. This may take a while...")
    torchmetrics_accurary_score = torchmetrics.Accuracy(task="binary").to(device)
    torchmetrics_precision_score = torchmetrics.Precision(task="binary").to(device)
    torchmetrics_f1_score = torchmetrics.F1Score(task="binary").to(device)
    confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

    truncate_input = config["truncate_input"]
    with torch.no_grad():
        start_time = time.time()
        for i, (x_train, y_train) in enumerate(dataset):
            x_train = np.array(
                [
                    torch.frombuffer(
                        seq[:truncate_input] + b"?" * (truncate_input - len(seq)),
                        dtype=torch.uint8,
                    )
                    for seq in x_train
                ]
            )
            x_train = torch.from_numpy(x_train)
            y_train = y_train.int()
            outputs = model.predict(x_train)

            predicted = torch.from_numpy(outputs)

            _accuracy_score = torchmetrics_accurary_score(predicted, y_train)
            _precision_score = torchmetrics_precision_score(predicted, y_train)
            _f1_score = torchmetrics_f1_score(predicted, y_train)
            _confmat_scores = confmat(predicted, y_train)

            if i % (int(len(dataset) / 100) + 1) == 0:
                logger.info(f"Testing model. Current index {i}/{len(dataset)}")

        end_time = time.time() - start_time
        logger.info(f"It took {end_time:.3f} to validate model")

    return (
        torchmetrics_accurary_score.compute(),
        torchmetrics_precision_score.compute(),
        torchmetrics_f1_score.compute(),
        confmat.compute(),
    )


def train_classical(config):

    generator = torch.Generator()
    if config["seed"] != 0:
        generator.manual_seed(config["seed"])
    logger.info("Splitting dataset into train and validation. This may take a while...")

    whole_dataset = h5dataset.H5Dataset(config["dataset"])

    # train_dataset, testing_dataset, _ignore = random_split(
    train_dataset, testing_dataset = random_split(
        dataset=whole_dataset,
        lengths=[
            train_dataset_size,
            testing_dataset_size,
            # ignore_size,
        ],
        generator=generator,
    )

    train_dataset = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        pin_memory=True,
    )

    testing_dataset = DataLoader(
        testing_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        pin_memory=True,
    )

    truncate_input = config["truncate_input"]

    models = [
        GaussianNB(),
        SGDClassifier(),
        BernoulliNB(),
        MultinomialNB(),
        PassiveAggressiveClassifier(),
    ]

    for model in models:
        for index, (x_train, y_train) in enumerate(train_dataset):

            x_train = np.array(
                [
                    torch.frombuffer(
                        seq[:truncate_input] + b"0" * (truncate_input - len(seq)),
                        dtype=torch.uint8,
                    )
                    for seq in x_train
                ]
            )
            x_train = torch.from_numpy(x_train)
            y_train = y_train.int()

            model.partial_fit(x_train, y_train, classes=[0, 1])

            if index % (int(len(train_dataset) / 100) + 1) == 0:
                logger.info(f"Training. Current index {index}/{len(train_dataset)}")

        logger.info("Finished training")

        accuracy, precision, f1, confmat = test_classical(
            model=model, dataset=testing_dataset, config=config
        )

        confmat_values = [item for sublist in confmat.tolist() for item in sublist]
        results = {
            "model": model.__class__.__name__,
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "f1": f1.item(),
            "confmat": confmat_values,
        }

        save_result(
            result=results, save_dir=config["output"], experiment_name=results["model"]
        )

        logger.info(f"Model {results['model']} results {results}")


if __name__ == "__main__":

    config = {
        "dataset": str(args.dataset),
        "seed": args.seed,
        "output": str(args.output),
        "runs_dir": str(args.runs_dir),
        "truncate_input": 128,
        "batch_size": 512,
    }

    train_classical(config=config)
