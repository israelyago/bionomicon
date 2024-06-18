import os
import time
from itertools import product

import arguments
import h5dataset
import logs
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from model_loader import load_checkpoint
from model_saver import save_checkpoint
from dict_hash import sha256

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from torch.utils.tensorboard import SummaryWriter
import sys

import torchmetrics

from model import TransformerModel

logger = logs.get_logger("train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using {device} device for torch")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info("Saving last model before shutdown")
        save_checkpoint(
            checkpoint=checkpoint,
            save_dir=args.output,
            experiment_name=experiment_name,
            name_without_extension="last",
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

args = arguments.get_args()
whole_dataset = h5dataset.H5Dataset(args.dataset)

g = torch.Generator()
if args.seed != 0:
    g.manual_seed(args.seed)

search_space = {
    "lr": [1e-4, 1e-5],
    "batch_size": [32, 64],
    "emsize": [32, 128, 512],
    "d_hid": [256, 512, 1024],
    "nlayers": [8, 10],
    "nhead": [8, 16],
    "truncate_input": [128, 512],
}


def all_combinations(params):
    return [dict(zip(params.keys(), values)) for values in product(*params.values())]


model_configs = all_combinations(search_space)
logger.info(f"Total combinations before filtering: {len(model_configs)}")
model_configs = [
    # Filter out invalid configurations
    config
    for config in model_configs
    if config["emsize"] % config["nhead"] == 0
]
logger.info(f"Total combinations after filtering: {len(model_configs)}")

train_dataset_size, validation_dataset_size, testing_dataset_size = [
    0.002,
    0.001,
    0.001,
]
ignore_size = 1 - train_dataset_size - validation_dataset_size - testing_dataset_size
for model_config_index, model_config in enumerate(model_configs):

    logger.info(f"Model configuration {model_config_index + 1} / {len(model_configs)}")
    checkpoint = {
        "current_iter": 1,
        "current_epoch": 1,
        "total_loss": 0,
        "model": model_config,
    }

    experiment_name = sha256(model_config)
    logger.info(f"Experiment name: {experiment_name}")
    tf_writer = SummaryWriter(f"runs/{experiment_name}")

    logger.info("Splitting dataset into train and validation. This may take a while...")
    train_dataset, validation_dataset, testing_dataset, _ignore = random_split(
        dataset=whole_dataset,
        lengths=[
            train_dataset_size,
            validation_dataset_size,
            testing_dataset_size,
            ignore_size,
        ],
        generator=g,
    )

    logger.info("Creating a data loader")
    train_dataset = DataLoader(
        train_dataset,
        batch_size=checkpoint["model"]["batch_size"],
        shuffle=True,
        generator=g,
        pin_memory=True,
    )

    validation_dataset = DataLoader(
        validation_dataset,
        batch_size=checkpoint["model"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    testing_dataset = DataLoader(
        testing_dataset,
        batch_size=checkpoint["model"]["batch_size"],
    )

    amino_acids = {
        "<unk>": 0,
        "<pad>": 1,
        "A": 2,
        "B": 3,
        "C": 4,
        "D": 5,
        "E": 6,
        "F": 7,
        "G": 8,
        "H": 9,
        "I": 10,
        "J": 11,
        "K": 12,
        "L": 13,
        "M": 14,
        "N": 15,
        "O": 16,
        "P": 17,
        "Q": 18,
        "R": 19,
        "S": 20,
        "T": 21,
        "U": 22,
        "V": 23,
        "W": 24,
        "X": 25,
        "Y": 26,
        "Z": 27,
    }

    def get_tokens(sequence: bytes):
        sequence = sequence.decode("utf-8")
        try:
            tokens = [amino_acids[amino_acid] for amino_acid in sequence]
            tokens = tokens[0 : checkpoint["model"]["truncate_input"]]  # Truncate
            tokens += [amino_acids["<pad>"]] * (
                checkpoint["model"]["truncate_input"] - len(tokens)
            )  # Padding

            return torch.tensor(tokens, dtype=torch.int, device=device)
        except Exception as e:
            logger.error(f"There is something wrong with the sequence {sequence}")
            logger.error(f"Amino acid not recognized: {e}")

    dropout = 0.1  # dropout probability

    ntokens = len(amino_acids)  # size of vocabulary

    model = TransformerModel(
        ntokens,
        checkpoint["model"]["emsize"],
        checkpoint["model"]["nhead"],
        checkpoint["model"]["d_hid"],
        checkpoint["model"]["nlayers"],
        dropout,
    ).to(device)
    model.train()
    checkpoint["model"]["state_dict"] = model.state_dict()

    # Complete dataset
    # class_weights = torch.tensor(
    #     [211290413 / 184948058, 211290413 / 26342355], dtype=torch.float32
    # )

    # Balanced dataset
    class_weights = torch.tensor(
        [52430723 / 26342355, 52430723 / 26088368], dtype=torch.float
    )

    # Balanced-tiny dataset
    # class_weights = torch.tensor([1570420 / 816433, 1570420 / 753987], dtype=torch.float)

    class_weights = 1.0 / (class_weights + 0.1)  # Apply smoothing

    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["model"]["lr"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 10

    def get_model_metrics(dataset):

        logger.info("Calculating metrics for current model. This may take a while...")
        torchmetrics_accurary_score = torchmetrics.Accuracy(task="binary").to(device)
        torchmetrics_precision_score = torchmetrics.Precision(task="binary").to(device)
        torchmetrics_f1_score = torchmetrics.F1Score(task="binary").to(device)
        confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

        with torch.no_grad():
            start_time = time.time()
            max_iter = int(len(dataset))
            for i, (sequences, labels) in enumerate(dataset):
                labels = labels.to(device)
                labels = labels.long()
                tokens = torch.stack([get_tokens(seq) for seq in sequences]).to(device)
                outputs = model(tokens)

                _, predicted = torch.max(outputs, 1)

                _accuracy_score = torchmetrics_accurary_score(predicted, labels)
                _precision_score = torchmetrics_precision_score(predicted, labels)
                _f1_score = torchmetrics_f1_score(predicted, labels)
                _confmat_scores = confmat(predicted, labels)
                if i > max_iter:
                    break

            end_time = time.time() - start_time
            logger.info(
                f"It took {end_time:.3f} to validate model (max_iter: {max_iter})"
            )
        return (
            torchmetrics_accurary_score.compute(),
            torchmetrics_precision_score.compute(),
            torchmetrics_f1_score.compute(),
            confmat.compute(),
        )

    log_interval_in_steps = 100
    validation_interval_in_steps = 50000
    checkpoint_interval_in_steps = 50000
    # exec_time = time.time()

    # optimizer_name = type(optimizer).__name__
    # dataset_name = pathlib.Path(args.dataset).stem
    # model_size = "big" if args.big else "small"
    # experiment_name = f"{dataset_name}-{optimizer_name}-{model_size}-{epochs}ep-{batch_size}batch-{lr}lr{TRUNCATE_INPUT}seq{exec_time:.0f}"
    # experiment_name = f"{dataset_name}-{optimizer_name}-{model_size}-{checkpoint["batch_size"]}batch-{lr}lr{checkpoint["truncate_input"]}seq"
    # tf_writer = SummaryWriter(f"runs/{experiment_name}")

    last_checkpoint = load_checkpoint(
        checkpoint_dir=args.output, experiment_name=experiment_name
    )
    if last_checkpoint:
        model.load_state_dict(last_checkpoint["model"]["state_dict"])
        checkpoint = last_checkpoint
        logger.info("Using last checkpoint")

    # logger.info("Validating model baseline F1 score")
    # training_f1_score = get_f1_model_score(validation_dataset)
    # tf_writer.add_scalar(f"F1/validation/network", training_f1_score, 0)
    # logger.info(f"Baseline F1 score: {training_f1_score:5.6f}")

    logger.info("Starting training")
    start_time = time.time()
    num_batches = len(train_dataset)
    skip_nth_iters = checkpoint["current_iter"] - 1 % num_batches

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./runs/{experiment_name}"
        ),
    ) as profiler:

        training_torchmetrics_accuracy_score = torchmetrics.Accuracy(task="binary").to(
            device
        )
        training_torchmetrics_precision_score = torchmetrics.Precision(
            task="binary"
        ).to(device)
        training_torchmetrics_f1_score = torchmetrics.F1Score(task="binary").to(device)

        for epoch in range(checkpoint["current_epoch"], epochs + 1):
            logger.info(f"Current epoch {epoch}")
            for batch_index, (sequences, labels) in enumerate(train_dataset):
                if skip_nth_iters > 0:
                    skip_nth_iters -= 1
                    continue
                labels = labels.long().to(device)

                tokens = torch.stack([get_tokens(seq) for seq in sequences])
                outputs = model(tokens)  # .to(device)

                targets = nn.functional.one_hot(
                    labels, num_classes=2
                ).float()  # For nn.BCEWithLogitsLoss

                loss = criterion(outputs, targets)

                _, predicted = torch.max(outputs, 1)

                training_accuracy_score = training_torchmetrics_accuracy_score(
                    predicted, labels
                ).item()
                training_precision_score = training_torchmetrics_precision_score(
                    predicted, labels
                ).item()
                training_f1_score = training_torchmetrics_f1_score(
                    predicted, labels
                ).item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                checkpoint["total_loss"] += loss.item()

                total_average_loss = (
                    checkpoint["total_loss"] / checkpoint["current_iter"]
                )
                average_log_loss = loss.mean().item()

                tf_writer.add_scalar(
                    "Train/accuracy",
                    training_accuracy_score,
                    checkpoint["current_iter"],
                )
                tf_writer.add_scalar(
                    "Train/precision",
                    training_precision_score,
                    checkpoint["current_iter"],
                )
                tf_writer.add_scalar(
                    "Train/F1", training_f1_score, checkpoint["current_iter"]
                )

                tf_writer.add_scalar("Train/loss/raw", loss, checkpoint["current_iter"])
                tf_writer.add_scalar(
                    "Train/loss/log_average",
                    average_log_loss,
                    checkpoint["current_iter"],
                )
                tf_writer.add_scalar(
                    "Train/loss/total_average",
                    total_average_loss,
                    checkpoint["current_iter"],
                )

                if checkpoint["current_iter"] % log_interval_in_steps == 0:
                    ms_per_batch = (
                        (time.time() - start_time) * 1000 / log_interval_in_steps
                    )
                    batch_nth = batch_index + 1
                    logger.info(
                        f"| epoch {epoch:2d}"
                        f" | {batch_nth:5d}/{num_batches:5d} batches"
                        f" | ms/batch {ms_per_batch:5.2f}"
                        # f" | lr {lr:02.2f}"
                        f" | avg. log loss {average_log_loss:5.6f}"
                        f" | f1 training {training_f1_score:5.6f}"
                    )
                    start_time = time.time()

                if checkpoint["current_iter"] % validation_interval_in_steps == 0:
                    start_val_time = time.time()
                    (
                        training_accuracy_score,
                        training_precision_score,
                        training_f1_score,
                        _confmat,
                    ) = get_model_metrics(validation_dataset)
                    tf_writer.add_scalar(
                        f"Validation/accuracy",
                        training_accuracy_score,
                        checkpoint["current_iter"],
                    )
                    tf_writer.add_scalar(
                        f"Validation/precision",
                        training_precision_score,
                        checkpoint["current_iter"],
                    )
                    tf_writer.add_scalar(
                        f"Validation/F1",
                        training_f1_score,
                        checkpoint["current_iter"],
                    )
                    delta = time.time() - start_val_time
                    start_time += delta

                if checkpoint["current_iter"] % checkpoint_interval_in_steps == 0:
                    checkpoint["model"]["state_dict"] = model.state_dict()
                    save_checkpoint(
                        checkpoint=checkpoint,
                        save_dir=args.output,
                        experiment_name=experiment_name,
                    )

                checkpoint["current_iter"] += 1
                profiler.step()

            checkpoint["current_epoch"] += 1
    training_accuracy_score, training_precision_score, training_f1_score, confmat = (
        get_model_metrics(testing_dataset)
    )
    logger.info(
        f"Final model Accuracy score: {training_accuracy_score} (testing dataset)"
    )
    logger.info(
        f"Final model Precision score: {training_precision_score} (testing dataset)"
    )
    logger.info(f"Final model F1 score: {training_f1_score} (testing dataset)")
    TN, FP, FN, TP = confmat[0, 0], confmat[0, 1], confmat[1, 0], confmat[1, 1]
    logger.info(
        f"Confusion matrix scores: TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP} (testing dataset)"
    )
    save_checkpoint(
        checkpoint=checkpoint,
        save_dir=args.output,
        experiment_name=experiment_name,
        name_without_extension="last",
    )
