import os
import time
import sys
import arguments
import h5dataset
import logs
import torch
import pathlib
from itertools import product
from torch import nn
from torch.utils.data import DataLoader, random_split
from model_loader import load_checkpoint
from model_saver import save_checkpoint
from dict_hash import sha256

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from model import TransformerModel

logger = logs.get_logger("train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using {device} device for torch")

checkpoint = None
experiment_name = None

args = arguments.get_args()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info(
            "Saving last model before shutdown",
            extra={"experiment_name": experiment_name},
        )
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

train_dataset_size, validation_dataset_size, testing_dataset_size = [
    0.01,
    0.01,
    0.01,
]
ignore_size = 1 - train_dataset_size - validation_dataset_size - testing_dataset_size
EPOCHS = 3


def all_combinations(params):
    return [dict(zip(params.keys(), values)) for values in product(*params.values())]


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


def get_tokens(sequence: bytes, max_seq_length: int):
    sequence = sequence.decode("utf-8")
    try:
        tokens = [amino_acids[amino_acid] for amino_acid in sequence]
        tokens = tokens[0:max_seq_length]  # Truncate
        tokens += [amino_acids["<pad>"]] * (max_seq_length - len(tokens))  # Padding

        return torch.tensor(tokens, dtype=torch.int, device=device)
    except Exception as e:
        logger.error(f"There is something wrong with the sequence {sequence}")
        logger.error(f"Amino acid not recognized: {e}")


def test_bionomicon(model, dataset, config):

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
            tokens = torch.stack(
                [
                    get_tokens(seq, max_seq_length=config["truncate_input"])
                    for seq in sequences
                ]
            ).to(device)
            outputs = model(tokens)

            _, predicted = torch.max(outputs, 1)

            _accuracy_score = torchmetrics_accurary_score(predicted, labels)
            _precision_score = torchmetrics_precision_score(predicted, labels)
            _f1_score = torchmetrics_f1_score(predicted, labels)
            _confmat_scores = confmat(predicted, labels)
            if i > max_iter:
                break

        end_time = time.time() - start_time
        logger.info(f"It took {end_time:.3f} to validate model (max_iter: {max_iter})")

    return (
        torchmetrics_accurary_score.compute(),
        torchmetrics_precision_score.compute(),
        torchmetrics_f1_score.compute(),
        confmat.compute(),
    )


def train_bionomicon(config):

    experiment_name = config["experiment_name"]

    logger.info(f"Experiment name: {experiment_name}")
    my_checkpoint = {
        "current_iter": 1,
        "current_epoch": 1,
        "total_loss": 0,
        "experiment_name": experiment_name,
        "model": config,
    }

    experiment_dir = f"runs/{experiment_name}"
    if config["runs_dir"]:
        experiment_dir = str(pathlib.Path(config["runs_dir"], experiment_name))
    tf_writer = SummaryWriter(experiment_dir)

    generator = torch.Generator()
    if config["seed"] != 0:
        generator.manual_seed(config["seed"])
    logger.info("Splitting dataset into train and validation. This may take a while...")

    whole_dataset = h5dataset.H5Dataset(config["dataset"])
    train_dataset, validation_dataset, testing_dataset, _ignore = random_split(
        dataset=whole_dataset,
        lengths=[
            train_dataset_size,
            validation_dataset_size,
            testing_dataset_size,
            ignore_size,
        ],
        generator=generator,
    )

    logger.info("Creating a data loader")
    train_dataset = DataLoader(
        train_dataset,
        batch_size=my_checkpoint["model"]["batch_size"],
        shuffle=True,
        generator=generator,
        pin_memory=True,
    )

    validation_dataset = DataLoader(
        validation_dataset,
        batch_size=my_checkpoint["model"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    testing_dataset = DataLoader(
        testing_dataset,
        batch_size=my_checkpoint["model"]["batch_size"],
    )

    dropout = 0.1  # dropout probability

    ntokens = len(amino_acids)  # size of vocabulary

    model = TransformerModel(
        ntokens,
        my_checkpoint["model"]["emsize"],
        my_checkpoint["model"]["nhead"],
        my_checkpoint["model"]["d_hid"],
        my_checkpoint["model"]["nlayers"],
        dropout,
    ).to(device)
    model.train()
    my_checkpoint["model"]["state_dict"] = model.state_dict()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=my_checkpoint["model"]["lr"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")

    log_interval_in_steps = 100
    validation_interval_in_steps = 50000
    checkpoint_interval_in_steps = 50000

    last_checkpoint = load_checkpoint(
        checkpoint_dir=config["output"], experiment_name=experiment_name
    )
    if last_checkpoint:
        model.load_state_dict(last_checkpoint["model"]["state_dict"])
        my_checkpoint = last_checkpoint
        logger.info("Using last checkpoint")

    # logger.info("Validating model baseline F1 score")
    # training_f1_score = get_f1_model_score(validation_dataset)
    # tf_writer.add_scalar(f"F1/validation/network", training_f1_score, 0)
    # logger.info(f"Baseline F1 score: {training_f1_score:5.6f}")

    logger.info("Starting training")
    start_time = time.time()
    num_batches = len(train_dataset)
    skip_nth_iters = my_checkpoint["current_iter"] - 1 % num_batches

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

        for epoch in range(my_checkpoint["current_epoch"], EPOCHS + 1):
            logger.info(f"Current epoch {epoch}")
            for batch_index, (sequences, labels) in enumerate(train_dataset):
                if skip_nth_iters > 0:
                    skip_nth_iters -= 1
                    continue
                labels = labels.long().to(device)

                tokens = torch.stack(
                    [get_tokens(seq, config["truncate_input"]) for seq in sequences]
                )
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

                my_checkpoint["total_loss"] += loss.item()

                total_average_loss = (
                    my_checkpoint["total_loss"] / my_checkpoint["current_iter"]
                )
                average_log_loss = loss.mean().item()

                tf_writer.add_scalar(
                    "Train/accuracy",
                    training_accuracy_score,
                    my_checkpoint["current_iter"],
                )
                tf_writer.add_scalar(
                    "Train/precision",
                    training_precision_score,
                    my_checkpoint["current_iter"],
                )
                tf_writer.add_scalar(
                    "Train/F1", training_f1_score, my_checkpoint["current_iter"]
                )

                tf_writer.add_scalar(
                    "Train/loss/raw", loss, my_checkpoint["current_iter"]
                )
                tf_writer.add_scalar(
                    "Train/loss/log_average",
                    average_log_loss,
                    my_checkpoint["current_iter"],
                )
                tf_writer.add_scalar(
                    "Train/loss/total_average",
                    total_average_loss,
                    my_checkpoint["current_iter"],
                )

                if my_checkpoint["current_iter"] % log_interval_in_steps == 0:
                    ms_per_batch = (
                        (time.time() - start_time) * 1000 / log_interval_in_steps
                    )
                    batch_nth = batch_index + 1
                    logger.info(
                        f"| exp. nth {config['experiment_number']}"
                        f" | epoch {epoch:2d}"
                        f" | {batch_nth:5d}/{num_batches:5d} batches"
                        f" | ms/batch {ms_per_batch:5.2f}"
                        # f" | lr {lr:02.2f}"
                        f" | avg. log loss {average_log_loss:5.6f}"
                        f" | f1 training {training_f1_score:5.6f}"
                    )
                    start_time = time.time()

                # if checkpoint["current_iter"] % validation_interval_in_steps == 0:
                #     start_val_time = time.time()
                #     (
                #         training_accuracy_score,
                #         training_precision_score,
                #         training_f1_score,
                #         _confmat,
                #     ) = test_bionomicon(model=model, dataset=validation_dataset, config=config)
                #     tf_writer.add_scalar(
                #         f"Validation/accuracy",
                #         training_accuracy_score,
                #         checkpoint["current_iter"],
                #     )
                #     tf_writer.add_scalar(
                #         f"Validation/precision",
                #         training_precision_score,
                #         checkpoint["current_iter"],
                #     )
                #     tf_writer.add_scalar(
                #         f"Validation/F1",
                #         training_f1_score,
                #         checkpoint["current_iter"],
                #     )
                #     delta = time.time() - start_val_time
                #     start_time += delta

                if my_checkpoint["current_iter"] % checkpoint_interval_in_steps == 0:
                    my_checkpoint["model"]["state_dict"] = model.state_dict()
                    save_checkpoint(
                        checkpoint=my_checkpoint,
                        save_dir=config["output"],
                        experiment_name=experiment_name,
                    )

                my_checkpoint["current_iter"] += 1
                profiler.step()

            val_accuracy_score, val_precision_score, val_f1_score, confmat = (
                test_bionomicon(model=model, dataset=validation_dataset, config=config)
            )
            tf_writer.add_scalar(
                f"Validation/accuracy",
                val_accuracy_score,
                my_checkpoint["current_epoch"],
            )
            tf_writer.add_scalar(
                f"Validation/precision",
                val_precision_score,
                my_checkpoint["current_epoch"],
            )
            tf_writer.add_scalar(
                f"Validation/F1",
                val_f1_score,
                my_checkpoint["current_epoch"],
            )

            my_checkpoint["current_epoch"] += 1
    # training_accuracy_score, training_precision_score, training_f1_score, confmat = (
    #     test_bionomicon(model=model, dataset=testing_dataset, config=config)
    # )
    # logger.info(
    #     f"Final model Accuracy score: {training_accuracy_score} (testing dataset)"
    # )
    # logger.info(
    #     f"Final model Precision score: {training_precision_score} (testing dataset)"
    # )
    # logger.info(f"Final model F1 score: {training_f1_score} (testing dataset)")
    # TN, FP, FN, TP = confmat[0, 0], confmat[0, 1], confmat[1, 0], confmat[1, 1]
    # logger.info(
    #     f"Confusion matrix scores: TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP} (testing dataset)"
    # )
    save_checkpoint(
        checkpoint=my_checkpoint,
        save_dir=config["output"],
        experiment_name=experiment_name,
        name_without_extension="last",
    )


if __name__ == "__main__":

    # search_space = {
    #     "lr": [1e-5, 1e-4],
    #     "batch_size": [32, 64],
    #     "emsize": [32, 512],
    #     "d_hid": [256, 1024],
    #     "nlayers": [8, 10],
    #     "nhead": [8, 16],
    #     "truncate_input": [128, 512],
    # }

    search_space = {
        "lr": [1e-5],
        "batch_size": [32],
        "emsize": [1024, 2048],
        "d_hid": [1024],
        "nlayers": [8],
        "nhead": [4, 8],
        "truncate_input": [512],
    }

    configs = all_combinations(search_space)
    configs = [
        # Filter out invalid configurations
        config
        for config in configs
        if config["emsize"] % config["nhead"] == 0
    ]
    external_config = {
        "dataset": str(args.dataset),
        "seed": args.seed,
        "output": str(args.output),
        "runs_dir": str(args.runs_dir),
    }

    for config_index, config in enumerate(configs):
        # name = sha256(config)
        name = (
            f"lr{config['lr']}"
            f"-batch_size{config['batch_size']}"
            f"-emsize{config['emsize']}"
            f"-d_hid{config['d_hid']}"
            f"-nlayers{config['nlayers']}"
            f"-nhead{config['nhead']}"
            f"-truncate_input{config['truncate_input']}"
        )
        external_config["experiment_name"] = name
        external_config["experiment_number"] = config_index + 1
        config.update(external_config)

    for config in configs:
        logger.info(
            f"Running experiment nth: {config['experiment_number']}/{len(configs)}"
        )
        train_bionomicon(config=config)
