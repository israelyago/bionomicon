import pathlib
import collections
import math
import os
import time
from datetime import datetime

import numpy as np
import arguments
import h5dataset
import logs
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, dataset, random_split
from torchtext.vocab import Vocab

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from torch.utils.tensorboard import SummaryWriter
import sys

from torchsampler import ImbalancedDatasetSampler

import torchmetrics

from model import TransformerModel

logger = logs.get_logger("train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using {device} device for torch")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info("Saving last model before shutdown")
        save_model(model, name_without_extension="last")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

args = arguments.get_args()
train_data = h5dataset.H5Dataset(args.dataset)

g = torch.Generator()
g.manual_seed(42)

batch_size = 32

logger.info("Splitting dataset into train and validation. This may take a while...")
train_data, val_dataset = random_split(
    dataset=train_data, lengths=[0.7, 0.3], generator=g
)

logger.info("Creating a data loader")
train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=g)

val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=g)

amino_acids = collections.Counter(
    {
        "A": 11,
        "V": 3,
        "L": 10,
        "I": 9,
        "F": 17,
        "W": 18,
        "M": 20,
        "P": 7,
        "D": 12,
        "E": 2,
        "K": 4,
        "R": 13,
        "G": 8,
        "S": 6,
        "T": 5,
        "C": 19,
        "Y": 15,
        "N": 14,
        "Q": 16,
        "H": 21,
    }
)


def get_tokens(sequence: bytes):
    sequence = sequence.decode("utf-8")
    tokens = [vocab.stoi[amino_acid] for amino_acid in sequence]
    tokens = tokens[0:128]  # Truncate
    tokens += [vocab.stoi["<pad>"]] * (128 - len(tokens))  # Padding
    return torch.LongTensor(tokens)


vocab = Vocab(counter=amino_acids, specials=["<unk>", "<pad>"])

ntokens = len(vocab)  # size of vocabulary
emsize = 10  # embedding dimension
d_hid = 10  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model.train()

# class_weights = torch.tensor(
#     [211290413 / 184948058, 211290413 / 26342355], dtype=torch.float32
# )
# class_weights = 1.0 / (class_weights + 0.1)  # Apply smoothing

# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.BCEWithLogitsLoss()
lr = 0.0001  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float("inf")
epochs = 3


def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [
        torch.tensor(vocab.stoi[token], dtype=torch.long)
        for item in raw_text_iter
        for token in get_tokens(item)
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# def evaluate(model: nn.Module, eval_data: Tensor) -> float:
#     model.eval()  # turn on evaluation mode
#     total_loss = 0.0
#     print("Evaluation size: ", eval_data.size(0))
#     with torch.no_grad():
#         for i in range(0, eval_data.size(0) - 1, 32):
#             for batch in train_data:
#                 data = batch[0]
#                 targets = batch[1].long()
#                 seq_len = data.size(0)
#                 output = model(data)
#                 total_loss += seq_len * criterion(output, targets).item()
#     return total_loss / (len(eval_data) - 1)


def test_model(iter_nth):

    torchmetrics_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2)
    # logger.info(f"Testing model (iter: {iter_nth})")
    with torch.no_grad():
        counter = 0
        for sequences, labels in val_dataset:
            # for sequences, labels in train_data:
            labels = labels.long()
            tokens = torch.stack([get_tokens(seq) for seq in sequences])
            outputs = model(tokens)

            _, predicted = torch.max(outputs, 1)

            values_summed = predicted.sum().item()
            if values_summed != 0:
                logger.info(f"Summed predicted values: {values_summed}")

            score = torchmetrics_f1_score(predicted, labels)

            if counter == 100:
                break
            counter += 1

    score = torchmetrics_f1_score.compute()
    # logger.info(f"F1 score from torchmetrics: {score}")
    tf_writer.add_scalar(f"F1/test/network", score, iter_nth)


def save_model(model, name_without_extension=None):
    dir = pathlib.Path(args.output, experiment_name)
    dir.mkdir(exist_ok=True)

    if name_without_extension is None:
        name_without_extension = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = f"{name_without_extension}.pth"
    file = pathlib.Path(dir, name)

    logger.info(
        f"Saving model as {experiment_name}/{name}",
        extra={"model_path": file, "model_name": name},
    )
    torch.save(model.state_dict(), file)


total_loss = 0.0
log_interval_in_steps = 200
checkpoint_interval_in_steps = log_interval_in_steps * 10
start_time = time.time()

num_batches = math.ceil(len(train_data) / batch_size)

experiment_name = f"Adam-{epochs}epoch-{batch_size}batch-{lr}lr"
tf_writer = SummaryWriter(f"runs/{experiment_name}")


def load_model_params():
    sub_path = pathlib.Path(experiment_name, "last.pth")
    model_path = pathlib.Path(args.output, sub_path)

    if model_path.exists():
        logger.info(f"Loading model params from {model_path}")
        return torch.load(model_path)
    logger.info(f"No checkpoint found. Starting training from scratch")
    return None


last_model_params = load_model_params()
if last_model_params:
    model.load_state_dict(last_model_params)

logger.info(f"Experiment name: {experiment_name}")

current_iter = 1
logger.info("Starting training")
for epoch in range(1, epochs + 1):
    logger.info("Current epoch " + str(epoch))
    for batch in train_data:
        sequences = batch[0]
        labels = batch[1].long()
        # sum = labels.sum().item()
        # if sum is not 32:
        #     logger.info("Labels sum " + str(sum))
        # continue
        tokens = torch.stack([get_tokens(seq) for seq in sequences])
        outputs = model(tokens)

        labels = nn.functional.one_hot(
            labels, num_classes=2
        ).float()  # For nn.BCEWithLogitsLoss

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        average_loss = total_loss / current_iter

        tf_writer.add_scalar("Loss/train/raw", loss, current_iter)
        tf_writer.add_scalar("Loss/train/average", average_loss, current_iter)

        if current_iter % log_interval_in_steps == 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval_in_steps
            cur_loss = total_loss / log_interval_in_steps
            # ppl = np.exp(cur_loss)
            # logger.info(
            #     f"| epoch {epoch:3d} | {current_iter:5d}/{num_batches:5d} batches | "
            #     f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
            #     f"loss {cur_loss:5.6f} | ppl {ppl:8.2f}"
            # )
            logger.info(
                f"| epoch {epoch:3d} | {current_iter:5d}/{num_batches:5d} batches | "
                f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.6f}"
            )
            total_loss = 0
            test_model(current_iter)
            start_time = time.time()

        if current_iter % checkpoint_interval_in_steps == 0:
            save_model(model=model)
        # if current_iter > 100:
        #     break
        current_iter += 1

# test_loss = evaluate(model, train_data)
# test_ppl = math.exp(test_loss)
# print("=" * 89)
# print(f"| End of training | test loss {test_loss:5.2f} | " f"test ppl {test_ppl:8.2f}")
# print("=" * 89)
