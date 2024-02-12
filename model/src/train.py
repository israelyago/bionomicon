import collections
import math
import os
import time

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, dataset
from torchtext.vocab import Vocab

import arguments
import h5dataset
import logs
from model import TransformerModel

logger = logs.get_logger("train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using {device} device for torch")

args = arguments.get_args()
train_data = h5dataset.H5Dataset(args.dataset)

g = torch.Generator()
g.manual_seed(42)

batch_size = 32


# logger.info("Splitting dataset into train and validation. This may take a while...")
# train_data, val_dataset = random_split(
#     dataset=train_data, lengths=[0.7, 0.3], generator=g
# )

logger.info("Creating a data loader")
train_data = DataLoader(train_data, batch_size=batch_size, shuffle=False, generator=g)
# val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=g)

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

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
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


model.train()

classifier_head = nn.Linear(22, 2)
total_loss = 0.0
log_interval = 200
start_time = time.time()

num_batches = math.ceil(len(train_data) / batch_size)
epoch = 1
current_iter = 1
logger.info("Starting training")
for batch in train_data:
    sequences = batch[0]
    truth_y = batch[1].long()
    tokens = torch.stack([get_tokens(seq) for seq in sequences])
    result = model(tokens)
    result_from_head = classifier_head(result)
    result_from_head = result_from_head.mean(dim=1)
    pred_probab = nn.Softmax(dim=-1)(result_from_head)

    loss = criterion(pred_probab, truth_y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()
    if current_iter % log_interval == 0:
        lr = scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
        cur_loss = total_loss / log_interval
        ppl = math.exp(cur_loss)
        print(
            f"| epoch {epoch:3d} | {current_iter:5d}/{num_batches:5d} batches | "
            f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
            f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
        )
        total_loss = 0
        start_time = time.time()

    current_iter += 1
