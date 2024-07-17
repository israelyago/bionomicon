import collections
import os

import arguments
import logs
import torch
from torchtext.vocab import Vocab

from model_loader import load_checkpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import sys

from model import TransformerModel

logger = logs.get_logger("inference")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

logger.info(f"Using {device} device for torch")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

args = arguments.get_args()


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

TRUNCATE_INPUT = 1024  # 128 / 512 / 768 / 1024


def get_tokens(sequence: bytes):
    sequence = sequence.decode("utf-8")
    tokens = [vocab.stoi[amino_acid] for amino_acid in sequence]
    tokens = tokens[0:TRUNCATE_INPUT]  # Truncate
    tokens += [vocab.stoi["<pad>"]] * (TRUNCATE_INPUT - len(tokens))  # Padding
    return torch.tensor(tokens, dtype=torch.int, device=device)


emsize = 8  # 10 # embedding dimension
d_hid = (
    128  # 10 # dimension of the feedforward network model in ``nn.TransformerEncoder``
)
nlayers = 6  # 2 # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # 2 # number of heads in ``nn.MultiheadAttention``

if args.big:
    emsize = 128
    d_hid = 512
    nlayers = 6
    nhead = 8


dropout = 0.1  # dropout probability
vocab = Vocab(counter=amino_acids, specials=["<unk>", "<pad>"])
ntokens = len(vocab)  # size of vocabulary

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model.eval()

experiment = input("Experiment name to load from:")
model_params = load_checkpoint(checkpoint_dir=args.output, experiment_name=experiment)

model.load_state_dict(model_params)

while True:
    sequence = input("Your amino acid sequence:")
    tokens = get_tokens(sequence.encode("utf-8"))
    tokens = torch.stack([tokens])

    outputs = model(tokens)
    _, predicted = torch.max(outputs, 1)

    label = "Enzyme" if predicted.item() == 1 else "Non-enzyme"
    logger.info(f"Your sequence is: {label}")
