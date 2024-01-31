import argparse
import pathlib

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "-r", "--release", help="Run as a release version", action="store_true"
)
_parser.add_argument(
    "-d",
    "--dataset",
    help="Dataset .hdf5 generated from parser project",
    required=True,
    type=pathlib.Path,
)
_parser.add_argument(
    "-o",
    "--output",
    help="Folder to output model to. Creates if needed",
    default="output",
    type=pathlib.Path,
)
_parser.add_argument(
    "-l",
    "--logs",
    help="Folder path to save the logs",
    default="logs",
    type=pathlib.Path,
)
_parser.add_argument(
    "-s", "--seed", help="Seed for random operations. Default: 42", type=int, default=42
)
_args = _parser.parse_args()


def get_args():
    return _args
