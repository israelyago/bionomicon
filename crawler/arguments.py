import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "-r", "--release", help="Run as a release version", action="store_true"
)
_parser.add_argument(
    "-o", "--output", help="Output folder. Creates if needed", default="data"
)
_parser.add_argument(
    "-l", "--logs", help="Folder path to save the logs", default="logs"
)
_parser.add_argument(
    "-b",
    "--batch-size",
    help="Batch size for downloading data. Default: 500 (API limitation)",
    type=int,
    default=500,
)
_args = _parser.parse_args()


def get_args():
    return _args
