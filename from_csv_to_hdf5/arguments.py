import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "-r", "--release", help="Run as a release version", action="store_true"
)
_parser.add_argument(
    "-i",
    "--input",
    help="Input .csv file created from extractor project",
    required=True,
)
_parser.add_argument("-o", "--out", help="Output .hdf5 file path", default="output")
_parser.add_argument(
    "-l", "--logs", help="Folder path to save the logs", default="logs"
)
_parser.add_argument(
    "--limit",
    help="Limit how many datapoints to extract. Default: 1mi",
    default=1e6,
    type=int,
)
_args = _parser.parse_args()


def get_args():
    return _args
