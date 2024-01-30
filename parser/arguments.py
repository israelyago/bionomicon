import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "-r", "--release", help="Run as a release version", action="store_true"
)
_parser.add_argument("-i", "--input", help="Input xml file from UniProt", required=True)
_parser.add_argument(
    "-o", "--output", help="Output folder. Creates if needed", default="output"
)
_parser.add_argument(
    "-l", "--logs", help="Folder path to save the logs", default="logs"
)
_args = _parser.parse_args()


def get_args():
    return _args
