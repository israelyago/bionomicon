import logging
import os
import pathlib
import sys
from logging.handlers import RotatingFileHandler

import arguments
import colorlog
from pythonjsonlogger import jsonlogger

MAX_LOG_FILE_SIZE = 100 * 1024**2  # 100 MiB
MAIN_LOG_FILE_NAME = "crawler"


def _get_stdout_handler() -> colorlog.StreamHandler:
    stdout = colorlog.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.INFO)

    stdout_fmt = colorlog.ColoredFormatter(
        "%(asctime)s (%(log_color)s%(levelname)s%(reset)s) %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    stdout.setFormatter(stdout_fmt)

    return stdout


def _get_file_handler(logs_path: pathlib.Path) -> RotatingFileHandler:
    file_handler = RotatingFileHandler(
        logs_path, backupCount=5, maxBytes=MAX_LOG_FILE_SIZE
    )

    file_fmt = jsonlogger.JsonFormatter(
        "%(name)s %(asctime)s %(levelname)s %(filename)s %(lineno)s %(process)d %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        rename_fields={"levelname": "severity", "asctime": "timestamp"},
    )

    file_handler.setFormatter(file_fmt)

    return file_handler


logger = logging.getLogger("crawler")

args = arguments.get_args()
logger.setLevel(logging.INFO)

if not args.release:
    logger.setLevel(logging.DEBUG)
    logger.info(f"Running in DEBUG mode.")

stdout = _get_stdout_handler()
logger.addHandler(stdout)

logs_path = pathlib.Path(args.logs)

if logs_path.exists() and not os.path.isdir(args.logs):
    logger.error(f"The path provided for logging ({args.logs}) is not a directory.")
    exit(1)
elif not logs_path.exists():
    logs_path.mkdir()

logs_main_file = logs_path.joinpath(MAIN_LOG_FILE_NAME)

file_handler = _get_file_handler(logs_main_file)
logger.addHandler(file_handler)


def get_logger(name: str):
    return logger.getChild(name)
