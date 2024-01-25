import pathlib
import sys

import arguments
import extractor
import logs
import save_data

logger = logs.get_logger("main")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

args = arguments.get_args()

INPUT_FILE = args.input
OUTPUT_FOLDER = pathlib.Path(args.output)

if OUTPUT_FOLDER.exists() and not OUTPUT_FOLDER.is_dir():
    logger.error(f"The output data path ({OUTPUT_FOLDER}) is not a folder.")
    sys.exit("Fatal error. Provided output folder is a file, should be a dir.")

pathlib.Path.mkdir(OUTPUT_FOLDER, exist_ok=True)

logger.info(f"Input file set to '{INPUT_FILE}'.")
logger.info(f"Output folder set to '{OUTPUT_FOLDER}'.")

data_extractor = extractor.DataExtractor(data_path=INPUT_FILE)
saver = save_data.DataSaver(OUTPUT_FOLDER)

data_to_save = data_extractor.get_data()

saver.save_to_csv_file(data_to_save, "sequences.csv")

logger.info("All data were extracted")
