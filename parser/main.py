import pathlib
import sys
from datetime import datetime

import arguments
import logs
import save_data

import extractor

logger = logs.get_logger("main")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

args = arguments.get_args()

INPUT_FILE = args.input
OUT_FILE_PATH = pathlib.Path(args.out)
BATCH_SIZE = 10000
HDF5_DRIVER = "core"
if args.release:
    BATCH_SIZE = 100000
    HDF5_DRIVER = "sec2"

if OUT_FILE_PATH.exists() and not OUT_FILE_PATH.is_file():
    logger.error(f"The output file path ({OUT_FILE_PATH}) is not a file")
    sys.exit("Fatal error. Provided output file path is not a file")

pathlib.Path.mkdir(OUT_FILE_PATH.parent, exist_ok=True)

logger.info(f"Input file set to {INPUT_FILE}")
logger.info(f"Output file to {OUT_FILE_PATH}")
logger.info(f"Batch size set to {BATCH_SIZE}")
logger.info(f"Driver for HDF5: {HDF5_DRIVER}")

data_extractor = extractor.DataExtractor(data_path=INPUT_FILE)
saver = save_data.DataSaver(OUT_FILE_PATH, driver=HDF5_DRIVER)

data_buffer = []
total_data_processed = 0
start_processing_time = datetime.now()

for biological_data in data_extractor.next():
    data_buffer.append(biological_data)

    if len(data_buffer) >= BATCH_SIZE:
        saver.save_to_hdf5_file(data_buffer)
        total_data_processed += len(data_buffer)
        data_buffer.clear()

if len(data_buffer) > 0:
    saver.save_to_hdf5_file(data_buffer)
    total_data_processed += len(data_buffer)
    data_buffer.clear()

end_processing_time = datetime.now() - start_processing_time
logger.info(
    f"It took {end_processing_time} to process {total_data_processed} data points"
)

if HDF5_DRIVER == "core":
    logger.info(
        "HDF5 file will be closing, and all the data will be flushed to disk. This may take a while..."
    )
