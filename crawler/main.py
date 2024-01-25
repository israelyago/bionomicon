import pathlib
import sys

import arguments
import downloader
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

BATCH_SIZE = args.batch_size
OUTPUT_FOLDER = pathlib.Path(args.output)

if not args.release:
    BATCH_SIZE = 5

if OUTPUT_FOLDER.exists() and not OUTPUT_FOLDER.is_dir():
    logger.error(f"The output data path ({OUTPUT_FOLDER}) is not a folder.")
    sys.exit("Fatal error. Provided output folder is a file, should be a dir.")

pathlib.Path.mkdir(OUTPUT_FOLDER, exist_ok=True)

logger.info(f"Output folder set to '{OUTPUT_FOLDER}'.")
logger.info(f"Download batch set to {BATCH_SIZE}.")

uniprot_downloader = downloader.UniProtDownloader(
    batch_size=BATCH_SIZE, download_type="enzyme"
)

saver = save_data.DataSaver(OUTPUT_FOLDER)
save_cursor_every_x_batches = 10
current_batch = 1
while True:
    data_to_save = uniprot_downloader.get_next_batch()
    if data_to_save is None:
        break

    # saver.save_in_indiviual_files(data_to_save)
    saver.save_to_csv_file(data_to_save, "sequences.csv")
    # saver.save_to_hdf5_file(data_to_save, "sequences.hdf5")

    # if current_batch % save_cursor_every_x_batches == 0:
    cursor = uniprot_downloader.get_cursor_in_memory()
    uniprot_downloader.update_cursor_in_file(cursor)
    logger.info(f"Updated cursor in file to {cursor}")

    current_batch += 1


logger.info("All data were collected")
