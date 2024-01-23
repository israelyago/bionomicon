import pathlib
import sys

import arguments
import logs
import requests
import save_data
from data import BiologicalData

logger = logs.get_logger("main")


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

args = arguments.get_args()

DOWNLOAD_BATCH = args.batch_size
OUTPUT_FOLDER = pathlib.Path(args.output)

if not args.release:
    DOWNLOAD_BATCH = 5

if OUTPUT_FOLDER.exists() and not OUTPUT_FOLDER.is_dir():
    logger.error(f"The output data path ({OUTPUT_FOLDER}) is not a folder.")
    sys.exit("Fatal error. Provided output folder is a file, should be a dir.")

pathlib.Path.mkdir(OUTPUT_FOLDER, exist_ok=True)

logger.info(f"Output folder set to '{OUTPUT_FOLDER}'.")
logger.info(f"Download batch set to {DOWNLOAD_BATCH}.")


def get_link():
    return f"https://rest.uniprot.org/uniprotkb/search"


def get_request_params_for_enzymes(cursor=None):
    params = {
        "format": "json",
        "query": "(cc_catalytic_activity:*)",
        "fields": "cc_catalytic_activity,sequence",
        "size": DOWNLOAD_BATCH,
    }
    if cursor is not None:
        params["cursor"] = cursor
    return params


def extract_information(response_body):
    results = response_body["results"]
    relevant_information = []

    for result in results:
        primary_accession = result["primaryAccession"]
        sequence = result["sequence"]["value"]

        relevant_information.append((primary_accession, sequence))

    return relevant_information


link = get_link()
params = get_request_params_for_enzymes()

response = requests.get(link, params=params)

next_link = None
if "Link" in response.headers:
    next_link = response.headers["Link"]

body = response.json()
data_points = extract_information(body)
data_to_save = []
for data in data_points:
    data_to_save.append(BiologicalData(*data, True))

saver = save_data.DataSaver(OUTPUT_FOLDER)
saver.save_in_indiviual_files(data_to_save)
saver.save_to_csv_file(data_to_save, "peptides_sequences.csv")
saver.save_to_hdf5_file(data_to_save, "peptides_sequences.hdf5")

logger.info("All data were collected")
