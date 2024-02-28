import csv
import pathlib

import logs
from data import BiologicalData

logger = logs.get_logger("extractor")


class DataExtractor:
    def __init__(self, data_path: str) -> None:
        file_path = pathlib.Path(data_path)
        if not file_path.exists():
            raise ValueError(f"File does not exist ({file_path})")
        if not file_path.is_file():
            raise ValueError(
                f"Are you sure the provided path is a file? {file_path}. Symlink not supported"
            )

        self._csv_reader = csv.reader(file_path.open("r"), delimiter=",")

    def next(self) -> BiologicalData | None:
        for nth, row in enumerate(self._csv_reader):
            try:
                yield BiologicalData(row[0], row[1], bool(int(row[2])))
            except IndexError as error:
                logger.error(
                    f"Could not access information from element index {nth}. Element: {row}"
                )
                logger.error(error)
