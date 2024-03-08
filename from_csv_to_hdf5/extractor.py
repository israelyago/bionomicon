import csv
import pathlib
import math

import logs
from data import BiologicalData

logger = logs.get_logger("extractor")


class DataExtractor:
    def __init__(self, data_path: str, limit_data_points: int) -> None:
        file_path = pathlib.Path(data_path)
        if not file_path.exists():
            raise ValueError(f"File does not exist ({file_path})")
        if not file_path.is_file():
            raise ValueError(
                f"Are you sure the provided path is a file? {file_path}. Symlink not supported"
            )

        self._csv_reader = csv.reader(file_path.open("r"), delimiter=",")
        self._enzymes_to_collect = math.floor(limit_data_points / 2)
        self._non_enzymes_to_collect = self._enzymes_to_collect

    def next(self) -> BiologicalData | None:
        for nth, row in enumerate(self._csv_reader):
            try:
                is_enzyme = bool(int(row[2]))
                if is_enzyme:
                    if self._enzymes_to_collect <= 0:
                        continue
                    self._enzymes_to_collect -= 1
                else:
                    if self._non_enzymes_to_collect <= 0:
                        continue
                    self._non_enzymes_to_collect -= 1
                yield BiologicalData(row[0], row[1], is_enzyme)
            except IndexError as error:
                logger.error(
                    f"Could not access information from element index {nth}. Element: {row}"
                )
                logger.error(error)
