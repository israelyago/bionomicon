import csv
import logging
import pathlib
from datetime import datetime
from typing import List

import h5py

from data import BiologicalData

logger = logging.getLogger("crawler.data_saver")


class DataSaver:
    def __init__(self, output_folder: pathlib.Path, driver="core") -> None:
        self._output_folder = output_folder

        file_path = pathlib.Path.joinpath(self._output_folder, "sequences.hdf5")

        logger.info(f"Opening HDF5 file. May take some time...")
        start_time = datetime.now()
        if driver == "core":
            self._hdf5_file = h5py.File(
                file_path, "a", driver=driver, backing_store=True
            )
        else:
            self._hdf5_file = h5py.File(file_path, "a", driver="sec2")
        end_time = datetime.now() - start_time

        logger.info(f"It took {end_time} to open the file hdf5")

    def save_in_indiviual_files(self, data_to_save: List[BiologicalData]):
        for to_save in data_to_save:
            file_path = pathlib.Path.joinpath(
                self._output_folder, f"{to_save.get_identifier()}.txt"
            )
            is_enzyme = int(to_save.is_enzyme())
            content = f"{to_save.get_sequence()},{is_enzyme}"
            with file_path.open("w") as file:
                file.write(content)

    def save_to_csv_file(self, biological_data: List[BiologicalData], file_name: str):
        file_path = pathlib.Path.joinpath(self._output_folder, file_name)
        with file_path.open("a") as csv_file:
            stream = csv.writer(csv_file)

            for bio_data in biological_data:
                seq = bio_data.get_sequence()
                is_enzyme = int(bio_data.is_enzyme())
                stream.writerow((seq, is_enzyme))

    def save_to_hdf5_file(
        self, biological_data: List[BiologicalData], file_name: str, driver="core"
    ):
        sequences_list = [data.get_sequence() for data in biological_data]
        is_enzyme_list = [bool(data.is_enzyme()) for data in biological_data]

        off_set = len(sequences_list)

        group = self._hdf5_file.require_group("default")
        h5py_string_type = h5py.string_dtype(encoding="utf-8")
        sequences_dataset = group.require_dataset(
            "sequences", shape=(0,), maxshape=(None,), dtype=h5py_string_type
        )
        is_enzyme_dataset = group.require_dataset(
            "is_enzyme", shape=(0,), maxshape=(None,), dtype=bool
        )

        sequences_dataset.resize(sequences_dataset.shape[0] + off_set, axis=0)
        sequences_dataset[-off_set:] = sequences_list

        is_enzyme_dataset.resize(is_enzyme_dataset.shape[0] + off_set, axis=0)
        is_enzyme_dataset[-off_set:] = is_enzyme_list
