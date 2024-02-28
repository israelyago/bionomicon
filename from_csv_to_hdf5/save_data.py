import csv
import logging
import pathlib
from datetime import datetime
from typing import List

import h5py

from data import BiologicalData

logger = logging.getLogger("crawler.data_saver")


class DataSaver:
    def __init__(self, save_to: pathlib.Path, driver="core") -> None:
        self._output_file_path = save_to

        logger.info(f"Opening HDF5 file. May take some time...")
        start_time = datetime.now()
        if driver == "core":
            self._hdf5_file = h5py.File(
                save_to, "a", driver=driver, backing_store=True
            )
        else:
            self._hdf5_file = h5py.File(save_to, "a", driver="sec2")
        end_time = datetime.now() - start_time

        logger.info(f"It took {end_time} to open the file hdf5")

    def save_to_hdf5_file(self, biological_data: List[BiologicalData]):
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
