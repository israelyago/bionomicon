import pathlib

import h5py
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, h5_path: pathlib.Path):
        self._hdf5_file = h5py.File(h5_path, "r", driver="sec2")

        enzymes_size = self._hdf5_file["default"]["is_enzyme"].size
        sequences_size = self._hdf5_file["default"]["sequences"].size

        msg = """Your database is missing data?
            Expected equal size for is_enzymes ({}) and sequences ({})""".format(
            enzymes_size, sequences_size
        )
        assert enzymes_size == sequences_size, msg

        self._len = enzymes_size

    def __getitem__(self, index):
        item = (
            self._hdf5_file["default"]["sequences"][index],
            self._hdf5_file["default"]["is_enzyme"][index],
        )
        return item

    def __len__(self):
        return self._len
