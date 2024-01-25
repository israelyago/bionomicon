import pathlib
import xml.etree.ElementTree as ET
from sys import getsizeof
from typing import List

from data import BiologicalData

import logs

logger = logs.get_logger("extractor")

NAMESPACE_DICT = {"uniprot": "http://uniprot.org/uniprot"}


class DataExtractor:
    def __init__(self, data_path: str) -> None:
        file_path = pathlib.Path(data_path)
        if not file_path.exists():
            raise ValueError(f"File does not exist ({file_path})")
        if not file_path.is_file():
            raise ValueError(
                f"Are you sure the provided path is a file? {file_path}. Symlink not supported"
            )

        tree = ET.parse(file_path)
        root = tree.getroot()

        self._root = root

    def get_data(self) -> List[BiologicalData] | None:
        biological_data = []
        for child in self._root.findall("uniprot:entry", NAMESPACE_DICT):
            data = self._extract_entry(child)
            if data is None:
                continue
            biological_data.append(data)

        return biological_data

    def _extract_entry(self, entry: ET.Element) -> BiologicalData | None:
        accession_element = entry.find("uniprot:name", NAMESPACE_DICT)
        primary_accession = accession_element.text.strip()

        sequence_element = entry.find("uniprot:sequence", NAMESPACE_DICT)
        if sequence_element is None:
            msg = f"Sequence element not found (id: {primary_accession})."
            logger.error(msg)
            return None

        sequence = sequence_element.text.strip()
        for comment in entry.findall("uniprot:comment", NAMESPACE_DICT):
            if comment.get("type") == "catalytic activity":
                return BiologicalData(primary_accession, sequence, True)

        return BiologicalData(primary_accession, sequence, False)
