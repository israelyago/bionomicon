import pathlib
import xml.etree.ElementTree as ET
from dataclasses import InitVar, dataclass
from sys import getsizeof
from typing import List

import logs
from bigxml import (
    HandlerTypeHelper,
    Parser,
    Streamable,
    XMLElement,
    XMLElementAttributes,
    XMLText,
    xml_handle_element,
    xml_handle_text,
)
from data import BiologicalData

logger = logs.get_logger("extractor")

NAMESPACE_DICT = {"uniprot": "http://uniprot.org/uniprot"}


@xml_handle_element("uniprot", "entry")
@dataclass
class UniProtEntry:
    node: InitVar
    primary_accession: str = ""
    sequence: str = ""
    is_enzyme: bool = False

    def __post_init__(self, node):
        pass

    @xml_handle_element("accession")
    def handle_accession(self, node: XMLElement):
        if self.primary_accession == "":
            self.primary_accession = node.text.strip()

    @xml_handle_element("sequence")
    def handle_sequence(self, node: XMLElement):
        if self.sequence == "":
            self.sequence = node.text.strip()

    @xml_handle_element("comment")
    def handle_comment(self, node: XMLElement):
        if node.attributes["type"] == "catalytic activity":
            self.is_enzyme = True

    def to_biological_data(self) -> BiologicalData:
        return BiologicalData(
            primary_accession=self.primary_accession,
            sequence=self.sequence,
            is_enzyme=self.is_enzyme,
        )


class DataExtractor:
    def __init__(self, data_path: str) -> None:
        file_path = pathlib.Path(data_path)
        if not file_path.exists():
            raise ValueError(f"File does not exist ({file_path})")
        if not file_path.is_file():
            raise ValueError(
                f"Are you sure the provided path is a file? {file_path}. Symlink not supported"
            )

        self._file_path = file_path

    def get_data(self) -> List[BiologicalData] | None:
        biological_data = []
        with open(self._file_path, "rb") as big_file:
            for item in Parser(big_file).iter_from(UniProtEntry):
                biological_data.append(item.to_biological_data())

        return biological_data
