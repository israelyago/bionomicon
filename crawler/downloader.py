import pathlib
import re
from typing import List
from urllib.parse import parse_qs, urlencode, urlparse

import logs
import requests
from data import BiologicalData

logger = logs.get_logger("downloader")

LINK_HEADER_REGEX = r"<(.*)>;"
CURSOR_FOLDER_PATH = "."


class UniProtDownloader:
    def __init__(self, batch_size: int, download_type: str) -> None:
        valid_download_types = ["enzyme", "non-enzyme"]
        if download_type not in valid_download_types:
            message = f"download_type should be one of {valid_download_types}, got '{download_type}'"
            logger.error(message)
            raise ValueError(message)

        self._batch_size = batch_size
        self._next_link = None
        self._no_more_data = False
        self._link_header_pattern = re.compile(LINK_HEADER_REGEX)
        self._download_type = download_type

        self._cursor_file_path = pathlib.Path(CURSOR_FOLDER_PATH).joinpath(
            f"cursor-{download_type}.txt"
        )
        with open(self._cursor_file_path, "a+") as cursor_file:
            cursor_file.seek(0)
            cursor = cursor_file.read()
            if not cursor:
                return
            base_link = self._get_base_link()
            params = self._get_request_params()
            params["cursor"] = cursor
            params_encoded = urlencode(params)
            self._next_link = f"{base_link}?{params_encoded}"
            logger.info(f"Cursor {cursor} found. Starting download from it.")

    def _get_base_link(self):
        return f"https://rest.uniprot.org/uniprotkb/search"

    def _get_request_params(self):
        params = {
            "format": "json",
            "query": "(cc_catalytic_activity:*)",
            "fields": "sequence",
            "size": self._batch_size,
        }
        if self._download_type == "non-enzymes":
            params = {
                "format": "json",
                "query": "NOT+(cc_catalytic_activity:*)",
                "fields": "sequence",
                "size": self._batch_size,
            }
        return params

    def _extract_information(self, response_body):
        results = response_body["results"]
        relevant_information = []

        for result in results:
            primary_accession = result["primaryAccession"]
            sequence = result["sequence"]["value"]

            relevant_information.append((primary_accession, sequence))

        return relevant_information

    def _grab_next_link_from_header_string(self, header_str: str) -> str:
        result = self._link_header_pattern.match(header_str)
        return result.group(1)

    def get_cursor_in_memory(self) -> str | None:
        if self._next_link is None:
            return None
        parsed = urlparse(self._next_link)
        query_params = parse_qs(parsed.query)
        if "cursor" in query_params:
            return query_params["cursor"][0]
        return None

    def update_cursor_in_file(self, cursor: str | None):
        with open(self._cursor_file_path, "w") as cursor_file:
            cursor_file.write(cursor)

    def _get_next_data_points(self) -> List | None:
        if self._no_more_data:
            return None

        link = self._get_base_link() if self._next_link is None else self._next_link

        params = self._get_request_params()

        response = None
        if self._next_link is None:
            response = requests.get(link, params=params)
        else:
            response = requests.get(link)

        if "next" in response.links:
            self._next_link = response.links["next"]["url"]
        else:
            self._no_more_data = True

        body = response.json()
        data_points = self._extract_information(body)

        return data_points

    def get_next_batch(self) -> List[BiologicalData] | None:
        is_enzyme = self._download_type == "enzyme"
        data_points = self._get_next_data_points()
        if data_points == None:
            return None

        return [BiologicalData(*data, is_enzyme) for data in data_points]
