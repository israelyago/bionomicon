class BiologicalData:
    def __init__(self, primary_accession: str, sequence: str, is_enzyme: bool):
        self._primary_accession = primary_accession
        self._sequence = sequence
        self._sequence_is_enzyme = is_enzyme

    def get_identifier(self) -> str:
        return self._primary_accession

    def get_sequence(self) -> str:
        return self._sequence

    def is_enzyme(self) -> bool:
        return self._sequence_is_enzyme
