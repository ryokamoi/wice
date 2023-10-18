from typing import Literal, TypedDict

DevTest = Literal["dev", "test"]

ThreeLabels = Literal["supported", "partially_supported", "not_supported"]


class RawData(TypedDict):
    label: ThreeLabels
    supporting_sentences: list[list[int]]
    claim: str
    evidence: list[str]
    meta: dict


class Chunks(TypedDict):
    chunks_list: list[list[str]]
    sentence_idx_list: list[list[int]]


class ProcessedData(TypedDict):
    label: str
    claim: str
    evidence: str
    meta: dict[str, str]
