"""Iterative two-record retrieval from the question and inventory text.

Does not use gold lockers, answer letters, or dataset evidence indices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HOP_QUESTION = re.compile(
    r"Which locker contains package ([A-Za-z][A-Za-z0-9]*)\?",
    re.IGNORECASE,
)
PACKAGE_LINE = re.compile(
    r"(?m)^\d+\.\s*Package ([A-Za-z][A-Za-z0-9]*) belongs to shipment ([A-Za-z][A-Za-z0-9]*)\.$"
)
SHIPMENT_LINE = re.compile(
    r"(?m)^\d+\.\s*Shipment ([A-Za-z][A-Za-z0-9]*) is stored in locker (\d+)\.$"
)


@dataclass(frozen=True)
class HopRetrieval:
    package_name: str | None
    shipment_name: str | None
    locker: int | None
    package_sentence: str | None
    shipment_sentence: str | None
    package_line_index: int | None
    shipment_line_index: int | None
    found_package: bool
    found_shipment: bool
    found_both: bool


def package_name_from_hop_question(question: str) -> str:
    match = HOP_QUESTION.search(question or "")
    if match is None:
        raise ValueError(f"Could not parse a package name from hop question: {question!r}")
    return match.group(1)


def retrieve_two_records(question: str, inventory_text: str) -> HopRetrieval:
    try:
        package_name = package_name_from_hop_question(question)
    except ValueError:
        return HopRetrieval(None, None, None, None, None, None, None, False, False, False)

    pkg_hits = []
    for match in PACKAGE_LINE.finditer(inventory_text or ""):
        if match.group(1) == package_name:
            pkg_hits.append(match)
    if len(pkg_hits) != 1:
        return HopRetrieval(package_name, None, None, None, None, None, None, False, False, False)
    pkg_match = pkg_hits[0]
    shipment_name = pkg_match.group(2)
    package_sentence = f"Package {package_name} belongs to shipment {shipment_name}."
    package_line_index = (inventory_text[: pkg_match.start()].count("\n"))

    ship_hits = []
    for match in SHIPMENT_LINE.finditer(inventory_text or ""):
        if match.group(1) == shipment_name:
            ship_hits.append(match)
    if len(ship_hits) != 1:
        return HopRetrieval(
            package_name,
            shipment_name,
            None,
            package_sentence,
            None,
            package_line_index,
            None,
            True,
            False,
            False,
        )
    ship_match = ship_hits[0]
    locker = int(ship_match.group(2))
    shipment_sentence = f"Shipment {shipment_name} is stored in locker {locker}."
    shipment_line_index = (inventory_text[: ship_match.start()].count("\n"))
    return HopRetrieval(
        package_name,
        shipment_name,
        locker,
        package_sentence,
        shipment_sentence,
        package_line_index,
        shipment_line_index,
        True,
        True,
        True,
    )


def retrieved_message(retrieval: HopRetrieval) -> str:
    parts = [s for s in (retrieval.package_sentence, retrieval.shipment_sentence) if s]
    return "\n".join(parts)


def oracle_message(ex) -> str:
    lines = ex.true_inventory_text.splitlines()
    pkg = lines[ex.evidence_package_line].split(". ", 1)[1]
    ship = lines[ex.evidence_shipment_line].split(". ", 1)[1]
    return f"{pkg}\n{ship}"
