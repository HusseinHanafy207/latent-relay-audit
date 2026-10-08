"""Question-based record matching from A's prompt text.

Uses the receiver question to find a package name, then keeps that inventory
line in the sender prompt. Does not take a gold letter, locker, or record index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from src.evidence_spans import TokenSpans, token_spans_for_line_text

PACKAGE_IN_QUESTION = re.compile(r"Which locker is package ([A-Za-z][A-Za-z0-9]*) in\?", re.IGNORECASE)
INVENTORY_LINE = re.compile(r"(?m)^(\d+)\. Package ([A-Za-z][A-Za-z0-9]*) is in locker (\d+)\.$")


def package_name_from_question(question: str) -> str:
    match = PACKAGE_IN_QUESTION.search(question or "")
    if match is None:
        raise ValueError(f"Could not parse a package name from question: {question!r}")
    return match.group(1)


def other_package_name(prompt_text: str, exclude_name: str) -> str:
    """Pick a deterministic other inventory package. Does not use a gold index."""
    names = [match.group(2) for match in INVENTORY_LINE.finditer(prompt_text or "")]
    others = [name for name in names if name != exclude_name]
    if not others:
        raise ValueError(f"No other package besides {exclude_name!r}")
    return others[len(others) // 2]


def inventory_line_for_package(prompt_text: str, package_name: str) -> str:
    hits = []
    for match in INVENTORY_LINE.finditer(prompt_text or ""):
        if match.group(2) == package_name:
            hits.append(match)
    if len(hits) != 1:
        raise ValueError(f"Expected one inventory line for package {package_name!r}, found {len(hits)}")
    return hits[0].group(0)


def matched_fact_sentence(question: str, inventory_text: str) -> str:
    """Bare fact line from the question and inventory. No gold index or locker."""
    package_name = package_name_from_question(question)
    line_text = inventory_line_for_package(inventory_text, package_name)
    parsed = INVENTORY_LINE.search(line_text)
    if parsed is None:
        raise ValueError(f"Matched line did not parse: {line_text!r}")
    return f"Package {parsed.group(2)} is in locker {int(parsed.group(3))}."


@dataclass(frozen=True)
class MatchedRecord:
    package_name: str
    line_text: str
    locker_text: str
    line_number: int
    spans: TokenSpans


def match_record_from_question(
    *,
    question: str,
    sender_prompt_text: str,
    rendered: str,
    input_ids: Sequence[int],
    tokenizer: Any,
) -> MatchedRecord:
    package_name = package_name_from_question(question)
    line_text = inventory_line_for_package(sender_prompt_text, package_name)
    parsed = INVENTORY_LINE.search(line_text)
    if parsed is None:
        raise ValueError(f"Matched line did not parse: {line_text!r}")
    locker_text = parsed.group(3)
    spans = token_spans_for_line_text(
        tokenizer,
        rendered,
        input_ids,
        line_text,
        package_name,
        locker_text,
        record_index=int(parsed.group(1)) - 1,
    )
    return MatchedRecord(
        package_name=package_name,
        line_text=line_text,
        locker_text=locker_text,
        line_number=int(parsed.group(1)),
        spans=spans,
    )
