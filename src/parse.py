"""Parse short locker answers. Malformed output is incorrect."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# First-match rule used in Stages A–validate. Kept so original scores can be
# compared after the shared extractor. Do not use this for new primary scores.
ANSWER_LINE = re.compile(
    r"(?:answer\s*:\s*)?(?P<letter>[ABCD])\)\s*locker\s*(?P<locker>\d+)",
    re.IGNORECASE,
)
LETTER_ONLY = re.compile(r"\b(?P<letter>[ABCD])\)", re.IGNORECASE)
LOCKER_ONLY = re.compile(r"locker\s*(?P<locker>\d+)", re.IGNORECASE)

ANSWER_PREFIXED = re.compile(
    r"answer\s*:\s*\(?(?P<letter>[ABCD])\)\s*locker\s*(?P<locker>\d+)",
    re.IGNORECASE,
)
PAIR = re.compile(
    r"\(?(?P<letter>[ABCD])\)\s*locker\s*(?P<locker>\d+)",
    re.IGNORECASE,
)
ANSWER_MARK = re.compile(r"answer\s*:", re.IGNORECASE)
LETTER_LEAD = re.compile(r"^\s*\(?(?P<letter>[ABCD])\)?", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAnswer:
    letter: Optional[str]
    locker: Optional[int]
    kind: str
    parse_ok: bool
    raw: str


def _from_groups(letter: Optional[str], locker: Optional[int], raw: str) -> ParsedAnswer:
    parse_ok = letter is not None and locker is not None
    return ParsedAnswer(
        letter=letter,
        locker=locker,
        kind="unparsed",
        parse_ok=parse_ok,
        raw=raw,
    )


def parse_answer_legacy(text: str) -> ParsedAnswer:
    """Original extractor: first letter/locker pair, including choice echoes."""
    raw = (text or "").strip()
    match = ANSWER_LINE.search(raw)
    letter = match.group("letter").upper() if match else None
    locker = int(match.group("locker")) if match else None
    if letter is None:
        m2 = LETTER_ONLY.search(raw)
        if m2:
            letter = m2.group("letter").upper()
    if locker is None:
        m3 = LOCKER_ONLY.search(raw)
        if m3:
            locker = int(m3.group("locker"))
    return _from_groups(letter, locker, raw)


def parse_answer(text: str) -> ParsedAnswer:
    """Shared extractor for every method.

    Prefer the last ``Answer: <letter>) locker <n>`` line. If that prefix is
    present but incomplete, parse only the text after the last ``Answer:``.
    Unlabeled choice lists (several letter/locker pairs, no Answer line) stay
    unparsed. This is not a recency-only patch.
    """
    raw = (text or "").strip()
    prefixed = list(ANSWER_PREFIXED.finditer(raw))
    if prefixed:
        match = prefixed[-1]
        return _from_groups(match.group("letter").upper(), int(match.group("locker")), raw)

    mark = None
    for found in ANSWER_MARK.finditer(raw):
        mark = found
    region = raw[mark.end() :] if mark is not None else raw
    pairs = list(PAIR.finditer(region))
    if mark is not None:
        if pairs:
            match = pairs[-1]
            return _from_groups(match.group("letter").upper(), int(match.group("locker")), raw)
        letter = None
        locker = None
        lead = LETTER_LEAD.match(region)
        if lead:
            letter = lead.group("letter").upper()
        lockers = list(LOCKER_ONLY.finditer(region))
        if lockers:
            locker = int(lockers[-1].group("locker"))
        return _from_groups(letter, locker, raw)
    if len(pairs) == 1:
        match = pairs[0]
        return _from_groups(match.group("letter").upper(), int(match.group("locker")), raw)
    letter = None
    locker = None
    if len(pairs) == 0:
        m2 = LETTER_ONLY.search(region)
        if m2:
            letter = m2.group("letter").upper()
        m3 = LOCKER_ONLY.search(region)
        if m3:
            locker = int(m3.group("locker"))
    return _from_groups(letter, locker, raw)


def label_answer(
    parsed: ParsedAnswer,
    *,
    gold_letter: str,
    gold_locker: int,
    donor_letter: str,
    donor_locker: int,
) -> ParsedAnswer:
    if not parsed.parse_ok:
        kind = "unparsed"
    elif parsed.letter == gold_letter and parsed.locker == gold_locker:
        kind = "true"
    elif parsed.letter == donor_letter and parsed.locker == donor_locker:
        kind = "donor"
    elif parsed.locker == gold_locker or parsed.letter == gold_letter:
        kind = "true_partial"
    elif parsed.locker == donor_locker or parsed.letter == donor_letter:
        kind = "donor_partial"
    else:
        kind = "other"
    return ParsedAnswer(
        letter=parsed.letter,
        locker=parsed.locker,
        kind=kind,
        parse_ok=parsed.parse_ok,
        raw=parsed.raw,
    )


def label_both(
    text: str,
    *,
    gold_letter: str,
    gold_locker: int,
    donor_letter: str,
    donor_locker: int,
) -> tuple[ParsedAnswer, ParsedAnswer]:
    kwargs = dict(
        gold_letter=gold_letter,
        gold_locker=gold_locker,
        donor_letter=donor_letter,
        donor_locker=donor_locker,
    )
    current = label_answer(parse_answer(text), **kwargs)
    legacy = label_answer(parse_answer_legacy(text), **kwargs)
    return current, legacy


def flags_from_kind(kind: str) -> dict[str, bool]:
    """Exact true and exact donor are the primary rates; partials are separate."""
    return {
        "correct": kind == "true",
        "follows_donor": kind == "donor",
        "true_partial": kind == "true_partial",
        "donor_partial": kind == "donor_partial",
    }
