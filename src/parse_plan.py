"""Parse short carrier answers. Malformed output is incorrect."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CARRIERS = ("P", "Q", "R")
LETTERS = "ABC"

ANSWER_PREFIXED = re.compile(
    r"answer\s*:\s*\(?(?P<letter>[ABC])\)\s*(?:carrier\s*)?(?P<carrier>[PQR])\b",
    re.IGNORECASE,
)
PAIR = re.compile(
    r"\(?(?P<letter>[ABC])\)\s*(?:carrier\s*)?(?P<carrier>[PQR])\b",
    re.IGNORECASE,
)
ANSWER_MARK = re.compile(r"answer\s*:", re.IGNORECASE)
CARRIER_ONLY = re.compile(r"\b(?:carrier\s*)?(?P<carrier>[PQR])\b", re.IGNORECASE)
LETTER_LEAD = re.compile(r"^\s*\(?(?P<letter>[ABC])\)?", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedPlan:
    letter: Optional[str]
    carrier: Optional[str]
    kind: str
    parse_ok: bool
    raw: str


def _from_groups(letter: Optional[str], carrier: Optional[str], raw: str) -> ParsedPlan:
    if letter is not None:
        letter = letter.upper()
    if carrier is not None:
        carrier = carrier.upper()
    letter_ok = letter is not None and letter in LETTERS
    carrier_ok = carrier is not None and carrier in CARRIERS
    return ParsedPlan(
        letter=letter if letter_ok else None,
        carrier=carrier if carrier_ok else None,
        kind="unparsed",
        parse_ok=letter_ok and carrier_ok,
        raw=raw,
    )


def parse_plan_answer(text: str) -> ParsedPlan:
    """Prefer the last ``Answer: <letter>) <carrier>`` line."""
    raw = (text or "").strip()
    prefixed = list(ANSWER_PREFIXED.finditer(raw))
    if prefixed:
        match = prefixed[-1]
        return _from_groups(match.group("letter"), match.group("carrier"), raw)

    mark = None
    for found in ANSWER_MARK.finditer(raw):
        mark = found
    region = raw[mark.end() :] if mark is not None else raw
    pairs = list(PAIR.finditer(region))
    if mark is not None:
        if pairs:
            match = pairs[-1]
            return _from_groups(match.group("letter"), match.group("carrier"), raw)
        letter = None
        carrier = None
        lead = LETTER_LEAD.match(region)
        if lead:
            letter = lead.group("letter")
        carriers = list(CARRIER_ONLY.finditer(region))
        if carriers:
            carrier = carriers[-1].group("carrier")
        return _from_groups(letter, carrier, raw)
    if len(pairs) == 1:
        match = pairs[0]
        return _from_groups(match.group("letter"), match.group("carrier"), raw)
    letter = None
    carrier = None
    if len(pairs) == 0:
        lead = LETTER_LEAD.search(region)
        if lead:
            letter = lead.group("letter")
        carriers = list(CARRIER_ONLY.finditer(region))
        if len(carriers) == 1:
            carrier = carriers[0].group("carrier")
    return _from_groups(letter, carrier, raw)


def label_plan_answer(
    parsed: ParsedPlan,
    *,
    gold_letter: str,
    gold_carrier: str,
    trap_letter: str,
    trap_carrier: str,
) -> ParsedPlan:
    if not parsed.parse_ok:
        kind = "unparsed"
    elif parsed.letter == gold_letter and parsed.carrier == gold_carrier:
        kind = "true"
    elif parsed.letter == trap_letter and parsed.carrier == trap_carrier:
        kind = "trap"
    elif parsed.carrier == gold_carrier or parsed.letter == gold_letter:
        kind = "true_partial"
    elif parsed.carrier == trap_carrier or parsed.letter == trap_letter:
        kind = "trap_partial"
    else:
        kind = "other"
    return ParsedPlan(
        letter=parsed.letter,
        carrier=parsed.carrier,
        kind=kind,
        parse_ok=parsed.parse_ok,
        raw=parsed.raw,
    )


def flags_from_plan_kind(kind: str) -> dict[str, bool]:
    return {
        "correct": kind == "true",
        "follows_trap": kind == "trap",
        "true_partial": kind == "true_partial",
        "trap_partial": kind == "trap_partial",
    }


STATUS_LINE = re.compile(
    r"(?P<carrier>[PQR])\s*:\s*(?P<status>eligible|ineligible)\b",
    re.IGNORECASE,
)
OLD_TEMPLATE_STATUSES = {"P": "eligible", "Q": "ineligible", "R": "eligible"}


def parse_eligibility_statuses(text: str) -> dict[str, str | None]:
    """Last eligible/ineligible label per carrier in A's message."""
    found: dict[str, str | None] = {c: None for c in CARRIERS}
    for match in STATUS_LINE.finditer(text or ""):
        found[match.group("carrier").upper()] = match.group("status").lower()
    return found


def score_eligibility_message(text: str, gold: dict[str, str]) -> dict:
    pred = parse_eligibility_statuses(text)
    n_ok = sum(pred[c] == gold[c] for c in gold)
    parsed_n = sum(pred[c] is not None for c in gold)
    matches_old = pred == OLD_TEMPLATE_STATUSES
    return {
        "gold_statuses": gold,
        "pred_statuses": pred,
        "n_status_correct": n_ok,
        "n_status_parsed": parsed_n,
        "all_status_correct": n_ok == 3,
        "copied_old_template": matches_old,
        "incorrect_old_template_copy": matches_old and n_ok != 3,
        "correct": n_ok == 3,
    }
