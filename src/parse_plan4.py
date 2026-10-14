"""Parse four-carrier booking answers and A's q/eligibility lines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CARRIERS = ("P", "Q", "R", "S")
LETTERS = "ABCD"

ANSWER_PREFIXED = re.compile(
    r"answer\s*:\s*\(?(?P<letter>[ABCD])\)\s*(?:carrier\s*)?(?P<carrier>[PQRS])\b",
    re.IGNORECASE,
)
PAIR = re.compile(
    r"\(?(?P<letter>[ABCD])\)\s*(?:carrier\s*)?(?P<carrier>[PQRS])\b",
    re.IGNORECASE,
)
ANSWER_MARK = re.compile(r"answer\s*:", re.IGNORECASE)
CARRIER_ONLY = re.compile(r"\b(?:carrier\s*)?(?P<carrier>[PQRS])\b", re.IGNORECASE)
LETTER_LEAD = re.compile(r"^\s*\(?(?P<letter>[ABCD])\)?", re.IGNORECASE)
RESULT_LINE = re.compile(
    r"(?P<carrier>[PQRS])\s*:\s*q\s*=\s*(?P<q>-?\d+)\s*\|\s*(?P<status>eligible|ineligible)\b",
    re.IGNORECASE,
)
RESULT_LINE_OPTIONAL_Q = re.compile(
    r"(?P<carrier>[PQRS])\s*:\s*(?:q\s*=\s*)?(?P<q>-?\d+)\s*\|\s*(?P<status>eligible|ineligible)\b",
    re.IGNORECASE,
)
RESULTS_MARK = re.compile(r"(?im)^\s*(?:\#{1,3}\s*)?\*{0,2}RESULTS\*{0,2}\s*$")
ARITHMETIC = re.compile(
    r"[=×*+\-/]|floor\s*\(|min\s*\(|max\s*\(|productive hours|\bh\s*=|\bq\s*=",
    re.IGNORECASE,
)
CHEAPEST_AFTER = re.compile(
    r"(?:cheapest|lowest[- ](?:cost|price)|lowest\s+price)(?:\s+\w+){0,12}\s+"
    r"(?:is\s+)?(?:carrier\s*)?\*{0,2}([PQRS])\*{0,2}\b",
    re.IGNORECASE,
)
CHEAPEST_BEFORE = re.compile(
    r"\*{0,2}([PQRS])\*{0,2}(?:\s+\w+){0,8}\s+(?:is|as)\s+(?:the\s+)?"
    r"(?:cheapest|lowest[- ](?:cost|price))",
    re.IGNORECASE,
)
CHOICE_LINE = re.compile(
    r"(?im)^(?:choice|pick|selected)\s*:\s*(?:carrier\s*)?\*{0,2}([PQRS])\*{0,2}\b",
)
B_PLAN_LINE = re.compile(
    r"(?P<carrier>[PQRS])\s*:\s*(?P<status>eligible|ineligible)\b"
    r"(?:\s*[|,]\s*\$?\s*(?P<price>-?\d+))?"
    r"(?:\s*\|\s*(?P<flag>candidate|dropped))?",
    re.IGNORECASE,
)
THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
CHAT_MARK = re.compile(r"<\|im_(?:start|end)\|>")


@dataclass(frozen=True)
class ParsedPlan4:
    letter: Optional[str]
    carrier: Optional[str]
    kind: str
    parse_ok: bool
    raw: str


def _from_groups(letter: Optional[str], carrier: Optional[str], raw: str) -> ParsedPlan4:
    if letter is not None:
        letter = letter.upper()
    if carrier is not None:
        carrier = carrier.upper()
    letter_ok = letter is not None and letter in LETTERS
    carrier_ok = carrier is not None and carrier in CARRIERS
    return ParsedPlan4(
        letter=letter if letter_ok else None,
        carrier=carrier if carrier_ok else None,
        kind="unparsed",
        parse_ok=letter_ok and carrier_ok,
        raw=raw,
    )


def parse_plan4_answer(text: str) -> ParsedPlan4:
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


def label_plan4_answer(
    parsed: ParsedPlan4,
    *,
    gold_letter: str,
    gold_carrier: str,
    trap_letter: str,
    trap_carrier: str,
) -> ParsedPlan4:
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
    return ParsedPlan4(
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


def result_text_for_scoring(text: str) -> str:
    """Prefer the marked RESULTS block when it contains result lines."""
    match = RESULTS_MARK.search(text or "")
    if match:
        rest = (text or "")[match.end() :]
        if RESULT_LINE.search(rest):
            return rest
    return text or ""


def written_working_metrics(text: str) -> dict:
    """Did the written condition actually produce working, not just four result lines?"""
    raw = text or ""
    nonempty = [ln for ln in raw.splitlines() if ln.strip()]
    mark = RESULTS_MARK.search(raw)
    first_result = RESULT_LINE.search(raw)
    if mark:
        prefix = raw[: mark.start()]
    elif first_result:
        prefix = raw[: first_result.start()]
    else:
        prefix = raw
    working_lines = [ln for ln in prefix.splitlines() if ln.strip()]
    has_block = mark is not None
    has_arith = bool(ARITHMETIC.search(prefix))
    n_result = len(RESULT_LINE.findall(raw))
    produced = bool(has_block or len(working_lines) >= 2 or (has_arith and len(nonempty) > 4))
    return {
        "n_nonempty_lines": len(nonempty),
        "n_working_lines": len(working_lines),
        "n_result_lines": n_result,
        "has_results_block": has_block,
        "has_arithmetic": has_arith,
        "produced_working": produced,
    }


def cheapest_named_in_plan(text: str) -> Optional[str]:
    found: list[tuple[int, str]] = []
    for match in CHOICE_LINE.finditer(text or ""):
        found.append((match.start(), match.group(1).upper()))
    for match in CHEAPEST_AFTER.finditer(text or ""):
        found.append((match.start(), match.group(1).upper()))
    for match in CHEAPEST_BEFORE.finditer(text or ""):
        found.append((match.start(), match.group(1).upper()))
    if not found:
        return None
    found.sort()
    return found[-1][1]


def plan_extraction(plan_text: str, pred_carrier: Optional[str]) -> dict:
    """Missing extraction is not disagreement."""
    named = cheapest_named_in_plan(plan_text)
    if named is None:
        status = "missing"
        agrees = None
    elif pred_carrier and named == pred_carrier.upper():
        status = "agrees"
        agrees = True
    else:
        status = "disagrees"
        agrees = False
    return {
        "plan_named_cheapest": named,
        "plan_extraction_status": status,
        "plan_answer_agrees": agrees,
    }


def parse_intermediate_lines(text: str, *, allow_missing_q_equals: bool = False) -> dict[str, dict]:
    """Last q/eligibility line per carrier."""
    found = {c: {"q": None, "status": None} for c in CARRIERS}
    pattern = RESULT_LINE_OPTIONAL_Q if allow_missing_q_equals else RESULT_LINE
    for match in pattern.finditer(text or ""):
        carrier = match.group("carrier").upper()
        found[carrier] = {"q": int(match.group("q")), "status": match.group("status").lower()}
    return found


def handoff_from_written(text: str) -> str:
    """Four result lines for B. Drops scratchpad working when a RESULTS block exists."""
    region = result_text_for_scoring(text)
    found: dict[str, str] = {}
    for match in RESULT_LINE.finditer(region):
        found[match.group("carrier").upper()] = match.group(0).strip()
    if len(found) == 4:
        return "\n".join(found[c] for c in CARRIERS)
    for match in RESULT_LINE_OPTIONAL_Q.finditer(region):
        found.setdefault(match.group("carrier").upper(), match.group(0).strip())
    if found:
        return "\n".join(found.get(c, f"{c}: unparsed") for c in CARRIERS)
    return (region or text or "").strip()


def n_missing_q_equals(text: str) -> int:
    scored = result_text_for_scoring(text)
    strict = parse_intermediate_lines(scored, allow_missing_q_equals=False)
    loose = parse_intermediate_lines(scored, allow_missing_q_equals=True)
    return sum(loose[c]["q"] is not None and strict[c]["q"] is None for c in CARRIERS)


def score_sender_message(
    text: str,
    gold_q: dict[str, int],
    gold_status: dict[str, str],
    *,
    allow_missing_q_equals: bool = False,
) -> dict:
    pred = parse_intermediate_lines(
        result_text_for_scoring(text), allow_missing_q_equals=allow_missing_q_equals
    )
    n_q = sum(pred[c]["q"] == gold_q[c] for c in CARRIERS)
    n_status = sum(pred[c]["status"] == gold_status[c] for c in CARRIERS)
    n_parsed = sum(pred[c]["status"] is not None and pred[c]["q"] is not None for c in CARRIERS)
    all_q = n_q == 4
    all_status = n_status == 4
    loose = score_sender_message(
        text, gold_q, gold_status, allow_missing_q_equals=True
    ) if not allow_missing_q_equals else None
    out = {
        "pred_q": {c: pred[c]["q"] for c in CARRIERS},
        "pred_status": {c: pred[c]["status"] for c in CARRIERS},
        "gold_q": gold_q,
        "gold_status": gold_status,
        "n_q_correct": n_q,
        "n_status_correct": n_status,
        "n_carriers_parsed": n_parsed,
        "all_q_correct": all_q,
        "all_status_correct": all_status,
        "all_four_correct": all_q and all_status,
        "malformed": n_parsed != 4,
        "correct": all_q and all_status,
        "n_missing_q_equals": n_missing_q_equals(text),
    }
    if loose is not None:
        out["all_four_correct_optional_q"] = loose["all_four_correct"]
        out["malformed_optional_q"] = loose["malformed"]
    return out


def is_think_collapse(text: str) -> bool:
    """True when the output is only thinking tags and whitespace."""
    raw = text or ""
    if not raw.strip():
        return False
    return not THINK_TAG.sub("", raw).strip()


def selected_content_flags(text: str) -> dict:
    raw = text or ""
    return {
        "has_result_line": bool(RESULT_LINE.search(raw)),
        "has_chat_marker": bool(CHAT_MARK.search(raw)),
        "has_q_equals": bool(re.search(r"\bq\s*=", raw, re.IGNORECASE)),
    }


def parse_b_plan_lines(text: str) -> dict[str, dict]:
    """Last eligibility/price/candidate line per carrier in B's plan format.

    B is asked for `P: eligible | $60 | candidate`, not A's `q=` lines.
    """
    found = {c: {"status": None, "price": None, "flag": None} for c in CARRIERS}
    for match in B_PLAN_LINE.finditer(text or ""):
        carrier = match.group("carrier").upper()
        flag = match.group("flag")
        price = match.group("price")
        found[carrier] = {
            "status": match.group("status").lower(),
            "price": int(price) if price is not None else None,
            "flag": flag.lower() if flag else None,
        }
    return found


def score_b_plan(
    plan_text: str,
    *,
    gold_status: dict[str, str],
    prices: dict[str, int],
    budget: int,
    pred_carrier: Optional[str] = None,
) -> dict:
    """Score B's plan on eligibility, prices, candidate flags, and Choice.

    Does not score capacities: B was never asked to print q=.
    """
    pred = parse_b_plan_lines(plan_text)
    gold_flag = {
        c: (
            "candidate"
            if gold_status[c] == "eligible" and int(prices[c]) <= int(budget)
            else "dropped"
        )
        for c in CARRIERS
    }
    n_status = sum(pred[c]["status"] == gold_status[c] for c in CARRIERS)
    n_price = sum(pred[c]["price"] == int(prices[c]) for c in CARRIERS)
    n_flag = sum(pred[c]["flag"] == gold_flag[c] for c in CARRIERS)
    n_parsed = sum(pred[c]["status"] is not None for c in CARRIERS)
    named = cheapest_named_in_plan(plan_text)
    own_candidates = [
        c
        for c in CARRIERS
        if pred[c]["flag"] == "candidate" and pred[c]["price"] is not None
    ]
    own_cheapest = (
        min(own_candidates, key=lambda c: (int(pred[c]["price"]), c)) if own_candidates else None
    )
    gold_choice = min(
        (c for c in CARRIERS if gold_flag[c] == "candidate"),
        key=lambda c: (int(prices[c]), c),
    )
    extracted = plan_extraction(plan_text, pred_carrier)
    return {
        "pred_status": {c: pred[c]["status"] for c in CARRIERS},
        "pred_price": {c: pred[c]["price"] for c in CARRIERS},
        "pred_flag": {c: pred[c]["flag"] for c in CARRIERS},
        "gold_status": dict(gold_status),
        "gold_flag": gold_flag,
        "n_status_correct": n_status,
        "n_price_correct": n_price,
        "n_flag_correct": n_flag,
        "n_carriers_parsed": n_parsed,
        "all_status_correct": n_status == 4,
        "all_price_correct": n_price == 4,
        "all_flag_correct": n_flag == 4,
        "all_four_status": n_status == 4,
        "choice": named,
        "choice_matches_gold": named == gold_choice if named else False,
        "own_cheapest": own_cheapest,
        "choice_matches_own_cheapest": named == own_cheapest if named and own_cheapest else False,
        "think_collapse": is_think_collapse(plan_text),
        **extracted,
    }
