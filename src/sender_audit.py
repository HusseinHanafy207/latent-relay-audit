"""Audit A's written working and RESULT explanations against the source sheet.

The ordinary sender scorer checks final q and eligibility labels only. This
module checks copied facts, arithmetic, capacity comparisons, eligibility
reasoning, and the short RESULT reasons. Labels use only A's text and the
operational source. They do not use B's success.

Do not correct or regenerate A here. A corrected-message experiment would be
a separate condition.
"""

from __future__ import annotations

import re
from typing import Optional

from src.parse_plan4 import RESULT_LINE, RESULTS_MARK, parse_intermediate_lines, result_text_for_scoring
from src.plan4_task import CARRIERS, Plan4Example, gold_eligibility_statuses, gold_quantities, productive_hours

VERDICTS = ("consistent", "incorrect", "ambiguous")

HEAD = re.compile(r"(?im)^(?:#{1,3}\s*)?\*{0,2}Carrier\s+([PQRS])\*{0,2}\s*:")
HOURS_ARITH = re.compile(
    r"max\(\s*0\s*,\s*(\d+)\s*[-−]\s*(\d+)\s*[-−]\s*(\d+)\s*\)\s*=\s*(-?\d+)",
    re.IGNORECASE,
)
HOURS_GIVEN = re.compile(r"productive hours\s+(\d+)", re.IGNORECASE)
HOURS_SUB = re.compile(
    r"productive hours\s*[:=]\s*(\d+)\s*[-−]\s*(\d+)\s*[-−]\s*(\d+)\s*=\s*(-?\d+)",
    re.IGNORECASE,
)
HOURS_SIMPLE = re.compile(
    r"productive hours\s*[:=]\s*(-?\d+)(?!\s*[-−+])",
    re.IGNORECASE,
)
CAP_ARITH = re.compile(
    r"min\(\s*(\d+)\s*,\s*floor\(\s*(\d+)\s*[*×]\s*(\d+)\s*\)\s*\)"
    r"(?:\s*=\s*min\(\s*\d+\s*,\s*(\d+)\s*\))?\s*=\s*(\d+)",
    re.IGNORECASE,
)
L1_CAP = re.compile(
    r"effective capacity\s*[:=]\s*(\d+)(?:\s*boxes)?(?:\s*\(given vehicle capacity\))?",
    re.IGNORECASE,
)
NUM_CMP = re.compile(
    r"effective capacity\s*\(\s*(?P<q>\d+)\s*\)\s*(?:is\s+)?"
    r"(?P<pred>not at least|at least|less than|below|greater than or equal(?: to)?|≥|>=|>|<)"
    r"\s+(?:the\s+)?(?:shipment size\s*)?\(?\s*(?P<boxes>\d+)\s*\)?",
    re.IGNORECASE,
)
SYM_CMP = re.compile(
    r"effective capacity\s*\(\s*(?P<q>\d+)\s*\)\s*"
    r"(?P<pred>≥|>=|≤|<=|<|>)\s*"
    r"(?:shipment size\s*)?\(?\s*(?P<boxes>\d+)\s*\)?",
    re.IGNORECASE,
)
VEHICLE_CAP = re.compile(r"vehicle capacity\s+(\d+)", re.IGNORECASE)
RATE = re.compile(r"loading rate\s+(\d+)", re.IGNORECASE)
WORK_H = re.compile(r"available working time\s+(\d+)", re.IGNORECASE)
TRAVEL = re.compile(r"travel time\s+(\d+)", re.IGNORECASE)
BREAK = re.compile(r"break time\s+(\d+)", re.IGNORECASE)


def _issue(code: str, detail: str, *, severity: str = "incorrect") -> dict:
    return {"code": code, "detail": detail, "severity": severity}


def _split_working_and_results(text: str) -> tuple[str, str]:
    raw = text or ""
    mark = RESULTS_MARK.search(raw)
    if mark:
        return raw[: mark.start()], raw[mark.end() :]
    return raw, raw


def _carrier_blocks(working: str) -> dict[str, str]:
    matches = list(HEAD.finditer(working or ""))
    blocks = {c: "" for c in CARRIERS}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(working)
        blocks[match.group(1).upper()] = working[match.start() : end]
    return blocks


def _capacity_ok(q: int, boxes: int) -> bool:
    return int(q) >= int(boxes)


def _pred_holds(pred: str, q: int, boxes: int) -> Optional[bool]:
    token = (pred or "").strip().lower().replace("≥", ">=").replace("≤", "<=")
    if token in {"less than", "below", "<", "not at least"}:
        return int(q) < int(boxes)
    if token in {"at least", "greater than or equal", "greater than or equal to", ">="}:
        return int(q) >= int(boxes)
    if token == ">":
        return int(q) > int(boxes)
    if token in {"<="}:
        return int(q) <= int(boxes)
    return None


def _claims(text: str) -> list[str]:
    """Atomic eligibility claims. Order matters: negative phrases first."""
    raw = text or ""
    lower = raw.lower()
    found: list[str] = []
    if re.search(r"driver(?:\s+is)?\s+unavailable", lower):
        found.append("driver_fail")
    elif re.search(r"driver(?:\s+is)?\s+available", lower):
        found.append("driver_ok")
    if re.search(
        r"no refrigeration|refrigeration is not present|refrigeration is not|"
        r"not present when required|does not have refrigeration",
        lower,
    ):
        found.append("fridge_fail")
    elif re.search(r"refrigeration unit working|refrigeration is present", lower):
        found.append("fridge_ok")
    if re.search(
        r"effective capacity.{0,80}(not at least|less than|insufficient|below)|"
        r"(not at least|less than|insufficient).{0,40}shipment",
        lower,
    ):
        found.append("cap_fail")
    elif re.search(r"effective capacity.{0,80}(\bat least\b|≥|>=)|at least shipment", lower):
        found.append("cap_ok")
    return found


def _claim_true(kind: str, *, q: int, boxes: int, fridge: bool, driver: bool, refrigerated: bool) -> bool:
    if kind == "cap_ok":
        return _capacity_ok(q, boxes)
    if kind == "cap_fail":
        return not _capacity_ok(q, boxes)
    if kind == "fridge_ok":
        return bool(fridge)
    if kind == "fridge_fail":
        return bool(refrigerated) and (not bool(fridge))
    if kind == "driver_ok":
        return bool(driver)
    if kind == "driver_fail":
        return not bool(driver)
    return False


def _heading_facts(block: str, ex: Plan4Example, carrier: str) -> list[dict]:
    issues = []
    facts = ex.a_facts[carrier]
    first = (block.splitlines() or [""])[0]
    cap = VEHICLE_CAP.search(first)
    want_cap = int(facts["capacity"] if ex.difficulty == "L1" else facts["vehicle_capacity"])
    if cap and int(cap.group(1)) != want_cap:
        issues.append(
            _issue(
                "wrong_copied_fact",
                f"{carrier} heading vehicle capacity {cap.group(1)} != source {want_cap}",
            )
        )
    if ex.difficulty != "L1":
        hours = HOURS_GIVEN.search(first)
        rate = RATE.search(first)
        if ex.difficulty == "L2":
            want_h = int(facts["hours"])
            if hours and int(hours.group(1)) != want_h:
                issues.append(
                    _issue("wrong_copied_fact", f"{carrier} heading hours {hours.group(1)} != source {want_h}")
                )
        if rate and int(rate.group(1)) != int(facts["rate"]):
            issues.append(
                _issue("wrong_copied_fact", f"{carrier} heading rate {rate.group(1)} != source {facts['rate']}")
            )
    if ex.difficulty == "L3":
        for rx, key, label in (
            (WORK_H, "H", "working time"),
            (TRAVEL, "tau", "travel"),
            (BREAK, "b", "break"),
        ):
            match = rx.search(first)
            if match and int(match.group(1)) != int(facts[key]):
                issues.append(
                    _issue("wrong_copied_fact", f"{carrier} heading {label} {match.group(1)} != source {facts[key]}")
                )
    if re.search(r"no refrigeration unit", first, re.I) and facts.get("refrigeration"):
        issues.append(_issue("wrong_copied_fact", f"{carrier} heading denies refrigeration"))
    if re.search(r"refrigeration unit working", first, re.I) and not facts.get("refrigeration"):
        issues.append(_issue("wrong_copied_fact", f"{carrier} heading asserts refrigeration"))
    if re.search(r"driver unavailable", first, re.I) and facts.get("driver"):
        issues.append(_issue("wrong_copied_fact", f"{carrier} heading denies driver"))
    if re.search(r"driver available", first, re.I) and not facts.get("driver"):
        issues.append(_issue("wrong_copied_fact", f"{carrier} heading asserts driver"))
    return issues


def _check_arithmetic(block: str, ex: Plan4Example, carrier: str) -> list[dict]:
    issues = []
    facts = ex.a_facts[carrier]
    gold_h = productive_hours(facts, difficulty=ex.difficulty)
    gold_q = int(ex.intermediates[carrier]["q"])
    hours_m = HOURS_ARITH.search(block) or HOURS_SUB.search(block)
    if hours_m:
        h_s, tau_s, b_s, out_s = (int(hours_m.group(i)) for i in range(1, 5))
        if ex.difficulty != "L3":
            issues.append(_issue("wrong_hours_arithmetic", f"{carrier} computed H-tau-b but difficulty is {ex.difficulty}"))
        else:
            if (h_s, tau_s, b_s) != (int(facts["H"]), int(facts["tau"]), int(facts["b"])):
                issues.append(
                    _issue(
                        "wrong_hours_arithmetic",
                        f"{carrier} hours inputs {(h_s, tau_s, b_s)} != source "
                        f"{(facts['H'], facts['tau'], facts['b'])}",
                    )
                )
            want = max(0, int(facts["H"]) - int(facts["tau"]) - int(facts["b"]))
            if out_s != want:
                issues.append(_issue("wrong_hours_arithmetic", f"{carrier} hours result {out_s} != {want}"))
    elif gold_h is not None:
        given = HOURS_SIMPLE.search(block)
        if given and int(given.group(1)) != int(gold_h):
            issues.append(_issue("wrong_hours_arithmetic", f"{carrier} stated hours {given.group(1)} != {gold_h}"))
    cap_m = CAP_ARITH.search(block)
    if cap_m:
        cap_s, a_s, b_s, floor_s, q_s = (
            int(cap_m.group(1)),
            int(cap_m.group(2)),
            int(cap_m.group(3)),
            cap_m.group(4),
            int(cap_m.group(5)),
        )
        want_cap = int(facts["capacity"] if ex.difficulty == "L1" else facts["vehicle_capacity"])
        if cap_s != want_cap:
            issues.append(_issue("wrong_capacity_arithmetic", f"{carrier} min() capacity {cap_s} != source {want_cap}"))
        product = a_s * b_s
        if floor_s is not None and int(floor_s) != product:
            issues.append(_issue("wrong_capacity_arithmetic", f"{carrier} floor product {floor_s} != {product}"))
        want_q = min(cap_s, product)
        if q_s != want_q:
            issues.append(_issue("wrong_capacity_arithmetic", f"{carrier} min() result {q_s} != {want_q}"))
        if q_s != gold_q:
            issues.append(_issue("wrong_capacity_arithmetic", f"{carrier} arithmetic q {q_s} != gold {gold_q}"))
    elif ex.difficulty == "L1":
        l1 = L1_CAP.search(block)
        if l1 and int(l1.group(1)) != gold_q:
            issues.append(_issue("wrong_capacity_arithmetic", f"{carrier} copied capacity {l1.group(1)} != {gold_q}"))
    return issues


def _check_comparisons(block: str, ex: Plan4Example, carrier: str) -> list[dict]:
    issues = []
    gold_q = int(ex.intermediates[carrier]["q"])
    boxes = int(ex.boxes)
    for match in list(NUM_CMP.finditer(block)) + list(SYM_CMP.finditer(block)):
        after = block[match.end() : match.end() + 24]
        if re.match(r"\s*\?\s*No\b", after, re.IGNORECASE):
            continue
        q_s = int(match.group("q"))
        boxes_s = int(match.group("boxes"))
        pred = match.group("pred")
        if q_s != gold_q or boxes_s != boxes:
            issues.append(
                _issue(
                    "false_capacity_comparison",
                    f"{carrier} comparison uses q={q_s}, boxes={boxes_s}; source is q={gold_q}, boxes={boxes}",
                )
            )
        holds = _pred_holds(pred, q_s, boxes_s)
        if holds is False:
            issues.append(
                _issue(
                    "false_capacity_comparison",
                    f"{carrier} states {q_s} {pred} {boxes_s}, which is false",
                )
            )
        elif holds is None:
            issues.append(
                _issue(
                    "unparsed_comparison",
                    f"{carrier} comparison predicate {pred!r} was not recognized",
                    severity="ambiguous",
                )
            )
    return issues


def _check_claims(text: str, ex: Plan4Example, carrier: str, *, where: str) -> list[dict]:
    issues = []
    gold_q = int(ex.intermediates[carrier]["q"])
    boxes = int(ex.boxes)
    fridge = bool(ex.a_facts[carrier]["refrigeration"])
    driver = bool(ex.a_facts[carrier]["driver"])
    for kind in _claims(text):
        if not _claim_true(
            kind,
            q=gold_q,
            boxes=boxes,
            fridge=fridge,
            driver=driver,
            refrigerated=bool(ex.refrigerated),
        ):
            issues.append(
                _issue(
                    "false_result_reason" if where == "result" else "false_eligibility_reason",
                    f"{carrier} {where} claims {kind}, which is false of the source",
                )
            )
    return issues


def audit_sender_text(text: str, ex: Plan4Example) -> dict:
    """Classify one frozen sender transcript against the operational source."""
    working, results = _split_working_and_results(text)
    blocks = _carrier_blocks(working)
    parsed = parse_intermediate_lines(result_text_for_scoring(text))
    gold_q = gold_quantities(ex)
    gold_st = gold_eligibility_statuses(ex)
    issues: list[dict] = []
    per = {}
    for carrier in CARRIERS:
        block = blocks.get(carrier) or ""
        carrier_issues: list[dict] = []
        if not block.strip():
            carrier_issues.append(
                _issue("missing_working_block", f"{carrier} has no working block", severity="ambiguous")
            )
        else:
            carrier_issues.extend(_heading_facts(block, ex, carrier))
            carrier_issues.extend(_check_arithmetic(block, ex, carrier))
            carrier_issues.extend(_check_comparisons(block, ex, carrier))
            carrier_issues.extend(_check_claims(block, ex, carrier, where="working"))
        pred_q = parsed[carrier]["q"]
        pred_st = parsed[carrier]["status"]
        if pred_q is None or pred_st is None:
            carrier_issues.append(_issue("missing_result_line", f"{carrier} result line missing"))
        else:
            if int(pred_q) != int(gold_q[carrier]):
                carrier_issues.append(
                    _issue("wrong_q", f"{carrier} result q={pred_q} != gold {gold_q[carrier]}")
                )
            if pred_st != gold_st[carrier]:
                carrier_issues.append(
                    _issue("wrong_status", f"{carrier} result {pred_st} != gold {gold_st[carrier]}")
                )
            reason = ""
            for raw_line in (results or text).splitlines():
                match = RESULT_LINE.search(raw_line)
                if match and match.group("carrier").upper() == carrier:
                    parts = raw_line.split("|")
                    reason = parts[-1].strip() if len(parts) >= 3 else ""
            if reason:
                carrier_issues.extend(_check_claims(reason, ex, carrier, where="result"))
            else:
                carrier_issues.append(
                    _issue("missing_result_reason", f"{carrier} result reason is empty", severity="ambiguous")
                )
        per[carrier] = {
            "n_issues": len(carrier_issues),
            "issues": carrier_issues,
            "has_working_block": bool(block.strip()),
            "pred_q": pred_q,
            "pred_status": pred_st,
            "gold_q": gold_q[carrier],
            "gold_status": gold_st[carrier],
        }
        issues.extend(carrier_issues)
    severities = {item["severity"] for item in issues}
    if "incorrect" in severities:
        verdict = "incorrect"
    elif "ambiguous" in severities:
        verdict = "ambiguous"
    else:
        verdict = "consistent"
    return {
        "verdict": verdict,
        "n_issues": len(issues),
        "n_incorrect": sum(item["severity"] == "incorrect" for item in issues),
        "n_ambiguous": sum(item["severity"] == "ambiguous" for item in issues),
        "issues": issues,
        "by_carrier": per,
        "used_b_outcome": False,
        "all_q_correct": all(parsed[c]["q"] == gold_q[c] for c in CARRIERS),
        "all_status_correct": all(parsed[c]["status"] == gold_st[c] for c in CARRIERS),
    }


def audit_example(ex: Plan4Example, written: str) -> dict:
    rec = audit_sender_text(written, ex)
    rec.update(
        {
            "example_id": ex.example_id,
            "family_id": ex.family_id,
            "variant": ex.variant,
            "difficulty": ex.difficulty,
        }
    )
    return rec
