"""Four-carrier delivery task with three computation levels.

A has operational facts and must determine what each carrier can deliver.
B has prices and budget and chooses the cheapest feasible carrier.
The solver is the only source of gold labels. Those labels may be formatted
for the oracle-to-B diagnostic. They must not be copied into ordinary A prompts.

Level 1: capacity, refrigeration, and driver are given directly.
Level 2: effective capacity is min(vehicle capacity, floor(rate * productive hours)).
Level 3: productive hours are max(0, working time - travel - breaks), then Level 2.

Study 0/1 three-carrier files are not rewritten.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR

CARRIERS = ("P", "Q", "R", "S")
LETTERS = "ABCD"
LEVELS = ("L1", "L2", "L3")
PLAN4_DEV_SEED = 2050
N_PLAN4_FAMILIES = 30
N_PER_LEVEL = 10
PLAN4_FRESH_SEED = 2060
N_PLAN4_FRESH_FAMILIES = 30
N_FRESH_PER_LEVEL = 10
PLAN4_HOLD_SEED = 2070
N_PLAN4_HOLD_FAMILIES = 30
N_HOLD_PER_LEVEL = 10
PRIOR_PLAN4_FILES = ("plan4_dev.jsonl", "plan4_fresh.jsonl")
# Six development families for the prompt/cache diagnostic. Two per level.
# Mix make_eligible/make_ineligible and price_up/price_down. Frozen ids.
PLAN4_DIAG_FAMILIES = (
    "plan4_dev-fam-006",  # L1 make_eligible price_up
    "plan4_dev-fam-010",  # L1 make_ineligible price_down
    "plan4_dev-fam-001",  # L2 make_ineligible price_down
    "plan4_dev-fam-004",  # L2 make_eligible price_up
    "plan4_dev-fam-000",  # L3 make_ineligible price_up
    "plan4_dev-fam-008",  # L3 make_eligible price_down
)
# One frozen family per level for the sparse-cache continuation diagnostic.
# Taken from PLAN4_DIAG_FAMILIES, not chosen after seeing span failures.
PLAN4_KVDIAG_FAMILIES = (
    "plan4_dev-fam-006",  # L1
    "plan4_dev-fam-001",  # L2
    "plan4_dev-fam-000",  # L3
)
SINK_INITIAL = 4
QUESTION = (
    "Choose the lowest-cost carrier that satisfies all operational requirements "
    "and does not exceed the budget."
)


@dataclass(frozen=True)
class EvalResult:
    eligible: tuple[str, ...]
    ineligible_reasons: dict[str, str]
    gold_carrier: str
    trap_carrier: str
    affordable_eligible: tuple[str, ...]
    intermediates: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Plan4Example:
    example_id: str
    family_id: str
    split: str
    variant: str
    difficulty: str
    boxes: int
    refrigerated: bool
    request_line: str
    a_facts: dict[str, Any]
    b_facts: dict[str, Any]
    a_text: str
    b_text: str
    question: str
    choices: list[str]
    choices_text: str
    gold_carrier: str
    gold_letter: str
    trap_carrier: str
    trap_letter: str
    alt_carrier: str
    spare_carrier: str
    a_change: str
    b_change: str
    intermediates: dict[str, dict[str, Any]]


def request_line(boxes: int, refrigerated: bool) -> str:
    kind = "refrigerated boxes" if refrigerated else "boxes"
    return f"Shipment request: {boxes} {kind}."


def _letter(index: int) -> str:
    return LETTERS[index]


def productive_hours(facts: dict[str, Any], *, difficulty: str) -> int | None:
    if difficulty == "L1":
        return None
    if difficulty == "L2":
        return int(facts["hours"])
    if difficulty == "L3":
        return max(0, int(facts["H"]) - int(facts["tau"]) - int(facts["b"]))
    raise ValueError(f"Unknown difficulty {difficulty}")


def effective_capacity(facts: dict[str, Any], *, difficulty: str) -> int:
    """Exact intermediate quantity A must derive (L1 copies the given capacity)."""
    if difficulty == "L1":
        return int(facts["capacity"])
    hours = productive_hours(facts, difficulty=difficulty)
    assert hours is not None
    return min(int(facts["vehicle_capacity"]), math.floor(int(facts["rate"]) * hours))


def carrier_result(
    request: dict[str, Any],
    facts: dict[str, Any],
    *,
    difficulty: str,
) -> dict[str, Any]:
    boxes = int(request["boxes"])
    refrigerated = bool(request["refrigerated"])
    hours = productive_hours(facts, difficulty=difficulty)
    quantity = effective_capacity(facts, difficulty=difficulty)
    if quantity < boxes:
        ok = False
        reason = f"effective capacity {quantity} boxes is below the {boxes}-box shipment"
    elif refrigerated and not facts.get("refrigeration"):
        ok = False
        reason = "no refrigeration unit"
    elif not facts.get("driver"):
        ok = False
        reason = "driver unavailable"
    else:
        ok = True
        reason = f"effective capacity {quantity} boxes; refrigeration and driver requirements are met"
    return {
        "hours": hours,
        "q": quantity,
        "eligible": ok,
        "reason": reason,
    }


def evaluate(
    request: dict[str, Any],
    a_facts: dict[str, Any],
    b_facts: dict[str, Any],
    *,
    difficulty: str,
) -> EvalResult:
    intermediates: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    reasons: dict[str, str] = {}
    for carrier in CARRIERS:
        rec = carrier_result(request, a_facts[carrier], difficulty=difficulty)
        intermediates[carrier] = rec
        if rec["eligible"]:
            eligible.append(carrier)
        else:
            reasons[carrier] = rec["reason"]
    prices = b_facts["prices"]
    budget = int(b_facts["budget"])
    under_budget = [c for c in CARRIERS if int(prices[c]) <= budget]
    if not under_budget:
        raise ValueError("No carrier is under budget")
    trap = min(under_budget, key=lambda c: (int(prices[c]), c))
    affordable = [c for c in eligible if int(prices[c]) <= budget]
    if not affordable:
        raise ValueError("No eligible carrier is under budget")
    gold = min(affordable, key=lambda c: (int(prices[c]), c))
    return EvalResult(
        eligible=tuple(eligible),
        ineligible_reasons=reasons,
        gold_carrier=gold,
        trap_carrier=trap,
        affordable_eligible=tuple(sorted(affordable, key=lambda c: (int(prices[c]), c))),
        intermediates=intermediates,
    )


def gold_eligibility_statuses(ex: Plan4Example) -> dict[str, str]:
    return {c: ("eligible" if ex.intermediates[c]["eligible"] else "ineligible") for c in CARRIERS}


def gold_quantities(ex: Plan4Example) -> dict[str, int]:
    return {c: int(ex.intermediates[c]["q"]) for c in CARRIERS}


def oracle_intermediate_message(ex: Plan4Example) -> str:
    """Correct q and eligibility for B. Diagnostic only. Does not name the gold booking."""
    lines = []
    for carrier in CARRIERS:
        rec = ex.intermediates[carrier]
        label = "eligible" if rec["eligible"] else "ineligible"
        lines.append(f"{carrier}: q={int(rec['q'])} | {label} | {rec['reason']}")
    text = "\n".join(lines)
    if f"book {ex.gold_carrier}" in text.lower():
        raise RuntimeError("Oracle message named the booking decision")
    return text


def _yes_no(flag: bool, yes: str, no: str) -> str:
    return yes if flag else no


def _render_a_text(request_ln: str, a_facts: dict[str, Any], *, difficulty: str, order: list[str]) -> str:
    lines = [request_ln, ""]
    for carrier in order:
        facts = a_facts[carrier]
        fridge = _yes_no(bool(facts["refrigeration"]), "refrigeration unit working", "no refrigeration unit")
        driver = _yes_no(bool(facts["driver"]), "driver available", "driver unavailable")
        if difficulty == "L1":
            lines.append(
                f"Carrier {carrier}: vehicle capacity {int(facts['capacity'])} boxes; {fridge}; {driver}."
            )
        elif difficulty == "L2":
            lines.append(
                f"Carrier {carrier}: vehicle capacity {int(facts['vehicle_capacity'])} boxes; "
                f"productive hours {int(facts['hours'])}; "
                f"loading rate {int(facts['rate'])} boxes per hour; {fridge}; {driver}."
            )
        else:
            lines.append(
                f"Carrier {carrier}: vehicle capacity {int(facts['vehicle_capacity'])} boxes; "
                f"available working time {int(facts['H'])} hours; "
                f"travel time {int(facts['tau'])} hours; "
                f"break time {int(facts['b'])} hours; "
                f"loading rate {int(facts['rate'])} boxes per hour; {fridge}; {driver}."
            )
    text = "\n".join(lines)
    if "effective capacity" in text.lower():
        raise RuntimeError("A's sheet leaked effective capacity")
    return text


def _render_b_text(request_ln: str, b_facts: dict[str, Any], order: list[str]) -> str:
    lines = [request_ln, ""]
    for carrier in order:
        lines.append(f"Carrier {carrier} costs ${int(b_facts['prices'][carrier])}.")
    lines.append(f"Budget: ${int(b_facts['budget'])}.")
    return "\n".join(lines)


def _ok(facts: dict[str, Any], boxes: int, difficulty: str, *, want: bool) -> bool:
    rec = carrier_result({"boxes": boxes, "refrigerated": True}, facts, difficulty=difficulty)
    return bool(rec["eligible"]) is want


def _eligible_facts(rng: random.Random, boxes: int, difficulty: str) -> dict[str, Any]:
    for _ in range(80):
        if difficulty == "L1":
            facts = {
                "capacity": rng.randint(boxes, boxes + 6),
                "refrigeration": True,
                "driver": True,
            }
        elif difficulty == "L2":
            rate = rng.choice([3, 4, 5])
            hours = rng.choice([3, 4, 5])
            cap = rng.choice([boxes, boxes + 2, boxes + 4, boxes + 8])
            facts = {
                "vehicle_capacity": cap,
                "hours": hours,
                "rate": rate,
                "refrigeration": True,
                "driver": True,
            }
        else:
            tau = rng.choice([0, 1])
            brk = rng.choice([0, 1])
            work = tau + brk + rng.choice([3, 4, 5])
            rate = rng.choice([3, 4, 5])
            cap = rng.choice([boxes, boxes + 2, boxes + 4, boxes + 8])
            facts = {
                "vehicle_capacity": cap,
                "H": work,
                "tau": tau,
                "b": brk,
                "rate": rate,
                "refrigeration": True,
                "driver": True,
            }
        if _ok(facts, boxes, difficulty, want=True):
            return facts
    raise ValueError("Could not sample eligible facts")


def _ineligible_facts(rng: random.Random, boxes: int, difficulty: str, *, mode: str) -> dict[str, Any]:
    for _ in range(80):
        facts = _eligible_facts(rng, boxes, difficulty)
        if mode == "fridge":
            facts["refrigeration"] = False
        elif mode == "driver":
            facts["driver"] = False
        elif difficulty == "L1":
            facts["capacity"] = rng.randint(1, max(1, boxes - 1))
        elif difficulty == "L2":
            facts["hours"] = rng.choice([1, 2])
            facts["rate"] = rng.choice([1, 2])
            facts["vehicle_capacity"] = rng.choice([boxes, boxes + 4, boxes + 8])
        else:
            facts["tau"] = rng.choice([2, 3])
            facts["b"] = rng.choice([1, 2])
            facts["H"] = facts["tau"] + facts["b"] + rng.choice([0, 1])
            facts["rate"] = rng.choice([2, 3, 4])
            facts["vehicle_capacity"] = rng.choice([boxes, boxes + 8, 20])
        if _ok(facts, boxes, difficulty, want=False):
            return facts
    raise ValueError("Could not sample ineligible facts")


def _make_eligible(facts: dict[str, Any], boxes: int, difficulty: str) -> dict[str, Any]:
    out = dict(facts)
    out["refrigeration"] = True
    out["driver"] = True
    if difficulty == "L1":
        out["capacity"] = max(int(out.get("capacity", 1)), boxes)
    elif difficulty == "L2":
        out["vehicle_capacity"] = max(int(out["vehicle_capacity"]), boxes)
        out["rate"] = max(int(out["rate"]), 4)
        out["hours"] = max(int(out["hours"]), math.ceil(boxes / out["rate"]))
    else:
        out["vehicle_capacity"] = max(int(out["vehicle_capacity"]), boxes)
        out["rate"] = max(int(out["rate"]), 4)
        need = math.ceil(boxes / out["rate"])
        out["tau"] = min(int(out["tau"]), 1)
        out["b"] = min(int(out["b"]), 1)
        out["H"] = max(int(out["H"]), out["tau"] + out["b"] + need)
    if not _ok(out, boxes, difficulty, want=True):
        raise ValueError("make_eligible failed")
    return out


def _make_ineligible(facts: dict[str, Any], boxes: int, difficulty: str) -> dict[str, Any]:
    out = dict(facts)
    if difficulty == "L1":
        mode = "capacity" if int(out["capacity"]) >= boxes else "fridge"
        if mode == "fridge":
            out["refrigeration"] = False
        else:
            out["capacity"] = max(1, boxes - 2)
    elif difficulty == "L2":
        out["hours"] = 1
        out["rate"] = 2
        out["vehicle_capacity"] = max(int(out["vehicle_capacity"]), boxes)
    else:
        # Productive hours 0, so q=0 regardless of vehicle capacity.
        out["tau"] = 2
        out["b"] = 1
        out["H"] = 3
        out["rate"] = 4
        out["vehicle_capacity"] = max(int(out["vehicle_capacity"]), 20)
        out["refrigeration"] = True
        out["driver"] = True
    if not _ok(out, boxes, difficulty, want=False):
        raise ValueError("make_ineligible failed")
    return out


def _choices_block(order: list[str]) -> str:
    return "\n".join(f"{_letter(i)}) {carrier}" for i, carrier in enumerate(order))


def _build_example(
    *,
    split: str,
    family_id: str,
    variant: str,
    difficulty: str,
    boxes: int,
    a_facts: dict[str, Any],
    b_facts: dict[str, Any],
    order: list[str],
    alt_carrier: str,
    spare_carrier: str,
    a_change: str,
    b_change: str,
) -> Plan4Example:
    request = {"boxes": boxes, "refrigerated": True}
    req = request_line(boxes, True)
    ev = evaluate(request, a_facts, b_facts, difficulty=difficulty)
    return Plan4Example(
        example_id=f"{family_id}-{variant}",
        family_id=family_id,
        split=split,
        variant=variant,
        difficulty=difficulty,
        boxes=boxes,
        refrigerated=True,
        request_line=req,
        a_facts=a_facts,
        b_facts=b_facts,
        a_text=_render_a_text(req, a_facts, difficulty=difficulty, order=order),
        b_text=_render_b_text(req, b_facts, order=order),
        question=QUESTION,
        choices=list(order),
        choices_text=_choices_block(order),
        gold_carrier=ev.gold_carrier,
        gold_letter=_letter(order.index(ev.gold_carrier)),
        trap_carrier=ev.trap_carrier,
        trap_letter=_letter(order.index(ev.trap_carrier)),
        alt_carrier=alt_carrier,
        spare_carrier=spare_carrier,
        a_change=a_change,
        b_change=b_change,
        intermediates=ev.intermediates,
    )


def _sample_family(
    rng: random.Random,
    split: str,
    fam_index: int,
    difficulty: str,
    a_change: str,
    b_change: str,
) -> tuple[Plan4Example, Plan4Example, Plan4Example]:
    trap, gold, alt, spare = rng.sample(list(CARRIERS), 4)
    boxes = rng.choice([8, 10, 12])
    trap_price = rng.choice([30, 35, 40])
    gold_price = trap_price + rng.choice([20, 25])
    alt_price = gold_price + rng.choice([15, 20])
    budget = alt_price + rng.choice([10, 15, 20])
    spare_price = budget + rng.choice([5, 10])
    trap_mode, spare_mode = rng.sample(["capacity", "fridge", "driver"], 2)
    a_facts = {
        trap: _ineligible_facts(rng, boxes, difficulty, mode=trap_mode),
        gold: _eligible_facts(rng, boxes, difficulty),
        alt: _eligible_facts(rng, boxes, difficulty),
        spare: _ineligible_facts(rng, boxes, difficulty, mode=spare_mode),
    }
    b_facts = {
        "prices": {trap: trap_price, gold: gold_price, alt: alt_price, spare: spare_price},
        "budget": budget,
    }
    order = list(CARRIERS)
    rng.shuffle(order)
    family_id = f"{split}-fam-{fam_index:03d}"
    base = _build_example(
        split=split,
        family_id=family_id,
        variant="base",
        difficulty=difficulty,
        boxes=boxes,
        a_facts=a_facts,
        b_facts=b_facts,
        order=order,
        alt_carrier=alt,
        spare_carrier=spare,
        a_change=a_change,
        b_change=b_change,
    )
    if base.gold_carrier != gold or base.trap_carrier != trap:
        raise ValueError("Base gold/trap mismatch")
    if len(evaluate({"boxes": boxes, "refrigerated": True}, a_facts, b_facts, difficulty=difficulty).eligible) < 2:
        raise ValueError("Need at least two eligible carriers")

    a_flip_facts = {c: dict(vals) for c, vals in a_facts.items()}
    if a_change == "make_eligible":
        a_flip_facts[trap] = _make_eligible(a_flip_facts[trap], boxes, difficulty)
        want_gold = trap
    elif a_change == "make_ineligible":
        a_flip_facts[gold] = _make_ineligible(a_flip_facts[gold], boxes, difficulty)
        want_gold = alt
    else:
        raise ValueError(f"Unknown a_change {a_change}")
    a_flip = _build_example(
        split=split,
        family_id=family_id,
        variant="a_flip",
        difficulty=difficulty,
        boxes=boxes,
        a_facts=a_flip_facts,
        b_facts=b_facts,
        order=order,
        alt_carrier=alt,
        spare_carrier=spare,
        a_change=a_change,
        b_change=b_change,
    )
    if a_flip.gold_carrier != want_gold:
        raise ValueError("A-change did not produce the intended gold")
    if a_flip.b_text != base.b_text:
        raise ValueError("A-change changed B's document")

    b_flip_prices = dict(b_facts["prices"])
    if b_change == "price_down":
        alt_new = gold_price - rng.choice([5, 10])
        if alt_new <= trap_price:
            raise ValueError("Price decrease collided with the trap")
        b_flip_prices[alt] = alt_new
    elif b_change == "price_up":
        gold_new = alt_price + rng.choice([5, 10])
        if gold_new > budget:
            raise ValueError("Price increase exceeded the frozen budget")
        b_flip_prices[gold] = gold_new
    else:
        raise ValueError(f"Unknown b_change {b_change}")
    b_flip = _build_example(
        split=split,
        family_id=family_id,
        variant="b_flip",
        difficulty=difficulty,
        boxes=boxes,
        a_facts=a_facts,
        b_facts={"prices": b_flip_prices, "budget": budget},
        order=order,
        alt_carrier=alt,
        spare_carrier=spare,
        a_change=a_change,
        b_change=b_change,
    )
    if b_flip.gold_carrier != alt:
        raise ValueError("B-change did not make the alternate carrier gold")
    if b_flip.a_text != base.a_text:
        raise ValueError("B-change changed A's document")
    if b_flip.a_facts != base.a_facts:
        raise ValueError("B-change changed A's facts")
    return base, a_flip, b_flip


def assert_plan4_invariants(ex: Plan4Example) -> None:
    request = {"boxes": ex.boxes, "refrigerated": ex.refrigerated}
    ev = evaluate(request, ex.a_facts, ex.b_facts, difficulty=ex.difficulty)
    if ev.gold_carrier != ex.gold_carrier or ev.trap_carrier != ex.trap_carrier:
        raise AssertionError("Stored gold/trap does not match the solver")
    if ev.intermediates != ex.intermediates:
        raise AssertionError("Stored intermediates do not match the solver")
    if set(ex.choices) != set(CARRIERS) or len(ex.choices) != 4:
        raise AssertionError("Choices must be a permutation of P,Q,R,S")
    if ex.choices_text.splitlines()[LETTERS.index(ex.gold_letter)] != f"{ex.gold_letter}) {ex.gold_carrier}":
        raise AssertionError("Gold letter does not match the choice list")
    if len(ev.eligible) < 2 and ex.variant != "a_flip":
        raise AssertionError("Base/B-change must leave at least two eligible carriers")
    if ex.variant != "a_flip" and ev.trap_carrier in ev.eligible:
        raise AssertionError("Trap must be ineligible on base/B-change")
    if ex.variant == "a_flip" and ex.a_change == "make_eligible" and ev.gold_carrier != ev.trap_carrier:
        raise AssertionError("Eligibilizing A-change should make the old trap gold")
    a_rest = ex.a_text.replace(ex.request_line, "")
    b_rest = ex.b_text.replace(ex.request_line, "")
    if "$" in a_rest or "Budget" in a_rest:
        raise AssertionError("A's document contains prices")
    for banned in ("capacity", "driver", "refrigeration", "hours", "loading"):
        if banned in b_rest.lower():
            raise AssertionError(f"B's document contains operational term {banned}")
    if "effective capacity" in ex.a_text.lower() or "q=" in ex.a_text:
        raise AssertionError("A's document contains derived capacity")
    if ex.request_line not in ex.a_text or ex.request_line not in ex.b_text:
        raise AssertionError("Shared request line missing")
    oracle = oracle_intermediate_message(ex)
    if "$" in oracle or "Budget" in oracle:
        raise AssertionError("Oracle message contains prices")


def generate_split(
    split: str,
    n_families: int,
    seed: int,
    *,
    n_per_level: int = N_PER_LEVEL,
) -> list[Plan4Example]:
    if n_per_level * 3 != n_families:
        raise ValueError("n_families must be 3 * n_per_level")
    difficulties = (["L1"] * n_per_level) + (["L2"] * n_per_level) + (["L3"] * n_per_level)
    a_pool = (["make_eligible"] * (n_per_level // 2) + ["make_ineligible"] * (n_per_level - n_per_level // 2))
    b_pool = (["price_down"] * (n_per_level // 2) + ["price_up"] * (n_per_level - n_per_level // 2))
    for attempt in range(8000):
        rng = random.Random(seed + attempt * 10007)
        level_order = list(difficulties)
        rng.shuffle(level_order)
        kinds_by_level = {
            level: list(zip(rng.sample(a_pool, len(a_pool)), rng.sample(b_pool, len(b_pool))))
            for level in LEVELS
        }
        cursor = {level: 0 for level in LEVELS}
        families: list[tuple[Plan4Example, Plan4Example, Plan4Example]] = []
        try:
            for i, difficulty in enumerate(level_order):
                a_change, b_change = kinds_by_level[difficulty][cursor[difficulty]]
                cursor[difficulty] += 1
                families.append(_sample_family(rng, split, i, difficulty, a_change, b_change))
        except ValueError:
            continue
        bases = [trip[0] for trip in families]
        letters = [ex.gold_letter for ex in bases]
        counts = {L: letters.count(L) for L in LETTERS}
        if min(counts.values()) < n_families // 4 - 1 or max(counts.values()) > n_families // 4 + 2:
            continue
        out: list[Plan4Example] = []
        for base, a_flip, b_flip in families:
            for ex in (base, a_flip, b_flip):
                assert_plan4_invariants(ex)
            if a_flip.a_change != base.a_change or b_flip.b_change != base.b_change:
                raise AssertionError("Family change kinds drifted")
            out.extend((base, a_flip, b_flip))
        got = {level: sum(ex.difficulty == level for ex in bases) for level in LEVELS}
        if any(got[level] != n_per_level for level in LEVELS):
            continue
        return out
    raise RuntimeError(f"Could not generate a balanced {split} split")


def example_to_dict(ex: Plan4Example) -> dict[str, Any]:
    return asdict(ex)


def example_from_dict(row: dict[str, Any]) -> Plan4Example:
    return Plan4Example(**row)


def write_jsonl(path: Path, examples: list[Plan4Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def load_plan4_jsonl(path: Path) -> list[Plan4Example]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(example_from_dict(json.loads(line)))
    return rows


def a_state_key(ex: Plan4Example) -> str:
    """Base and B-change share A's information; A-change does not."""
    return f"{ex.family_id}-base" if ex.variant != "a_flip" else ex.example_id


def subset_plan4_diag(examples: list[Plan4Example]) -> list[Plan4Example]:
    wanted = set(PLAN4_DIAG_FAMILIES)
    selected = [ex for ex in examples if ex.family_id in wanted]
    found = {ex.family_id for ex in selected}
    missing = [fid for fid in PLAN4_DIAG_FAMILIES if fid not in found]
    if missing:
        raise ValueError(f"Diagnostic families missing from split: {missing}")
    return selected


def subset_plan4_kvdiag(examples: list[Plan4Example]) -> list[Plan4Example]:
    wanted = set(PLAN4_KVDIAG_FAMILIES)
    selected = [ex for ex in examples if ex.family_id in wanted]
    found = {ex.family_id for ex in selected}
    missing = [fid for fid in PLAN4_KVDIAG_FAMILIES if fid not in found]
    if missing:
        raise ValueError(f"KV diagnostic families missing from split: {missing}")
    order = {fid: i for i, fid in enumerate(PLAN4_KVDIAG_FAMILIES)}
    selected.sort(key=lambda ex: (order[ex.family_id], ex.variant))
    return selected


def sheet_fingerprint(ex: Plan4Example) -> tuple[str, ...]:
    return (
        ex.difficulty,
        str(ex.boxes),
        json.dumps(ex.a_facts, sort_keys=True),
        json.dumps(ex.b_facts, sort_keys=True),
        ex.a_text,
        ex.b_text,
    )


def generate_disjoint_split(
    split: str,
    *,
    n_families: int,
    seed: int,
    n_per_level: int,
    banned: list[Plan4Example],
) -> list[Plan4Example]:
    """Same shape as plan4_dev, no overlapping sheets with `banned`."""
    banned_fp = {sheet_fingerprint(ex) for ex in banned}
    for bump in range(80):
        examples = generate_split(
            split,
            n_families,
            seed + bump * 100003,
            n_per_level=n_per_level,
        )
        if any(sheet_fingerprint(ex) in banned_fp for ex in examples):
            continue
        if any(ex.split != split for ex in examples):
            continue
        return examples
    raise RuntimeError(f"Could not generate a disjoint {split} split")


def generate_fresh_disjoint(dev_examples: list[Plan4Example]) -> list[Plan4Example]:
    """Fresh families, same shape as plan4_dev, no overlapping sheets."""
    return generate_disjoint_split(
        "plan4_fresh",
        n_families=N_PLAN4_FRESH_FAMILIES,
        seed=PLAN4_FRESH_SEED,
        n_per_level=N_FRESH_PER_LEVEL,
        banned=dev_examples,
    )


def generate_hold_disjoint(prior_examples: list[Plan4Example]) -> list[Plan4Example]:
    """Held-out families, disjoint from plan4_dev and plan4_fresh."""
    return generate_disjoint_split(
        "plan4_hold",
        n_families=N_PLAN4_HOLD_FAMILIES,
        seed=PLAN4_HOLD_SEED,
        n_per_level=N_HOLD_PER_LEVEL,
        banned=prior_examples,
    )


def prior_plan4_examples() -> list[Plan4Example]:
    rows: list[Plan4Example] = []
    for name in PRIOR_PLAN4_FILES:
        path = DATA_DIR / name
        if path.is_file():
            rows.extend(load_plan4_jsonl(path))
    return rows


def overlap_with_prior(
    examples: list[Plan4Example], prior: list[Plan4Example] | None = None
) -> dict[str, int]:
    prior = prior_plan4_examples() if prior is None else prior
    banned_fp = {sheet_fingerprint(ex) for ex in prior}
    banned_ids = {ex.example_id for ex in prior}
    banned_fams = {ex.family_id for ex in prior}
    return {
        "n_prior": len(prior),
        "n_sheet_overlap": sum(sheet_fingerprint(ex) in banned_fp for ex in examples),
        "n_example_id_overlap": sum(ex.example_id in banned_ids for ex in examples),
        "n_family_id_overlap": sum(ex.family_id in banned_fams for ex in examples),
    }


def mismatch_source_id(ex: Plan4Example, examples: list[Plan4Example]) -> str:
    families = []
    seen = set()
    for item in examples:
        if item.family_id not in seen:
            seen.add(item.family_id)
            families.append(item.family_id)
    idx = families.index(ex.family_id)
    other = families[(idx + 1) % len(families)]
    return f"{other}-base"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate plan4_dev, plan4_fresh, or plan4_hold. Does not rewrite plan_dev or a frozen test split."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Write plan4_fresh.jsonl only. Does not rewrite plan4_dev.",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Write plan4_hold.jsonl only. Does not rewrite plan4_dev or plan4_fresh.",
    )
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and args.hold:
        print("Pass only one of --fresh or --hold.")
        return 2
    if args.hold:
        hold_path = DATA_DIR / "plan4_hold.jsonl"
        if not args.force and hold_path.exists():
            print(f"{hold_path} already exists. Pass --force --hold to regenerate.")
            return 0
        prior = prior_plan4_examples()
        if len(prior) < 180:
            print("Need plan4_dev.jsonl and plan4_fresh.jsonl before generating plan4_hold.")
            return 2
        examples = generate_hold_disjoint(prior)
        report = overlap_with_prior(examples, prior)
        if any(report[k] for k in ("n_sheet_overlap", "n_example_id_overlap", "n_family_id_overlap")):
            print(f"Generated hold split still overlaps prior examples: {report}")
            return 2
        write_jsonl(hold_path, examples)
        print(
            f"Wrote {hold_path} ({len(examples)} rows, {N_PLAN4_HOLD_FAMILIES} families, "
            f"{N_HOLD_PER_LEVEL} per level). plan4_dev and plan4_fresh were not rewritten."
        )
        return 0
    if args.fresh:
        fresh_path = DATA_DIR / "plan4_fresh.jsonl"
        if not args.force and fresh_path.exists():
            print(f"{fresh_path} already exists. Pass --force --fresh to regenerate.")
            return 0
        dev_path = DATA_DIR / "plan4_dev.jsonl"
        if not dev_path.is_file():
            print(f"Missing {dev_path}. Generate plan4_dev first.")
            return 2
        examples = generate_fresh_disjoint(load_plan4_jsonl(dev_path))
        write_jsonl(fresh_path, examples)
        print(
            f"Wrote {fresh_path} ({len(examples)} rows, {N_PLAN4_FRESH_FAMILIES} families, "
            f"{N_FRESH_PER_LEVEL} per level). plan4_dev was not rewritten."
        )
        return 0
    dev_path = DATA_DIR / "plan4_dev.jsonl"
    if not args.force and dev_path.exists():
        print(f"{dev_path} already exists. Pass --force to regenerate.")
        return 0
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    write_jsonl(dev_path, examples)
    print(
        f"Wrote {dev_path} ({len(examples)} rows, {N_PLAN4_FAMILIES} families, "
        f"{N_PER_LEVEL} per level). plan_dev and plan_test were not rewritten."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
