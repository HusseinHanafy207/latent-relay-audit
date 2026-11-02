"""12-carrier timing split. Does not rewrite plan4_dev, plan4_fresh, or plan4_hold.

Same operational rules as the frozen four-carrier task. Base variants only.
Accuracy claims stay on the 90-example hold confirmation; this split exists
to measure preparation, receiver prefill, and decode as the sheet grows.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR
from src.plan4_task import (
    LEVELS,
    QUESTION,
    _eligible_facts,
    _ineligible_facts,
    _render_a_text,
    _render_b_text,
    carrier_result,
    request_line,
)

CARRIERS_12 = ("P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "N")
N_CARRIERS_12 = len(CARRIERS_12)
PLAN4_TIMING12_SEED = 2080
N_TIMING_FAMILIES = 9
N_TIMING_PER_LEVEL = 3
N_CARRIERS_4 = 4


@dataclass(frozen=True)
class TimingExample:
    example_id: str
    family_id: str
    split: str
    variant: str
    difficulty: str
    n_carriers: int
    carriers: tuple[str, ...]
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
    letter_of: dict[str, str]
    gold_carrier: str
    gold_letter: str
    trap_carrier: str
    trap_letter: str
    alt_carrier: str
    spare_carrier: str
    a_change: str
    b_change: str
    intermediates: dict[str, dict[str, Any]]


def _letter_map(order: list[str]) -> dict[str, str]:
    if len(order) > len(string.ascii_uppercase):
        raise ValueError("Too many carriers for A–Z choice letters.")
    return {carrier: string.ascii_uppercase[i] for i, carrier in enumerate(order)}


def _choices_block(order: list[str], letter_of: dict[str, str]) -> str:
    return "\n".join(f"{letter_of[carrier]}) {carrier}" for carrier in order)


def evaluate_carriers(
    request: dict[str, Any],
    a_facts: dict[str, Any],
    b_facts: dict[str, Any],
    *,
    difficulty: str,
    carriers: tuple[str, ...],
):
    intermediates: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    reasons: dict[str, str] = {}
    for carrier in carriers:
        rec = carrier_result(request, a_facts[carrier], difficulty=difficulty)
        intermediates[carrier] = rec
        if rec["eligible"]:
            eligible.append(carrier)
        else:
            reasons[carrier] = rec["reason"]
    prices = b_facts["prices"]
    budget = int(b_facts["budget"])
    under_budget = [c for c in carriers if int(prices[c]) <= budget]
    if not under_budget:
        raise ValueError("No carrier is under budget")
    trap = min(under_budget, key=lambda c: (int(prices[c]), c))
    affordable = [c for c in eligible if int(prices[c]) <= budget]
    if not affordable:
        raise ValueError("No eligible carrier is under budget")
    gold = min(affordable, key=lambda c: (int(prices[c]), c))
    return {
        "eligible": tuple(eligible),
        "ineligible_reasons": reasons,
        "gold_carrier": gold,
        "trap_carrier": trap,
        "affordable_eligible": tuple(sorted(affordable, key=lambda c: (int(prices[c]), c))),
        "intermediates": intermediates,
    }


def gold_eligibility_statuses_n(ex: TimingExample) -> dict[str, str]:
    return {
        c: ("eligible" if ex.intermediates[c]["eligible"] else "ineligible") for c in ex.carriers
    }


def gold_quantities_n(ex: TimingExample) -> dict[str, int]:
    return {c: int(ex.intermediates[c]["q"]) for c in ex.carriers}


def _build_timing_example(
    *,
    split: str,
    family_id: str,
    difficulty: str,
    boxes: int,
    a_facts: dict[str, Any],
    b_facts: dict[str, Any],
    order: list[str],
    alt_carrier: str,
    spare_carrier: str,
) -> TimingExample:
    carriers = tuple(order)
    request = {"boxes": boxes, "refrigerated": True}
    req = request_line(boxes, True)
    ev = evaluate_carriers(
        request, a_facts, b_facts, difficulty=difficulty, carriers=carriers
    )
    letter_of = _letter_map(order)
    return TimingExample(
        example_id=f"{family_id}-base",
        family_id=family_id,
        split=split,
        variant="base",
        difficulty=difficulty,
        n_carriers=len(order),
        carriers=carriers,
        boxes=boxes,
        refrigerated=True,
        request_line=req,
        a_facts=a_facts,
        b_facts=b_facts,
        a_text=_render_a_text(req, a_facts, difficulty=difficulty, order=order),
        b_text=_render_b_text(req, b_facts, order=order),
        question=QUESTION,
        choices=list(order),
        choices_text=_choices_block(order, letter_of),
        letter_of=letter_of,
        gold_carrier=ev["gold_carrier"],
        gold_letter=letter_of[ev["gold_carrier"]],
        trap_carrier=ev["trap_carrier"],
        trap_letter=letter_of[ev["trap_carrier"]],
        alt_carrier=alt_carrier,
        spare_carrier=spare_carrier,
        a_change="none",
        b_change="none",
        intermediates=ev["intermediates"],
    )


def assert_timing12_invariants(ex: TimingExample) -> None:
    if ex.n_carriers != N_CARRIERS_12 or tuple(ex.carriers) != tuple(ex.choices):
        raise AssertionError("Timing-12 example does not have 12 carriers")
    if set(ex.carriers) != set(CARRIERS_12):
        raise AssertionError("Timing-12 carriers drifted from CARRIERS_12")
    ev = evaluate_carriers(
        {"boxes": ex.boxes, "refrigerated": True},
        ex.a_facts,
        ex.b_facts,
        difficulty=ex.difficulty,
        carriers=ex.carriers,
    )
    if ev["gold_carrier"] != ex.gold_carrier or ev["trap_carrier"] != ex.trap_carrier:
        raise AssertionError("Stored gold/trap does not match the solver")
    if len(ev["affordable_eligible"]) < 2:
        raise AssertionError("Need at least two affordable eligible carriers")
    if ev["trap_carrier"] in ev["eligible"]:
        raise AssertionError("Trap must be ineligible")
    a_rest = ex.a_text.replace(ex.request_line, "")
    b_rest = ex.b_text.replace(ex.request_line, "")
    if "$" in a_rest or "Budget" in a_rest:
        raise AssertionError("A's document contains prices")
    for banned in ("capacity", "driver", "refrigeration", "hours", "loading"):
        if banned in b_rest.lower():
            raise AssertionError(f"B's document contains operational term {banned}")


def _sample_timing12_family(
    rng: random.Random,
    split: str,
    fam_index: int,
    difficulty: str,
) -> TimingExample:
    order = list(CARRIERS_12)
    rng.shuffle(order)
    trap, gold, alt, spare, *rest = rng.sample(order, 12)
    boxes = rng.choice([8, 10, 12])
    trap_price = rng.choice([30, 35, 40])
    gold_price = trap_price + rng.choice([20, 25])
    alt_price = gold_price + rng.choice([15, 20])
    budget = alt_price + rng.choice([10, 15, 20])
    spare_price = budget + rng.choice([5, 10])
    modes = ["capacity", "fridge", "driver"]
    a_facts = {
        trap: _ineligible_facts(rng, boxes, difficulty, mode=rng.choice(modes)),
        gold: _eligible_facts(rng, boxes, difficulty),
        alt: _eligible_facts(rng, boxes, difficulty),
        spare: _ineligible_facts(rng, boxes, difficulty, mode=rng.choice(modes)),
    }
    b_facts_prices = {
        trap: trap_price,
        gold: gold_price,
        alt: alt_price,
        spare: spare_price,
    }
    for i, carrier in enumerate(rest):
        if i < 4:
            a_facts[carrier] = _ineligible_facts(rng, boxes, difficulty, mode=rng.choice(modes))
            b_facts_prices[carrier] = trap_price + rng.choice([5, 10, 15])
        else:
            a_facts[carrier] = _eligible_facts(rng, boxes, difficulty)
            b_facts_prices[carrier] = budget + rng.choice([5, 10, 15, 20])
    b_facts = {"prices": b_facts_prices, "budget": budget}
    ex = _build_timing_example(
        split=split,
        family_id=f"{split}-fam-{fam_index:03d}",
        difficulty=difficulty,
        boxes=boxes,
        a_facts=a_facts,
        b_facts=b_facts,
        order=order,
        alt_carrier=alt,
        spare_carrier=spare,
    )
    if ex.gold_carrier != gold or ex.trap_carrier != trap:
        raise ValueError("Gold/trap mismatch after sampling")
    assert_timing12_invariants(ex)
    return ex


def generate_timing12(
    n_families: int = N_TIMING_FAMILIES,
    seed: int = PLAN4_TIMING12_SEED,
    *,
    n_per_level: int = N_TIMING_PER_LEVEL,
) -> list[TimingExample]:
    if n_per_level * 3 != n_families:
        raise ValueError("n_families must be 3 * n_per_level")
    difficulties = (["L1"] * n_per_level) + (["L2"] * n_per_level) + (["L3"] * n_per_level)
    for attempt in range(8000):
        rng = random.Random(seed + attempt * 10007)
        level_order = list(difficulties)
        rng.shuffle(level_order)
        try:
            out = [
                _sample_timing12_family(rng, "plan4_timing12", i, difficulty)
                for i, difficulty in enumerate(level_order)
            ]
        except ValueError:
            continue
        got = {level: sum(ex.difficulty == level for ex in out) for level in LEVELS}
        if any(got[level] != n_per_level for level in LEVELS):
            continue
        return out
    raise RuntimeError("Could not generate a balanced 12-carrier timing split")


def example_to_dict(ex: TimingExample) -> dict[str, Any]:
    rec = asdict(ex)
    rec["carriers"] = list(ex.carriers)
    return rec


def example_from_dict(row: dict[str, Any]) -> TimingExample:
    row = dict(row)
    row["carriers"] = tuple(row["carriers"])
    return TimingExample(**row)


def write_timing12_jsonl(path: Path, examples: list[TimingExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def load_timing12_jsonl(path: Path) -> list[TimingExample]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(example_from_dict(json.loads(line)))
    return rows


def ensure_timing12_jsonl(path: Path | None = None) -> list[TimingExample]:
    dest = DATA_DIR / "plan4_timing12.jsonl" if path is None else path
    if dest.is_file():
        examples = load_timing12_jsonl(dest)
        if len(examples) == N_TIMING_FAMILIES:
            for ex in examples:
                assert_timing12_invariants(ex)
            return examples
    examples = generate_timing12()
    write_timing12_jsonl(dest, examples)
    return examples


def select_timing4_examples(hold_examples, *, n_per_level: int = N_TIMING_PER_LEVEL):
    """First n_per_level base families per level, in file order. Not chosen after seeing times."""
    picked: dict[str, list] = {level: [] for level in LEVELS}
    out = []
    for ex in hold_examples:
        if ex.variant != "base":
            continue
        bucket = picked[ex.difficulty]
        if len(bucket) >= n_per_level:
            continue
        bucket.append(ex.family_id)
        out.append(ex)
    if any(len(picked[level]) != n_per_level for level in LEVELS):
        raise RuntimeError("plan4_hold does not have three unused base families per level.")
    return out
