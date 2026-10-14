"""Joint delivery-planning task: A has operations, B has money.

The evaluator computes eligibility and the gold carrier. Those labels may be
stored for scoring and may be formatted for the oracle_to_b diagnostic. They
must not be copied into Agent A's ordinary message or any other non-oracle
channel.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR

CARRIERS = ("P", "Q", "R")
PLAN_DEV_SEED = 2040
N_PLAN_FAMILIES = 12
N_EASY = 6
N_COMBINE = 6
PLAN_FRESH_SEED = 2042
N_PLAN_FRESH_FAMILIES = 6
N_FRESH_EASY = 3
N_FRESH_COMBINE = 3
QUESTION = (
    "Choose the lowest-cost carrier that satisfies all operational requirements "
    "and does not exceed the budget."
)

# Difficulty is frozen here. Do not retune examples until a method is chosen.
DIFFICULTIES = ("easy", "combine")


@dataclass(frozen=True)
class EvalResult:
    eligible: tuple[str, ...]
    ineligible_reasons: dict[str, str]
    gold_carrier: str
    trap_carrier: str
    affordable_eligible: tuple[str, ...]


@dataclass(frozen=True)
class PlanExample:
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
    other_carrier: str


def request_line(boxes: int, refrigerated: bool) -> str:
    kind = "refrigerated boxes" if refrigerated else "boxes"
    return f"Shipment request: {boxes} {kind}."


def _letter(index: int) -> str:
    return "ABC"[index]


def is_eligible(request: dict[str, Any], facts: dict[str, Any], *, difficulty: str) -> tuple[bool, str]:
    boxes = int(request["boxes"])
    refrigerated = bool(request["refrigerated"])
    if difficulty == "easy":
        if facts.get("cleared"):
            return True, "cleared for this load"
        if refrigerated and not facts.get("refrigeration"):
            return False, "not cleared for refrigerated cargo"
        if int(facts["capacity"]) < boxes:
            return False, "not cleared: vehicle capacity is too small"
        if not facts.get("driver"):
            return False, "not cleared: driver unavailable"
        return False, "not cleared for this load"
    if int(facts["capacity"]) < boxes:
        return False, "vehicle capacity is too small"
    if refrigerated and not facts.get("refrigeration"):
        return False, "no refrigeration unit"
    if not facts.get("driver"):
        return False, "driver unavailable"
    return True, "meets capacity, refrigeration, and driver rules"


def evaluate(
    request: dict[str, Any],
    a_facts: dict[str, Any],
    b_facts: dict[str, Any],
    *,
    difficulty: str,
) -> EvalResult:
    reasons: dict[str, str] = {}
    eligible: list[str] = []
    for carrier in CARRIERS:
        ok, reason = is_eligible(request, a_facts[carrier], difficulty=difficulty)
        if ok:
            eligible.append(carrier)
        else:
            reasons[carrier] = reason
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
    )


def gold_eligibility_statuses(ex: PlanExample) -> dict[str, str]:
    request = {"boxes": ex.boxes, "refrigerated": ex.refrigerated}
    out: dict[str, str] = {}
    for carrier in CARRIERS:
        ok, _reason = is_eligible(request, ex.a_facts[carrier], difficulty=ex.difficulty)
        out[carrier] = "eligible" if ok else "ineligible"
    return out


def oracle_intermediate_message(ex: PlanExample) -> str:
    """Correct eligibility for B. Diagnostic only. Does not name the gold carrier."""
    request = {"boxes": ex.boxes, "refrigerated": ex.refrigerated}
    lines = []
    for carrier in CARRIERS:
        ok, reason = is_eligible(request, ex.a_facts[carrier], difficulty=ex.difficulty)
        label = "eligible" if ok else "ineligible"
        lines.append(f"{carrier}: {label} | {reason}")
    text = "\n".join(lines)
    if f"book {ex.gold_carrier}" in text.lower():
        raise RuntimeError("Oracle message named the booking decision")
    return text


def _carrier_phrase(carriers: list[str]) -> str:
    if len(carriers) == 1:
        return f"Carrier {carriers[0]}"
    if len(carriers) == 2:
        return f"Carriers {carriers[0]} and {carriers[1]}"
    return f"Carriers {carriers[0]}, {carriers[1]}, and {carriers[2]}"


def oracle_sentence_message(ex: PlanExample) -> str:
    """Correct eligibility as ordinary sentences. Does not name the gold carrier."""
    statuses = gold_eligibility_statuses(ex)
    eligible = [c for c in CARRIERS if statuses[c] == "eligible"]
    ineligible = [c for c in CARRIERS if statuses[c] == "ineligible"]
    parts = []
    if ineligible:
        parts.append(f"{_carrier_phrase(ineligible)} cannot be used.")
    if eligible:
        verb = "meets the operational requirements" if len(eligible) == 1 else "meet the operational requirements"
        parts.append(f"{_carrier_phrase(eligible)} {verb}.")
    text = " ".join(parts)
    if not text:
        raise RuntimeError("Sentence oracle produced an empty message")
    if f"book {ex.gold_carrier}" in text.lower():
        raise RuntimeError("Sentence oracle named the booking decision")
    return text


def _render_a_text(request_ln: str, a_facts: dict[str, Any], *, difficulty: str, order: list[str]) -> str:
    lines = [request_ln, ""]
    for carrier in order:
        facts = a_facts[carrier]
        if difficulty == "easy":
            if facts.get("cleared"):
                lines.append(f"Carrier {carrier} is cleared to carry this refrigerated load.")
            else:
                lines.append(f"Carrier {carrier} is not cleared for refrigerated cargo.")
        else:
            fridge = "refrigeration unit working" if facts.get("refrigeration") else "no refrigeration unit"
            driver = "driver available" if facts.get("driver") else "driver unavailable"
            lines.append(
                f"Carrier {carrier}: vehicle capacity {int(facts['capacity'])} boxes; {fridge}; {driver}."
            )
    return "\n".join(lines)


def _render_b_text(request_ln: str, b_facts: dict[str, Any], order: list[str]) -> str:
    lines = [request_ln, ""]
    for carrier in order:
        lines.append(f"Carrier {carrier} costs ${int(b_facts['prices'][carrier])}.")
    lines.append(f"Budget: ${int(b_facts['budget'])}.")
    return "\n".join(lines)


def _eligible_ops(rng: random.Random, boxes: int) -> dict[str, Any]:
    return {
        "capacity": rng.randint(boxes, boxes + 6),
        "refrigeration": True,
        "driver": True,
        "cleared": True,
    }


def _ineligible_ops(rng: random.Random, boxes: int, *, difficulty: str) -> dict[str, Any]:
    facts = {
        "capacity": rng.randint(boxes, boxes + 6),
        "refrigeration": True,
        "driver": True,
        "cleared": False,
    }
    if difficulty == "easy":
        facts["refrigeration"] = False
        return facts
    mode = rng.choice(["fridge", "capacity", "driver"])
    if mode == "fridge":
        facts["refrigeration"] = False
    elif mode == "capacity":
        facts["capacity"] = rng.randint(1, max(1, boxes - 1))
    else:
        facts["driver"] = False
    return facts


def _make_eligible(facts: dict[str, Any], boxes: int) -> dict[str, Any]:
    out = dict(facts)
    out["cleared"] = True
    out["refrigeration"] = True
    out["driver"] = True
    out["capacity"] = max(int(out["capacity"]), boxes)
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
    other_carrier: str,
) -> PlanExample:
    request = {"boxes": boxes, "refrigerated": True}
    req = request_line(boxes, True)
    ev = evaluate(request, a_facts, b_facts, difficulty=difficulty)
    gold_letter = _letter(order.index(ev.gold_carrier))
    trap_letter = _letter(order.index(ev.trap_carrier))
    return PlanExample(
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
        gold_letter=gold_letter,
        trap_carrier=ev.trap_carrier,
        trap_letter=trap_letter,
        other_carrier=other_carrier,
    )


def _sample_base(rng: random.Random, split: str, fam_index: int, difficulty: str) -> tuple[PlanExample, PlanExample, PlanExample]:
    trap, gold, other = rng.sample(list(CARRIERS), 3)
    boxes = rng.choice([6, 7, 8, 9, 10])
    trap_price = rng.choice([30, 35, 40, 45])
    gold_price = trap_price + rng.choice([20, 25, 30])
    other_price = gold_price + rng.choice([15, 20, 25])
    budget = rng.choice([gold_price, gold_price + 5, other_price - 5, other_price])
    budget = max(int(budget), int(gold_price))
    other_new = gold_price - rng.choice([5, 10, 15])
    if other_new <= trap_price:
        other_new = trap_price + rng.choice([5, 10])
    if other_new >= gold_price or other_new > budget:
        raise ValueError("B-flip price is not a unique cheaper gold")

    a_facts = {
        trap: _ineligible_ops(rng, boxes, difficulty=difficulty),
        gold: _eligible_ops(rng, boxes),
        other: _eligible_ops(rng, boxes),
    }
    b_facts = {"prices": {trap: trap_price, gold: gold_price, other: other_price}, "budget": budget}
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
        other_carrier=other,
    )
    if base.gold_carrier != gold or base.trap_carrier != trap:
        raise ValueError("Base gold/trap mismatch")
    if len(evaluate({"boxes": boxes, "refrigerated": True}, a_facts, b_facts, difficulty=difficulty).eligible) < 2:
        raise ValueError("A must have at least two eligible carriers")

    a_flip_facts = {c: dict(vals) for c, vals in a_facts.items()}
    a_flip_facts[trap] = _make_eligible(a_flip_facts[trap], boxes)
    a_flip = _build_example(
        split=split,
        family_id=family_id,
        variant="a_flip",
        difficulty=difficulty,
        boxes=boxes,
        a_facts=a_flip_facts,
        b_facts=b_facts,
        order=order,
        other_carrier=other,
    )
    if a_flip.gold_carrier != trap:
        raise ValueError("A-flip did not change the gold carrier")
    if a_flip.b_text != base.b_text:
        raise ValueError("A-flip changed B's document")

    b_flip_prices = dict(b_facts["prices"])
    b_flip_prices[other] = other_new
    b_flip_facts = {"prices": b_flip_prices, "budget": budget}
    b_flip = _build_example(
        split=split,
        family_id=family_id,
        variant="b_flip",
        difficulty=difficulty,
        boxes=boxes,
        a_facts=a_facts,
        b_facts=b_flip_facts,
        order=order,
        other_carrier=other,
    )
    if b_flip.gold_carrier != other:
        raise ValueError("B-flip did not change the gold carrier")
    if b_flip.a_text != base.a_text:
        raise ValueError("B-flip changed A's document")
    return base, a_flip, b_flip


def assert_plan_invariants(ex: PlanExample) -> None:
    request = {"boxes": ex.boxes, "refrigerated": ex.refrigerated}
    ev = evaluate(request, ex.a_facts, ex.b_facts, difficulty=ex.difficulty)
    if ev.gold_carrier != ex.gold_carrier:
        raise AssertionError("Stored gold does not match the evaluator")
    if ev.trap_carrier != ex.trap_carrier:
        raise AssertionError("Stored trap does not match the evaluator")
    if ex.gold_carrier not in ex.choices or ex.trap_carrier not in ex.choices:
        raise AssertionError("Gold or trap missing from choices")
    if ex.choices_text.splitlines()[list("ABC").index(ex.gold_letter)] != f"{ex.gold_letter}) {ex.gold_carrier}":
        raise AssertionError("Gold letter does not match the choice list")
    if len(ev.eligible) < 2 and ex.variant != "a_flip":
        raise AssertionError("Base/B-flip must leave A with at least two eligible carriers")
    if ex.variant != "a_flip" and ev.trap_carrier in ev.eligible:
        raise AssertionError("B's cheapest-under-budget carrier must be ineligible on base/B-flip")
    if ex.variant == "a_flip" and ev.gold_carrier != ev.trap_carrier:
        raise AssertionError("After A-flip the cheapest under budget should be gold")
    a_rest = ex.a_text.replace(ex.request_line, "")
    b_rest = ex.b_text.replace(ex.request_line, "")
    if "$" in a_rest or "Budget" in a_rest:
        raise AssertionError("A's document contains prices")
    for banned in ("capacity", "driver", "cleared", "refrigeration unit"):
        if banned in b_rest.lower():
            raise AssertionError(f"B's document contains operational term {banned}")
    if ex.request_line not in ex.a_text or ex.request_line not in ex.b_text:
        raise AssertionError("Shared request line missing")
    oracle = oracle_intermediate_message(ex)
    if "$" in oracle or "Budget" in oracle:
        raise AssertionError("Oracle eligibility message contains prices")
    if ex.gold_carrier not in oracle and ex.variant == "base":
        # gold is eligible, so its name appears in the eligible list; that is not the booking
        pass


def generate_split(
    split: str,
    n_families: int,
    seed: int,
    *,
    n_easy: int | None = None,
    n_combine: int | None = None,
) -> list[PlanExample]:
    if n_easy is None and n_combine is None:
        if n_families != N_PLAN_FAMILIES:
            raise ValueError(f"plan_dev freezes {N_PLAN_FAMILIES} families; got {n_families}")
        n_easy, n_combine = N_EASY, N_COMBINE
    if n_easy is None or n_combine is None:
        raise ValueError("Pass both n_easy and n_combine")
    if n_easy + n_combine != n_families:
        raise ValueError("n_easy + n_combine must equal n_families")
    if n_families % 3 != 0:
        raise ValueError("n_families must be divisible by 3 for letter balance")
    difficulties = ["easy"] * n_easy + ["combine"] * n_combine
    for attempt in range(4000):
        rng = random.Random(seed + attempt * 10007)
        rng.shuffle(difficulties)
        families: list[tuple[PlanExample, PlanExample, PlanExample]] = []
        try:
            for i, difficulty in enumerate(difficulties):
                families.append(_sample_base(rng, split, i, difficulty))
        except ValueError:
            continue
        bases = [trip[0] for trip in families]
        letters = [ex.gold_letter for ex in bases]
        counts = {L: letters.count(L) for L in "ABC"}
        if not all(v == n_families // 3 for v in counts.values()):
            continue
        out: list[PlanExample] = []
        for base, a_flip, b_flip in families:
            for ex in (base, a_flip, b_flip):
                assert_plan_invariants(ex)
            out.extend((base, a_flip, b_flip))
        got_easy = sum(ex.difficulty == "easy" for ex in bases)
        got_combine = sum(ex.difficulty == "combine" for ex in bases)
        if got_easy != n_easy or got_combine != n_combine:
            continue
        return out
    raise RuntimeError(f"Could not generate a balanced {split} split")


def example_to_dict(ex: PlanExample) -> dict[str, Any]:
    return asdict(ex)


def example_from_dict(row: dict[str, Any]) -> PlanExample:
    return PlanExample(**row)


def write_jsonl(path: Path, examples: list[PlanExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def load_plan_jsonl(path: Path) -> list[PlanExample]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(example_from_dict(json.loads(line)))
    return rows


def mismatch_source_id(ex: PlanExample, examples: list[PlanExample]) -> str:
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
    parser = argparse.ArgumentParser(description="Generate plan_dev or plan_fresh. Does not write a frozen test split.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Write plan_fresh.jsonl only. Does not rewrite plan_dev.")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        fresh_path = DATA_DIR / "plan_fresh.jsonl"
        if not args.force and fresh_path.exists():
            print(f"{fresh_path} already exists. Pass --force --fresh to regenerate.")
            return 0
        examples = generate_split(
            "plan_fresh",
            N_PLAN_FRESH_FAMILIES,
            PLAN_FRESH_SEED,
            n_easy=N_FRESH_EASY,
            n_combine=N_FRESH_COMBINE,
        )
        write_jsonl(fresh_path, examples)
        print(
            f"Wrote {fresh_path} ({len(examples)} rows, {N_PLAN_FRESH_FAMILIES} families). "
            "plan_dev and plan_test were not rewritten."
        )
        return 0
    dev_path = DATA_DIR / "plan_dev.jsonl"
    if not args.force and dev_path.exists():
        print(f"{dev_path} already exists. Pass --force to regenerate.")
        return 0
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    write_jsonl(dev_path, examples)
    print(f"Wrote {dev_path} ({len(examples)} rows, {N_PLAN_FAMILIES} families). No plan_test file was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
