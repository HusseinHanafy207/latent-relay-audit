"""Twelve-carrier families for the query-adapter pilot.

Three disjoint splits with their own seeds; plan4_timing12 (the nine diagnostic
cases) is never reused or rewritten.
- train: base sheets only (adapter fitting)
- dev: base + A-change pairs (checkpoint selection and the advance decision)
- eval: base + A-change pairs (untouched until a frozen candidate exists)

An A-change pair changes one carrier's operational facts in A's document so the
gold carrier changes, while B's document stays identical. Same rules as the
four-carrier A-change: make the trap eligible (it becomes gold) or make the gold
ineligible (the alternate becomes gold).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

from src.paths import DATA_DIR
from src.plan4_task import LEVELS, _make_eligible, _make_ineligible
from src.plan4_timing_task import (
    N_CARRIERS_12,
    TimingExample,
    _build_timing_example,
    _sample_timing12_family,
    assert_timing12_invariants,
    example_from_dict,
    example_to_dict,
    gold_eligibility_statuses_n,
)

QADAPT_SPLITS = {
    "train": {"name": "plan4_qadapt_train", "seed": 2090, "n_per_level": 32, "a_flip": False},
    "dev": {"name": "plan4_qadapt_dev", "seed": 2091, "n_per_level": 4, "a_flip": True},
    "eval": {"name": "plan4_qadapt_eval", "seed": 2092, "n_per_level": 8, "a_flip": True},
}
A_CHANGES = ("make_eligible", "make_ineligible")
MAX_FAMILY_TRIES = 500


def split_path(split: str) -> Path:
    return DATA_DIR / f"{QADAPT_SPLITS[split]['name']}.jsonl"


def build_a_flip(base: TimingExample, a_change: str) -> TimingExample:
    facts = {c: dict(v) for c, v in base.a_facts.items()}
    if a_change == "make_eligible":
        facts[base.trap_carrier] = _make_eligible(facts[base.trap_carrier], base.boxes, base.difficulty)
        want = base.trap_carrier
    elif a_change == "make_ineligible":
        facts[base.gold_carrier] = _make_ineligible(facts[base.gold_carrier], base.boxes, base.difficulty)
        want = base.alt_carrier
    else:
        raise ValueError(f"Unknown a_change {a_change}")
    ex = _build_timing_example(
        split=base.split,
        family_id=base.family_id,
        difficulty=base.difficulty,
        boxes=base.boxes,
        a_facts=facts,
        b_facts=base.b_facts,
        order=list(base.carriers),
        alt_carrier=base.alt_carrier,
        spare_carrier=base.spare_carrier,
    )
    ex = replace(ex, example_id=f"{base.family_id}-a_flip", variant="a_flip", a_change=a_change)
    if ex.gold_carrier != want:
        raise ValueError("A-change did not produce the intended gold carrier")
    return ex


def assert_a_flip_invariants(base: TimingExample, flip: TimingExample) -> None:
    if flip.variant != "a_flip" or flip.family_id != base.family_id:
        raise AssertionError("A-change variant is not paired with its base")
    if flip.b_text != base.b_text or flip.b_facts != base.b_facts:
        raise AssertionError("A-change altered B's document")
    changed = [c for c in base.carriers if base.a_facts[c] != flip.a_facts[c]]
    if len(changed) != 1:
        raise AssertionError(f"A-change must alter exactly one carrier, altered {changed}")
    gb, gf = gold_eligibility_statuses_n(base), gold_eligibility_statuses_n(flip)
    if sum(gb[c] != gf[c] for c in base.carriers) != 1:
        raise AssertionError("A-change must flip exactly one eligibility status")
    if flip.gold_carrier == base.gold_carrier:
        raise AssertionError("A-change did not change the gold carrier")
    if "$" in flip.a_text.replace(flip.request_line, "") or "Budget" in flip.a_text:
        raise AssertionError("A's document contains prices")


def changed_carrier(base: TimingExample, flip: TimingExample) -> str:
    return next(c for c in base.carriers if base.a_facts[c] != flip.a_facts[c])


def _family(rng: random.Random, split_name: str, index: int, difficulty: str, a_change: str | None):
    for _ in range(MAX_FAMILY_TRIES):
        try:
            base = _sample_timing12_family(rng, split_name, index, difficulty)
            if a_change is None:
                return [base]
            base = replace(base, a_change=a_change)
            flip = build_a_flip(base, a_change)
            assert_a_flip_invariants(base, flip)
            return [base, flip]
        except ValueError:
            continue
    raise RuntimeError(f"Could not sample family {index} of {split_name}")


def generate_split(split: str) -> list[TimingExample]:
    spec = QADAPT_SPLITS[split]
    rng = random.Random(spec["seed"])
    levels = [lvl for lvl in LEVELS for _ in range(spec["n_per_level"])]
    rng.shuffle(levels)
    seen = {lvl: 0 for lvl in LEVELS}
    out = []
    for i, lvl in enumerate(levels):
        a_change = A_CHANGES[seen[lvl] % 2] if spec["a_flip"] else None
        seen[lvl] += 1
        out.extend(_family(rng, spec["name"], i, lvl, a_change))
    return out


def assert_split_invariants(split: str, examples: list[TimingExample]) -> None:
    spec = QADAPT_SPLITS[split]
    bases = [ex for ex in examples if ex.variant == "base"]
    flips = {ex.family_id: ex for ex in examples if ex.variant == "a_flip"}
    if len(bases) != 3 * spec["n_per_level"]:
        raise AssertionError(f"{split}: expected {3 * spec['n_per_level']} base sheets, got {len(bases)}")
    for lvl in LEVELS:
        if sum(ex.difficulty == lvl for ex in bases) != spec["n_per_level"]:
            raise AssertionError(f"{split}: level {lvl} is unbalanced")
    for ex in bases:
        if ex.split != spec["name"] or ex.n_carriers != N_CARRIERS_12:
            raise AssertionError(f"{split}: wrong split name or width for {ex.example_id}")
        assert_timing12_invariants(ex)
    if spec["a_flip"]:
        if set(flips) != {ex.family_id for ex in bases}:
            raise AssertionError(f"{split}: every base sheet needs exactly one A-change pair")
        for ex in bases:
            assert_a_flip_invariants(ex, flips[ex.family_id])
        for lvl in LEVELS:
            kinds = [flips[ex.family_id].a_change for ex in bases if ex.difficulty == lvl]
            if kinds.count("make_eligible") != kinds.count("make_ineligible"):
                raise AssertionError(f"{split}: A-change kinds unbalanced at {lvl}")
    elif flips:
        raise AssertionError(f"{split}: train must contain base sheets only")


def sheet_signature(ex: TimingExample) -> str:
    body = json.dumps(
        {"carriers": list(ex.carriers), "boxes": ex.boxes, "a": ex.a_facts, "b": ex.b_facts, "d": ex.difficulty},
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def assert_disjoint(splits: dict[str, list[TimingExample]], others: list[TimingExample]) -> None:
    owner: dict[str, str] = {}
    for name, examples in splits.items():
        for ex in examples:
            if ex.variant != "base":
                continue
            sig = sheet_signature(ex)
            if sig in owner:
                raise AssertionError(f"Sheet in {name} duplicates one in {owner[sig]}")
            owner[sig] = name
    for ex in others:
        if sheet_signature(ex) in owner:
            raise AssertionError(f"A pilot sheet duplicates diagnostic sheet {ex.example_id}")


def mismatch_partner(examples: list[TimingExample]) -> dict[str, str]:
    """Base example -> base example of the next family at the same level (cyclic, file order)."""
    bases = [ex for ex in examples if ex.variant == "base"]
    out = {}
    for lvl in LEVELS:
        group = [ex for ex in bases if ex.difficulty == lvl]
        for i, ex in enumerate(group):
            out[ex.example_id] = group[(i + 1) % len(group)].example_id
    return out


def write_split(path: Path, examples: list[TimingExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for ex in examples:
            handle.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def load_split(path: Path) -> list[TimingExample]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(example_from_dict(json.loads(line)))
    return rows


def ensure_split(split: str, path: Path | None = None) -> list[TimingExample]:
    """Load the frozen split; write it only if the file does not exist."""
    dest = split_path(split) if path is None else path
    if dest.is_file():
        examples = load_split(dest)
    else:
        examples = generate_split(split)
        write_split(dest, examples)
    assert_split_invariants(split, examples)
    return examples


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
