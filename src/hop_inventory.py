"""Two-record hop inventories: package → shipment → locker.

Does not rewrite data/dev.jsonl or data/test.jsonl. hop_test.jsonl is written
so IDs stay frozen, but the development runner must not load it.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.inventory import FILLER_PREFIXES, NUCLEI, ONSETS, _package_name
from src.paths import DATA_DIR

HOP_DEV_SEED = 2030
HOP_TEST_SEED = 2031
N_HOP_DEV = 10
N_HOP_TEST = 40
N_HOP_PACKAGES = 10
N_HOP_LINES = 20
N_CHOICES = 4


@dataclass(frozen=True)
class HopExample:
    example_id: str
    split: str
    packages: list[str]
    shipments: list[str]
    lockers: list[int]
    queried_package: str
    queried_shipment: str
    queried_index: int
    true_locker: int
    alt_locker: int
    donor_letter: str
    gold_letter: str
    donor_locker: int
    choices: list[int]
    choices_text: str
    question: str
    true_inventory_text: str
    filler_text: str
    evidence_package_line: int
    evidence_shipment_line: int
    evidence_package_sentence: str
    evidence_shipment_sentence: str
    n_lines: int


def _letter(index: int) -> str:
    return "ABCD"[index]


def _render_filler(rng: random.Random, n_lines: int) -> str:
    lines = []
    used: set[tuple[str, str]] = set()
    for i in range(n_lines):
        while True:
            tag = rng.choice(FILLER_PREFIXES) + str(rng.randint(10, 99))
            crate = rng.choice(ONSETS) + rng.choice(NUCLEI) + str(rng.randint(100, 999))
            key = (tag, crate)
            if key not in used:
                used.add(key)
                break
        lines.append(f"{i + 1}. {tag} holds crate {crate}.")
    return "\n".join(lines)


def _build_one(rng: random.Random, split: str, index: int, used_names: set[str]) -> HopExample:
    packages = [_package_name(rng, used_names) for _ in range(N_HOP_PACKAGES)]
    shipments = [_package_name(rng, used_names) for _ in range(N_HOP_PACKAGES)]
    lockers = rng.sample(range(1, 81), N_HOP_PACKAGES)
    queried_index = rng.randrange(N_HOP_PACKAGES)
    queried_package = packages[queried_index]
    queried_shipment = shipments[queried_index]
    true_locker = lockers[queried_index]

    records: list[tuple] = []
    for i in range(N_HOP_PACKAGES):
        records.append(("package", packages[i], shipments[i]))
        records.append(("shipment", shipments[i], lockers[i]))
    rng.shuffle(records)

    lines = []
    package_line = shipment_line = None
    package_sentence = shipment_sentence = None
    for i, rec in enumerate(records):
        if rec[0] == "package":
            sentence = f"Package {rec[1]} belongs to shipment {rec[2]}."
            if rec[1] == queried_package:
                package_line = i
                package_sentence = sentence
        else:
            sentence = f"Shipment {rec[1]} is stored in locker {rec[2]}."
            if rec[1] == queried_shipment:
                shipment_line = i
                shipment_sentence = sentence
        lines.append(f"{i + 1}. {sentence}")
    if package_line is None or shipment_line is None or not package_sentence or not shipment_sentence:
        raise RuntimeError("Failed to place both evidence records")

    other_lockers = [loc for i, loc in enumerate(lockers) if i != queried_index]
    distractors = rng.sample(other_lockers, 3)
    choices = [true_locker, *distractors]
    rng.shuffle(choices)
    gold_letter = _letter(choices.index(true_locker))
    donor_locker = distractors[0]
    donor_letter = _letter(choices.index(donor_locker))
    choice_lines = [f"{_letter(i)}) locker {loc}" for i, loc in enumerate(choices)]
    question = f"Which locker contains package {queried_package}?"
    inventory = "\n".join(lines)
    filler = _render_filler(rng, N_HOP_LINES)
    return HopExample(
        example_id=f"{split}-{index:03d}",
        split=split,
        packages=packages,
        shipments=shipments,
        lockers=lockers,
        queried_package=queried_package,
        queried_shipment=queried_shipment,
        queried_index=queried_index,
        true_locker=true_locker,
        alt_locker=donor_locker,
        donor_letter=donor_letter,
        gold_letter=gold_letter,
        donor_locker=donor_locker,
        choices=choices,
        choices_text="\n".join(choice_lines),
        question=question,
        true_inventory_text=inventory,
        filler_text=filler,
        evidence_package_line=package_line,
        evidence_shipment_line=shipment_line,
        evidence_package_sentence=package_sentence,
        evidence_shipment_sentence=shipment_sentence,
        n_lines=N_HOP_LINES,
    )


def _letter_ok(letters: list[str], n: int) -> bool:
    counts = {L: letters.count(L) for L in "ABCD"}
    if n == N_HOP_TEST:
        return all(v == n // 4 for v in counts.values())
    if n == N_HOP_DEV:
        return min(counts.values()) >= 2 and max(counts.values()) <= 3
    return True


def generate_split(split: str, n: int, seed: int) -> list[HopExample]:
    for attempt in range(2000):
        rng = random.Random(seed + attempt * 10007)
        used: set[str] = set()
        examples = [_build_one(rng, split, i, used) for i in range(n)]
        letters = [ex.gold_letter for ex in examples]
        if _letter_ok(letters, n):
            return examples
    raise RuntimeError(f"Could not balance gold letters for {split} n={n}")


def example_to_dict(ex: HopExample) -> dict[str, Any]:
    return asdict(ex)


def example_from_dict(row: dict[str, Any]) -> HopExample:
    return HopExample(**row)


def write_jsonl(path: Path, examples: list[HopExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def load_hop_jsonl(path: Path) -> list[HopExample]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(example_from_dict(json.loads(line)))
    return rows


def assert_hop_no_leak(ex: HopExample) -> None:
    filler_l = ex.filler_text.lower()
    for banned in ("locker", "package", "shipment"):
        if banned in filler_l:
            raise AssertionError(f"Filler mentions {banned}")
    for name in [*ex.packages, *ex.shipments]:
        if name.lower() in filler_l:
            raise AssertionError(f"Filler contains name {name}")
    if str(ex.true_locker) in ex.question:
        raise AssertionError("Question contains the gold locker")
    if ex.true_inventory_text.count("\n") + 1 != N_HOP_LINES:
        raise AssertionError("Inventory length is not 20 lines")
    if len(set(ex.packages)) != N_HOP_PACKAGES:
        raise AssertionError("Package names are not unique")
    if len(set(ex.shipments)) != N_HOP_PACKAGES:
        raise AssertionError("Shipment names are not unique")
    if set(ex.packages) & set(ex.shipments):
        raise AssertionError("Package and shipment names overlap")
    if len(set(ex.lockers)) != N_HOP_PACKAGES:
        raise AssertionError("Lockers are not unique")
    if len(set(ex.choices)) != 4:
        raise AssertionError("Choices are not unique")
    if ex.true_locker not in ex.choices:
        raise AssertionError("Gold locker missing from choices")
    lines = ex.true_inventory_text.splitlines()
    pkg = lines[ex.evidence_package_line].split(". ", 1)[1]
    ship = lines[ex.evidence_shipment_line].split(". ", 1)[1]
    if pkg != ex.evidence_package_sentence:
        raise AssertionError("Package evidence index does not match the stored sentence")
    if ship != ex.evidence_shipment_sentence:
        raise AssertionError("Shipment evidence index does not match the stored sentence")


def hop_evidence_absent_from_prompt(prompt: str, ex: HopExample) -> bool:
    if ex.true_inventory_text in prompt:
        return False
    if ex.evidence_package_sentence in prompt:
        return False
    if ex.evidence_shipment_sentence in prompt:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate hop_dev / hop_test. Never rewrites one-record splits.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dev_path = DATA_DIR / "hop_dev.jsonl"
    test_path = DATA_DIR / "hop_test.jsonl"
    if args.dev_only:
        if dev_path.exists() and not args.force:
            print(f"{dev_path} already exists. Pass --force --dev-only to regenerate.")
            return 0
        examples = generate_split("hop_dev", N_HOP_DEV, HOP_DEV_SEED)
        for ex in examples:
            assert_hop_no_leak(ex)
        write_jsonl(dev_path, examples)
        print(f"Wrote {dev_path} ({N_HOP_DEV}). hop_test was not rewritten.")
        return 0
    if not args.force and dev_path.exists() and test_path.exists():
        print("Hop files already exist. Pass --force to regenerate.")
        return 0
    dev = generate_split("hop_dev", N_HOP_DEV, HOP_DEV_SEED)
    test = generate_split("hop_test", N_HOP_TEST, HOP_TEST_SEED)
    for ex in [*dev, *test]:
        assert_hop_no_leak(ex)
    write_jsonl(dev_path, dev)
    write_jsonl(test_path, test)
    print(f"Wrote {dev_path} ({N_HOP_DEV}) and {test_path} ({N_HOP_TEST}). One-record splits were not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
