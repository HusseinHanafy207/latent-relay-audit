"""Synthetic package–locker inventories with matched mismatch variants."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.paths import DATA_DIR

DEV_SEED = 2026
TEST_SEED = 2027
VALIDATE_SEED = 2028
N_DEV = 20
N_TEST = 80
N_VALIDATE = 20
N_ASSIGNMENTS = 20
N_CHOICES = 4

ONSETS = [
    "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
    "br", "kl", "tr", "vr", "sk", "pl",
]
NUCLEI = ["a", "e", "i", "o", "u", "ae", "eo"]
CODAS = ["", "n", "r", "l", "x", "v", "z"]

FILLER_PREFIXES = [
    "Dock", "Bay", "Aisle", "Ramp", "Yard", "Gate", "Bin", "Rack",
]


@dataclass(frozen=True)
class InventoryExample:
    example_id: str
    split: str
    packages: list[str]
    true_lockers: list[int]
    alt_lockers: list[int]
    queried_package: str
    queried_index: int
    true_locker: int
    alt_locker: int
    choices: list[int]
    gold_letter: str
    donor_letter: str
    supporting_line_index: int
    true_inventory_text: str
    alt_inventory_text: str
    filler_text: str
    question: str
    choices_text: str


def _syllable(rng: random.Random) -> str:
    return rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)


def _package_name(rng: random.Random, used: set[str]) -> str:
    for _ in range(1000):
        name = (_syllable(rng) + _syllable(rng)).capitalize()
        if name.lower() not in used and 4 <= len(name) <= 12:
            used.add(name.lower())
            return name
    raise RuntimeError("Could not sample a unique package name")


def _render_inventory(packages: list[str], lockers: list[int]) -> str:
    lines = [f"{i + 1}. Package {pkg} is in locker {loc}." for i, (pkg, loc) in enumerate(zip(packages, lockers))]
    return "\n".join(lines)


def _render_filler(rng: random.Random, n_lines: int) -> str:
    """Same line count and similar shape; no package or locker vocabulary."""
    lines = []
    used = set()
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


def _letter(index: int) -> str:
    return "ABCD"[index]


def _build_one(rng: random.Random, split: str, index: int, used_names: set[str]) -> InventoryExample:
    packages = [_package_name(rng, used_names) for _ in range(N_ASSIGNMENTS)]
    lockers = rng.sample(range(1, 81), N_ASSIGNMENTS)
    queried_index = rng.randrange(N_ASSIGNMENTS)
    queried_package = packages[queried_index]
    true_lockers = list(lockers)
    true_locker = true_lockers[queried_index]

    swap_with = (queried_index + 1 + rng.randrange(N_ASSIGNMENTS - 1)) % N_ASSIGNMENTS
    alt_lockers = list(true_lockers)
    alt_lockers[queried_index], alt_lockers[swap_with] = alt_lockers[swap_with], alt_lockers[queried_index]
    alt_locker = alt_lockers[queried_index]
    if alt_locker == true_locker:
        raise RuntimeError("Matched alternative did not change the queried locker")

    other_lockers = [loc for i, loc in enumerate(true_lockers) if i not in {queried_index, swap_with}]
    distractors = rng.sample(other_lockers, 2)
    choices = [true_locker, alt_locker, distractors[0], distractors[1]]
    rng.shuffle(choices)
    gold_letter = _letter(choices.index(true_locker))
    donor_letter = _letter(choices.index(alt_locker))

    choice_lines = [f"{_letter(i)}) locker {loc}" for i, loc in enumerate(choices)]
    question = f"Which locker is package {queried_package} in?"

    return InventoryExample(
        example_id=f"{split}-{index:03d}",
        split=split,
        packages=packages,
        true_lockers=true_lockers,
        alt_lockers=alt_lockers,
        queried_package=queried_package,
        queried_index=queried_index,
        true_locker=true_locker,
        alt_locker=alt_locker,
        choices=choices,
        gold_letter=gold_letter,
        donor_letter=donor_letter,
        supporting_line_index=queried_index,
        true_inventory_text=_render_inventory(packages, true_lockers),
        alt_inventory_text=_render_inventory(packages, alt_lockers),
        filler_text=_render_filler(rng, N_ASSIGNMENTS),
        question=question,
        choices_text="\n".join(choice_lines),
    )


def generate_split(split: str, n: int, seed: int) -> list[InventoryExample]:
    rng = random.Random(seed)
    used: set[str] = set()
    examples = [_build_one(rng, split, i, used) for i in range(n)]
    letters = [ex.gold_letter for ex in examples]
    counts = {L: letters.count(L) for L in "ABCD"}
    if split == "test" and min(counts.values()) < n // 8:
        raise RuntimeError(f"Test label balance looks off: {counts}")
    return examples


def example_to_dict(ex: InventoryExample) -> dict[str, Any]:
    return asdict(ex)


def example_from_dict(row: dict[str, Any]) -> InventoryExample:
    return InventoryExample(**row)


def write_jsonl(path: Path, examples: list[InventoryExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(example_to_dict(ex), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[InventoryExample]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(example_from_dict(json.loads(line)))
    return rows


def assert_no_leak(ex: InventoryExample) -> None:
    filler_l = ex.filler_text.lower()
    if "locker" in filler_l or "package" in filler_l:
        raise AssertionError("Filler mentions package/locker vocabulary")
    for pkg in ex.packages:
        if pkg.lower() in filler_l:
            raise AssertionError(f"Filler contains package name {pkg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate frozen inventory splits.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Write data/validate.jsonl only. Never rewrite the frozen dev/test splits.",
    )
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    validate_path = DATA_DIR / "validate.jsonl"
    if args.validate_only:
        if validate_path.exists() and not args.force:
            print(f"{validate_path} already exists. Pass --force --validate-only to regenerate.")
            return 0
        write_jsonl(validate_path, generate_split("validate", N_VALIDATE, VALIDATE_SEED))
        print(f"Wrote {validate_path} ({N_VALIDATE}). Frozen dev/test were not touched.")
        return 0
    targets = {
        "dev": DATA_DIR / "dev.jsonl",
        "test": DATA_DIR / "test.jsonl",
    }
    if not args.force and all(p.exists() for p in targets.values()):
        print("Inventory files already exist. Pass --force to regenerate.")
        return 0
    write_jsonl(targets["dev"], generate_split("dev", N_DEV, DEV_SEED))
    write_jsonl(targets["test"], generate_split("test", N_TEST, TEST_SEED))
    print(f"Wrote {targets['dev']} ({N_DEV}) and {targets['test']} ({N_TEST})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
