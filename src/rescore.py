"""Re-score saved evaluations with the shared extractor. Does not rewrite files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.parse import flags_from_kind, label_both
from src.stage_select import _load_rows


def rescore_row(row: dict) -> dict:
    current, legacy = label_both(
        row.get("raw_text") or "",
        gold_letter=row["gold_letter"],
        gold_locker=int(row["gold_locker"]),
        donor_letter=row["donor_letter"],
        donor_locker=int(row["donor_locker"]),
    )
    cur_flags = flags_from_kind(current.kind)
    leg_flags = flags_from_kind(legacy.kind)
    return {
        "example_id": row["example_id"],
        "condition": row["condition"],
        "kind_original": row.get("kind"),
        "correct_original": row.get("correct"),
        "pred_letter_original": row.get("pred_letter"),
        "pred_locker_original": row.get("pred_locker"),
        "kind": current.kind,
        "correct": cur_flags["correct"],
        "pred_letter": current.letter,
        "pred_locker": current.locker,
        "kind_legacy": legacy.kind,
        "correct_legacy": leg_flags["correct"],
        "pred_letter_legacy": legacy.letter,
        "pred_locker_legacy": legacy.locker,
        "changed": bool(row.get("correct")) != cur_flags["correct"],
    }


def summarize_rescore(rows: list[dict]) -> dict:
    scored = [rescore_row(row) for row in rows]
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for item in scored:
        by_cond[item["condition"]].append(item)
    accuracy = {}
    for cond, items in by_cond.items():
        n = len(items)
        n_orig = sum(bool(item["correct_original"]) for item in items)
        n_new = sum(bool(item["correct"]) for item in items)
        n_leg = sum(bool(item["correct_legacy"]) for item in items)
        accuracy[cond] = {
            "n": n,
            "correct_original": n_orig,
            "accuracy_original": n_orig / n if n else None,
            "correct": n_new,
            "accuracy": n_new / n if n else None,
            "correct_legacy": n_leg,
            "accuracy_legacy": n_leg / n if n else None,
            "n_changed": sum(bool(item["changed"]) for item in items),
        }
    flips = [item for item in scored if item["changed"]]
    return {
        "n_rows": len(scored),
        "accuracy": accuracy,
        "flips": flips,
        "note": (
            "Original scores stay in the jsonl. Primary accuracy uses the shared "
            "extractor. Legacy is the first-match rule from earlier stages."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-score evaluations.jsonl without rewriting it.")
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    rows = _load_rows(args.jsonl)
    report = summarize_rescore(rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
