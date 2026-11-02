"""Audit frozen A transcripts, then rescore the support comparison on the consistent subset.

No GPU. Does not regenerate A or rewrite evaluations.jsonl / summary.json.
plan4_hold stays out.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import a_state_key, assert_plan4_invariants, load_plan4_jsonl
from src.sender_audit import audit_example
from src.stage_plan0 import _load_messages
from src.stage_plan4_compress import _family_delta
from src.stage_plan4_support import (
    CONDS,
    N_BOOT,
    BOOT_SEED,
    PAIRED,
    RAND_CONDS,
    _family_delta_vs_mean,
    _summarize,
)
from src.stage_select import _load_rows

PRESERVED = (
    "results_stage_plan4_support/evaluations.jsonl",
    "results_stage_plan4_support/summary.json",
    "results_stage_plan4_support/a_artifacts.jsonl",
    "results_stage_plan4_iface",
    "results_stage_plan4_hold",
)


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    candidates = [
        REPO_ROOT / "results_stage_plan4_support",
        RESULTS_DIR / "stage_plan4_support" / "run",
        Path("/kaggle/working/results_stage_plan4_support"),
    ]
    src = next((p for p in candidates if (p / "a_artifacts.jsonl").is_file()), candidates[0])
    return data, src


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _counts(rows: list[dict], cond: str) -> dict:
    items = [row for row in rows if row.get("condition") == cond]
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "parse_ok": sum(bool(row.get("parse_ok")) for row in items),
        "truncated": sum(bool(row.get("truncated") or row.get("plan_truncated")) for row in items),
        "plan_all_status_correct": sum(bool(row.get("plan_all_status_correct")) for row in items),
        "correct": sum(bool(row.get("correct")) for row in items),
        "correct_with_wrong_eligibility": sum(bool(row.get("correct_with_wrong_eligibility")) for row in items),
        "plan_ineligible_marked_eligible": sum(int(row.get("plan_ineligible_marked_eligible") or 0) for row in items),
        "plan_eligible_marked_ineligible": sum(int(row.get("plan_eligible_marked_ineligible") or 0) for row in items),
    }


def _family_elig(rows: list[dict]) -> dict:
    out = {
        name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED, key="plan_all_status_correct")
        for name, a, b in PAIRED
    }
    out["support_vs_rand_mean"] = _family_delta_vs_mean(
        rows, "compact_support", RAND_CONDS, n_boot=N_BOOT, seed=BOOT_SEED
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen A transcripts on plan4_dev. No GPU.")
    parser.add_argument("--split", default="plan4_dev")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Sender audit uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, src = _paths()
    examples = load_plan4_jsonl(data)
    for ex in examples:
        assert_plan4_invariants(ex)
    by_id = {ex.example_id: ex for ex in examples}
    artifacts = _load_messages(src / "a_artifacts.jsonl")
    if len(artifacts) != 60:
        print(f"Expected 60 unique frozen A sheets, got {len(artifacts)} in {src}.")
        return 2

    audits = []
    for rec in artifacts.values():
        key = rec.get("a_state_key") or rec["example_id"]
        ex = by_id.get(key) or by_id[rec["example_id"]]
        written = rec.get("written") or ""
        if not written:
            print(f"Missing written transcript for {key}.")
            return 2
        pack = audit_example(ex, written)
        pack["a_state_key"] = key
        pack["ordinary_scorer_all_four"] = bool((rec.get("written_score") or {}).get("all_four_correct"))
        pack["ordinary_scorer_all_status"] = bool((rec.get("written_score") or {}).get("all_status_correct"))
        audits.append(pack)

    verdicts = Counter(item["verdict"] for item in audits)
    consistent_keys = {item["a_state_key"] for item in audits if item["verdict"] == "consistent"}
    issue_codes = Counter(issue["code"] for item in audits for issue in item.get("issues") or [])

    eval_path = src / "evaluations.jsonl"
    rows = _load_rows(eval_path) if eval_path.is_file() else []
    subset_rows = []
    for row in rows:
        ex = by_id.get(row["example_id"])
        key = row.get("a_state_key") or (a_state_key(ex) if ex is not None else None)
        if key in consistent_keys:
            subset_rows.append(row)

    construction = None
    cpath = src / "construction.json"
    if cpath.is_file():
        construction = json.loads(cpath.read_text(encoding="utf-8"))

    full_receiver = {cond: _counts(rows, cond) for cond in CONDS} if rows else {}
    subset_receiver = {cond: _counts(subset_rows, cond) for cond in CONDS} if subset_rows else {}
    subset_summary = _summarize(subset_rows, construction) if subset_rows else {"n_rows": 0}

    out = {
        "n_unique_a_sheets": len(audits),
        "verdicts": dict(verdicts),
        "n_consistent_sheets": verdicts.get("consistent", 0),
        "n_incorrect_sheets": verdicts.get("incorrect", 0),
        "n_ambiguous_sheets": verdicts.get("ambiguous", 0),
        "issue_codes": dict(issue_codes),
        "n_eval_rows_full": len(rows),
        "n_eval_rows_consistent_subset": len(subset_rows),
        "n_examples_consistent_subset": len({row["example_id"] for row in subset_rows}),
        "n_families_consistent_subset": len({row["family_id"] for row in subset_rows}),
        "used_b_outcome": any(item.get("used_b_outcome") for item in audits),
        "preserved_original_outputs": True,
        "regenerated_a": False,
        "note": (
            "The ordinary scorer checks RESULT capacities and eligibility labels. "
            "This audit also checks copied facts, arithmetic, capacity comparisons, "
            "eligibility reasoning, and RESULT explanations against the source sheet. "
            "Original A transcripts and the 720-row comparison were not rewritten."
        ),
        "full_set_receiver": full_receiver,
        "consistent_subset_receiver": subset_receiver,
        "consistent_subset_family_eligibility": _family_elig(subset_rows) if subset_rows else {},
        "full_set_family_eligibility": _family_elig(rows) if rows else {},
        "consistent_subset_summary": {
            "n_rows": subset_summary.get("n_rows"),
            "n_examples": subset_summary.get("n_examples"),
            "n_families": subset_summary.get("n_families"),
            "receiver": subset_summary.get("receiver"),
            "family_clustered_eligibility": subset_summary.get("family_clustered_eligibility"),
            "family_clustered_delta": subset_summary.get("family_clustered_delta"),
            "decision": subset_summary.get("decision"),
        },
        "incorrect_sheets": [
            {
                "a_state_key": item["a_state_key"],
                "difficulty": item.get("difficulty"),
                "ordinary_scorer_all_four": item.get("ordinary_scorer_all_four"),
                "ordinary_scorer_all_status": item.get("ordinary_scorer_all_status"),
                "codes": sorted({issue["code"] for issue in item.get("issues") or [] if issue["severity"] == "incorrect"}),
                "details": [issue["detail"] for issue in item.get("issues") or [] if issue["severity"] == "incorrect"][:8],
            }
            for item in audits
            if item["verdict"] == "incorrect"
        ],
        "ambiguous_sheets": [
            {
                "a_state_key": item["a_state_key"],
                "codes": sorted({issue["code"] for issue in item.get("issues") or []}),
                "details": [issue["detail"] for issue in item.get("issues") or []][:8],
            }
            for item in audits
            if item["verdict"] == "ambiguous"
        ],
        "do_not_resume": list(PRESERVED),
    }
    audit_jsonl = src / "sender_audit.jsonl"
    with audit_jsonl.open("w", encoding="utf-8") as handle:
        for item in audits:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    _write_json(src / "sender_audit_summary.json", out)
    print(
        json.dumps(
            {
                "verdicts": out["verdicts"],
                "n_examples_consistent_subset": out["n_examples_consistent_subset"],
                "issue_codes": out["issue_codes"],
                "full_labels": {
                    cond: (full_receiver.get(cond) or {}).get("plan_all_status_correct")
                    for cond in ("compact", "compact_support", "text_result", "kv")
                },
                "subset_labels": {
                    cond: (subset_receiver.get(cond) or {}).get("plan_all_status_correct")
                    for cond in ("compact", "compact_support", "text_result", "kv")
                },
                "wrote": [str(src / "sender_audit.jsonl"), str(src / "sender_audit_summary.json")],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
