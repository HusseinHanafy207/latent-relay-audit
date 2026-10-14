"""Recompute diagnostic metrics from saved plan4_diag outputs. No GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.parse_plan4 import (
    n_missing_q_equals,
    plan_extraction,
    score_sender_message,
    written_working_metrics,
)
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import gold_eligibility_statuses, gold_quantities, load_plan4_jsonl
from src.stage_plan4_diag import _summarize
from src.stage_select import _load_rows


def _dest() -> Path:
    kaggle = Path("/kaggle/working/results_stage_plan4_diag")
    if kaggle.is_dir() and (kaggle / "evaluations.jsonl").is_file():
        return kaggle
    local = Path("results_stage_plan4_diag")
    if (local / "evaluations.jsonl").is_file():
        return local
    return RESULTS_DIR / "stage_plan4_diag" / "run"


def rescore(dest: Path | None = None) -> dict:
    dest = dest or _dest()
    rows = _load_rows(dest / "evaluations.jsonl")
    artifacts = _load_rows(dest / "a_artifacts.jsonl")
    compares = _load_rows(dest / "cache_compare.jsonl")
    examples = {ex.example_id: ex for ex in load_plan4_jsonl(DATA_DIR / "plan4_dev.jsonl")}

    for row in rows:
        ex = examples[row["example_id"]]
        text = row.get("message_text") or row.get("raw_text") or ""
        if row.get("condition") in {"a_direct", "a_written", "a_latent40"}:
            scored = score_sender_message(text, gold_quantities(ex), gold_eligibility_statuses(ex))
            working = written_working_metrics(text)
            row.update(scored)
            row.update(working)
            row["n_missing_q_equals"] = n_missing_q_equals(text)
        elif row.get("condition") == "b_oracle":
            extracted = plan_extraction(row.get("plan_text") or "", row.get("pred_carrier"))
            row.update(extracted)

    unique = []
    for art in artifacts:
        ex = examples[art["example_id"]]
        rec = {
            "example_id": art["example_id"],
            "family_id": art["family_id"],
            "difficulty": art["difficulty"],
            "variant": art.get("variant"),
            "written_ordinary": score_sender_message(
                art.get("written") or "", gold_quantities(ex), gold_eligibility_statuses(ex)
            ),
            "written_cached": score_sender_message(
                art.get("written_cached") or "", gold_quantities(ex), gold_eligibility_statuses(ex)
            ),
            "written_working": written_working_metrics(art.get("written") or ""),
            "latent40": score_sender_message(
                art.get("latent40") or "", gold_quantities(ex), gold_eligibility_statuses(ex)
            ),
            "latent40_optional_q": score_sender_message(
                art.get("latent40") or "",
                gold_quantities(ex),
                gold_eligibility_statuses(ex),
                allow_missing_q_equals=True,
            ),
            "direct": score_sender_message(
                art.get("direct") or "", gold_quantities(ex), gold_eligibility_statuses(ex)
            ),
            "n_missing_q_equals_latent40": n_missing_q_equals(art.get("latent40") or ""),
        }
        unique.append(rec)

    def _count(items, key, inner=None):
        if inner is None:
            return sum(bool(item[key]["all_four_correct"]) for item in items)
        return sum(bool(item[inner]["all_four_correct"]) for item in items)

    n_u = len(unique)
    parse_diag = {
        "n_unique_a_sheets": n_u,
        "all_four_direct": _count(unique, "direct"),
        "all_four_written_ordinary": _count(unique, "written_ordinary"),
        "all_four_written_cached": _count(unique, "written_cached"),
        "all_four_latent40": _count(unique, "latent40"),
        "all_four_latent40_optional_q": _count(unique, "latent40_optional_q"),
        "n_unique_missing_q_equals_latent40": sum(item["n_missing_q_equals_latent40"] > 0 for item in unique),
        "has_results_block_unique": sum(bool(item["written_working"]["has_results_block"]) for item in unique),
        "produced_working_unique": sum(bool(item["written_working"]["produced_working"]) for item in unique),
        "note": (
            "Primary sender scores still require q=. "
            "all_four_latent40_optional_q is a parsing diagnostic only. "
            "plan_extraction_missing is not disagreement. "
            "plan_truncated is separate from answer truncated."
        ),
        "by_difficulty_unique": {},
    }
    for level in ("L1", "L2", "L3"):
        sub = [item for item in unique if item["difficulty"] == level]
        parse_diag["by_difficulty_unique"][level] = {
            "n": len(sub),
            "direct": _count(sub, "direct"),
            "written_ordinary": _count(sub, "written_ordinary"),
            "written_cached": _count(sub, "written_cached"),
            "latent40": _count(sub, "latent40"),
            "latent40_optional_q": _count(sub, "latent40_optional_q"),
        }

    report = _summarize(rows, compares)
    report["parse_diagnostics"] = parse_diag
    report["rescored_from_saved_outputs"] = True
    (dest / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (dest / "parse_diagnostics.json").write_text(json.dumps(parse_diag, indent=2), encoding="utf-8")
    with (dest / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    dest = _dest()
    if not (dest / "evaluations.jsonl").is_file():
        print(f"Missing evaluations in {dest}")
        return 2
    report = rescore(dest)
    print(json.dumps(report.get("parse_diagnostics"), indent=2))
    print("receiver", json.dumps(report.get("receiver_oracle"), indent=2))
    print(f"Wrote {dest / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
