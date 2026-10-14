"""Rescore saved plan4_span B plans. No GPU. Does not rewrite original files.

B's plan format is eligibility | price | candidate, not A's q= lines.
Do not score capacity recovery.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.parse_plan4 import (
    is_think_collapse,
    score_b_plan,
    selected_content_flags,
)
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import gold_eligibility_statuses, load_plan4_jsonl
from src.stage_plan4_span import CONDS
from src.stage_select import _load_rows

ORIGINAL_NAMES = ("evaluations.jsonl", "summary.json", "manifest.json", "a_artifacts.jsonl")


def _dest() -> Path:
    kaggle = Path("/kaggle/working/results_stage_plan4_span")
    if kaggle.is_dir() and (kaggle / "evaluations.jsonl").is_file():
        return kaggle
    local = REPO_ROOT / "results_stage_plan4_span"
    if (local / "evaluations.jsonl").is_file():
        return local
    return RESULTS_DIR / "stage_plan4_span" / "run"


def _block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "final_correct": sum(bool(item.get("correct")) for item in items),
        "all_status_correct": sum(bool(item.get("plan_all_status_correct")) for item in items),
        "all_price_correct": sum(bool(item.get("plan_all_price_correct")) for item in items),
        "all_flag_correct": sum(bool(item.get("plan_all_flag_correct")) for item in items),
        "choice_matches_gold": sum(bool(item.get("plan_choice_matches_gold")) for item in items),
        "choice_matches_own_cheapest": sum(bool(item.get("plan_choice_matches_own_cheapest")) for item in items),
        "plan_answer_agrees": sum(item.get("plan_answer_agrees") is True for item in items),
        "plan_answer_disagrees": sum(item.get("plan_answer_agrees") is False for item in items),
        "plan_answer_missing": sum(item.get("plan_extraction_status") == "missing" for item in items),
        "think_collapse": sum(bool(item.get("think_collapse")) for item in items),
        "selected_has_result_line": sum(bool(item.get("selected_has_result_line")) for item in items),
        "selected_has_chat_marker": sum(bool(item.get("selected_has_chat_marker")) for item in items),
        "mean_n_status_correct": sum(int(item.get("plan_n_status_correct") or 0) for item in items) / n,
        "mean_n_price_correct": sum(int(item.get("plan_n_price_correct") or 0) for item in items) / n,
        "parse_ok": sum(bool(item.get("parse_ok")) for item in items),
    }


def rescore(dest: Path | None = None) -> dict:
    dest = dest or _dest()
    eval_path = dest / "evaluations.jsonl"
    rows = _load_rows(eval_path)
    examples = {ex.example_id: ex for ex in load_plan4_jsonl(DATA_DIR / "plan4_dev.jsonl")}
    original_sizes = {name: (dest / name).stat().st_size if (dest / name).is_file() else None for name in ORIGINAL_NAMES}

    scored_rows = []
    for row in rows:
        ex = examples[row["example_id"]]
        plan_text = row.get("plan_text") or ""
        raw_text = row.get("raw_text") or ""
        selected = row.get("selected_text") or row.get("message_text") or ""
        plan = score_b_plan(
            plan_text,
            gold_status=gold_eligibility_statuses(ex),
            prices={c: int(ex.b_facts["prices"][c]) for c in ("P", "Q", "R", "S")},
            budget=int(ex.b_facts["budget"]),
            pred_carrier=row.get("pred_carrier"),
        )
        flags = selected_content_flags(selected)
        rec = {
            "example_id": row["example_id"],
            "family_id": row["family_id"],
            "variant": row.get("variant"),
            "difficulty": row.get("difficulty"),
            "condition": row["condition"],
            "correct": row.get("correct"),
            "kind": row.get("kind"),
            "parse_ok": row.get("parse_ok"),
            "pred_carrier": row.get("pred_carrier"),
            "gold_carrier": row.get("gold_carrier"),
            "plan_text": plan_text,
            "raw_text": raw_text,
            "plan_all_status_correct": plan["all_status_correct"],
            "plan_all_price_correct": plan["all_price_correct"],
            "plan_all_flag_correct": plan["all_flag_correct"],
            "plan_n_status_correct": plan["n_status_correct"],
            "plan_n_price_correct": plan["n_price_correct"],
            "plan_n_flag_correct": plan["n_flag_correct"],
            "plan_n_carriers_parsed": plan["n_carriers_parsed"],
            "plan_pred_status": plan["pred_status"],
            "plan_pred_price": plan["pred_price"],
            "plan_pred_flag": plan["pred_flag"],
            "plan_choice": plan["choice"],
            "plan_choice_matches_gold": plan["choice_matches_gold"],
            "plan_choice_matches_own_cheapest": plan["choice_matches_own_cheapest"],
            "plan_named_cheapest": plan["plan_named_cheapest"],
            "plan_extraction_status": plan["plan_extraction_status"],
            "plan_answer_agrees": plan["plan_answer_agrees"],
            "think_collapse": plan["think_collapse"] or is_think_collapse(raw_text),
            "selected_has_result_line": flags["has_result_line"],
            "selected_has_chat_marker": flags["has_chat_marker"],
            "selected_has_q_equals": flags["has_q_equals"],
        }
        scored_rows.append(rec)

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for rec in scored_rows:
        by_cond[rec["condition"]].append(rec)
    report = {
        "n_rows": len(scored_rows),
        "rescored_from_saved_outputs": True,
        "original_files_preserved": True,
        "note": (
            "B plans scored on eligibility, prices, candidate/dropped, and Choice. "
            "Capacities are not scored: B was never asked to print q=. "
            "Original evaluations.jsonl and summary.json are not rewritten."
        ),
        "receiver": {cond: _block(by_cond.get(cond, [])) for cond in CONDS},
        "lead": (
            "Corrected plan scoring. Remaining full-KV and result-text errors can be "
            "downstream decision failures after B already wrote the right eligibility labels."
        ),
    }
    (dest / "plan_rescore.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (dest / "plan_rescore.jsonl").open("w", encoding="utf-8") as handle:
        for rec in scored_rows:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for name, size in original_sizes.items():
        path = dest / name
        if size is not None and path.stat().st_size != size:
            raise RuntimeError(f"Rescore must not rewrite {name}")
    return report


def main() -> int:
    dest = _dest()
    if not (dest / "evaluations.jsonl").is_file():
        print(f"Missing evaluations in {dest}")
        return 2
    report = rescore(dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {dest / 'plan_rescore.json'} without changing original span files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
