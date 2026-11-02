import json
from pathlib import Path

from src.plan4_task import gold_eligibility_statuses, gold_quantities, load_plan4_jsonl
from src.paths import DATA_DIR

examples = {ex.example_id: ex for ex in load_plan4_jsonl(DATA_DIR / "plan4_dev.jsonl")}
path = Path("results_stage_plan4_support/a_artifacts.jsonl")
arts = [json.loads(line) for line in path.open(encoding="utf-8")]
out = []
for rec in arts:
    ex = examples[rec["example_id"]]
    out.append(
        "\n".join(
            [
                "=" * 80,
                f"{rec['example_id']} {rec['difficulty']} boxes={ex.boxes} refrig={ex.refrigerated}",
                f"gold_q={gold_quantities(ex)} gold_st={gold_eligibility_statuses(ex)}",
                f"pred_q={rec.get('written_score', {}).get('pred_q')} pred_st={rec.get('written_score', {}).get('pred_status')}",
                "A_TEXT:",
                ex.a_text,
                "WRITTEN:",
                rec["written"],
            ]
        )
    )
Path("_tmp_written.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote", len(arts), "to _tmp_written.txt")