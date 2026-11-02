"""Matched reconstruction of twelve-carrier compact messages."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.plan4_timing_task import gold_eligibility_statuses_n, load_timing12_jsonl
from src.prompts_plan4 import b_reason_kv_instruct_prompt
from src.prompts_plan4_timing import b_reason_kv_instruct_prompt_n
from src.stage_plan4_recon12 import CONDS, FROZEN_PLACEMENT, INTERPRETATION_GUIDE, PRESERVED, SPEC, main
from src.stage_plan4_timing import label_direction_counts
from src.two_agent import TwoAgentInventory


def test_recon12_conditions_and_frozen_a(monkeypatch):
    assert CONDS == ("full_kv", "text_result", "generic_r", "generic_fresh_r", "reconstruct")
    assert SPEC["generic_fresh_r"]["r_source"] == "fresh_after_support"
    assert SPEC["reconstruct"]["cache_source"] == "reconstruct_compact"
    assert "same support cache" in FROZEN_PLACEMENT
    assert "changing the result states is enough" in INTERPRETATION_GUIDE
    assert "framing and presentation" in INTERPRETATION_GUIDE
    assert "results_stage_plan4_timing" in PRESERVED
    source = inspect.getsource(__import__("src.stage_plan4_recon12", fromlist=["main"]))
    assert "extend_cache_with_tokens" in source
    assert "prefill_token_sequence" in source
    assert "generate_text_keep_cache" not in source
    assert "Do not load plan4_dev" in source
    assert "parity" in source
    assert "b_reason_kv_instruct_prompt(" in source
    extend = inspect.getsource(TwoAgentInventory.extend_cache_with_tokens)
    assert "Does not gather A's states" in extend or "added tokens" in extend.lower() or "Forward added tokens" in extend
    monkeypatch.setattr("sys.argv", ["stage_plan4_recon12", "--split", "plan4_dev"])
    assert main() == 2


def test_four_carrier_parity_keeps_original_prompts():
    from src.plan4_task import load_plan4_jsonl
    from src.plan4_timing_task import select_timing4_examples

    hold = select_timing4_examples(load_plan4_jsonl(Path("data/plan4_hold.jsonl")))
    prompt = b_reason_kv_instruct_prompt(hold[0])
    assert "Write exactly four short lines" in prompt
    twelve = load_timing12_jsonl(Path("data/plan4_timing12.jsonl"))[0]
    assert "Write exactly 12 short lines" in b_reason_kv_instruct_prompt_n(twelve)


def test_timing_generic_r_direction_is_44_and_1():
    examples = {ex.example_id: ex for ex in load_timing12_jsonl(Path("data/plan4_timing12.jsonl"))}
    rows = [
        json.loads(line)
        for line in Path("results_stage_plan4_timing/evaluations.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    ie = ei = 0
    n = 0
    for row in rows:
        if int(row.get("n_carriers") or 4) != 12 or row.get("condition") != "generic_r":
            continue
        extra = label_direction_counts(
            row.get("plan_pred_status") or {},
            gold_eligibility_statuses_n(examples[row["example_id"]]),
        )
        ie += extra["plan_ineligible_marked_eligible"]
        ei += extra["plan_eligible_marked_ineligible"]
        n += 1
    assert n == 9
    assert ie == 44
    assert ei == 1
    source = inspect.getsource(__import__("src.stage_plan4_timing", fromlist=["_score_b_n"])._score_b_n)
    assert "label_direction_counts" in source
    assert label_direction_counts(
        {"P": "eligible", "Q": "ineligible", "R": "eligible"},
        {"P": "ineligible", "Q": "eligible", "R": "eligible"},
    ) == {
        "gold_status": {"P": "ineligible", "Q": "eligible", "R": "eligible"},
        "plan_ineligible_marked_eligible": 1,
        "plan_eligible_marked_ineligible": 1,
    }
    summary = json.loads(Path("results_stage_plan4_timing/summary.json").read_text(encoding="utf-8"))
    twelve = summary["by_n_carriers"]["12"]["receiver"]["generic_r"]
    assert twelve["plan_ineligible_marked_eligible"] == 44
    assert twelve["plan_eligible_marked_ineligible"] == 1
    assert "does not explain" in summary["twelve_carrier_label_error_note"]
    assert "rewrite_labels" in inspect.getsource(__import__("src.stage_plan4_timing", fromlist=["main"]))
