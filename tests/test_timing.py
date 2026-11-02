"""4- vs 12-carrier timing of the frozen generic+R candidate."""

from __future__ import annotations

import inspect
from pathlib import Path

from src.plan4_task import CARRIERS, load_plan4_jsonl
from src.plan4_timing_task import (
    CARRIERS_12,
    N_CARRIERS_12,
    N_TIMING_FAMILIES,
    PLAN4_TIMING12_SEED,
    assert_timing12_invariants,
    generate_timing12,
    load_timing12_jsonl,
    select_timing4_examples,
)
from src.prompts_plan4 import b_reason_kv_instruct_prompt
from src.prompts_plan4_timing import b_reason_kv_instruct_prompt_n, written_result_handoff_n
from src.stage_plan4_confirm import INTERPRETATION_GUIDE as CONFIRM_GUIDE
from src.stage_plan4_confirm import _development_refs
from src.stage_plan4_timing import (
    CONDS,
    CONFIRM_ACCURACY,
    INTERPRETATION_GUIDE,
    PRESERVED,
    SPEC,
    _process_time_s,
    main,
)
from src.two_agent import TwoAgentInventory


def test_timing_keeps_candidate_frozen_and_splits_b_calls():
    assert CONDS == ("generic_r", "result_kv", "full_kv", "text_result")
    assert "generic_mismatch" not in CONDS
    assert SPEC["generic_r"]["cache_source"] == "generic_instr_same_R"
    assert CONFIRM_ACCURACY["labels"]["generic_r"] == 83
    assert CONFIRM_ACCURACY["labels"]["full_kv"] == 88
    assert CONFIRM_ACCURACY["labels"]["text_result"] == 87
    assert CONFIRM_ACCURACY["choices"]["text_result"] == 74
    assert CONFIRM_ACCURACY["generic_r_minus_full_kv_pp"] == -5.56
    assert "unestablished" in CONFIRM_ACCURACY["note"]
    assert "network traffic" in INTERPRETATION_GUIDE
    assert "prepare once" in INTERPRETATION_GUIDE or "prepared once" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_timing", fromlist=["main"]))
    assert "generate_from_cache_split" in source
    assert "generate_text_message_split" in source
    assert "rewrite_labels" in source
    assert "plan_ineligible_marked_eligible" in source
    assert "plan4_dev.jsonl" in source
    assert "do_not_overwrite" in source
    assert "results_stage_plan4_hold" in PRESERVED
    assert "results_stage_plan4_confirm" in PRESERVED
    mix = inspect.getsource(TwoAgentInventory.generate_from_cache_split)
    assert "prefill_s" in mix
    assert "decode_s" in mix
    text_split = inspect.getsource(TwoAgentInventory.generate_text_message_split)
    assert "prefill_s" in text_split
    assert _process_time_s(
        {
            "sender_generation_time_s": 29.0,
            "rebuild_time_s": 1.0,
            "construct_s": 0.2,
            "plan_time_s": 3.0,
            "receiver_latency_s": 2.0,
        }
    ) == 6.2


def test_timing4_uses_previously_evaluated_hold_bases(monkeypatch):
    hold = load_plan4_jsonl(Path("data/plan4_hold.jsonl"))
    selected = select_timing4_examples(hold)
    assert len(selected) == N_TIMING_FAMILIES
    assert all(ex.split == "plan4_hold" and ex.variant == "base" for ex in selected)
    assert "plan4_hold-fam-004" in {ex.family_id for ex in selected}
    prompt = b_reason_kv_instruct_prompt(selected[0])
    assert "Write exactly four short lines" in prompt
    assert "T: <eligible" not in prompt
    assert "previously evaluated" in CONFIRM_GUIDE
    assert "on untouched families" not in CONFIRM_GUIDE
    refs = _development_refs()
    assert refs["text_result"]["correct"] == 79
    monkeypatch.setattr("sys.argv", ["stage_plan4_timing", "--split", "plan4_dev"])
    assert main() == 2


def test_timing12_is_new_and_does_not_rewrite_frozen_splits():
    path = Path("data/plan4_timing12.jsonl")
    examples = load_timing12_jsonl(path)
    assert len(examples) == N_TIMING_FAMILIES
    assert examples[0].n_carriers == N_CARRIERS_12
    assert tuple(CARRIERS_12) != CARRIERS
    for ex in examples:
        assert_timing12_invariants(ex)
        assert ex.split == "plan4_timing12"
        assert ex.variant == "base"
        prompt = b_reason_kv_instruct_prompt_n(ex)
        assert f"Write exactly {N_CARRIERS_12} short lines" in prompt
        assert "Write exactly four short lines" not in prompt
        handoff = written_result_handoff_n(ex.carriers)
        assert f"exactly {N_CARRIERS_12} result lines" in handoff
        assert "exactly four result lines" not in handoff
    regenerated = generate_timing12(seed=PLAN4_TIMING12_SEED)
    assert [ex.example_id for ex in regenerated] == [ex.example_id for ex in examples]
    frozen = (
        Path("data/plan4_dev.jsonl").read_bytes(),
        Path("data/plan4_fresh.jsonl").read_bytes(),
        Path("data/plan4_hold.jsonl").read_bytes(),
    )
    assert all(len(blob) > 0 for blob in frozen)
    nb = Path("notebooks/kaggle_stage_plan4_timing.ipynb")
    assert nb.is_file()
    text = nb.read_text(encoding="utf-8")
    assert "--phase" in text and "check" in text
    assert "write12" in text
    assert "compare" in text
    assert "previously evaluated" in text
