"""Held-out confirmation of the frozen generic+R candidate."""

from __future__ import annotations

import inspect

from src.plan4_task import (
    N_PLAN4_HOLD_FAMILIES,
    PLAN4_HOLD_SEED,
    a_state_key,
    generate_split,
)
from src.prompts_plan4 import FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.stage_plan4_confirm import (
    CONDS,
    FROZEN_PLACEMENT,
    INTERPRETATION_GUIDE,
    PRESERVED,
    SPEC,
    _development_refs,
    main,
)
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_rdep import mismatch_a_key
from src.two_agent import TwoAgentInventory


def test_confirm_uses_hold_and_keeps_five_conditions(monkeypatch):
    assert LAYOUT == "sep"
    assert EXTRA_SEP is True
    assert CONDS == ("generic_r", "result_kv", "full_kv", "text_result", "generic_mismatch")
    assert SPEC["generic_r"]["r_source"] == "this"
    assert SPEC["result_kv"]["cache_source"] == "mix_result_a_k_a_v"
    assert SPEC["full_kv"]["cache_source"] == "rebuilt_full"
    assert SPEC["text_result"]["kind"] == "text"
    assert SPEC["generic_mismatch"]["r_source"] == "paired_a_flip"
    assert "previously evaluated" in INTERPRETATION_GUIDE
    assert "on untouched families" not in INTERPRETATION_GUIDE
    assert "fresh independent confirmation" in INTERPRETATION_GUIDE
    assert "generic" in FROZEN_PLACEMENT.lower()
    source = inspect.getsource(__import__("src.stage_plan4_confirm", fromlist=["main"]))
    assert "plan4_hold.jsonl" in source
    assert "plan4_dev.jsonl" in source
    assert "do_not_load" in source
    assert "generate_text_keep_cache" not in source
    assert "rewrite_check" in source
    assert "on untouched families" not in source
    assert "previously evaluated" in source
    refs = _development_refs()
    assert refs["text_result"]["plan_all_status_correct"] == 90
    assert refs["text_result"]["correct"] == 79
    assert "79/90" in refs["text_result"]["source"]
    assert "calculation block is missing carrier headings" not in source
    assert "results_stage_plan4_hold" in PRESERVED
    assert "results_stage_plan4_rdep" in PRESERVED
    assert "insert_after_framing" not in source
    assert "bool(ok)" in inspect.getsource(__import__("src.stage_plan4_rdep", fromlist=["_check_row"])._check_row)
    mix_src = inspect.getsource(TwoAgentInventory.mix_sender_kv_at)
    assert "positions" in mix_src
    examples = generate_split("plan4_hold", N_PLAN4_HOLD_FAMILIES, PLAN4_HOLD_SEED)
    assert len(examples) == 90
    assert all(ex.split == "plan4_hold" for ex in examples)
    assert FRAMING_INSTRUCTION in b_reason_kv_instruct_prompt(examples[0])
    base = next(ex for ex in examples if ex.variant == "base")
    partner_id = mismatch_a_key(base)
    assert partner_id.startswith("plan4_hold")
    assert a_state_key(base) == f"{base.family_id}-base"
    from src.stage_plan4_confirm import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_hold.jsonl"
    assert "stage_plan4_confirm" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_confirm", "--split", "plan4_dev"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_confirm", "--split", "plan4_fresh"])
    assert main() == 2
