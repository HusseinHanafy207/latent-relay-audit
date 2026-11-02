"""Generic support with correct R, paired A-flip R, or no R."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.plan4_task import (
    CARRIERS,
    N_PLAN4_FAMILIES,
    PLAN4_DEV_SEED,
    a_state_key,
    generate_split,
    gold_eligibility_statuses,
)
from src.prompts_plan4 import FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_rdep import (
    CONDS,
    FROZEN_PLACEMENT,
    INTERPRETATION_GUIDE,
    PRESERVED,
    SPEC,
    coerce_sheet_ok,
    construction_ok_from_row,
    instruction_construction_ready,
    main,
    mismatch_a_key,
    n_status_agree,
    partner_a_key,
)
from src.two_agent import TwoAgentInventory


def test_sheet_ok_coerces_partner_id_to_boolean():
    assert coerce_sheet_ok("plan4_dev-fam-000-a_flip") is True
    assert coerce_sheet_ok("") is False
    assert coerce_sheet_ok(True) is True
    assert coerce_sheet_ok(False) is False
    source = inspect.getsource(__import__("src.stage_plan4_rdep", fromlist=["_check_row"])._check_row)
    assert "bool(state.get(\"partner_key\"))" in source
    assert "\"ok\": bool(ok)" in source or '"ok": bool(ok)' in source


def test_saved_rdep_ok_fields_are_boolean():
    path = Path(__file__).resolve().parents[1] / "results_stage_plan4_rdep" / "check_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    assert len(rows) == 60
    for row in rows:
        assert isinstance(row.get("ok"), bool), row.get("a_state_key")
        assert row["ok"] is True
        assert row.get("partner_key")
        assert row["ok"] is not row["partner_key"]


def test_construction_ok_ignores_calc_heading_pack_failures():
    spans = {
        "n_instruction": 64,
        "instruction_before_facts": True,
        "instruction_contains_rule": True,
        "causal_prefix": {"private_facts": False},
        "result_carriers": ["P", "Q", "R", "S"],
        "ok": False,
        "failures": ["calculation block is missing carrier headings"],
    }
    assert instruction_construction_ready(spans) is True
    row = {
        "ids_equal": True,
        "result_equal": True,
        "prefix_equal": True,
        "cursor_equal": True,
        "no_r_ok": True,
        "mismatch_ok": True,
        "result_kv_prefix_held": True,
        "instruction_before_facts": True,
        "instruction_contains_rule": True,
        "partner_key": "plan4_hold-fam-004-a_flip",
        "n_instruction": 64,
        "n_result": 81,
        "private_facts": False,
        "ok": False,
        "span_extract_ok": False,
    }
    assert construction_ok_from_row(row) is True
    source = inspect.getsource(__import__("src.stage_plan4_rdep", fromlist=["_check_row"])._check_row)
    gate = source.split("ok = (", 1)[1].split("return", 1)[0]
    assert "instr_ready" in gate
    assert "spans.get(\"ok\")" not in gate


def test_a_flip_pairs_keep_b_prompt_owner():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    by_id = {ex.example_id: ex for ex in examples}
    n_pairs = 0
    for ex in examples:
        partner = by_id[mismatch_a_key(ex)]
        assert partner.family_id == ex.family_id
        if ex.variant == "a_flip":
            assert partner.variant == "base"
            assert a_state_key(ex) != a_state_key(partner)
        else:
            assert partner.variant == "a_flip"
            assert a_state_key(ex) == f"{ex.family_id}-base"
        assert partner_a_key(a_state_key(ex)) == a_state_key(partner)
        n_pairs += 1
    assert n_pairs == 90
    base = next(ex for ex in examples if ex.variant == "base")
    a_flip = next(ex for ex in examples if ex.family_id == base.family_id and ex.variant == "a_flip")
    b_flip = next(ex for ex in examples if ex.family_id == base.family_id and ex.variant == "b_flip")
    assert mismatch_a_key(base) == mismatch_a_key(b_flip) == a_state_key(a_flip)
    assert gold_eligibility_statuses(base) == gold_eligibility_statuses(b_flip)
    assert gold_eligibility_statuses(base) != gold_eligibility_statuses(a_flip)


def test_status_agree_counts_substituted_labels():
    gold = {c: "eligible" for c in CARRIERS}
    pred = {"P": "eligible", "Q": "ineligible", "R": "eligible", "S": "eligible"}
    assert n_status_agree(pred, gold) == 3
    assert n_status_agree({}, gold) == 0
    assert n_status_agree(pred, None) == 0


def test_rdep_conditions_hold_generic_support_and_hold_out(monkeypatch):
    assert LAYOUT == "sep"
    assert EXTRA_SEP is True
    assert CONDS == ("generic_r", "generic_mismatch", "generic_no_r")
    assert SPEC["generic_r"]["r_source"] == "this"
    assert SPEC["generic_mismatch"]["r_source"] == "paired_a_flip"
    assert SPEC["generic_no_r"]["r_source"] == "none"
    assert "A-flip" in FROZEN_PLACEMENT or "a_flip" in FROZEN_PLACEMENT.lower() or "paired" in FROZEN_PLACEMENT.lower()
    assert "example-specific" in INTERPRETATION_GUIDE or "result states" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_rdep", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "generate_text_keep_cache" not in source
    assert "append_packed_slots" in source
    assert "results_stage_plan4_nbound" in PRESERVED
    assert "results_stage_plan4_generic" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "insert_after_framing" not in source
    mix_src = inspect.getsource(TwoAgentInventory.append_packed_slots)
    assert "without rerotating" in mix_src or "without rerotate" in mix_src
    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    assert FRAMING_INSTRUCTION in b_reason_kv_instruct_prompt(ex)
    from src.stage_plan4_rdep import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_rdep" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_rdep", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_rdep", "--split", "plan4_fresh"])
    assert main() == 2
