"""Instruction-only extra block, equal-size random controls, and the development runner."""

from __future__ import annotations

import inspect

import torch

from src.boundary_layout import compact_slot_ranges, compose_compact_ids, compose_compact_with_extra
from src.cache_utils import cache_slice_delta
from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split
from src.prompts_plan4 import (
    ELIGIBILITY_RULE,
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
    insert_after_framing,
)
from src.stage_plan4_instr import (
    CONDS,
    EQUAL_SIZE,
    EXTRA_SEP,
    FROZEN_PLACEMENT,
    INTERPRETATION_GUIDE,
    PRESERVED,
    RAND_CONDS,
    SPEC,
    main,
)
from src.stage_plan4_kvswap import LAYOUT
from src.support_select import INSTRUCTION_RANDOM_SEEDS, extract_instruction_pack
from src.two_agent import TwoAgentInventory
from tests.test_support import _pieces_and_decode


def test_prefix_slice_allows_longer_extra_cache():
    short = {
        "layers": [
            (torch.arange(15, dtype=torch.float32).view(1, 1, 5, 3), torch.arange(15, dtype=torch.float32).view(1, 1, 5, 3)),
        ]
    }
    long = {
        "layers": [
            (
                torch.cat([short["layers"][0][0], torch.ones(1, 1, 4, 3)], dim=2),
                torch.cat([short["layers"][0][1], torch.zeros(1, 1, 4, 3)], dim=2),
            )
        ]
    }
    held = cache_slice_delta(long, short, [0, 1, 2, 3, 4])
    assert held["equal"] is True
    long["layers"][0][0][0, 0, 1, 0] = 99
    changed = cache_slice_delta(long, short, [0, 1, 2, 3, 4])
    assert changed["equal"] is False


def test_instruction_pack_matches_paragraph_not_calculations():
    ids, decode, n_prompt, n_full, pieces = _pieces_and_decode()
    pack = extract_instruction_pack(
        ids, n_prompt=n_prompt, n_full=n_full, decode=decode, a_state_key="sheet-a"
    )
    other = extract_instruction_pack(
        ids, n_prompt=n_prompt, n_full=n_full, decode=decode, a_state_key="sheet-a"
    )
    different = extract_instruction_pack(
        ids, n_prompt=n_prompt, n_full=n_full, decode=decode, a_state_key="sheet-b"
    )
    assert pack["ok"], pack["failures"]
    assert pack["instruction_contains_rule"]
    assert pack["instruction_before_facts"]
    assert pack["n_instruction"] == 1
    assert pack["n_instruction"] < pack["n_extra"]
    assert ELIGIBILITY_RULE in pack["instruction"]["text"]
    assert not (set(pack["instruction_pos"]) & set(pack["retained_pos"]))
    assert pack["equal_instruction_random_counts"]
    for seed in INSTRUCTION_RANDOM_SEEDS:
        rec = pack["instruction_random"][str(seed)]
        assert rec["n"] == pack["n_instruction"]
        assert not (set(rec["positions"]) & set(pack["retained_pos"]))
        assert rec["positions"] == other["instruction_random"][str(seed)]["positions"]
        assert "overlap_instruction" in rec
        assert "overlap_calc" in rec
    assert pack["instruction_random"]["1"]["positions"] != different["instruction_random"]["1"]["positions"] or pack[
        "n_instruction"
    ] <= 1


def test_extra_layout_keeps_shared_prefix_and_matched_result_slots():
    initial = [1, 2, 3, 4]
    result = [10, 11, 12]
    extra = [20, 21, 22]
    sep = [198]
    composed = compose_compact_with_extra(
        initial,
        extra,
        result,
        sep_ids=sep,
        extra_pos=[8, 9, 10],
        result_pos=[20, 21, 22],
        extra_sep=True,
    )
    slots = compact_slot_ranges(composed)
    baseline = compose_compact_ids(initial, result, sep_ids=sep, close_ids=[0], layout="sep")
    base_slots = compact_slot_ranges(baseline)
    assert EXTRA_SEP is True
    assert composed["ids"][:5] == baseline["ids"][:5]
    assert slots["prefix"] == base_slots["prefix"] == [0, 1, 2, 3, 4]
    assert slots["n_extra"] == 3
    assert slots["n_extra_sep"] == 1
    assert slots["result"] == [9, 10, 11]
    assert slots["result_start"] == 9
    assert composed["ids"][slots["result_start"] :] == result
    assert composed["ids"][5:8] == extra
    other = compose_compact_with_extra(
        initial,
        [30, 31, 32],
        result,
        sep_ids=sep,
        extra_pos=[40, 41, 42],
        result_pos=[20, 21, 22],
        extra_sep=True,
    )
    other_slots = compact_slot_ranges(other)
    assert other_slots["result_start"] == slots["result_start"]
    assert other["n"] == composed["n"]


def test_instruction_text_is_inserted_after_framing_only():
    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    paragraph = ELIGIBILITY_RULE
    plan = insert_after_framing(b_reason_kv_instruct_prompt(ex), paragraph)
    answer = insert_after_framing(b_answer_kv_instruct_prompt(ex, "notes"), paragraph)
    assert plan.find(FRAMING_INSTRUCTION) < plan.find(paragraph)
    assert answer.find(FRAMING_INSTRUCTION) < answer.find(paragraph)
    assert plan.count(paragraph) == 1
    assert "Write exactly four short lines" in plan
    assert b_reason_kv_instruct_prompt(ex).count(FRAMING_INSTRUCTION) == 1


def test_seven_conditions_and_hold_stays_out(monkeypatch):
    assert LAYOUT == "sep"
    assert CONDS == ("r", "instr_kv", "r_instr_text", "rand_1", "rand_2", "rand_3", "reconstruct")
    assert RAND_CONDS == ("rand_1", "rand_2", "rand_3")
    assert EQUAL_SIZE == ("instr_kv", "rand_1", "rand_2", "rand_3")
    assert SPEC["r"]["cache_source"] == "mix_result_a_k_a_v"
    assert SPEC["instr_kv"]["extra"] == "instruction"
    assert SPEC["r_instr_text"]["instruction_text"] is True
    assert "random" in INTERPRETATION_GUIDE
    assert "calculation-block" in INTERPRETATION_GUIDE
    assert "reconstructed five-token prefix" in FROZEN_PLACEMENT
    source = inspect.getsource(__import__("src.stage_plan4_instr", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "generate_text_keep_cache" not in source
    assert "results_stage_plan4_kvresult" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "assemble_compact_cache" in source
    assert "extract_instruction_pack" in source
    mix_src = inspect.getsource(TwoAgentInventory.assemble_compact_cache)
    assert "generate_text_keep_cache" not in mix_src
    from src.stage_plan4_instr import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_instr" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_instr", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_instr", "--split", "plan4_fresh"])
    assert main() == 2
