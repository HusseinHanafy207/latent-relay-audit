"""Transferred instruction KV vs a fresh forward of the same tokens, with R held fixed."""

from __future__ import annotations

import inspect

import torch

from src.boundary_layout import compact_slot_ranges, compose_compact_with_extra
from src.cache_utils import cache_slice_delta, concat_cache_blobs, n_kv_positions
from src.prompts_plan4 import FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.span_select import gather_cache_positions
from src.stage_plan4_ifresh import CONDS, FROZEN_PLACEMENT, INTERPRETATION_GUIDE, PRESERVED, SPEC, main
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvswap import LAYOUT
from src.two_agent import TwoAgentInventory


def _blob(fill: float, seq: int):
    return {
        "layers": [
            (torch.full((1, 2, seq, 3), fill), torch.full((1, 2, seq, 3), fill + 0.5)),
        ]
    }


def test_appended_result_slots_stay_identical():
    prefix = _blob(1.0, 5)
    extra = _blob(3.0, 4)
    result = _blob(7.0, 3)
    transferred = concat_cache_blobs([prefix, extra, result])
    fresh_head = concat_cache_blobs([prefix, _blob(9.0, 4)])
    spliced = concat_cache_blobs([fresh_head, gather_cache_positions(transferred, [9, 10, 11])])
    assert n_kv_positions(spliced) == n_kv_positions(transferred) == 12
    assert cache_slice_delta(spliced, transferred, [0, 1, 2, 3, 4])["equal"] is True
    assert cache_slice_delta(spliced, transferred, [9, 10, 11])["equal"] is True
    assert cache_slice_delta(spliced, transferred, [5, 6, 7, 8])["equal"] is False


def test_two_conditions_hold_r_and_prompt_fixed(monkeypatch):
    assert LAYOUT == "sep"
    assert EXTRA_SEP is True
    assert CONDS == ("instr_kv", "instr_fresh")
    assert SPEC["instr_kv"]["instruction"] == "transfer"
    assert SPEC["instr_fresh"]["instruction"] == "fresh"
    assert "same transferred R" in FROZEN_PLACEMENT
    assert "preceding context" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_ifresh", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "generate_text_keep_cache" not in source
    assert "results_stage_plan4_instr" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "append_packed_slots" in source
    assert "b_reason_kv_instruct_prompt" in source
    assert "insert_after_framing" not in source
    assert FRAMING_INSTRUCTION
    mix_src = inspect.getsource(TwoAgentInventory.append_packed_slots)
    assert "gather_cache_positions" in mix_src
    assert "rerotate_cache_keys" not in mix_src
    from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split

    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    prompt = b_reason_kv_instruct_prompt(ex)
    assert FRAMING_INSTRUCTION in prompt
    from src.stage_plan4_ifresh import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_ifresh" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_ifresh", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_ifresh", "--split", "plan4_fresh"])
    assert main() == 2


def test_fresh_layout_keeps_instruction_tokens_and_result_start():
    composed = compose_compact_with_extra(
        [1, 2, 3, 4],
        [20, 21, 22],
        [10, 11, 12],
        sep_ids=[198],
        extra_pos=[8, 9, 10],
        result_pos=[20, 21, 22],
        extra_sep=True,
    )
    slots = compact_slot_ranges(composed)
    assert slots["n_prefix"] == 5
    assert slots["result_start"] == 5 + 3 + 1
    assert composed["ids"][5:8] == [20, 21, 22]
    assert composed["ids"][slots["result_start"] :] == [10, 11, 12]
