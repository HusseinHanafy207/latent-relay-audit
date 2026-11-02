"""Packed-prefix comparison and result-slot-only key/value mix."""

from __future__ import annotations

import inspect

import torch

from src.boundary_layout import compact_slot_ranges, compose_compact_ids
from src.cache_utils import cache_slice_delta, mix_cache_kv_at_positions, n_kv_positions
from src.prompts_plan4 import FRAMING_INSTRUCTION
from src.span_select import caches_equal
from src.stage_plan4_kvresult import CONDS, INTERPRETATION_GUIDE, PREFIX_SOURCE, PRESERVED, SPEC, main
from src.stage_plan4_kvswap import LAYOUT
from src.two_agent import TwoAgentInventory


def _blob(key_fill: float, value_fill: float, seq: int = 6):
    return {
        "layers": [
            (torch.full((1, 2, seq, 3), key_fill), torch.full((1, 2, seq, 3), value_fill)),
        ]
    }


def test_prefix_slots_are_initial_plus_separator():
    rec = compose_compact_ids([1, 2, 3, 4], [10, 11, 12], sep_ids=[198], close_ids=[0], layout="sep")
    slots = compact_slot_ranges(rec)
    assert slots["prefix"] == [0, 1, 2, 3, 4]
    assert slots["result"] == [5, 6, 7]
    assert rec["ids"][4] == 198
    assert rec["ids"][5:] == [10, 11, 12]


def test_slice_delta_and_result_slot_mix_hold_prefix():
    transfer = _blob(1.0, 2.0)
    recon = _blob(7.0, 8.0)
    prefix = cache_slice_delta(transfer, recon, [0, 1, 2, 3, 4])
    assert prefix["n_positions"] == 5
    assert prefix["equal"] is False
    same = cache_slice_delta(recon, recon, [0, 1, 2, 3, 4])
    assert same["equal"] is True
    mixed = mix_cache_kv_at_positions(transfer, transfer, base=recon, positions=[5])
    assert n_kv_positions(mixed) == 6
    prefix_held = cache_slice_delta(mixed, recon, [0, 1, 2, 3, 4])
    assert prefix_held["equal"] is True
    result_from_a = cache_slice_delta(mixed, transfer, [5])
    assert result_from_a["equal"] is True
    identity = mix_cache_kv_at_positions(recon, recon, base=recon, positions=[5])
    assert caches_equal(identity, recon)


def test_four_result_slot_conditions_keep_reconstruct_prefix():
    assert LAYOUT == "sep"
    assert PREFIX_SOURCE == "reconstruct"
    assert CONDS == ("a_k_a_v", "recon_k_a_v", "a_k_recon_v", "reconstruct")
    assert SPEC["a_k_a_v"]["keys"] == SPEC["a_k_a_v"]["values"] == "A"
    assert SPEC["reconstruct"]["keys"] == SPEC["reconstruct"]["values"] == "reconstruct"
    assert "result-line" in INTERPRETATION_GUIDE


def test_kvresult_uses_dev_keeps_hold_out_and_does_not_resume_kvswap(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_kvresult", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "results_stage_plan4_kvswap" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "mix_sender_kv_at" in source
    assert "generate_text_keep_cache" not in source
    assert FRAMING_INSTRUCTION
    mix_src = inspect.getsource(TwoAgentInventory.mix_sender_kv_at)
    assert "mix_cache_kv_at_positions" in mix_src
    assert "generate_text_keep_cache" not in mix_src
    from src.stage_plan4_kvresult import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_kvresult" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_kvresult", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_kvresult", "--split", "plan4_fresh"])
    assert main() == 2
