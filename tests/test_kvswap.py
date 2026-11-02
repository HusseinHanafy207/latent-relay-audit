"""Key/value mix diagnostic on the frozen sep compact layout."""

from __future__ import annotations

import inspect

import torch

from src.cache_utils import mix_cache_kv, n_kv_positions
from src.prompts_plan4 import FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.span_select import caches_equal
from src.stage_plan4_kvswap import CONDS, INTERPRETATION_GUIDE, LAYOUT, PRESERVED, SPEC, main
from src.two_agent import TwoAgentInventory


def _blob(key_fill: float, value_fill: float, seq: int = 4):
    return {
        "layers": [
            (torch.full((1, 2, seq, 3), key_fill), torch.full((1, 2, seq, 3), value_fill)),
            (torch.full((1, 2, seq, 3), key_fill + 1), torch.full((1, 2, seq, 3), value_fill + 1)),
        ]
    }


def test_mix_takes_keys_from_one_cache_and_values_from_the_other():
    a = _blob(1.0, 2.0)
    r = _blob(7.0, 8.0)
    mixed = mix_cache_kv(r, a)
    assert n_kv_positions(mixed) == 4
    assert torch.equal(mixed["layers"][0][0], r["layers"][0][0])
    assert torch.equal(mixed["layers"][0][1], a["layers"][0][1])
    assert not torch.equal(mixed["layers"][0][1], r["layers"][0][1])
    assert caches_equal(mix_cache_kv(a, a), a)
    assert caches_equal(mix_cache_kv(r, r), r)
    assert not caches_equal(mix_cache_kv(r, a), a)
    assert not caches_equal(mix_cache_kv(r, a), r)


def test_four_kv_conditions_and_frozen_sep_layout():
    assert LAYOUT == "sep"
    assert CONDS == ("transfer", "recon_k_a_v", "a_k_recon_v", "reconstruct")
    assert SPEC["transfer"] == {"keys": "A", "values": "A", "cache_source": "transfer_gather_rerotate_extend"}
    assert SPEC["recon_k_a_v"]["keys"] == "reconstruct"
    assert SPEC["recon_k_a_v"]["values"] == "A"
    assert SPEC["a_k_recon_v"]["keys"] == "A"
    assert SPEC["a_k_recon_v"]["values"] == "reconstruct"
    assert SPEC["reconstruct"]["keys"] == SPEC["reconstruct"]["values"] == "reconstruct"
    assert "attends" in INTERPRETATION_GUIDE
    assert "compatibility" in INTERPRETATION_GUIDE


def test_b_instructions_and_sep_layout_stay_frozen():
    from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split

    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    prompt = b_reason_kv_instruct_prompt(ex)
    assert FRAMING_INSTRUCTION in prompt
    source = inspect.getsource(__import__("src.stage_plan4_kvswap", fromlist=["main"]))
    assert 'LAYOUT = "sep"' in source
    assert "b_reason_kv_instruct_prompt" in source
    assert "mix_sender_kv" in source
    assert "_build_recon" in source
    assert "_build_transfer" in source


def test_kvswap_uses_dev_keeps_hold_out_and_does_not_regenerate_a(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_kvswap", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "results_stage_plan4_boundary" in PRESERVED
    assert "results_stage_plan4_recon" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "generate_text_keep_cache" not in source
    mix_src = inspect.getsource(TwoAgentInventory.mix_sender_kv)
    assert "generate_text_keep_cache" not in mix_src
    assert "mix_cache_kv" in mix_src
    from src.stage_plan4_kvswap import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_kvswap" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_kvswap", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_kvswap", "--split", "plan4_fresh"])
    assert main() == 2
