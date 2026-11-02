"""Compact message-boundary layouts on frozen cons_instr_sink token IDs."""

from __future__ import annotations

import inspect

import torch

from src.boundary_layout import (
    LAYOUTS,
    compose_compact_ids,
    junction_flags,
    layout_flags,
    split_sink_positions,
)
from src.cache_utils import concat_cache_blobs, n_kv_positions
from src.prompts_plan4 import FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.stage_plan4_boundary import CONDS, INTERPRETATION_GUIDE, PAYLOAD_NOTE, PRESERVED, candidate_qualifies, main
from src.stage_plan4_recon import _compact_spans
from src.two_agent import TwoAgentInventory


def test_four_layouts_only_add_sep_or_close():
    initial = [1, 2, 3, 4]
    result = [10, 11, 12]
    sep = [198]
    close = [151645, 198]
    current = compose_compact_ids(initial, result, sep_ids=sep, close_ids=close, layout="current")
    sep_l = compose_compact_ids(initial, result, sep_ids=sep, close_ids=close, layout="sep")
    close_l = compose_compact_ids(initial, result, sep_ids=sep, close_ids=close, layout="close")
    both = compose_compact_ids(initial, result, sep_ids=sep, close_ids=close, layout="both")
    assert current["ids"] == initial + result
    assert current["n_added"] == 0
    assert sep_l["ids"] == initial + sep + result
    assert sep_l["ids"][4] == 198
    assert close_l["ids"] == initial + result + close
    assert both["ids"] == initial + sep + result + close
    for rec in (current, sep_l, close_l, both):
        assert rec["result_ids"] == result
    assert CONDS == LAYOUTS == ("current", "sep", "close", "both")
    assert layout_flags("sep")["separator"] and not layout_flags("sep")["turn_close"]
    assert layout_flags("close")["turn_close"] and not layout_flags("close")["separator"]


def test_sink_split_keeps_first_four_then_results():
    ids = list(range(12))

    def decode(seq):
        pieces = ["<|im_start|>", "user", "\n", "You"] + ["x\n"] * 4 + [
            "P: q=10 | eligible | ok\n",
            "Q: q=4 | ineligible | driver\n",
            "R: q=8 | ineligible | fridge\n",
            "S: q=12 | eligible | ok\n",
        ]
        return "".join(pieces[int(i)] for i in seq)

    pack = _compact_spans(ids, n_prompt=4, n_full=12, decode=decode)
    split = split_sink_positions(pack["sink_pos"])
    assert split["initial_pos"] == [0, 1, 2, 3]
    composed = compose_compact_ids(
        [ids[p] for p in split["initial_pos"]],
        [ids[p] for p in split["result_pos"]],
        sep_ids=[99],
        close_ids=[100],
        layout="current",
    )
    assert composed["ids"] == pack["selected_ids"]


def test_junction_flags_separate_youp_from_newline_and_im_end():
    current = "<|im_start|>user\nYouP: q=10 | Eligible | ok\nQ: q=4 | Ineligible"
    sep = "<|im_start|>user\nYou\nP: q=10 | Eligible | ok\nQ: q=4 | Ineligible"
    close = current + "<|im_end|>\n"
    assert junction_flags(current)["has_youp"]
    assert not junction_flags(current)["has_you_newline_p"]
    assert not junction_flags(current)["has_im_end"]
    assert junction_flags(sep)["has_you_newline_p"]
    assert not junction_flags(sep)["has_youp"]
    assert junction_flags(close)["has_youp"]
    assert junction_flags(close)["has_im_end"]


def test_concat_cache_blobs_along_sequence():
    left = {"layers": [(torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 3, 4))]}
    right = {"layers": [(torch.full((1, 2, 5, 4), 2.0), torch.full((1, 2, 5, 4), 3.0))]}
    out = concat_cache_blobs([left, right])
    assert n_kv_positions(out) == 8
    assert torch.equal(out["layers"][0][0][:, :, :3], left["layers"][0][0])
    assert torch.equal(out["layers"][0][0][:, :, 3:], right["layers"][0][0])


def test_b_instructions_unchanged_and_text_not_requalified_here():
    from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split

    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    prompt = b_reason_kv_instruct_prompt(ex)
    assert FRAMING_INSTRUCTION in prompt
    source = inspect.getsource(__import__("src.stage_plan4_boundary", fromlist=["main"]))
    assert "b_reason_kv_instruct_prompt" in source
    assert "FRAMING_INSTRUCTION" in source
    text = {"n": 90, "parse_ok": 90, "truncated": 0, "plan_all_status_correct": 90}
    good = {"n": 90, "parse_ok": 90, "truncated": 0, "plan_all_status_correct": 86}
    low = dict(good)
    low["plan_all_status_correct"] = 85
    assert candidate_qualifies(good, text)
    assert not candidate_qualifies(low, text)


def test_boundary_uses_dev_keeps_hold_out_and_does_not_regenerate_a(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_boundary", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "results_stage_plan4_recon" in PRESERVED
    assert "results_stage_plan4_iface" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "_rebuild_a" in source
    assert "prefill_token_sequence" in source
    assert "transfer_boundary_cache" in source
    assert "generate_text_keep_cache" not in source
    assert "inspect.jsonl" in source
    assert "skip_special_tokens=False" in inspect.getsource(
        __import__("src.boundary_layout", fromlist=["render_handoff_view"])
    )
    assert "YouP" in INTERPRETATION_GUIDE or "presentation" in INTERPRETATION_GUIDE
    assert "transmitted payload" in PAYLOAD_NOTE.lower() or "transmit token IDs" in PAYLOAD_NOTE
    rebuild_src = inspect.getsource(TwoAgentInventory.rebuild_cache_from_ids)
    prefill_src = inspect.getsource(TwoAgentInventory.prefill_token_sequence)
    extend_src = inspect.getsource(TwoAgentInventory.extend_cache_with_tokens)
    assert "generate_text_keep_cache" not in rebuild_src
    assert "generate_text_keep_cache" not in prefill_src
    assert "generate_text_keep_cache" not in extend_src
    from src.stage_plan4_boundary import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_boundary" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_boundary", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_boundary", "--split", "plan4_fresh"])
    assert main() == 2
    monkeypatch.setattr(
        "sys.argv", ["stage_plan4_boundary", "--split", "plan4_dev", "--phase", "transfer"]
    )
    assert main() == 2
