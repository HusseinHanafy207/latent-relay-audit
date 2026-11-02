"""Matched reconstruction of frozen cons_instr_sink token IDs."""

from __future__ import annotations

import inspect

from src.prompts_plan4 import FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt, b_reason_with_message_prompt
from src.stage_plan4_recon import CONDS, INTERPRETATION_GUIDE, PRESERVED, SPEC, _compact_spans, _jobs, main
from src.two_agent import TwoAgentInventory


def _pieces_and_decode():
    prompt = ["<|im_start|>", "user", "\n", "You are Agent A.\n"]
    written = [
        "**Carrier P:**\n",
        "- capacity ok\n",
        "**RESULTS**\n",
        "P: q=10 | eligible | ok\n",
        "Q: q=4 | ineligible | driver\n",
        "R: q=8 | ineligible | fridge\n",
        "S: q=12 | eligible | ok\n",
    ]
    pieces = prompt + written
    n_prompt = len(prompt)

    def decode(ids):
        return "".join(pieces[int(i)] for i in ids)

    return list(range(len(pieces))), decode, n_prompt, len(pieces)


def test_compact_ids_include_results_and_initial_positions():
    ids, decode, n_prompt, n_full = _pieces_and_decode()
    pack = _compact_spans(ids, n_prompt=n_prompt, n_full=n_full, decode=decode)
    assert pack["result_fallback"] is False
    assert set(range(4)).issubset(pack["sink_pos"])
    assert pack["sink_pos"] == sorted(pack["sink_pos"])
    assert pack["selected_ids"] == [ids[p] for p in pack["sink_pos"]]
    assert "P: q=10" in pack["result_text"]
    assert pack["n_sink_extra"] == 4


def test_three_matched_conditions():
    assert CONDS == ("transfer", "reconstruct", "text_result")
    assert [job["condition"] for job in _jobs()] == list(CONDS)
    assert SPEC["transfer"]["layout"] == "consecutive"
    assert SPEC["reconstruct"]["layout"] == "consecutive"
    assert SPEC["transfer"]["framing"] == "instruct"
    assert SPEC["reconstruct"]["framing"] == "instruct"
    assert SPEC["transfer"]["positions"] == SPEC["reconstruct"]["positions"] == "result_sink"
    assert SPEC["text_result"]["kind"] == "text"
    assert "reusing representations" in INTERPRETATION_GUIDE
    assert "compact message layout" in INTERPRETATION_GUIDE


def test_kv_paths_share_instruct_framing_text_does_not():
    from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split

    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    kv = b_reason_kv_instruct_prompt(ex)
    text = b_reason_with_message_prompt(ex, "P: q=1 | eligible")
    assert FRAMING_INSTRUCTION in kv
    assert FRAMING_INSTRUCTION not in text


def test_recon_uses_dev_keeps_hold_out_and_does_not_regenerate_a(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_recon", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "results_stage_plan4_iface" in PRESERVED
    assert "results_stage_plan4_support" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "rebuild_cache_from_ids" in source
    assert "prefill_token_sequence" in source
    assert "keep_shared_positions" in source
    assert "generate_text_keep_cache" not in source
    assert "FRAMING_INSTRUCTION" in source
    assert "rope_op_check" in source
    assert "CacheBack" in source
    rebuild_src = inspect.getsource(TwoAgentInventory.rebuild_cache_from_ids)
    prefill_src = inspect.getsource(TwoAgentInventory.prefill_token_sequence)
    assert "generate_text_keep_cache" not in rebuild_src
    assert "generate_text_keep_cache" not in prefill_src
    assert "_prefill_prompt" in rebuild_src
    assert "_prefill_prompt" in prefill_src
    from src.stage_plan4_recon import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_recon" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_recon", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_recon", "--split", "plan4_fresh"])
    assert main() == 2
