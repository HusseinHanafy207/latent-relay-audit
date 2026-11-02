"""Relevant-support spans, equal-size random controls, and the development runner."""

from __future__ import annotations

import inspect

from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split
from src.prompts_plan4 import ELIGIBILITY_RULE, FRAMING_INSTRUCTION
from src.stage_plan4_support import (
    COMPACT_KV,
    CONDS,
    LABEL_MARGIN,
    PRESERVED,
    RAND_CONDS,
    REFS,
    SPEC,
    candidate_qualifies,
    choose_candidate,
    main,
)
from src.support_select import RANDOM_SEEDS, extract_support_pack, merge_positions, sample_equal_random
from src.two_agent import TwoAgentInventory


def _pieces_and_decode():
    prompt = [
        "<|im_start|>",
        "user",
        "\n",
        "You",
        " are Agent A. ",
        ELIGIBILITY_RULE,
        " later prompt text ",
        ELIGIBILITY_RULE,
        " end.\n",
    ]
    written = [
        "**Carrier P:**\n",
        "- capacity ok\n",
        "**Carrier Q:**\n",
        "- driver unavailable\n",
        "**Carrier R:**\n",
        "- refrigeration missing\n",
        "**Carrier S:**\n",
        "- all three true\n",
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

    return list(range(len(pieces))), decode, n_prompt, len(pieces), pieces


def test_extracts_first_instruction_and_calc_before_results():
    ids, decode, n_prompt, n_full, pieces = _pieces_and_decode()
    pack = extract_support_pack(
        ids, n_prompt=n_prompt, n_full=n_full, decode=decode, a_state_key="sheet-a"
    )
    assert pack["ok"], pack["failures"]
    assert pack["instruction_occurrence"] == "first"
    instr = pack["instruction"]
    assert ELIGIBILITY_RULE in instr["text"]
    assert instr["start"] == pieces.index(ELIGIBILITY_RULE)
    assert instr["end"] == instr["start"] + 1
    second = n_prompt - 2
    assert pieces[second] == ELIGIBILITY_RULE
    assert second not in pack["instruction_pos"]
    assert "**RESULTS**" not in (pack["calc"]["text"] or "")
    assert "Carrier Q" in pack["calc"]["text"]
    assert pack["calc"]["end"] == n_prompt + 8
    assert pack["result_carriers"] == ["P", "Q", "R", "S"]
    assert "P: q=10" in pack["result_text"]
    assert set(range(4)).issubset(pack["baseline_pos"])
    assert pack["n_extra"] == len(pack["support_extra_pos"])
    assert len(pack["support_pos"]) == len(pack["baseline_pos"]) + pack["n_extra"]
    assert not pack["used_gold_or_b_private"]
    assert pack["result_fallback"] is False


def test_random_matches_extra_count_and_is_shared_by_a_state():
    ids, decode, n_prompt, n_full, _pieces = _pieces_and_decode()
    pack = extract_support_pack(
        ids, n_prompt=n_prompt, n_full=n_full, decode=decode, a_state_key="sheet-a"
    )
    other = extract_support_pack(
        ids, n_prompt=n_prompt, n_full=n_full, decode=decode, a_state_key="sheet-a"
    )
    different = extract_support_pack(
        ids, n_prompt=n_prompt, n_full=n_full, decode=decode, a_state_key="sheet-b"
    )
    assert pack["equal_random_counts"]
    for seed in RANDOM_SEEDS:
        rec = pack["random"][str(seed)]
        assert rec["n"] == pack["n_extra"]
        assert len(rec["union_positions"]) == len(pack["support_pos"])
        assert not (set(rec["positions"]) & set(pack["baseline_pos"]))
        assert rec["positions"] == other["random"][str(seed)]["positions"]
    assert pack["random"]["2101"]["positions"] != different["random"]["2101"]["positions"] or pack["n_extra"] <= 1


def test_random_may_overlap_support_and_is_not_resampled():
    rec = sample_equal_random(
        n_full=20,
        baseline=[0, 1, 2, 3, 18, 19],
        extra_count=4,
        extra_runs=[(6, 10)],
        seed=2101,
        a_state_key="sheet-a",
    )
    assert rec["n"] == 4
    assert rec["sampling_mode"] in {"block_matched", "uniform"}
    union = merge_positions([0, 1, 2, 3, 18, 19], rec["positions"])
    assert len(union) == 10


def test_missing_results_is_recorded_not_substituted():
    pieces = ["<a>", "b", "c", "d", "working without a marker\n", "P: q=1 | eligible | x\n"]
    ids = list(range(len(pieces)))

    def decode(seq):
        return "".join(pieces[int(i)] for i in seq)

    pack = extract_support_pack(ids, n_prompt=4, n_full=len(ids), decode=decode, a_state_key="bad")
    assert not pack["ok"]
    assert any("RESULTS" in msg for msg in pack["failures"])
    assert pack["result_fallback"] is False


def test_support_select_does_not_read_gold_or_b_private():
    source = inspect.getsource(__import__("src.support_select", fromlist=["extract_support_pack"]))
    assert "gold_eligibility_statuses" not in source
    assert "gold_quantities" not in source
    assert "b_facts" not in source
    assert "plan_all_status_correct" not in source


def test_eight_conditions_and_qualification_rule():
    assert CONDS == (
        "kv",
        "text_result",
        "compact",
        "compact_support",
        "compact_rand_2101",
        "compact_rand_2102",
        "compact_rand_2103",
        "text_support",
    )
    assert RAND_CONDS == ("compact_rand_2101", "compact_rand_2102", "compact_rand_2103")
    assert REFS == ("kv", "text_result", "text_support")
    assert SPEC["compact"]["layout"] == "consecutive"
    assert SPEC["compact"]["framing"] == "instruct"
    assert SPEC["compact_support"]["positions"] == "support"
    for name in COMPACT_KV:
        assert SPEC[name]["layout"] == "consecutive"
        assert SPEC[name]["framing"] == "instruct"
    kv = {"n": 90, "parse_ok": 90, "truncated": 0, "plan_all_status_correct": 90}
    text = {"n": 90, "parse_ok": 90, "truncated": 0, "plan_all_status_correct": 90}
    good = {
        "n": 90,
        "parse_ok": 90,
        "truncated": 0,
        "plan_all_status_correct": 86,
    }
    low = dict(good)
    low["plan_all_status_correct"] = 85
    assert LABEL_MARGIN == 5.0
    assert candidate_qualifies(good, kv, text)
    assert not candidate_qualifies(low, kv, text)
    decision = choose_candidate({"kv": kv, "text_result": text, "compact_support": good})
    assert decision["selected"] == "compact_support"
    decision_low = choose_candidate({"kv": kv, "text_result": text, "compact_support": low})
    assert decision_low["selected"] is None


def test_support_uses_dev_keeps_hold_out_and_does_not_regenerate_a(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_support", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "results_stage_plan4_iface" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "rebuild_cache_from_ids" in source
    assert "generate_text_keep_cache" not in source
    assert "FRAMING_INSTRUCTION" in source
    assert FRAMING_INSTRUCTION
    assert "extract_support_pack" in source
    rebuild_src = inspect.getsource(TwoAgentInventory.rebuild_cache_from_ids)
    assert "generate_text_keep_cache" not in rebuild_src
    assert "_prefill_prompt" in rebuild_src
    from src.stage_plan4_support import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_support" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_support", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_support", "--split", "plan4_fresh"])
    assert main() == 2


def test_instruct_framing_unchanged_on_dev_example():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    from src.prompts_plan4 import b_reason_kv_instruct_prompt

    prompt = b_reason_kv_instruct_prompt(examples[0])
    assert FRAMING_INSTRUCTION in prompt
