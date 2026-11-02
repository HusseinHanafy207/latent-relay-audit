"""Matched-length neutral contexts for the eligibility instruction, R held fixed."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.generic_select import N_EXPECTED_TEMPLATES
from src.neutral_select import (
    BANNED_PHRASES,
    CHAT_HEADER,
    NEUTRAL_PASSAGES,
    QWEN_CHAT_HEADER_IDS,
    assert_clean_text,
    splice_neutral_prefix,
)
from src.plan4_task import a_state_key, load_plan4_jsonl
from src.prompts_plan4 import ELIGIBILITY_RULE, FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_neutral import CONDS, FROZEN_PLACEMENT, INTERPRETATION_GUIDE, PRESERVED, SPEC, main
from src.two_agent import TwoAgentInventory


REPO = Path(__file__).resolve().parents[1]

HEADER_MAP = {9: "<|im_start|>", 8: "user", 7: "\n"}


def _decode(seq):
    return "".join(HEADER_MAP.get(int(i), chr(65 + (int(i) % 26))) for i in seq)


def _encode(text: str):
    return [200 + i for i in range(len(text))]


def test_passages_were_chosen_before_results_and_are_clean():
    assert set(NEUTRAL_PASSAGES) == {"neutral_1", "neutral_2"}
    assert NEUTRAL_PASSAGES["neutral_1"] != NEUTRAL_PASSAGES["neutral_2"]
    for name, text in NEUTRAL_PASSAGES.items():
        assert len(text) > 400, name
        assert_clean_text(text, where=name)
        for phrase in BANNED_PHRASES:
            assert phrase.lower() not in text.lower()
        assert "Shipment:" not in text
        assert ELIGIBILITY_RULE not in text


def test_splice_keeps_header_instruction_and_length():
    header = [9, 8, 7]
    body = list(range(20, 40))
    instr = [100, 101, 102]
    ids = header + body + instr
    rec = splice_neutral_prefix(
        ids,
        instruction_start=23,
        instruction_end=26,
        passage=NEUTRAL_PASSAGES["neutral_1"],
        encode=_encode,
        decode=_decode,
        passage_id="neutral_1",
    )
    assert rec["header_len"] == 3
    assert rec["ids"][:3] == header
    assert rec["ids"][23:26] == instr
    assert len(rec["ids"]) == len(ids)
    assert rec["ids"][3:23] != body
    assert rec["used_pad"] is False
    assert _decode(rec["ids"][:3]) == CHAT_HEADER


def test_splice_rejects_pad_tokens_and_task_leak():
    header = [9, 8, 7]
    ids = header + list(range(20, 30)) + [100]
    try:
        splice_neutral_prefix(
            ids,
            instruction_start=13,
            instruction_end=14,
            passage=NEUTRAL_PASSAGES["neutral_1"],
            encode=lambda _text: [4] * 20,
            decode=_decode,
            pad_token_id=4,
        )
        assert False, "expected pad tokens to fail"
    except RuntimeError as exc:
        assert "pad" in str(exc).lower() or "mask" in str(exc).lower()
    try:
        splice_neutral_prefix(
            ids,
            instruction_start=13,
            instruction_end=14,
            passage="A note about Shipment: 12 boxes and eligible carriers.",
            encode=_encode,
            decode=_decode,
        )
        assert False, "expected task leak to fail"
    except RuntimeError as exc:
        assert "task-related" in str(exc).lower() or "shipment" in str(exc).lower()


def test_splice_does_not_pad_a_short_passage():
    header = [9, 8, 7]
    ids = header + list(range(20, 40)) + [100]
    try:
        splice_neutral_prefix(
            ids,
            instruction_start=23,
            instruction_end=24,
            passage="short",
            encode=lambda text: [30, 31],
            decode=_decode,
        )
        assert False, "expected short passage to fail"
    except RuntimeError as exc:
        assert "pad" in str(exc).lower() or "need" in str(exc).lower()


def test_saved_templates_keep_chat_header_and_instruction_span():
    cons = json.loads((REPO / "results_stage_plan4_generic" / "construction.json").read_text(encoding="utf-8"))
    templates = cons["templates"]
    assert len(templates) == N_EXPECTED_TEMPLATES == 3
    assert set(templates) == {"L1", "L2", "L3"}
    for name, rec in templates.items():
        assert tuple(rec["ids"][:3]) == QWEN_CHAT_HEADER_IDS
        assert rec["instruction_end"] - rec["instruction_start"] == 64
        assert rec["n"] == 20
        examples = {a_state_key(ex): ex for ex in load_plan4_jsonl(REPO / "data" / "plan4_dev.jsonl")}
        for key in rec["a_state_keys"]:
            assert examples[key].difficulty == name


def test_three_conditions_hold_r_and_keep_hold_out(monkeypatch):
    assert LAYOUT == "sep"
    assert EXTRA_SEP is True
    assert CONDS == ("instr_generic", "instr_neutral_1", "instr_neutral_2")
    assert SPEC["instr_generic"]["instruction"] == "task"
    assert SPEC["instr_neutral_1"]["passage_id"] == "neutral_1"
    assert SPEC["instr_neutral_2"]["passage_id"] == "neutral_2"
    assert "neutral" in FROZEN_PLACEMENT
    assert "pad" in FROZEN_PLACEMENT.lower() or "mask" in FROZEN_PLACEMENT.lower()
    assert "task context" in INTERPRETATION_GUIDE or "meaningful" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_neutral", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "generate_text_keep_cache" not in source
    assert "results_stage_plan4_generic" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "_build_generic_from_transferred" in source
    gather_src = inspect.getsource(
        __import__("src.stage_plan4_generic", fromlist=["_build_generic_from_transferred"])._build_generic_from_transferred
    )
    assert "concat_gathered_block" in gather_src
    assert "insert_after_framing" not in source
    assert FRAMING_INSTRUCTION
    mix_src = inspect.getsource(TwoAgentInventory.concat_gathered_block)
    assert "rerotate_cache_keys" in mix_src
    from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split

    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    assert FRAMING_INSTRUCTION in b_reason_kv_instruct_prompt(ex)
    from src.stage_plan4_neutral import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_neutral" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_neutral", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_neutral", "--split", "plan4_fresh"])
    assert main() == 2
