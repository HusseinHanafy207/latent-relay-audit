"""Join-matched complete neutrals; keep the unmatched-boundary run in the record."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.nbound_select import (
    NEUTRAL_CLOSERS,
    NEUTRAL_SENTENCES,
    assert_clean_join,
    extract_join_suffix,
    splice_complete_neutral_prefix,
)
from src.neutral_select import CHAT_HEADER, assert_clean_text
from src.prompts_plan4 import ELIGIBILITY_RULE, FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_nbound import CONDS, FROZEN_PLACEMENT, INTERPRETATION_GUIDE, PRESERVED, SPEC, main


def _encode(text: str):
    return [100 + ord(c) for c in text]


def _decode(seq):
    out = []
    for i in seq:
        i = int(i)
        if i == 9:
            out.append("<|im_start|>")
        elif i == 8:
            out.append("user")
        elif i == 7:
            out.append("\n")
        elif i == 5:
            out.append("\n\n")
        else:
            out.append(chr(i - 100))
    return "".join(out)


def test_banks_stay_on_the_two_topics_and_are_clean():
    assert set(NEUTRAL_SENTENCES) == {"neutral_1", "neutral_2"}
    assert set(NEUTRAL_CLOSERS) == {"neutral_1", "neutral_2"}
    for name, sentences in NEUTRAL_SENTENCES.items():
        text = " ".join(sentences)
        assert_clean_text(text, where=name)
        assert "ridge" in text or "library" in text
        assert ELIGIBILITY_RULE not in text
        for closer in NEUTRAL_CLOSERS[name]:
            assert_clean_text(closer, where=f"{name}:{closer}")
            assert closer.endswith(".")


def test_extract_join_suffix_is_punctuation_and_break():
    rec = extract_join_suffix(_encode("boxes") + [5], _decode)
    assert rec["ids"] == [5]
    assert rec["text"] == "\n\n"
    rec = extract_join_suffix(_encode("boxes") + [100 + ord("."), 5], _decode)
    assert rec["text"].endswith("\n\n")
    assert "." in rec["text"]


def test_assert_clean_join_rejects_midword_crop():
    try:
        assert_clean_join(
            "Autumn turns the canopy copper, and",
            "A carrier is eligible only if all three are true.",
            where="cropped",
        )
        assert False, "expected mid-word join to fail"
    except RuntimeError as exc:
        assert "concatenates" in str(exc) or "paragraph" in str(exc)


def test_complete_splice_keeps_sentence_and_paragraph_join():
    header = [9, 8, 7]
    dummy = "xxxxx xxxxx"
    closer = "Rain falls."
    assert len(dummy) == len(closer) == 11
    content = _encode(dummy)
    suffix = [5]
    instr = _encode("A carrier is eligible")
    ids = header + content + suffix + instr
    start = 3 + len(content) + 1
    rec = splice_complete_neutral_prefix(
        ids,
        instruction_start=start,
        instruction_end=start + len(instr),
        passage_id="neutral_1",
        encode=_encode,
        decode=_decode,
    )
    assert rec["join_ok"] is True
    assert rec["cropped_mid_sentence"] is False
    assert rec["used_pad"] is False
    assert rec["ids"][:3] == header
    assert rec["ids"][3 + len(content) + 1 :] == instr
    assert rec["body_text"].endswith("\n\n")
    assert rec["body_text"].rstrip().endswith(".") or rec["suffix_text"].endswith("\n\n")
    assert "A carrier" in rec["decoded_join"]
    assert "andA carrier" not in rec["decoded_join"]
    assert _decode(rec["ids"][:3]) == CHAT_HEADER


def test_joins_export_reads_each_context_cell():
    from src.stage_plan4_nbound import _joins_from_contexts

    contexts = {
        "L1": {
            "instr_generic": {"decoded_join": "task-L1", "n_content": 72, "join_ok": True},
            "instr_neutral_1": {"decoded_join": "wood-L1", "n_content": 72, "join_ok": True},
            "instr_neutral_2": {"decoded_join": "lib-L1", "n_content": 72, "join_ok": True},
        },
        "L3": {
            "instr_generic": {"decoded_join": "task-L3", "n_content": 94, "join_ok": True},
            "instr_neutral_1": {"decoded_join": "wood-L3", "n_content": 94, "join_ok": True},
            "instr_neutral_2": {"decoded_join": "lib-L3", "n_content": 94, "join_ok": True},
        },
    }
    joins = _joins_from_contexts(contexts)
    texts = [cell["decoded_join"] for recs in joins.values() for cell in recs.values()]
    assert len(set(texts)) == 6
    assert joins["L1"]["instr_neutral_2"]["decoded_join"] == "lib-L1"
    assert joins["L3"]["instr_neutral_1"]["decoded_join"] == "wood-L3"


def test_saved_nbound_joins_follow_context_records():
    from src.stage_plan4_nbound import _joins_from_contexts

    root = Path(__file__).resolve().parents[1] / "results_stage_plan4_nbound"
    cons = json.loads((root / "construction.json").read_text(encoding="utf-8"))
    expected = _joins_from_contexts(cons["contexts"])
    assert cons["joins"] == expected
    texts = [cell["decoded_join"] for recs in cons["joins"].values() for cell in recs.values()]
    assert len({text for text in texts if text}) >= 6
    check = json.loads((root / "check_summary.json").read_text(encoding="utf-8"))
    assert (check.get("construction") or {}).get("joins") == expected
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert (summary.get("check") or {}).get("joins") == expected


def test_nbound_keeps_hold_out_and_prior_neutral_run(monkeypatch):
    assert LAYOUT == "sep"
    assert EXTRA_SEP is True
    assert CONDS == ("instr_generic", "instr_neutral_1", "instr_neutral_2")
    assert SPEC["instr_generic"]["instruction"] == "task"
    assert "join" in FROZEN_PLACEMENT.lower() or "paragraph" in FROZEN_PLACEMENT.lower()
    assert "unmatched" in INTERPRETATION_GUIDE or "boundary" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_nbound", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "generate_text_keep_cache" not in source
    assert "results_stage_plan4_neutral" in PRESERVED
    assert "results_stage_plan4_hold" in PRESERVED
    assert "splice_complete_neutral_prefix" in source
    assert "insert_after_framing" not in source
    assert FRAMING_INSTRUCTION
    from src.plan4_task import N_PLAN4_FAMILIES, PLAN4_DEV_SEED, generate_split

    ex = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0]
    assert FRAMING_INSTRUCTION in b_reason_kv_instruct_prompt(ex)
    from src.stage_plan4_nbound import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_nbound" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_nbound", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_nbound", "--split", "plan4_fresh"])
    assert main() == 2
