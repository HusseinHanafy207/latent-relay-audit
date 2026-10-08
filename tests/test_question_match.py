"""Question-name record matching. No GPU."""

from __future__ import annotations

import inspect

from src.evidence_spans import evidence_line_text
from src.inventory import load_jsonl
from src.paths import DATA_DIR
from src.prompts_inventory import probe_prompt, sender_prompt
from src.question_match import (
    inventory_line_for_package,
    match_record_from_question,
    package_name_from_question,
)
from src.stage_practical import CONDS, _jobs


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        ids = list(range(len(text)))
        offsets = [(i, i + 1) for i in range(len(text))]
        return {"input_ids": ids, "offset_mapping": offsets}


def test_jobs_are_practical_plus_oracle_ceiling():
    jobs = _jobs()
    assert [j["condition"] for j in jobs] == list(CONDS)
    assert {j["kind"] for j in jobs} == {"full", "headwise", "question_match", "probe", "oracle"}
    names = inspect.signature(match_record_from_question).parameters
    for banned in ("queried_index", "true_locker", "gold_letter", "queried_package"):
        assert banned not in names


def test_package_name_comes_from_the_question_on_dev():
    examples = load_jsonl(DATA_DIR / "dev.jsonl")
    tok = _CharTokenizer()
    for ex in examples:
        name = package_name_from_question(ex.question)
        assert name == ex.queried_package
        prompt = sender_prompt(ex.true_inventory_text, ex)
        line = inventory_line_for_package(prompt, name)
        assert line == evidence_line_text(ex)
        rendered = prompt
        ids = list(range(len(rendered)))
        matched = match_record_from_question(
            question=ex.question,
            sender_prompt_text=prompt,
            rendered=rendered,
            input_ids=ids,
            tokenizer=tok,
        )
        assert matched.line_text == evidence_line_text(ex)
        assert matched.package_name not in {"", ex.gold_letter}
        probe = probe_prompt(ex)
        assert "Answer:" not in probe
        assert ex.true_inventory_text not in probe
        assert f"Package {ex.queried_package} is in locker {ex.true_locker}." not in probe


def test_question_itself_is_not_an_inventory_line():
    question = "Which locker is package Luma in?"
    try:
        inventory_line_for_package(question, "Luma")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_stage_practical_refuses_held_out_split(monkeypatch):
    monkeypatch.setattr("sys.argv", ["stage_practical", "--split", "test"])
    from src.stage_practical import main

    assert main() == 2
