"""Inventory-only sender / question-withheld run. No GPU."""

from __future__ import annotations

from src.inventory import N_VALIDATE, VALIDATE_SEED, generate_split
from src.prompts_inventory import sender_prompt, sender_prompt_inventory_only
from src.stage_withhold import CONDS, _jobs


def test_withhold_jobs_are_full_recency_probes_and_match():
    jobs = _jobs()
    assert [j["condition"] for j in jobs] == list(CONDS)
    kinds = {j["kind"] for j in jobs}
    assert kinds == {"full", "recency", "probe", "generic_probe", "question_match"}
    recency = next(j for j in jobs if j["kind"] == "recency")
    assert recency["selection_mode"] == "recency"
    match = next(j for j in jobs if j["kind"] == "question_match")
    assert match["selection_mode"] == "force"


def test_name_match_finds_the_line_without_a_seeing_the_question():
    from src.evidence_spans import evidence_line_text
    from src.question_match import inventory_line_for_package, package_name_from_question

    ex = generate_split("validate", 1, VALIDATE_SEED)[0]
    blind = sender_prompt_inventory_only(ex.true_inventory_text)
    asked = package_name_from_question(ex.question)
    assert inventory_line_for_package(blind, asked) == evidence_line_text(ex)
    assert ex.question not in blind


def test_inventory_only_sender_omits_question_and_choices():
    examples = generate_split("validate", 5, VALIDATE_SEED)
    for ex in examples:
        with_q = sender_prompt(ex.true_inventory_text, ex)
        blind = sender_prompt_inventory_only(ex.true_inventory_text)
        assert ex.question in with_q
        assert "Choices:" in with_q
        assert ex.question not in blind
        assert "Choices:" not in blind
        assert ex.choices_text not in blind
        assert ex.true_inventory_text in blind
        assert f"Package {ex.queried_package} is in locker {ex.true_locker}." in blind


def test_seed_copies_existing_evaluations_once(tmp_path):
    from src.stage_withhold import _seed_evaluations

    dest = tmp_path / "dest"
    dest.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "evaluations.jsonl").write_text('{"n": 80}\n', encoding="utf-8")
    _seed_evaluations(dest, src)
    assert (dest / "evaluations.jsonl").read_text(encoding="utf-8") == '{"n": 80}\n'
    (src / "evaluations.jsonl").write_text("other\n", encoding="utf-8")
    _seed_evaluations(dest, src)
    assert (dest / "evaluations.jsonl").read_text(encoding="utf-8") == '{"n": 80}\n'


def test_stage_withhold_refuses_dev_and_test(monkeypatch):
    from src.stage_withhold import main

    monkeypatch.setattr("sys.argv", ["stage_withhold", "--split", "test"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_withhold", "--split", "dev"])
    assert main() == 2
