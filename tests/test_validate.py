"""Fresh validation split and frozen-selector controls. No GPU."""

from __future__ import annotations

from src.inventory import N_VALIDATE, VALIDATE_SEED, generate_split
from src.prompts_inventory import generic_probe_prompt, mismatch_probe_prompt, probe_prompt, sender_prompt
from src.question_match import other_package_name, package_name_from_question
from src.stage_validate import CONDS, _jobs


def test_validate_jobs_freeze_selectors_and_add_controls():
    jobs = _jobs()
    assert [j["condition"] for j in jobs] == list(CONDS)
    kinds = {j["kind"] for j in jobs}
    assert {"question_match", "probe", "mismatch_probe", "generic_probe", "recency"} <= kinds
    recency = next(j for j in jobs if j["kind"] == "recency")
    assert recency["selection_mode"] == "recency"


def test_validate_split_is_fresh_and_not_test():
    examples = generate_split("validate", N_VALIDATE, VALIDATE_SEED)
    assert len(examples) == 20
    assert all(ex.split == "validate" for ex in examples)
    assert all(ex.example_id.startswith("validate-") for ex in examples)
    test_ids = {ex.example_id for ex in generate_split("test", 80, 2027)}
    dev_ids = {ex.example_id for ex in generate_split("dev", 20, 2026)}
    val_ids = {ex.example_id for ex in examples}
    assert val_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(dev_ids)


def test_mismatch_and_generic_probes_do_not_use_the_asked_package():
    examples = generate_split("validate", 5, VALIDATE_SEED)
    for ex in examples:
        prompt = sender_prompt(ex.true_inventory_text, ex)
        asked = package_name_from_question(ex.question)
        other = other_package_name(prompt, asked)
        assert other != asked
        assert other in ex.packages
        mismatch = mismatch_probe_prompt(other, ex.choices_text)
        assert f"package {other}" in mismatch
        assert f"package {asked}" not in mismatch
        generic = generic_probe_prompt(ex)
        assert asked not in generic
        assert "queried package" in generic
        correct = probe_prompt(ex)
        assert asked in correct
        assert correct == f"{ex.question}\nChoices:\n{ex.choices_text}"


def test_stage_validate_refuses_dev_and_test(monkeypatch):
    from src.stage_validate import main

    monkeypatch.setattr("sys.argv", ["stage_validate", "--split", "test"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_validate", "--split", "dev"])
    assert main() == 2
