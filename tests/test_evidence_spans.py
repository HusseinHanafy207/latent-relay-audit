"""Evidence-line spans and retention audit. No GPU."""

from __future__ import annotations

from src.evidence_spans import (
    char_spans_for_record,
    evidence_line_text,
    kept_flags,
    match_slot_count,
    overlapping_token_indices,
    retention_grid,
    token_spans_for_record,
    unrelated_line_text,
    unrelated_record_index,
)
from src.inventory import load_jsonl
from src.paths import DATA_DIR
from src.prompts_inventory import sender_prompt
from src.stage_select import CONDS, _jobs


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        ids = list(range(len(text)))
        offsets = [(i, i + 1) for i in range(len(text))]
        return {"input_ids": ids, "offset_mapping": offsets}


def test_jobs_are_five_dev_conditions():
    jobs = _jobs()
    assert [j["condition"] for j in jobs] == list(CONDS)
    assert len(jobs) == 5
    assert {j["selection_mode"] for j in jobs} == {None, "attention", "random", "force"}


def test_evidence_and_unrelated_lines_are_unique_on_dev():
    examples = load_jsonl(DATA_DIR / "dev.jsonl")
    assert len(examples) == 20
    for ex in examples:
        prompt = sender_prompt(ex.true_inventory_text, ex)
        evidence = evidence_line_text(ex)
        unrelated = unrelated_line_text(ex)
        assert evidence != unrelated
        assert prompt.count(evidence) == 1
        assert prompt.count(unrelated) == 1
        assert unrelated_record_index(ex) != ex.queried_index
        chars = char_spans_for_record(prompt, ex, ex.queried_index)
        assert chars.package == (
            prompt.find(ex.queried_package, chars.line[0]),
            prompt.find(ex.queried_package, chars.line[0]) + len(ex.queried_package),
        )
        assert chars.line[0] <= chars.package[0] < chars.package[1] <= chars.line[1]
        assert chars.line[0] <= chars.locker[0] < chars.locker[1] <= chars.line[1]
        assert prompt[chars.locker[0] : chars.locker[1]] == str(ex.true_locker)
        question_pkg = prompt.find(f"package {ex.queried_package}", chars.line[1])
        assert question_pkg >= 0
        assert not (chars.package[0] <= question_pkg < chars.package[1])


def test_token_spans_use_the_inventory_line_not_the_question():
    ex = load_jsonl(DATA_DIR / "dev.jsonl")[0]
    prompt = sender_prompt(ex.true_inventory_text, ex)
    tok = _CharTokenizer()
    ids = list(range(len(prompt)))
    spans = token_spans_for_record(tok, prompt, ids, ex, ex.queried_index)
    chars = char_spans_for_record(prompt, ex, ex.queried_index)
    assert spans.package == overlapping_token_indices(
        [(i, i + 1) for i in range(len(prompt))], chars.package
    )
    question_at = prompt.rfind(ex.queried_package)
    assert question_at not in spans.package
    assert set(spans.package).issubset(set(spans.line))
    assert set(spans.locker).issubset(set(spans.line))


def test_kept_flags_require_every_token_on_that_head():
    from src.evidence_spans import TokenSpans

    spans = TokenSpans(record_index=0, line_text="x", line=[10, 11, 12, 13], package=[10, 11], locker=[13])
    flags = kept_flags([10, 11, 12], sink_len=4, spans=spans)
    assert flags["kept_package"] is True
    assert flags["kept_locker"] is False
    assert flags["kept_both"] is False
    flags = kept_flags([10, 11, 13], sink_len=4, spans=spans)
    assert flags["kept_both"] is True
    flags = kept_flags([11, 13], sink_len=4, spans=spans)
    assert flags["kept_package"] is False
    flags = kept_flags([], sink_len=14, spans=spans)
    assert flags["kept_line"] is True


def test_match_slot_count_covers_name_and_locker():
    from src.evidence_spans import TokenSpans

    source = TokenSpans(
        record_index=3,
        line_text="y",
        line=[20, 21, 22, 23, 24, 25, 26],
        package=[22],
        locker=[25],
    )
    got = match_slot_count(source, n_slots=4, sink_len=0)
    assert len(got) == 4
    assert 22 in got and 25 in got


def test_retention_grid_does_not_credit_another_head():
    from src.evidence_spans import TokenSpans

    spans = TokenSpans(record_index=0, line_text="x", line=[5, 6], package=[5], locker=[6])
    selected = [
        [[5], [6]],
        [[5, 6], [0, 1]],
    ]
    grid = retention_grid(selected, sink_len=0, spans=spans)
    assert grid["n_cells"] == 4
    assert grid["n_kept_package"] == 2
    assert grid["n_kept_locker"] == 2
    assert grid["n_kept_both"] == 1
    assert grid["kept_both"] == [[0, 0], [1, 0]]


def test_stage_select_refuses_held_out_split(monkeypatch):
    monkeypatch.setattr("sys.argv", ["stage_select", "--split", "test"])
    from src.stage_select import main

    assert main() == 2
