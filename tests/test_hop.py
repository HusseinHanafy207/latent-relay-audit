"""Two-record hop task. No GPU."""

from __future__ import annotations

from collections import Counter

from src.hop_inventory import (
    HOP_DEV_SEED,
    HOP_TEST_SEED,
    N_HOP_DEV,
    N_HOP_TEST,
    N_HOP_LINES,
    assert_hop_no_leak,
    generate_split,
    hop_evidence_absent_from_prompt,
)
from src.hop_retrieve import oracle_message, retrieve_two_records
from src.prompts_inventory import receiver_prompt, sender_helpful_message_prompt, sender_prompt_inventory_only
from src.stage_hop import CONDS, _jobs


def test_hop_dev_and_test_are_balanced_and_disjoint():
    dev = generate_split("hop_dev", N_HOP_DEV, HOP_DEV_SEED)
    test = generate_split("hop_test", N_HOP_TEST, HOP_TEST_SEED)
    assert len(dev) == 10
    assert len(test) == 40
    assert {ex.example_id for ex in dev}.isdisjoint({ex.example_id for ex in test})
    for ex in [*dev, *test]:
        assert_hop_no_leak(ex)
        assert ex.n_lines == N_HOP_LINES
        assert ex.question.startswith("Which locker contains package ")
        assert str(ex.true_locker) not in ex.question
    dev_letters = Counter(ex.gold_letter for ex in dev)
    test_letters = Counter(ex.gold_letter for ex in test)
    assert min(dev_letters.values()) >= 2
    assert max(dev_letters.values()) <= 3
    assert test_letters == {"A": 10, "B": 10, "C": 10, "D": 10}


def test_iterative_retriever_finds_both_gold_records_without_indices():
    import inspect

    names = inspect.signature(retrieve_two_records).parameters
    for banned in ("evidence_package_line", "evidence_shipment_line", "true_locker", "gold_letter", "queried_index"):
        assert banned not in names
    examples = generate_split("hop_dev", N_HOP_DEV, HOP_DEV_SEED)
    for ex in examples:
        hit = retrieve_two_records(ex.question, ex.true_inventory_text)
        assert hit.found_both
        assert hit.package_sentence == ex.evidence_package_sentence
        assert hit.shipment_sentence == ex.evidence_shipment_sentence
        assert hit.locker == ex.true_locker
        assert oracle_message(ex) == f"{ex.evidence_package_sentence}\n{ex.evidence_shipment_sentence}"


def test_sender_encodes_without_question_and_b_does_not_see_inventory():
    examples = generate_split("hop_dev", N_HOP_DEV, HOP_DEV_SEED)[:3]
    for ex in examples:
        sender = sender_prompt_inventory_only(ex.true_inventory_text)
        assert ex.question not in sender
        assert "Choices:" not in sender
        assert ex.true_inventory_text in sender
        kv_prompt = receiver_prompt(ex, inventory_text=None, filler_text=None, has_relay=True)
        assert hop_evidence_absent_from_prompt(kv_prompt, ex)
        assert "Warehouse notes" not in kv_prompt
        assert ex.filler_text not in kv_prompt
        none_prompt = receiver_prompt(ex, inventory_text=None, filler_text=None, has_relay=False)
        assert hop_evidence_absent_from_prompt(none_prompt, ex)
        assert "Warehouse notes" not in none_prompt
        assert ex.question in kv_prompt and ex.choices_text in kv_prompt
        assert ex.question in none_prompt and ex.choices_text in none_prompt
        gen = sender_helpful_message_prompt(ex.true_inventory_text, ex)
        assert ex.question in gen
        assert ex.true_inventory_text in gen


def test_hop_jobs_include_both_probe_budgets_and_dev_no_message():
    jobs = _jobs(include_no_message=True)
    assert [j["condition"] for j in jobs] == list(CONDS)
    assert all("_filler" not in j["condition"] for j in jobs)
    budgets = {j["kv_budget"] for j in jobs if j["kind"] == "probe"}
    assert budgets == {32, 64}
    eval_jobs = _jobs(include_no_message=False)
    assert "no_message" not in {j["condition"] for j in eval_jobs}
    assert len(eval_jobs) == 6


def test_process_time_includes_compression_and_found_both_is_retriever_only():
    from src.stage_hop import _cond_block, _process_time_s, _summarize

    row = {
        "prep_time_s": 1.0,
        "selection_time_s": 0.1,
        "compression_time_s": 0.2,
        "sender_generation_time_s": 0.0,
        "receiver_latency_s": 0.3,
    }
    assert abs(_process_time_s(row) - 1.6) < 1e-9
    iterative = {
        "example_id": "hop_dev-000",
        "condition": "iterative_text",
        "correct": True,
        "parse_ok": True,
        "truncated": False,
        "sender_truncated": False,
        "message_bytes": 80,
        "prep_time_s": 0,
        "selection_time_s": 0.001,
        "compression_time_s": 0,
        "sender_generation_time_s": 0,
        "receiver_latency_s": 0.5,
        "found_both": True,
    }
    full = {**iterative, "condition": "full_kv", "found_both": None}
    del full["found_both"]
    report = _summarize([iterative, full], split="hop_dev")
    assert report["accuracy"]["iterative_text"]["found_both"] == 1.0
    assert report["accuracy"]["full_kv"]["found_both"] is None
    assert "peak_gpu_gb_max" not in report["accuracy"]["full_kv"]


def test_stage_hop_refuses_one_record_splits(monkeypatch):
    from src.stage_hop import main

    monkeypatch.setattr("sys.argv", ["stage_hop", "--split", "test"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_hop", "--split", "dev"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_hop", "--split", "validate"])
    assert main() == 2
