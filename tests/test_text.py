"""Matched-sentence text baseline. No GPU."""

from __future__ import annotations

from src.inventory import load_jsonl
from src.paths import DATA_DIR
from src.prompts_inventory import retrieved_sentence_prompt
from src.question_match import matched_fact_sentence, package_name_from_question
from src.stage_text import CONDS, _jobs, _message_bytes, _process_time_s, _summarize


def test_text_jobs_are_sentence_only():
    jobs = _jobs()
    assert [j["condition"] for j in jobs] == list(CONDS)
    assert jobs[0]["kind"] == "text"
    assert jobs[0]["relay"] == "text"


def test_matcher_returns_the_bare_fact_sentence_on_dev():
    examples = load_jsonl(DATA_DIR / "dev.jsonl")
    assert len(examples) == 20
    for ex in examples:
        sentence = matched_fact_sentence(ex.question, ex.true_inventory_text)
        gold = f"Package {ex.queried_package} is in locker {ex.true_locker}."
        assert sentence == gold
        assert not sentence[0].isdigit()
        prompt = retrieved_sentence_prompt(ex, sentence)
        assert sentence in prompt
        assert "relayed memory" not in prompt.lower()
        asked = package_name_from_question(ex.question)
        for pkg in ex.packages:
            if pkg != asked:
                assert pkg not in prompt


def test_process_time_adds_match_sender_probe_compression_receiver():
    row = {
        "match_time_s": 0.001,
        "sender_latency_s": 1.0,
        "probe_time_s": 0.1,
        "compression_time_s": 0.07,
        "receiver_latency_s": 0.2,
        "message_bytes": 40,
        "relay_bytes": 0,
    }
    assert abs(_process_time_s(row) - 1.371) < 1e-9
    assert _message_bytes(row) == 40
    assert _message_bytes({"relay_bytes": 100}) == 100


def test_summarize_compares_text_to_existing_kv():
    text = {
        "example_id": "dev-000",
        "condition": "matched_sentence_text",
        "correct": True,
        "gold_letter": "A",
        "gold_locker": 1,
        "donor_letter": "B",
        "donor_locker": 2,
        "raw_text": "Answer: A) locker 1",
        "truncated": False,
        "message_bytes": 40,
        "match_time_s": 0.001,
        "sender_latency_s": 0.0,
        "probe_time_s": 0.0,
        "compression_time_s": 0.0,
        "receiver_latency_s": 0.2,
        "peak_gpu_gb": 1.0,
    }
    full = {
        **text,
        "example_id": "dev-000",
        "condition": "full_sender_only_filler",
        "message_bytes": None,
        "relay_bytes": 50_000_000,
        "sender_latency_s": 2.0,
        "receiver_latency_s": 0.3,
    }
    match = {**full, "condition": "question_match_sender_only_filler", "relay_bytes": 10_000_000}
    probe = {**full, "condition": "receiver_probe_sender_only_filler", "relay_bytes": 10_000_000, "probe_time_s": 0.1}
    report = _summarize([text], [full, match, probe])
    assert report["accuracy"]["matched_sentence_text"]["correct"] == 1
    assert report["comparison"]["full_sender_only_filler"]["mean_message_bytes"] == 50_000_000
    assert report["paired"]["text_vs_full"]["n"] == 1


def test_stage_text_refuses_test_and_validate(monkeypatch):
    from src.stage_text import main

    monkeypatch.setattr("sys.argv", ["stage_text", "--split", "test"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_text", "--split", "validate"])
    assert main() == 2
