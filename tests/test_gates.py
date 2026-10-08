"""Development-gate logic. No GPU."""

from __future__ import annotations

from src.gates import evaluate_gates, inspect_hobf_fp32


def _block(acc: float, follow: float = 0.0, truncated: float = 0.0, peak: float = 10.0) -> dict:
    return {
        "n": 20,
        "accuracy": acc,
        "follow_donor": follow,
        "truncated": truncated,
        "any_nonfinite_cache": False,
        "any_nonfinite_obf": False,
        "peak_gpu_gb_max": peak,
        "mean_compression_time_s": 1.7,
        "mean_compression_core_s": 1.7,
        "mean_receiver_latency_s": 6.0,
    }


def test_passing_gates_require_full_relay_before_mismatch():
    summary = {
        "direct_text": _block(0.9),
        "full_sender_only_filler": _block(0.8, follow=0.05),
        "full_sender_only_no_filler": _block(0.75),
        "mismatch_full_sender_only": _block(0.15, follow=0.7),
        "headwise_sender_only_filler": _block(0.4),
        "headwise_sender_only_no_filler": _block(0.35),
        "hobf_sender_only_filler": _block(0.45),
    }
    report = evaluate_gates(summary, n_examples=20)
    assert report["gates"]["direct_text"]["passed"]
    assert report["gates"]["full_relay_sender_only"]["passed"]
    assert report["gates"]["matched_mismatch"]["passed"]
    assert report["gates"]["filler_sensitivity"]["passed"]
    assert report["do_not_open_test"] is True
    assert report["extra_controls_complete"] is False
    assert "full relay" in report["next_finding"]


def test_mismatch_blocked_when_full_relay_is_at_chance():
    summary = {
        "direct_text": _block(0.9),
        "full_sender_only_filler": _block(0.25, follow=0.2),
        "full_sender_only_no_filler": _block(0.2),
        "mismatch_full_sender_only": _block(0.2, follow=0.25),
        "headwise_sender_only_filler": _block(0.2),
        "headwise_sender_only_no_filler": _block(0.2),
        "hobf_sender_only_filler": _block(0.2),
    }
    report = evaluate_gates(summary, n_examples=20)
    assert report["gates"]["full_relay_sender_only"]["passed"] is False
    assert report["gates"]["matched_mismatch"]["passed"] is False
    assert report["gates"]["matched_mismatch"]["blocked_by_full_relay"] is True
    assert report["all_passed"] is False


def test_stage_b_refuses_held_out_split():
    import sys

    from src.stage_b import main

    old = sys.argv
    sys.argv = ["stage_b", "--split", "test"]
    try:
        assert main() == 2
    finally:
        sys.argv = old


def test_hobf_source_keeps_sensitive_math_in_fp32():
    report = inspect_hobf_fp32()
    if report.get("error"):
        return
    assert report["ok"] is True


def test_extra_jobs_exist_and_unknown_only_exits():
    import sys

    from src.stage_b import _jobs, _summarize, main

    names = {job["condition"] for job in _jobs()}
    assert "no_relay_filler" in names
    assert "hobf_sender_only_no_filler" in names
    old = sys.argv
    sys.argv = ["stage_b", "--only", "not_a_real_condition"]
    try:
        assert main() == 2
    finally:
        sys.argv = old

    from pathlib import Path
    import json

    prior = Path("results_stage_b/evaluations.jsonl")
    if prior.is_file():
        rows = [json.loads(line) for line in prior.read_text(encoding="utf-8").splitlines() if line]
        summary = _summarize(rows)
        assert summary["mismatch_full_sender_only"]["follow_donor"] == 0.85
        assert summary["mismatch_full_sender_only"]["accuracy"] == 0.0
        assert summary["headwise_sender_only_no_filler"]["true_partial"] == 0.05
        assert summary["headwise_sender_only_no_filler"]["follow_donor"] == 0.1
