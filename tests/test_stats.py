"""Paired bootstrap and 5×2 contrasts. No GPU."""

from __future__ import annotations

from src.stats import accuracy_by_condition, contrasts, paired_bootstrap
from src.stage_c import _jobs


def test_jobs_are_ten_main_plus_sensitivity_plus_controls():
    class _Ex:
        pass

    jobs = _jobs(_Ex())
    mains = [j for j in jobs if j["main"]]
    sensitivity = [j for j in jobs if j.get("sensitivity")]
    controls = [j for j in jobs if not j["main"] and not j.get("sensitivity")]
    assert len(mains) == 10
    assert {j["condition"] for j in sensitivity} == {
        "full_sender_only_no_filler",
        "headwise_32_sender_only_no_filler",
        "hobf_32_sender_only_no_filler",
    }
    assert {j["condition"] for j in controls} == {"direct_text", "mismatch_full_sender_only"}
    assert len({j["condition"] for j in jobs}) == 15


def test_penalty_interaction_and_recovery():
    rows = []
    for i in range(4):
        eid = f"test-{i:03d}"
        rows.append({"example_id": eid, "condition": "full_sender_only", "correct": True})
        rows.append({"example_id": eid, "condition": "full_shared", "correct": True})
        rows.append({"example_id": eid, "condition": "headwise_32_sender_only", "correct": i < 1})
        rows.append({"example_id": eid, "condition": "headwise_32_shared", "correct": True})
        rows.append({"example_id": eid, "condition": "hobf_32_sender_only", "correct": i < 2})
        rows.append({"example_id": eid, "condition": "hobf_32_shared", "correct": True})
    table = accuracy_by_condition(rows)
    c = contrasts(table)
    assert c["compression_penalty_D"]["headwise_32_sender_only"] == 0.75
    assert c["compression_penalty_D"]["headwise_32_shared"] == 0.0
    assert c["interaction_I"]["headwise_32"] == 0.75
    assert c["obf_recovery_R"]["sender_only_32"] == 0.25
    assert c["obf_recovery_R"]["shared_32"] == 0.0


def test_bootstrap_resamples_examples_not_evals():
    grouped = {
        "a": {"full_sender_only": {"correct": True}, "headwise_32_sender_only": {"correct": False}},
        "b": {"full_sender_only": {"correct": True}, "headwise_32_sender_only": {"correct": True}},
    }
    report = paired_bootstrap(grouped, n_boot=200, seed=1, conditions=["full_sender_only", "headwise_32_sender_only"])
    assert report["n_examples"] == 2
    assert report["unit"] == "example"
    assert report["accuracy"]["full_sender_only"]["mean"] == 1.0
    lo, hi = report["accuracy"]["headwise_32_sender_only"]["ci95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_no_filler_sensitivity_contrasts():
    from src.stats import sensitivity_contrasts

    table = {
        "full_sender_only": 0.95,
        "full_sender_only_no_filler": 1.0,
        "headwise_32_sender_only": 0.5,
        "headwise_32_sender_only_no_filler": 0.65,
        "hobf_32_sender_only": 0.6,
        "hobf_32_sender_only_no_filler": 0.65,
    }
    s = sensitivity_contrasts(table)
    assert s["compression_penalty_D_no_filler"]["headwise_32_sender_only_no_filler"] == 0.35
    assert s["obf_recovery_R_no_filler_32"] == 0.0
    assert abs(s["filler_minus_no_filler"]["headwise_32"] - (0.5 - 0.65)) < 1e-12
