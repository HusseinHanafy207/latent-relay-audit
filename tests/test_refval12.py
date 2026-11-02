"""Value dependence of the reference-attention oracle."""

from __future__ import annotations

import inspect

import torch

import src.stage_plan4_refval12 as rv
from src.stage_plan4_refval12 import (
    BRANCHES,
    EDITS,
    INTERPRETATION_GUIDE,
    ORACLE,
    PERSIST_NAMES,
    PRESERVED,
    REFERENCE,
    _decision_change,
    _summarize,
    check_value_edit,
    edit_result_values,
    fixed_derangement,
    main,
)


def _blob(n=10, n_layers=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"layers": [(torch.randn(1, 2, n, 4, generator=g), torch.randn(1, 2, n, 4, generator=g)) for _ in range(n_layers)]}


def test_derangement_is_fixed_and_has_no_fixed_points():
    for n in (2, 5, 37):
        p = fixed_derangement(n)
        assert sorted(p) == list(range(n))
        assert all(i != j for i, j in enumerate(p))
        assert p == fixed_derangement(n)


def test_zero_edit_touches_only_result_values():
    base = _blob()
    result = [6, 7, 8, 9]
    before = [(k.clone(), v.clone()) for k, v in base["layers"]]
    z = edit_result_values(base, result, "zero")
    for (k, v), (k0, v0), (zk, zv) in zip(base["layers"], before, z["layers"]):
        assert torch.equal(k, k0) and torch.equal(v, v0)
        assert torch.equal(zk, k0)
        assert torch.equal(zv[:, :, :6], v0[:, :, :6])
        assert bool((zv[:, :, 6:] == 0).all())
    assert check_value_edit(base, z, result, "zero")["ok"]


def test_perm_edit_is_consistent_across_layers_and_heads():
    base = _blob()
    result = [6, 7, 8, 9]
    perm = fixed_derangement(len(result))
    p = edit_result_values(base, result, "perm", perm)
    for (k, v), (pk, pv) in zip(base["layers"], p["layers"]):
        assert torch.equal(pk, k)
        assert torch.equal(pv[:, :, :6], v[:, :, :6])
        for j, src in enumerate(perm):
            assert torch.equal(pv[:, :, result[j]], v[:, :, result[src]])
    assert check_value_edit(base, p, result, "perm", perm)["ok"]


def test_check_catches_key_or_outside_changes():
    base = _blob()
    result = [6, 7, 8, 9]
    z = edit_result_values(base, result, "zero")
    z["layers"][1][0][0, 0, 0, 0] += 1.0
    assert not check_value_edit(base, z, result, "zero")["keys_equal"]
    z2 = edit_result_values(base, result, "zero")
    z2["layers"][0][1][0, 0, 2, 0] += 1.0
    row = check_value_edit(base, z2, result, "zero")
    assert not row["values_outside_result_equal"] and not row["ok"]
    try:
        edit_result_values(base, result, "perm", [0, 0, 1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected a bad-permutation error")


def test_decision_change_counts():
    left = {"per_carrier": [{"carrier": "P", "decision": "eligible", "correct": True}, {"carrier": "Q", "decision": "eligible", "correct": False}]}
    right = {"per_carrier": [{"carrier": "P", "decision": "ineligible", "correct": False}, {"carrier": "Q", "decision": "ineligible", "correct": True}]}
    assert _decision_change(left, right) == {"n": 2, "changed": 2, "fixed": 1, "broken": 1}


def test_stage_scope_and_guards(monkeypatch):
    assert REFERENCE == "fk_fv" and ORACLE == "tk_tv_refattn"
    assert BRANCHES == ("tk_tv", "fk_fv", "tk_tv_refattn", "tk_tv_refattn_vzero", "tk_tv_refattn_vperm")
    assert EDITS == {"tk_tv_refattn_vzero": "zero", "tk_tv_refattn_vperm": "perm"}
    assert "results_stage_plan4_refgen12" in PRESERVED and "results_stage_plan4_refattn12" in PRESERVED
    assert "case_checks.jsonl" in PERSIST_NAMES
    assert "Teacher-forced" in INTERPRETATION_GUIDE
    source = inspect.getsource(rv)
    assert "full_kv" not in source
    assert "[ref_cache, target]" in source
    assert "reference_cache_unchanged" in source
    monkeypatch.setattr("sys.argv", ["stage_plan4_refval12", "--split", "plan4_dev"])
    assert main() == 2


def test_summary_retained_recovery():
    vals = {"tk_tv": (7, 0.4), "fk_fv": (12, 0.0), ORACLE: (12, 0.0), "tk_tv_refattn_vzero": (8, 0.3), "tk_tv_refattn_vperm": (9, 0.2)}
    rows = []
    for i in range(9):
        for b, (c, kl) in vals.items():
            rows.append(
                {
                    "example_id": f"e{i}",
                    "difficulty": "L1",
                    "branch": b,
                    "n_labels": 12,
                    "labels_correct": c,
                    "labels_agree_ref": c,
                    "ineligible_marked_eligible": 12 - c,
                    "eligible_marked_ineligible": 0,
                    "mean_kl": kl,
                    "mean_label_kl": kl,
                    "argmax_agree": 1.0 - kl,
                    "mean_nll_saved_tokens": kl,
                    "mean_gold_margin": 1.0,
                    "instrumentation_ok": True,
                    "decision_change_vs_oracle": None if b == ORACLE else {"n": 12, "changed": 12 - c, "fixed": 0, "broken": 12 - c},
                }
            )
    checks = [
        {"example_id": f"e{i}", "ok": True, "checks": {"oracle_reproduces_refattn12": {"decisions_equal": True, "labels_correct_equal": True, "mean_kl_abs_diff": 0.0}}}
        for i in range(9)
    ]
    s = _summarize(rows, checks)
    assert s["instrumentation_ok"] is True
    assert s["oracle_reproduces_refattn12"]["decisions_equal"] == 9
    r = s["recovery_retained"]
    assert abs(r["tk_tv_refattn_vzero"]["label_recovery_retained"] - 0.2) < 1e-9
    assert abs(r["tk_tv_refattn_vperm"]["label_recovery_retained"] - 0.4) < 1e-9
    assert abs(r["tk_tv_refattn_vperm"]["kl_recovery_retained"] - 0.5) < 1e-9
    d = s["paired_by_case"]["vzero_vs_oracle"]["labels_correct"]
    assert d["total_delta"] == -36 and d["cases_decreased"] == 9
    assert s["decision_change_vs_oracle"]["tk_tv_refattn_vzero"]["broken"] == 36
    assert "only through values" in s["mechanism_note"]
