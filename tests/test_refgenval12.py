"""Result-value edits under the reference-attention oracle, in free generation."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

import src.stage_plan4_refgenval12 as rgv
from src.stage_plan4_refgenval12 import (
    BRANCHES,
    EDITED,
    INTERPRETATION_GUIDE,
    LOCKSTEP,
    ORACLE,
    PERSIST_NAMES,
    PRESERVED,
    _refgen_rows,
    _summarize,
    label_confidence,
    main,
    result_lines_in_prompt,
    saved_permutation,
)
from src.stage_select import _load_rows

REFVAL_CHECKS = Path("results_stage_plan4_refval12/case_checks.jsonl")
REFGEN_EVALS = Path("results_stage_plan4_refgen12/evaluations.jsonl")


def test_saved_permutations_are_the_seeded_derangements():
    rows = {r["example_id"]: r for r in _load_rows(REFVAL_CHECKS)}
    assert len(rows) == 9
    for eid, row in rows.items():
        perm, check = saved_permutation(rows, eid, row["n_result"])
        assert check["ok"], eid
        assert all(i != j for i, j in enumerate(perm))
    _perm, bad = saved_permutation(rows, next(iter(rows)), 5)
    assert not bad["ok"]


def test_intact_refgen_rows_available_for_reproduction():
    rows = _refgen_rows(REFGEN_EVALS)
    assert len(rows) == 9
    for by in rows.values():
        for b in ("tk_tv", "fk_fv", ORACLE):
            assert by[b]["plan_text"] and by[b]["raw_text"]


def test_result_text_leak_check_ignores_own_plan():
    result_text = "Carrier P: eligible (capacity ok)\nCarrier Q: ineligible (no driver)\n"
    assert result_lines_in_prompt("Use the relayed results.\nShipment: 40 units", result_text) == []
    leaked = "Use the relayed results.\nCarrier Q: ineligible (no driver)\n"
    assert result_lines_in_prompt(leaked, result_text) == ["Carrier Q: ineligible (no driver)"]
    own_plan = "Carrier P: eligible (capacity ok)"
    answer = f"Your plan:\n{own_plan}\nNow answer."
    assert result_lines_in_prompt(answer, result_text, own_plan=own_plan) == []


def test_label_confidence():
    logits = torch.full((3, 10), -20.0)
    logits[0, 4] = 5.0
    logits[2, 5] = 0.0
    logits[2, 7] = 0.0
    logits[2, 8] = 0.0
    hist = {
        "labels": [{"token_index": 0}, {"token_index": 2}, {"token_index": None}],
        "pair": (4, 5),
    }
    c = label_confidence(logits, hist)
    assert c["n"] == 2 and c["n_below_0_5"] == 1
    assert abs(c["min_label_mass"] - 1 / 3) < 1e-4 and abs(c["mean_label_mass"] - 2 / 3) < 1e-4
    assert label_confidence(logits, {"labels": [], "pair": None})["method"] == "unavailable"


def test_stage_scope_and_guards(monkeypatch):
    assert BRANCHES == ("tk_tv", "fk_fv", "tk_tv_refattn", "tk_tv_refattn_vzero", "tk_tv_refattn_vperm")
    assert LOCKSTEP == {ORACLE: None, "tk_tv_refattn_vzero": "zero", "tk_tv_refattn_vperm": "perm"}
    assert EDITED == ("tk_tv_refattn_vzero", "tk_tv_refattn_vperm")
    for name in ("results_stage_plan4_refval12", "results_stage_plan4_refgen12", "results_stage_plan4_refattn12"):
        assert name in PRESERVED
    assert "case_checks.jsonl" in PERSIST_NAMES
    assert "close this diagnostic" in INTERPRETATION_GUIDE
    source = inspect.getsource(rgv)
    assert "_reference_plans" not in source
    assert "full_kv" not in source
    assert "lockstep_generate(runner, bundles" in source
    assert "_target_vs_full_sequence_replay" in source
    monkeypatch.setattr("sys.argv", ["stage_plan4_refgenval12", "--split", "plan4_dev"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_refgenval12", "--max_new_tokens", "64"])
    assert main() == 2


def _row(eid, cond, n_correct, ie, parse=True, conf=None):
    return {
        "example_id": eid,
        "condition": cond,
        "difficulty": "L1",
        "gold_status": {str(i): "eligible" for i in range(12)},
        "plan_pred_status": {str(i): "eligible" for i in range(12)},
        "plan_n_status_correct": n_correct,
        "plan_all_status_correct": n_correct == 12,
        "plan_ineligible_marked_eligible": ie,
        "plan_eligible_marked_ineligible": 0,
        "parse_ok": parse,
        "plan_truncated": False,
        "truncated": False,
        "correct": n_correct == 12,
        "n_plan_tokens": 130,
        "plan_text": "p",
        "raw_text": "r",
        "plan_label_confidence": conf or {},
    }


def test_summary_counts_and_retained_recovery():
    vals = {"tk_tv": (7, 5), "fk_fv": (12, 0), ORACLE: (12, 0), "tk_tv_refattn_vzero": (8, 4), "tk_tv_refattn_vperm": (12, 0)}
    conf = {"n": 12, "mean_label_mass": 0.9, "min_label_mass": 0.2, "n_below_0_5": 1}
    rows = [
        _row(f"e{i}", b, c, ie, parse=(b != "tk_tv_refattn_vzero" or i > 0), conf=conf if b in LOCKSTEP else None)
        for i in range(9)
        for b, (c, ie) in vals.items()
    ]
    checks = [
        {
            "example_id": f"e{i}",
            "ok": True,
            "checks": {"intact_branches_reproduce_refgen12": {"tk_tv": True, "fk_fv": True, ORACLE: True}},
            "comparisons": {
                "tk_tv_refattn_vzero": {"plan_identical_to_oracle": False, "labels_equal_oracle": False},
                "tk_tv_refattn_vperm": {"plan_identical_to_oracle": True, "labels_equal_oracle": True},
            },
            "divergence": {b: None for b in LOCKSTEP},
        }
        for i in range(9)
    ]
    refval = {"branches": {"tk_tv_refattn_vzero": {"labels_correct": 105, "n_labels": 108, "mean_kl_vs_reference": 0.004}}}
    s = _summarize(rows, checks, refval_summary=refval)
    assert s["instrumentation_ok"] is True
    assert s["intact_reproduce_refgen12"] == {"tk_tv": 9, "fk_fv": 9, ORACLE: 9}
    assert s["labels"]["tk_tv_refattn_vzero"]["labels_correct"] == 72
    r = s["recovery_retained"]
    assert abs(r["tk_tv_refattn_vzero"]["label_recovery_retained"] - 0.2) < 1e-9
    assert r["tk_tv_refattn_vperm"]["label_recovery_retained"] == 1.0
    assert s["malformed_output"]["tk_tv_refattn_vzero"]["parse_failed"] == 1
    assert s["plan_label_confidence"]["tk_tv_refattn_vzero"]["labels_below_0_5"] == 9
    assert s["edited_vs_oracle_text"]["tk_tv_refattn_vperm"]["plan_identical"] == 9
    d = s["paired_by_case"]["vzero_vs_oracle"]["labels_correct"]
    assert d["total_delta"] == -36 and d["cases_decreased"] == 9
    assert s["context"]["teacher_forced_refval12"]["tk_tv_refattn_vzero"]["labels_correct"] == 105
    assert "Free generation" in s["measurement_note"]
