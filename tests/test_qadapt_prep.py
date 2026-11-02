"""Query-adapter pilot preparation: frozen splits, A-change pairs, registered protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

import src.stage_plan4_qadapt_prep as prep
from src.manifest import load_pinned
from src.plan4_qadapt_task import (
    QADAPT_SPLITS,
    assert_disjoint,
    assert_split_invariants,
    build_a_flip,
    changed_carrier,
    file_sha256,
    generate_split,
    load_split,
    mismatch_partner,
    sheet_signature,
    split_path,
)
from src.plan4_timing_task import gold_eligibility_statuses_n, load_timing12_jsonl

TIMING12 = Path("data/plan4_timing12.jsonl")


@pytest.fixture(scope="module")
def splits():
    return {name: load_split(split_path(name)) for name in QADAPT_SPLITS}


def test_split_files_match_registered_hashes_and_regenerate_identically(splits):
    registered = load_pinned()["study_plan4_qadapt"]["data"]
    for name, examples in splits.items():
        assert file_sha256(split_path(name)) == registered[name]["sha256"], name
        assert_split_invariants(name, examples)
        fresh = generate_split(name)
        assert [sheet_signature(e) for e in fresh] == [sheet_signature(e) for e in examples]
        assert [e.example_id for e in fresh] == [e.example_id for e in examples]


def test_split_sizes(splits):
    sizes = {n: (sum(e.variant == "base" for e in v), sum(e.variant == "a_flip" for e in v)) for n, v in splits.items()}
    assert sizes == {"train": (96, 0), "dev": (12, 12), "eval": (24, 24)}


def test_splits_disjoint_from_each_other_and_the_nine_cases(splits):
    assert_disjoint(splits, load_timing12_jsonl(TIMING12))
    names = {e.split for v in splits.values() for e in v}
    assert names == {"plan4_qadapt_train", "plan4_qadapt_dev", "plan4_qadapt_eval"}


def test_a_flip_changes_only_a_and_moves_gold(splits):
    for name in ("dev", "eval"):
        bases = {e.family_id: e for e in splits[name] if e.variant == "base"}
        for flip in (e for e in splits[name] if e.variant == "a_flip"):
            base = bases[flip.family_id]
            assert flip.b_text == base.b_text and flip.b_facts == base.b_facts
            assert flip.carriers == base.carriers and flip.boxes == base.boxes
            carrier = changed_carrier(base, flip)
            gb, gf = gold_eligibility_statuses_n(base), gold_eligibility_statuses_n(flip)
            assert [c for c in base.carriers if gb[c] != gf[c]] == [carrier]
            if flip.a_change == "make_eligible":
                assert carrier == base.trap_carrier and flip.gold_carrier == base.trap_carrier
                assert gf[carrier] == "eligible"
            else:
                assert carrier == base.gold_carrier and flip.gold_carrier == base.alt_carrier
                assert gf[carrier] == "ineligible"
            assert base.a_change == flip.a_change


def test_build_a_flip_rejects_unknown_change(splits):
    with pytest.raises(ValueError):
        build_a_flip(splits["dev"][0], "swap")


def test_mismatch_partner_same_level_other_family(splits):
    for name in ("dev", "eval"):
        by_id = {e.example_id: e for e in splits[name]}
        pairs = mismatch_partner(splits[name])
        assert len(pairs) == sum(e.variant == "base" for e in splits[name])
        assert sorted(pairs.values()) == sorted(pairs)
        for a, b in pairs.items():
            assert a != b and by_id[a].difficulty == by_id[b].difficulty
            assert by_id[a].carriers and set(by_id[a].carriers) == set(by_id[b].carriers)
            ga, gb = gold_eligibility_statuses_n(by_id[a]), gold_eligibility_statuses_n(by_id[b])
            assert any(ga[c] != gb[c] for c in ga)


def test_data_summary_and_hash_guard(splits):
    summary = prep._data_summary(splits, load_timing12_jsonl(TIMING12))
    assert summary["disjoint_from_each_other_and_timing12"]
    assert summary["splits"]["eval"]["a_sheets_written_now"] is False
    assert summary["splits"]["train"]["a_sheets_written_now"] is True
    protocol = load_pinned()["study_plan4_qadapt"]
    assert prep.registered_hash_mismatches(summary, protocol) == []
    bad = {"data": {"dev": {"sha256": "0" * 64}}}
    assert prep.registered_hash_mismatches(summary, bad) == ["dev"]


def test_eval_a_sheets_not_written_here():
    assert prep.A_SPLITS == ("train", "dev")
    assert not any("eval" in name for name in prep.PERSIST_NAMES)


def test_compare_templates():
    t = {"ids": [1, 2, 3], "instruction_ids": [2], "instruction_start": 1, "n": 4}
    frozen = {"L1": dict(t), "L2": {**t, "ids": [1, 2, 4]}}
    assert prep.compare_templates({"L1": dict(t), "L2": {**t, "ids": [1, 2, 4]}}, frozen)["ok"]
    assert not prep.compare_templates({"L1": dict(t), "L2": dict(t)}, frozen)["ok"]
    assert not prep.compare_templates({"L1": dict(t)}, frozen)["ok"]
    assert not prep.compare_templates({"L1": {**t, "instruction_start": 2}, "L2": {**t, "ids": [1, 2, 4]}}, frozen)["ok"]


def _row(eid, cond, n, **kw):
    return {
        "example_id": eid,
        "condition": cond,
        "plan_n_status_correct": n,
        "gold_status": {str(i): "eligible" for i in range(12)},
        "plan_pred_status": {str(i): "eligible" for i in range(12)},
        "plan_time_s": 1.0,
        "receiver_latency_s": 0.5,
        "n_plan_tokens": 300,
        "n_new_tokens": 10,
        **kw,
    }


def test_teacher_summary_pairs_and_targets():
    rows = [
        _row("a", "fk_fv", 12, plan_truncated=True),
        _row("a", "tk_tv", 8),
        _row("b", "fk_fv", 10),
        _row("b", "tk_tv", 11),
        _row("c", "fk_fv", 9),
    ]
    s = prep._teacher_summary(rows)
    assert s["fk_fv"]["labels"]["labels_correct"] == 31 and s["tk_tv"]["labels"]["labels_correct"] == 19
    assert s["paired"] == {
        "n_cases": 2,
        "teacher_minus_student_labels": 3,
        "cases_teacher_better": 1,
        "cases_student_better": 1,
    }
    t = s["teacher_targets"]
    assert t["n"] == 3 and t["plan_truncated"] == 1 and t["filtered"] is False


def test_protocol_registered_before_training():
    p = load_pinned()["study_plan4_qadapt"]
    assert p["registered_before_training"] is True
    assert p["adapter"]["rank"] == 16 and p["adapter"]["trainable_params"] == 36 * (16 * 2560 + 4096 * 16)
    assert p["training"]["epochs"] == 3 and "earlier epoch" in p["selection"]["rule"]
    assert len(p["advance_rule_dev"]) == 5
    for folder in prep.PRESERVED:
        assert folder in p["do_not_overwrite"]


class _Bundle:
    def __init__(self, n=5):
        self.full_ids_cpu = torch.arange(n).view(1, -1)
        k = torch.arange(2 * n * 4, dtype=torch.float16).view(1, 2, n, 4)
        self.cache_cpu = {"layers": [(k, k + 1)]}


class _Runner:
    def clone_sender(self, b):
        return b


def _fake_stage(monkeypatch, tmp_path, splits):
    small = {"train": splits["train"][:2], "dev": splits["dev"][:2], "eval": []}
    for name in prep.A_SPLITS:
        recs = {
            prep.a_state_key(ex): {
                "example_id": ex.example_id,
                "written": "",
                "written_truncated": False,
                "written_score": {"n_status_correct": 11},
            }
            for ex in small[name]
        }
        prep._write_artifacts12(tmp_path / f"a_artifacts_{name}.jsonl", recs)
    monkeypatch.setattr(prep, "_setup", lambda *a: (None,) * 5 + ({"ok": True},))

    def fake_state(runner, ex, artifacts, setup):
        state = {"slots_extra": {"result_start": 2}, "kvmix": {"tk_tv": _Bundle(), "fk_fv": _Bundle()}}
        return state, {"support_held": True, "template_id": ex.difficulty}, {"ok": True, "n_kv": 5, "n_result": 3}, 0.1

    monkeypatch.setattr(prep, "_state", fake_state)
    monkeypatch.setattr(prep, "_persist", lambda *a: None)
    return small


def test_check_and_teacher_bookkeeping_with_fakes(monkeypatch, tmp_path, splits):
    small = _fake_stage(monkeypatch, tmp_path, splits)
    assert prep._run_check(None, tmp_path, None, small, _Runner(), {}, []) == 0
    check = json.loads((tmp_path / "check_summary.json").read_text(encoding="utf-8"))
    assert check["n"] == 4 and check["all_passed"]
    within = check["support_within_template"]
    assert sum(b["n"] for b in within.values()) == 4
    assert all(b["ids_equal"] == b["n"] and b["bytes_equal"] == b["n"] for b in within.values())
    assert check["a_flip_changed_carrier_written_correctly"]["n"] == 1

    def fake_row(runner, ex, state, cond, args):
        return _row(ex.example_id, cond, 12 if cond == "fk_fv" else 7)

    monkeypatch.setattr(prep, "_twelve_row", fake_row)
    args = type("A", (), {"limit": None})()
    assert prep._run_teacher(args, tmp_path, None, small, _Runner(), {}, []) == 0
    summary = json.loads((tmp_path / "teacher_summary.json").read_text(encoding="utf-8"))
    assert summary["paired"]["teacher_minus_student_labels"] == 10
    rows = prep._load_rows(tmp_path / "teacher_train.jsonl")
    assert {r["role"] for r in rows} == {"target", "before_training"} and len(rows) == 4
    assert prep._run_teacher(args, tmp_path, None, small, _Runner(), {}, []) == 0
    assert len(prep._load_rows(tmp_path / "teacher_train.jsonl")) == 4


def test_support_comparison_requires_ids_and_reports_tensor_noise():
    a, b = _Bundle(), _Bundle()
    k, v = b.cache_cpu["layers"][0]
    b.cache_cpu = {"layers": [(k + 0.001, v)]}
    ref = {"a_state_key": "first", "result_start": 2, "ids": prep._support_ids(a, 2), "cache": a.cache_cpu}
    near = prep._support_vs_ref(b, 2, ref)
    assert near["ids_equal"] and not near["bytes_equal"] and 0 < near["keys_max_abs"] < 0.01
    assert near["values_max_abs"] == 0
    b.full_ids_cpu = torch.tensor([[9, 1, 2, 3, 4]])
    assert not prep._support_vs_ref(b, 2, ref)["ids_equal"]
    assert not prep._support_vs_ref(a, 3, ref)["ids_equal"]


def test_main_rejects_changed_allowances(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prep", "--phase", "data", "--plan_max_new_12", "256"])
    assert prep.main() == 2
