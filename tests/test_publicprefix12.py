"""Complete public task prefix versus the instruction slice (twelve carriers)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

import src.stage_plan4_publicprefix12 as pp
from src.boundary_layout import EXTRA_SEP_KIND
from src.cache_utils import concat_cache_blobs
from src.paths import DATA_DIR
from src.plan4_timing_task import load_timing12_jsonl
from src.public_prefix import (
    CONTEXTS,
    boundary_record,
    matched_control_check,
    neutral_prefixes,
    packed_result_slots,
    private_prefix_hits,
)
from src.rope_rerotate import dummy_cos_sin, rerotate_cache_keys
from src.span_select import gather_cache_positions
from src.two_agent import TwoAgentInventory

REPO = DATA_DIR.parent
NBOUND = REPO / "results_stage_plan4_nbound" / "construction.json"
SEP = [271]
HEADER = [151644, 872, 198]
INSTR = [21, 22, 23, 24]


class _Rot(torch.nn.Module):
    def forward(self, x, pos):
        return dummy_cos_sin(pos.view(-1).tolist(), x.shape[-1])


def _runner():
    r = object.__new__(TwoAgentInventory)
    r.wrapper = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(rotary_emb=_Rot())))
    return r


def _blob(n, seed, n_layers=2):
    g = torch.Generator().manual_seed(seed)
    return {
        "layers": [
            (torch.randn(1, 2, n, 8, generator=g).half(), torch.randn(1, 2, n, 8, generator=g).half())
            for _ in range(n_layers)
        ]
    }


def _bundle(runner, n, ids, seed):
    return runner._bundle_from_packed(_blob(n, seed), ids, pos_cursor=n, source="test")


def _prefix_ids(ctx_tokens):
    return HEADER + ctx_tokens + INSTR


def _build(*, woodland_ctx=(31, 32), double_rotate=None, edit_value=None):
    runner = _runner()
    a_ids = list(range(1000, 1030))
    a = _bundle(runner, 30, a_ids, seed=1)
    result_pos = list(range(20, 26))
    r_ids = [a_ids[p] for p in result_pos]
    base_head = _bundle(runner, 70, list(range(2000, 2070)), seed=2)
    baseline = runner.concat_gathered_block(base_head, a, src_pos=result_pos, token_ids=r_ids, source="baseline")
    ctx_tokens = {"task": [11, 12], "woodland": list(woodland_ctx), "library": [41, 42]}
    heads, prefixes = {}, {}
    for i, ctx in enumerate(CONTEXTS):
        ids = _prefix_ids(ctx_tokens[ctx])
        head = _bundle(runner, len(ids) + len(SEP), ids + SEP, seed=10 + i)
        heads[ctx] = {"bundle": head, "n_head": len(ids) + len(SEP)}
        prefixes[ctx] = {
            "ids": ids,
            "header_len": len(HEADER),
            "instruction_start": len(HEADER) + 2,
            "instruction_end": len(ids),
        }
    caches = {pp.BASELINE: baseline, "full_kv": a}
    for cond in pp.LONG_TRANSFERRED:
        head = heads[pp.SPEC[cond]["context"]]["bundle"]
        caches[cond] = runner.concat_gathered_block(head, a, src_pos=result_pos, token_ids=r_ids, source=cond)
    task_head = heads["task"]["bundle"]
    fresh_blob = concat_cache_blobs([task_head.cache_cpu, _blob(len(r_ids), seed=99)])
    task_ids = [int(t) for t in task_head.full_ids_cpu.view(-1).tolist()]
    caches[pp.FRESH] = runner._bundle_from_packed(fresh_blob, task_ids + r_ids, pos_cursor=len(task_ids) + len(r_ids), source="fresh")
    n_head = heads["task"]["n_head"]
    slots = packed_result_slots(n_head, len(r_ids))
    if double_rotate:
        blob = caches[double_rotate].cache_cpu
        got = gather_cache_positions(blob, slots)
        twice = rerotate_cache_keys(got, old_positions=result_pos, new_positions=slots, model=runner.wrapper.model)
        for (k, _v), (tk, _tv) in zip(blob["layers"], twice["layers"]):
            k[:, :, slots] = tk
    if edit_value:
        caches[edit_value].cache_cpu["layers"][0][1][:, :, slots[0]] += 1.0
    pieces = [
        {"kind": "result", "ids": r_ids, "source_pos": result_pos},
        {"kind": EXTRA_SEP_KIND, "ids": SEP},
    ]
    state = {
        "key": "k0",
        "template": "L1",
        "pair": {
            "bundle": a,
            "composed_extra": {"pieces": pieces},
            "slots_extra": {"result": list(range(70, 76)), "result_start": 70},
        },
        "spans": {"result_text": "A: eligible\nB: ineligible"},
        "result_pos": result_pos,
        "r_ids": r_ids,
        "piece": pieces[0],
        "caches": caches,
        "assemble_s": {},
        "rebuild_s": 0.0,
    }
    setup = {"heads": {"L1": heads}, "prefixes": {"L1": prefixes}, "sep_ids": SEP}
    rec = {"full_ids": a_ids}
    return runner, state, setup, rec


def test_matched_construction_passes_all_checks():
    runner, state, setup, rec = _build()
    sel = pp._selection_check(state, rec, {"n_result": 6})
    matched = pp._matched_check(state, setup)
    transferred = pp._transferred_check(runner, state)
    support = pp._support_check(state, setup)
    assert sel["ok"], sel
    assert matched["ok"] and matched["result_start"] == 10 and matched["result_slots"] == [10, 15], matched
    assert transferred["ok"], transferred
    assert all(transferred["task_neutral_result_identical"].values())
    assert transferred["keys_max_abs_vs_baseline"] > 0
    assert not transferred["fresh_vs_transferred"]["equal"]
    assert support["ok"]


def test_rotating_twice_is_caught():
    runner, state, _setup, _rec = _build(double_rotate="woodland_prefix_r")
    row = pp._transferred_check(runner, state)
    assert not row["per_condition"]["woodland_prefix_r"]["keys_equal_single_rotation"]
    assert row["per_condition"]["woodland_prefix_r"]["values_equal_sender"]
    assert not row["task_neutral_result_identical"]["task_prefix_r_vs_woodland_prefix_r"]
    assert not row["ok"]


def test_changed_result_values_are_caught():
    runner, state, _setup, _rec = _build(edit_value=pp.CANDIDATE)
    row = pp._transferred_check(runner, state)
    assert not row["per_condition"][pp.CANDIDATE]["values_equal_sender"]
    assert not row["values_equal_baseline"]
    assert not row["ok"]


def test_unmatched_neutral_prefix_is_caught():
    _runner_, state, setup, _rec = _build(woodland_ctx=(31, 32, 33))
    row = pp._matched_check(state, setup)
    assert not row["lengths_equal"] and not row["ok"]


def test_changed_support_is_caught():
    _runner_, state, setup, _rec = _build()
    state["caches"]["library_prefix_r"].cache_cpu["layers"][1][0][:, :, 2] += 1.0
    row = pp._support_check(state, setup)
    assert not row["head_held"]["library_prefix_r"] and not row["ok"]


def test_selection_catches_wrong_ids_and_empty_text():
    _runner_, state, _setup, rec = _build()
    assert pp._selection_check(state, rec, {"n_result": 6})["ok"]
    assert not pp._selection_check(state, rec, {"n_result": 7})["ok"]
    state["r_ids"] = state["r_ids"][:-1] + [5]
    assert not pp._selection_check(state, rec, {"n_result": 6})["ids_match_record"]
    _runner_, state, _setup, rec = _build()
    state["spans"]["result_text"] = "  "
    assert not pp._selection_check(state, rec, {"n_result": 6})["ok"]


def test_packed_slots_follow_the_newline():
    assert packed_result_slots(141, 3) == [141, 142, 143]
    assert packed_result_slots(146, 1) == [146]
    assert packed_result_slots(163, 0) == []


def test_matched_control_failure_modes():
    task = HEADER + [11, 12] + INSTR
    kw = dict(header_len=3, instruction_start=5, instruction_end=9)
    assert matched_control_check(task, HEADER + [31, 32] + INSTR, **kw)["ok"]
    assert not matched_control_check(task, task, **kw)["context_differs"]
    assert not matched_control_check(task, HEADER + [31, 32] + [21, 22, 23, 99], **kw)["instruction_equal"]
    assert not matched_control_check(task, [151644, 872, 199, 31, 32] + INSTR, **kw)["header_equal"]


def test_boundary_record_with_character_vocabulary():
    vocab = {1: "<|im_start|>", 2: "user", 3: "\n", 4: "Rule.", 5: "\n\n", 6: "Ship", 7: "ment"}

    def decode(ids):
        return "".join(vocab[int(t)] for t in ids)

    row = boundary_record([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7], decode, n_look=2)
    assert row["prefix_matches_a"] and row["ends_with_paragraph_break"] and row["no_merge_across_boundary"]
    assert row["next_a_text"] == "Shipment"
    assert not boundary_record([1, 2, 3, 4], [1, 2, 3, 4, 5, 6], decode)["ends_with_paragraph_break"]


@pytest.mark.skipif(not NBOUND.is_file(), reason="nbound construction not available")
def test_saved_prefixes_are_public_and_matched():
    contexts = json.loads(NBOUND.read_text(encoding="utf-8"))["contexts"]
    examples = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    task_ids = {t: recs["instr_generic"]["ids"] for t, recs in contexts.items()}
    prefixes = neutral_prefixes(contexts, task_ids)
    assert {t: len(e["task"]["ids"]) for t, e in prefixes.items()} == {"L1": 140, "L2": 145, "L3": 162}
    for template, entry in prefixes.items():
        for ctx in CONTEXTS:
            assert len(entry[ctx]["ids"]) == len(entry["task"]["ids"])
        for key in ("instr_generic", "instr_neutral_1", "instr_neutral_2"):
            text = contexts[template][key]["decoded_prefix"]
            assert private_prefix_hits(text, examples) == [], (template, key)
            assert private_prefix_hits(text + "Shipment request: 40 units", examples)
            assert private_prefix_hits(text + "Carrier N is open", examples)
            assert private_prefix_hits(text + "\nZ: 12 units", examples)
    broken = dict(task_ids)
    broken["L2"] = list(broken["L2"])[:-1] + [0]
    with pytest.raises(RuntimeError):
        neutral_prefixes(contexts, broken)


def _rows(labels_by_cond: dict, all12_by_cond: dict) -> list[dict]:
    rows = []
    for cond in pp.CONDS:
        for i in range(9):
            n = labels_by_cond[cond][i]
            gold = {f"C{j}": ("eligible" if j % 2 else "ineligible") for j in range(12)}
            pred = {c: (g if j < n else ("ineligible" if g == "eligible" else "eligible")) for j, (c, g) in enumerate(gold.items())}
            rows.append(
                {
                    "example_id": f"ex{i}",
                    "condition": cond,
                    "difficulty": "easy",
                    "gold_status": gold,
                    "plan_pred_status": pred,
                    "plan_n_status_correct": n,
                    "plan_all_status_correct": all12_by_cond[cond][i],
                    "correct": True,
                    "parse_ok": True,
                    "plan_text": f"{cond}-{i}",
                    "raw_text": "",
                    "template_id": "L1",
                    "prefix_prepare_s": 0.5 if pp.SPEC[cond]["context"] else None,
                    "prefix_resident_mib": 19.8 if pp.SPEC[cond]["context"] else None,
                }
            )
    return rows


def test_summary_counts_pairs_and_recovery():
    labels = {c: [7] * 9 for c in pp.CONDS}
    all12 = {c: [False] * 9 for c in pp.CONDS}
    labels[pp.CANDIDATE] = [12] * 9
    all12[pp.CANDIDATE] = [True] * 9
    labels["full_kv"] = [12] * 9
    all12["full_kv"] = [True] * 9
    rows = _rows(labels, all12)
    recon = [dict(r, condition="generic_r") for r in rows if r["condition"] == pp.BASELINE]
    s = pp._summarize(rows, recon_rows=recon)
    assert s["n_rows"] == 63 and s["n_expected_rows"] == 63
    assert s["labels"][pp.BASELINE]["labels_correct"] == 63
    assert s["labels"][pp.CANDIDATE]["cases_all_twelve_correct"] == 9
    assert s["meets_recovery_standard"][pp.CANDIDATE] and not s["meets_recovery_standard"][pp.FRESH]
    tvb = s["paired_by_case"]["task_vs_baseline"]
    assert tvb["all_twelve_correct"]["n_cases"] == 9
    assert tvb["label_changes"]["fixed"] == 45 and tvb["label_changes"]["broken"] == 0
    assert s["vs_baseline_label_changes"][pp.CANDIDATE]["fixed"] == 45
    rep = s["reproduction_vs_recon12"][pp.BASELINE]
    assert rep["n_compared"] == 9 and rep["labels_here"] == rep["labels_reference"] == 63
    assert s["costs"][pp.CANDIDATE]["prefix_resident_mib_by_template"] == {"L1": 19.8}
    assert "not independent" in s["stats_note"]


def test_conditions_and_comparisons_are_registered():
    assert len(pp.CONDS) == 7 and pp.CONDS[0] == pp.BASELINE
    assert set(pp.LONG) == {pp.CANDIDATE, "woodland_prefix_r", "library_prefix_r", pp.FRESH}
    names = {n for n, _l, _r in pp.PAIRED}
    for needed in ("task_vs_baseline", "task_vs_woodland", "task_vs_library", "fresh_vs_transferred_task_prefix", "full_kv_vs_task", "text_vs_task"):
        assert needed in names
    pinned = json.loads((REPO / "configs" / "pinned.json").read_text(encoding="utf-8"))
    block = pinned["study_plan4_publicprefix12"]
    assert list(block["conditions"]) == list(pp.CONDS)
    assert set(pp.INTERPRETATION_GUIDE) == set(block["outcome_meanings"])
    assert "results_stage_plan4_publicprefix12" not in pp.PRESERVED
    assert "stage_plan4_publicprefix12" in str(pp._paths()[0])


@pytest.mark.parametrize(
    "argv",
    [
        ["--split", "plan4_dev"],
        ["--max_new_tokens", "64"],
        ["--plan_max_new_12", "512"],
    ],
)
def test_main_refuses_changed_split_or_allowances(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["stage"] + argv)
    assert pp.main() == 2
