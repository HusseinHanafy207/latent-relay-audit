"""Result-region K/V swap on the twelve-carrier generic-support message."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from src.cache_utils import cache_slice_delta, mix_cache_kv_at_positions
from src.span_select import caches_equal
from src.stage_plan4_kvmix12 import (
    CONDS,
    INTERPRETATION_GUIDE,
    PRESERVED,
    SPEC,
    _summarize,
    main,
)
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

RECON = Path("results_stage_plan4_recon12/evaluations.jsonl")
QUERY = Path("results_stage_plan4_query12/evaluations.jsonl")


def _blob(seq: int, key_fill, value_fill):
    return {
        "layers": [
            (
                torch.tensor(key_fill, dtype=torch.float16).view(1, 1, seq, 1).repeat(1, 2, 1, 3),
                torch.tensor(value_fill, dtype=torch.float16).view(1, 1, seq, 1).repeat(1, 2, 1, 3),
            )
        ]
    }


def test_swap_replaces_only_result_region_component():
    support, result = [0, 1, 2], [3, 4]
    transfer = _blob(5, [1, 1, 1, 2, 2], [5, 5, 5, 6, 6])
    fresh = _blob(5, [1, 1, 1, 8, 8], [5, 5, 5, 9, 9])
    src = {"transfer": transfer, "fresh": fresh}
    for cond in CONDS:
        spec = SPEC[cond]
        mixed = mix_cache_kv_at_positions(src[spec["keys"]], src[spec["values"]], base=transfer, positions=result)
        assert cache_slice_delta(mixed, transfer, support)["equal"]
        assert cache_slice_delta(mixed, src[spec["keys"]], result)["keys_equal"]
        assert cache_slice_delta(mixed, src[spec["values"]], result)["values_equal"]
    assert caches_equal(mix_cache_kv_at_positions(transfer, transfer, base=transfer, positions=result), transfer)
    assert caches_equal(mix_cache_kv_at_positions(fresh, fresh, base=transfer, positions=result), fresh)
    fk_tv = mix_cache_kv_at_positions(fresh, transfer, base=transfer, positions=result)
    assert not cache_slice_delta(fk_tv, transfer, result)["keys_equal"]
    assert cache_slice_delta(fk_tv, transfer, result)["values_equal"]


def test_stage_keeps_construction_frozen(monkeypatch):
    assert CONDS == ("tk_tv", "fk_tv", "tk_fv", "fk_fv")
    assert SPEC["tk_tv"]["reference"] == "generic_r"
    assert SPEC["fk_fv"]["reference"] == "generic_fresh_r"
    assert {"results_stage_plan4_query12", "results_stage_plan4_recon12"} <= set(PRESERVED)
    assert "do not establish" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_kvmix12", fromlist=["main"]))
    assert "mix_sender_kv_at" in source
    assert "rerotate" not in source.replace("RoPE is not applied again", "")
    assert "clone_sender" in source
    assert "generate_text_keep_cache" not in source
    mix_src = inspect.getsource(TwoAgentInventory.mix_sender_kv_at)
    assert "rerotate" not in mix_src
    monkeypatch.setattr("sys.argv", ["stage_plan4_kvmix12", "--split", "plan4_hold"])
    assert main() == 2


def test_summary_reproduction_and_reporting_from_references():
    recon = _load_rows(RECON)
    query12 = _load_rows(QUERY)
    rename = {"generic_r": "tk_tv", "generic_fresh_r": "fk_fv"}
    twelve = [{**r, "condition": rename[r["condition"]]} for r in recon if r["condition"] in rename]
    query = [{**r, "condition": rename[r["condition"]]} for r in query12 if r["condition"] in rename]
    s = _summarize(twelve, query, recon_rows=recon, query12_rows=query12)
    assert s["n_twelve_expected"] == 36 and s["n_query_expected"] == 432
    labels = s["twelve_line"]["labels"]
    assert labels["tk_tv"]["labels_correct"] == 63
    assert labels["fk_fv"]["labels_correct"] == 106
    assert labels["tk_tv"]["n_labels"] == 108
    assert labels["tk_tv"]["ineligible_marked_eligible"] == 44
    assert s["twelve_line"]["final_choice"]["fk_fv"]["final_choice_correct"] == 4
    assert s["single_label"]["labels"]["tk_tv"]["correct"] == 54
    assert s["single_label"]["labels"]["fk_fv"]["correct"] == 102
    for cond in ("tk_tv", "fk_fv"):
        rep = s["reproduction"][cond]
        assert rep["twelve_line_vs_recon12"]["identical_outputs"] == 9
        assert rep["single_label_vs_query12"]["identical_outputs"] == 108
    cross = s["twelve_line"]["label_crosstab"]["fk_fv_vs_tk_tv"]
    assert cross["fixed"] == 43 and cross["broken"] == 0
    qcross = s["single_label"]["label_crosstab"]["fk_fv_vs_tk_tv"]
    assert qcross["fixed"] == 48 and qcross["broken"] == 0
    delta = s["twelve_line"]["paired_by_case"]["fk_fv_vs_tk_tv"]["labels_correct"]
    assert delta["total_delta"] == 43
    assert "cases_increased" in delta and "cases_better" not in delta
    assert "Final-choice" in s["stats_note"]
