"""Reference-attention replay on the twelve-carrier transferred cache."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from src.attn_replay import (
    CONTROLLER,
    dist_stats,
    label_decision,
    relay_attention_forward,
    repeat_kv,
    token_index_at_char,
)
from src.plan4_timing_task import load_timing12_jsonl
from src.stage_plan4_refattn12 import (
    BRANCHES,
    INTERPRETATION_GUIDE,
    PRESERVED,
    REFERENCE,
    _summarize,
    label_tokens,
    main,
    response_labels,
)
from src.stage_select import _load_rows

KVMIX = Path("results_stage_plan4_kvmix12/evaluations.jsonl")


def _manual(query, key, value, mask, scaling, groups):
    k = repeat_kv(key, groups).float()
    v = repeat_kv(value, groups).float()
    w = torch.softmax(torch.matmul(query.float(), k.transpose(2, 3)) * scaling + mask, dim=-1)
    return w, torch.matmul(w, v)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    b, hq, hkv, q, kv, d = 2, 4, 2, 3, 7, 5
    query = torch.randn(b, hq, q, d, generator=g)
    key = torch.randn(b, hkv, kv, d, generator=g)
    value = torch.randn(b, hkv, kv, d, generator=g)
    allowed = torch.ones(q, kv, dtype=torch.bool)
    for i in range(q):
        allowed[i, kv - q + i + 1 :] = False
    mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min).view(1, 1, q, kv).expand(b, 1, q, kv)
    return query, key, value, allowed, mask


def test_off_mode_matches_eager_attention_with_gqa_and_both_mask_types():
    query, key, value, allowed, mask = _inputs()
    module = SimpleNamespace(num_key_value_groups=2, layer_idx=3)
    CONTROLLER.reset("off")
    out, weights = relay_attention_forward(module, query, key, value, mask, scaling=0.5)
    _w, ref = _manual(query, key, value, mask, 0.5, 2)
    assert weights is None
    assert torch.allclose(out, ref.transpose(1, 2), atol=1e-6)
    bool_mask = allowed.view(1, 1, *allowed.shape).expand(2, 1, *allowed.shape)
    out_b, _ = relay_attention_forward(module, query, key, value, bool_mask, scaling=0.5)
    assert torch.allclose(out_b, ref.transpose(1, 2), atol=1e-6)
    assert CONTROLLER.n_calls == 2 and CONTROLLER.n_replaced == 0 and CONTROLLER.layers_seen == {3}


def test_replace_gives_target_the_donor_weights_on_its_own_values():
    query, key, value, _allowed, mask = _inputs(1)
    module = SimpleNamespace(num_key_value_groups=2, layer_idx=0)
    CONTROLLER.reset("replace", donor=0, target=1)
    out, _ = relay_attention_forward(module, query, key, value, mask, scaling=0.5)
    w, ref = _manual(query, key, value, mask, 0.5, 2)
    v1 = repeat_kv(value, 2).float()[1]
    expected_target = torch.matmul(w[0], v1)
    assert torch.allclose(out[1], expected_target.transpose(0, 1), atol=1e-6)
    assert torch.allclose(out[0], ref[0].transpose(0, 1), atol=1e-6)
    assert not torch.allclose(out[1], ref[1].transpose(0, 1), atol=1e-4)
    assert CONTROLLER.n_replaced == 1
    same = torch.stack([query[1], query[1]]), torch.stack([key[1], key[1]]), torch.stack([value[1], value[1]])
    CONTROLLER.reset("replace")
    self_out, _ = relay_attention_forward(module, *same, mask, scaling=0.5)
    CONTROLLER.reset("off")
    plain, _ = relay_attention_forward(module, *same, mask, scaling=0.5)
    assert torch.equal(self_out, plain)


def test_helpers():
    offsets = [(0, 1), (1, 2), (2, 12), (12, 15)]
    assert token_index_at_char(offsets, 2) == 2
    assert token_index_at_char(offsets, 11) == 2
    assert token_index_at_char(offsets, 15) is None
    logits = torch.zeros(1, 6)
    logits[0, 4] = 3.0
    assert label_decision(logits[0], 4, 5)["decision"] == "eligible"
    assert label_decision(logits[0], 5, 4)["decision"] == "ineligible"
    st = dist_stats(logits, logits)
    assert float(st["kl"].max()) == 0.0 and bool(st["agree"].all())


def test_saved_reference_plans_locate_all_twelve_labels():
    examples = {ex.example_id: ex for ex in load_timing12_jsonl(Path("data/plan4_timing12.jsonl"))}
    refs = [r for r in _load_rows(KVMIX) if r["condition"] == REFERENCE]
    assert len(refs) == 9
    for row in refs:
        text = row["plan_text"]
        offsets = [(i, i + 1) for i in range(len(text))]
        ids = [ord(c) for c in text]
        labels = response_labels(text, examples[row["example_id"]].carriers, offsets, ids)
        assert len(labels) == 12
        assert {l["carrier"] for l in labels} == set(examples[row["example_id"]].carriers)
        assert all(text[l["token_index"]] in "ei" for l in labels)
        assert sorted(l["ref_text_label"] for l in labels) == sorted(row["plan_pred_status"].values())
        tok_e, tok_i = label_tokens(labels, tokenizer=None)
        assert (tok_e, tok_i) == (ord("e"), ord("i"))


def test_stage_is_a_matched_oracle_diagnostic(monkeypatch):
    assert REFERENCE == "fk_fv"
    assert BRANCHES == ("tk_tv", "fk_tv", "fk_fv", "tk_tv_refattn", "tk_tv_selfattn")
    assert "results_stage_plan4_kvmix12" in PRESERVED
    assert "not free-generation" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_refattn12", fromlist=["main"]))
    assert "full_kv" not in source
    assert "replay_attention" in source
    assert "logits_to_keep" in source
    replay = inspect.getsource(__import__("src.attn_replay", fromlist=["register"]))
    assert "AttentionMaskInterface.register" in replay
    monkeypatch.setattr("sys.argv", ["stage_plan4_refattn12", "--split", "plan4_dev"])
    assert main() == 2


def test_summary_recovery_and_grouping():
    rows = []
    vals = {"tk_tv": (6, 0.5), "fk_tv": (10, 0.2), "fk_fv": (12, 0.0), "tk_tv_refattn": (11, 0.1), "tk_tv_selfattn": (6, 0.5)}
    for i in range(9):
        for b, (correct, kl) in vals.items():
            rows.append(
                {
                    "example_id": f"e{i}",
                    "difficulty": "L1",
                    "branch": b,
                    "n_labels": 12,
                    "labels_correct": correct,
                    "labels_agree_ref": correct,
                    "ineligible_marked_eligible": 12 - correct,
                    "eligible_marked_ineligible": 0,
                    "mean_kl": kl,
                    "mean_label_kl": kl,
                    "argmax_agree": 1.0 - kl,
                    "mean_nll_saved_tokens": kl,
                    "mean_gold_margin": 1.0,
                    "instrumentation_ok": True,
                }
            )
    s = _summarize(rows, check={"all_passed": True}, compare_checks=[{"ok": True}])
    assert s["instrumentation_ok"] is True
    assert s["branches"]["tk_tv_refattn"]["labels_correct"] == 99
    rec = s["recovery_toward_reference"]
    assert abs(rec["label_gap_closed"] - 5 / 6) < 1e-9
    assert abs(rec["kl_reduction"] - 0.8) < 1e-9
    d = s["paired_by_case"]["tk_tv_refattn_vs_tk_tv"]["labels_correct"]
    assert d["total_delta"] == 45 and d["cases_increased"] == 9
    assert "not free-generation" in s["measurement_note"]
