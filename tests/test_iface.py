"""Prefix-logit stats, RoPE re-rotation, and the development interface factorial."""

from __future__ import annotations

import inspect

import torch

from src.prefix_logits import logit_delta_stats, within_numerical_baseline
from src.prompts_plan4 import (
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_answer_kv_prompt,
    b_reason_kv_instruct_prompt,
    b_reason_kv_prompt,
    prompt_contains_a_private,
    prompt_contains_solver_labels,
)
from src.rope_rerotate import (
    apply_rope,
    dummy_cos_sin,
    identity_rerotate_stats,
    invert_rope,
    rerotate_keys,
    rotate_half,
)
from src.stage_plan4_iface import (
    COMPACT,
    CONDS,
    DIAG,
    LABEL_MARGIN,
    PRESERVED,
    REFS,
    SPEC,
    candidate_qualifies,
    choose_candidate,
    main,
)
from src.plan4_task import (
    N_PLAN4_FAMILIES,
    PLAN4_DEV_SEED,
    gold_eligibility_statuses,
    generate_split,
)


def test_rotate_half_then_invert_recovers():
    x = torch.randn(1, 2, 4, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x)


def test_rerotate_from_p_to_q_matches_direct_q():
    k0 = torch.randn(1, 2, 4, 8)
    cos_old, sin_old = dummy_cos_sin([11, 20, 35, 40], 8)
    cos_new, sin_new = dummy_cos_sin([0, 1, 2, 3], 8)
    c_old, s_old = cos_old.unsqueeze(1), sin_old.unsqueeze(1)
    c_new, s_new = cos_new.unsqueeze(1), sin_new.unsqueeze(1)
    k_old = apply_rope(k0, c_old, s_old)
    got = rerotate_keys(k_old, cos_old, sin_old, cos_new, sin_new)
    want = apply_rope(k0, c_new, s_new)
    assert torch.allclose(got.float(), want, atol=1e-5, rtol=1e-5)
    recovered = invert_rope(k_old.float(), c_old.float(), s_old.float())
    assert torch.allclose(recovered, k0, atol=1e-5, rtol=1e-5)


def test_identity_rerotate_preserves_keys():
    k0 = torch.randn(1, 2, 4, 8)
    cos, sin = dummy_cos_sin([11, 20, 35, 40], 8)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    keys = apply_rope(k0, c, s)
    stats = identity_rerotate_stats(keys, cos, sin)
    assert stats["finite"]
    assert stats["max_abs"] < 1e-5


def test_forward_and_back_rerotate_recovers_keys():
    k0 = torch.randn(1, 2, 4, 8)
    cos_p, sin_p = dummy_cos_sin([11, 20, 35, 40], 8)
    cos_q, sin_q = dummy_cos_sin([0, 1, 2, 3], 8)
    k_p = apply_rope(k0, cos_p.unsqueeze(1), sin_p.unsqueeze(1))
    k_q = rerotate_keys(k_p, cos_p, sin_p, cos_q, sin_q)
    k_back = rerotate_keys(k_q, cos_q, sin_q, cos_p, sin_p)
    assert torch.allclose(k_back.float(), k_p.float(), atol=1e-5, rtol=1e-5)


def test_rerotate_rejects_nonfinite_keys():
    k = torch.zeros(1, 1, 2, 4)
    k[0, 0, 0, 0] = torch.inf
    cos, sin = dummy_cos_sin([0, 1], 4)
    try:
        rerotate_keys(k, cos, sin, cos, sin)
    except RuntimeError as err:
        assert "nonfinite" in str(err).lower()
    else:
        raise AssertionError("nonfinite keys must fail")


def test_logit_baseline_gate():
    a = torch.zeros(4, 6)
    b = a + 0.01
    c = a + 5.0
    repeat = logit_delta_stats(a, b)
    ordinary = logit_delta_stats(a, b)
    probe_ok = logit_delta_stats(a, a + 0.02)
    probe_bad = logit_delta_stats(a, c)
    assert within_numerical_baseline(probe_ok, repeat=repeat, ordinary=ordinary)
    assert not within_numerical_baseline(probe_bad, repeat=repeat, ordinary=ordinary)


def test_compact_conditions_are_the_factorial():
    layouts = {"original", "consecutive"}
    framings = {"baseline", "instruct"}
    positions = {"result", "result_sink"}
    seen = set()
    for name in COMPACT:
        spec = SPEC[name]
        seen.add((spec["layout"], spec["framing"], spec["positions"]))
        assert spec["kind"] == "kv"
    assert seen == {(layout, framing, pos) for layout in layouts for framing in framings for pos in positions}
    assert CONDS == REFS + COMPACT + DIAG
    assert SPEC["orig_base_four"]["positions"] == "sink"
    assert SPEC["kv"]["kind"] == "full"
    assert SPEC["text_result"]["kind"] == "text"
    assert LABEL_MARGIN == 5.0


def test_qualification_rule_and_smallest_message():
    kv = {"n": 90, "parse_ok": 90, "truncated": 0, "plan_all_status_correct": 81}
    text = {"n": 90, "parse_ok": 90, "truncated": 0, "plan_all_status_correct": 81}
    good = {
        "n": 90,
        "parse_ok": 90,
        "truncated": 0,
        "plan_all_status_correct": 77,
        "mean_message_bytes": 2e6,
        "mean_process_time_s": 4.0,
    }
    better = dict(good)
    better["mean_message_bytes"] = 1e6
    truncated = dict(good)
    truncated["truncated"] = 1
    low = dict(good)
    low["plan_all_status_correct"] = 70
    assert candidate_qualifies(good, kv, text)
    assert not candidate_qualifies(truncated, kv, text)
    assert not candidate_qualifies(low, kv, text)
    decision = choose_candidate(
        {
            "kv": kv,
            "text_result": text,
            "orig_base_sink": good,
            "cons_base_sink": better,
            **{name: low for name in COMPACT if name not in {"orig_base_sink", "cons_base_sink"}},
        }
    )
    assert decision["selected"] == "cons_base_sink"


def test_instruct_framing_does_not_add_task_answers():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    ex = examples[0]
    base = b_reason_kv_prompt(ex)
    instruct = b_reason_kv_instruct_prompt(ex)
    assert FRAMING_INSTRUCTION not in base
    assert FRAMING_INSTRUCTION in instruct
    assert FRAMING_INSTRUCTION in b_answer_kv_instruct_prompt(ex, "notes")
    assert FRAMING_INSTRUCTION not in b_answer_kv_prompt(ex, "notes")
    assert not prompt_contains_a_private(instruct, ex)
    assert not prompt_contains_solver_labels(instruct, ex)
    gold = gold_eligibility_statuses(ex)
    for carrier, status in gold.items():
        assert f"{carrier}: q=" not in instruct


def test_iface_uses_dev_and_keeps_hold_out(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_iface", fromlist=["main"]))
    assert "plan4_dev.jsonl" in source
    assert "plan4_hold.jsonl" in source
    assert "do_not_load" in source
    assert "results_stage_plan4_hold" in PRESERVED
    assert "results_stage_plan4_stable" in PRESERVED
    assert "keep_shared_positions" in source
    assert "compare_prefix_logits" in source
    assert source.count("_decode_fn(runner)") == 2
    assert "_decode_fn(tokenizer)" not in source
    assert "for ex in examples:\n        assert_plan4_invariants(ex)" in source
    assert "layout" in source
    assert "FRAMING_INSTRUCTION" in source
    from src.stage_plan4_iface import _paths

    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_iface" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_iface", "--split", "plan4_hold"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_iface", "--split", "plan4_fresh"])
    assert main() == 2
