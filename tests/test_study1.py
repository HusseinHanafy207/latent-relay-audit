"""Study 1 four-condition comparison. No GPU."""

from __future__ import annotations

import inspect

import torch

from src.cache_utils import clone_cache_blob, n_kv_positions
from src.parse_plan import score_eligibility_message
from src.plan_task import (
    N_PLAN_FAMILIES,
    PLAN_DEV_SEED,
    generate_split,
    gold_eligibility_statuses,
    mismatch_source_id,
)
from src.prompts_plan import (
    b_answer_after_message_plan_prompt,
    b_reason_with_message_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
)
from src.prompts_study1 import (
    INTERMEDIATE_HANDOFF,
    SOURCE_EVIDENCE_HANDOFF,
    a_local_task_prompt,
    b_kv_answer_prompt,
    b_kv_reason_prompt,
    intermediate_handoff_prompt,
    source_evidence_handoff_prompt,
    source_fact_coverage,
)
from src.stage_plan0 import A_MESSAGE_MAX_NEW, B_ANSWER_MAX_NEW, B_PLAN_MAX_NEW
from src.stage_study1 import (
    BASELINE_CONDS,
    CONDS,
    CORE_CONDS,
    LATENT_STEPS_C40,
    _delta_block,
    _paths,
    _process_time_s,
    _summarize,
)


def test_core_conditions_are_the_milestone_and_defer_compression():
    assert CORE_CONDS == ("text_c0", "text_c40", "kv_c0", "kv_c40")
    assert CONDS == CORE_CONDS + BASELINE_CONDS
    assert "source_evidence" in CONDS
    assert "no_message" in CONDS
    assert "mismatch_text_c40" in CONDS
    assert "mismatch_kv_c40" in CONDS
    source = inspect.getsource(__import__("src.stage_study1", fromlist=["main"]))
    assert "hop_retrieve" not in source
    assert "retrieve_two_records" not in source
    assert "make_compressor" not in source
    assert all("headwise" not in name and "hobf" not in name for name in CONDS)
    assert not any(name.startswith("compress") or "noise" in name for name in CONDS)
    assert LATENT_STEPS_C40 == 40
    data, out_root, _kaggle = _paths()
    assert data.name == "plan_dev.jsonl"
    assert "stage_study1" in str(out_root)
    assert "stage_plan0" not in str(out_root)


def test_c0_encode_is_local_task_without_handoff_or_b_prices():
    ex = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[0]
    prompt = a_local_task_prompt(ex)
    assert prompt_contains_a_private(prompt, ex) or ex.a_text in prompt
    assert not prompt_contains_b_private(prompt, ex)
    assert INTERMEDIATE_HANDOFF not in prompt
    assert SOURCE_EVIDENCE_HANDOFF not in prompt
    assert "Now write a message" not in prompt
    assert "Do not write a message yet" in prompt
    assert "P: <eligible or ineligible>" not in prompt
    if "$" not in ex.a_text:
        assert "$" not in prompt


def test_text_conditions_share_one_intermediate_handoff():
    assert intermediate_handoff_prompt() == INTERMEDIATE_HANDOFF
    assert intermediate_handoff_prompt() is not source_evidence_handoff_prompt()
    assert "supporting conditions" in INTERMEDIATE_HANDOFF
    assert "P: eligible" not in INTERMEDIATE_HANDOFF
    assert "Q: ineligible" not in INTERMEDIATE_HANDOFF
    assert "R: eligible" not in INTERMEDIATE_HANDOFF
    assert A_MESSAGE_MAX_NEW == 256
    assert B_PLAN_MAX_NEW == 128
    assert B_ANSWER_MAX_NEW == 48
    source = inspect.getsource(__import__("src.stage_study1", fromlist=["main"]))
    assert source.count("intermediate_handoff_prompt()") >= 2
    assert "text_c0" in source and "text_c40" in source


def test_source_evidence_is_a_selected_facts_not_a_retriever():
    assert "Leave the eligibility judgment to B" in SOURCE_EVIDENCE_HANDOFF
    assert "Do not write eligible or ineligible" in SOURCE_EVIDENCE_HANDOFF
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    combine = next(ex for ex in examples if ex.difficulty == "combine")
    covered = source_fact_coverage(combine.a_text, combine)
    assert covered["all_carriers_complete"]
    assert covered["n_carriers_named"] == 3
    omitted = source_fact_coverage("P looks fine.", combine)
    assert not omitted["all_carriers_complete"]
    assert omitted["n_carriers_complete"] < 3
    labeled = source_fact_coverage("P: eligible | enough boxes", combine)
    assert labeled["wrote_eligibility_label"]
    gold = gold_eligibility_statuses(combine)
    scored = score_eligibility_message(combine.a_text, gold)
    assert scored["n_status_parsed"] == 0


def test_b_reason_procedure_is_matched_across_text_and_kv():
    ex = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[0]
    message = "P: eligible | capacity, refrigeration, and driver are satisfied."
    text_plan = b_reason_with_message_prompt(ex, message)
    text_ans = b_answer_after_message_plan_prompt(ex, message, "notes")
    kv_plan = b_kv_reason_prompt(ex)
    kv_ans = b_kv_answer_prompt(ex, "notes")
    for prompt in (text_plan, kv_plan):
        assert "Do not write an Answer line yet" in prompt
        assert prompt_contains_b_private(prompt, ex)
        assert not prompt_contains_a_private(prompt, ex)
        assert "Stop after that line." not in prompt
    for prompt in (text_ans, kv_ans):
        assert "Answer: <letter>) <carrier>" in prompt
        assert prompt_contains_b_private(prompt, ex)
        assert not prompt_contains_a_private(prompt, ex)


def test_mismatch_comes_from_another_family():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    ex = examples[0]
    src = mismatch_source_id(ex, examples)
    assert src.endswith("-base")
    other = next(item for item in examples if item.example_id == src)
    assert other.family_id != ex.family_id


def test_clone_cache_blob_does_not_share_storage():
    blob = {"layers": [(torch.zeros(1, 2, 4, 3), torch.ones(1, 2, 4, 3))]}
    other = clone_cache_blob(blob)
    other["layers"][0][0].add_(5)
    assert float(blob["layers"][0][0].sum()) == 0.0
    assert n_kv_positions(blob) == 4
    assert n_kv_positions({"layers": []}) == 0


def test_delta_kv_and_delta_text_are_c40_minus_c0():
    rows = []
    for i, eid in enumerate(("ex-a", "ex-b", "ex-c", "ex-d")):
        rows.append(
            {
                "example_id": eid,
                "family_id": eid,
                "variant": "base",
                "difficulty": "combine",
                "condition": "text_c0",
                "correct": i < 1,
                "follows_trap": False,
                "parse_ok": True,
                "truncated": False,
                "sender_truncated": False,
                "feasible_affordable": False,
                "message_bytes": 10,
            }
        )
        rows.append(
            {
                "example_id": eid,
                "family_id": eid,
                "variant": "base",
                "difficulty": "combine",
                "condition": "text_c40",
                "correct": i < 3,
                "follows_trap": False,
                "parse_ok": True,
                "truncated": False,
                "sender_truncated": False,
                "feasible_affordable": False,
                "message_bytes": 12,
            }
        )
        rows.append(
            {
                "example_id": eid,
                "family_id": eid,
                "variant": "base",
                "difficulty": "combine",
                "condition": "kv_c0",
                "correct": i < 2,
                "follows_trap": False,
                "parse_ok": True,
                "truncated": False,
                "sender_truncated": False,
                "feasible_affordable": False,
                "message_bytes": 1000,
                "n_kv_positions": 80,
                "n_latent_positions": 0,
            }
        )
        rows.append(
            {
                "example_id": eid,
                "family_id": eid,
                "variant": "base",
                "difficulty": "combine",
                "condition": "kv_c40",
                "correct": i < 3,
                "follows_trap": False,
                "parse_ok": True,
                "truncated": False,
                "sender_truncated": False,
                "feasible_affordable": False,
                "message_bytes": 1100,
                "n_kv_positions": 120,
                "n_latent_positions": 40,
            }
        )
    report = _summarize(rows)
    assert report["delta_text"]["delta"] == 0.5
    assert report["delta_kv"]["delta"] == 0.25
    assert report["delta_text"]["b_fixes"] == 2
    assert report["delta_kv"]["b_fixes"] == 1
    assert "additional latent processing" in report["delta_kv"]["interpretation"]
    wrapped = _delta_block(report["paired"]["kv_c40_vs_kv_c0"])
    assert wrapped["delta"] == 0.25


def test_process_time_counts_rollout_and_leaves_compression_zero():
    row = {
        "encode_c0_s": 1.0,
        "rollout_c40_s": 2.0,
        "sender_generation_time_s": 3.0,
        "compression_time_s": 0.0,
        "plan_time_s": 4.0,
        "receiver_latency_s": 5.0,
    }
    assert _process_time_s(row) == 15.0
    text_c0 = dict(row)
    text_c0["rollout_c40_s"] = 0.0
    assert _process_time_s(text_c0) == 13.0


def test_stage_study1_refuses_anything_but_plan_dev(monkeypatch):
    from src.stage_study1 import main

    monkeypatch.setattr("sys.argv", ["stage_study1", "--split", "plan_fresh"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_study1", "--split", "plan_test"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_study1", "--split", "test"])
    assert main() == 2
