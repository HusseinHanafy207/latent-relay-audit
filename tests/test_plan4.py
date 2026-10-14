"""Four-carrier generator, solver, and sender-calibration wiring. No GPU."""

from __future__ import annotations

import inspect
import json
from collections import Counter

import pytest

from src.parse_plan4 import (
    cheapest_named_in_plan,
    handoff_from_written,
    is_think_collapse,
    parse_plan4_answer,
    plan_extraction,
    score_b_plan,
    score_sender_message,
    selected_content_flags,
    written_working_metrics,
)
from src.plan4_task import (
    CARRIERS,
    LETTERS,
    N_PLAN4_FAMILIES,
    N_PLAN4_FRESH_FAMILIES,
    N_PLAN4_HOLD_FAMILIES,
    PLAN4_DEV_SEED,
    PLAN4_HOLD_SEED,
    PLAN4_DIAG_FAMILIES,
    PLAN4_KVDIAG_FAMILIES,
    QUESTION,
    a_state_key,
    assert_plan4_invariants,
    carrier_result,
    effective_capacity,
    generate_hold_disjoint,
    generate_split,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
    overlap_with_prior,
    prior_plan4_examples,
    oracle_intermediate_message,
    productive_hours,
    sheet_fingerprint,
    subset_plan4_diag,
    subset_plan4_kvdiag,
)
from src.prompts_plan4 import (
    ELIGIBILITY_RULE,
    a_local_task_prompt,
    b_answer_after_brief_plan_prompt,
    b_answer_after_message_plan_prompt,
    b_answer_kv_prompt,
    b_answer_no_message_prompt,
    b_reason_brief_prompt,
    b_reason_kv_prompt,
    b_reason_no_message_prompt,
    b_reason_with_message_prompt,
    b_need_probe_prompt,
    compact_result_handoff,
    fuller_text_handoff,
    ordinary_sender_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.stage_plan4_cal import A_CONDS, B_COND, CONDS, LATENT_STEPS_C40, _paths


def test_level3_worked_example_is_eight_not_twenty():
    facts = {
        "vehicle_capacity": 20,
        "H": 5,
        "tau": 2,
        "b": 1,
        "rate": 4,
        "refrigeration": True,
        "driver": True,
    }
    assert productive_hours(facts, difficulty="L3") == 2
    assert effective_capacity(facts, difficulty="L3") == 8
    rec = carrier_result({"boxes": 12, "refrigerated": True}, facts, difficulty="L3")
    assert rec["q"] == 8
    assert rec["eligible"] is False


def test_plan4_dev_shape_and_balanced_changes():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    assert len(examples) == 90
    bases = [ex for ex in examples if ex.variant == "base"]
    assert len(bases) == 30
    assert Counter(ex.difficulty for ex in bases) == {"L1": 10, "L2": 10, "L3": 10}
    letters = Counter(ex.gold_letter for ex in bases)
    assert set(letters) <= set(LETTERS)
    assert min(letters.values()) >= 6
    families = {ex.family_id for ex in examples}
    assert len(families) == 30
    for level in ("L1", "L2", "L3"):
        level_bases = [ex for ex in bases if ex.difficulty == level]
        assert sum(ex.a_change == "make_eligible" for ex in level_bases) == 5
        assert sum(ex.a_change == "make_ineligible" for ex in level_bases) == 5
        assert sum(ex.b_change == "price_down" for ex in level_bases) == 5
        assert sum(ex.b_change == "price_up" for ex in level_bases) == 5
    for ex in examples:
        assert_plan4_invariants(ex)
        assert ex.split == "plan4_dev"
        assert ex.question == QUESTION
        assert set(ex.choices) == set(CARRIERS)


def test_base_and_b_change_share_a_state_key():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    by_fam = {}
    for ex in examples:
        by_fam.setdefault(ex.family_id, {})[ex.variant] = ex
    for fam in by_fam.values():
        assert fam["base"].a_text == fam["b_flip"].a_text
        assert fam["base"].a_facts == fam["b_flip"].a_facts
        assert a_state_key(fam["base"]) == a_state_key(fam["b_flip"]) == f"{fam['base'].family_id}-base"
        assert a_state_key(fam["a_flip"]) == fam["a_flip"].example_id
        assert fam["a_flip"].b_text == fam["base"].b_text
        assert fam["a_flip"].gold_carrier != fam["base"].gold_carrier
        assert fam["b_flip"].gold_carrier != fam["base"].gold_carrier
        assert fam["b_flip"].gold_carrier == fam["base"].alt_carrier


def test_derived_capacity_is_not_on_a_sheet():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    for ex in examples:
        if ex.difficulty == "L1":
            continue
        prompt = a_local_task_prompt(ex)
        assert "effective capacity" not in ex.a_text.lower()
        assert "q=" not in ex.a_text
        assert prompt_contains_a_private(prompt, ex) or ex.a_text in prompt
        assert not prompt_contains_b_private(prompt, ex)
        assert not prompt_contains_solver_labels(prompt, ex)
        assert compact_result_handoff() not in prompt
        gold_q = gold_quantities(ex)
        for carrier, qty in gold_q.items():
            assert f"q={qty}" not in ex.a_text


def test_oracle_has_quantities_not_booking():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    ex = examples[0]
    oracle = oracle_intermediate_message(ex)
    gold = gold_eligibility_statuses(ex)
    scored = score_sender_message(oracle, gold_quantities(ex), gold)
    assert scored["all_four_correct"]
    assert "book" not in oracle.lower()
    assert "$" not in oracle
    reason = b_reason_with_message_prompt(ex, oracle)
    answer = b_answer_after_message_plan_prompt(ex, oracle, "notes")
    assert oracle in reason and oracle in answer
    assert "Answer: <letter>) <carrier>" in answer
    assert "Stop after that line." not in reason
    assert prompt_contains_b_private(reason, ex)
    assert not prompt_contains_a_private(reason, ex)


def test_sender_message_parser_and_four_carrier_answers():
    gold_q = {"P": 8, "Q": 12, "R": 20, "S": 4}
    gold_s = {"P": "ineligible", "Q": "eligible", "R": "eligible", "S": "ineligible"}
    text = (
        "scratch\n"
        "P: q=8 | ineligible | two productive hours\n"
        "Q: q=12 | eligible | enough boxes\n"
        "R: q=20 | eligible | enough boxes\n"
        "S: q=4 | ineligible | too small\n"
    )
    scored = score_sender_message(text, gold_q, gold_s)
    assert scored["all_four_correct"]
    assert not scored["malformed"]
    parsed = parse_plan4_answer("Answer: D) S")
    assert parsed.parse_ok
    assert parsed.letter == "D"
    assert parsed.carrier == "S"
    empty = parse_plan4_answer("Answer:")
    assert empty.parse_ok is False


def test_latent_rollout_skips_attentions_and_uses_inference_mode():
    from src.two_agent import TwoAgentInventory

    src = inspect.getsource(TwoAgentInventory._run_latent_steps)
    assert "inference_mode" in src
    assert "output_attentions=False" in src or "output_attentions: bool = False" in src


def test_written_handoff_separates_working_from_results():
    compact = compact_result_handoff()
    written = written_result_handoff()
    assert compact != written
    assert "Write exactly four lines" in compact
    assert "Write exactly four lines" not in written
    assert "RESULTS" in written
    assert "Do not start with the four result lines." in written
    assert ELIGIBILITY_RULE in a_local_task_prompt(generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)[0])
    assert ELIGIBILITY_RULE in compact
    assert ELIGIBILITY_RULE in written
    joined = ordinary_sender_prompt("ENCODE", written)
    assert joined.startswith("ENCODE")
    assert written in joined
    assert compact not in joined


def test_written_parser_uses_results_block_and_detects_working():
    gold_q = {"P": 0, "Q": 12, "R": 20, "S": 4}
    gold_s = {"P": "ineligible", "Q": "eligible", "R": "eligible", "S": "ineligible"}
    text = (
        "P hours = max(0, 3-2-1) = 0; q = min(20, 4*0) = 0\n"
        "RESULTS\n"
        "P: q=0 | ineligible | zero productive hours\n"
        "Q: q=12 | eligible | enough boxes\n"
        "R: q=20 | eligible | enough boxes\n"
        "S: q=4 | ineligible | too small\n"
    )
    scored = score_sender_message(text, gold_q, gold_s)
    assert scored["all_four_correct"]
    metrics = written_working_metrics(text)
    assert metrics["has_results_block"]
    assert metrics["has_arithmetic"]
    assert metrics["produced_working"]
    assert metrics["n_working_lines"] >= 1
    four_only = (
        "P: q=20 | eligible | ok\n"
        "Q: q=12 | eligible | ok\n"
        "R: q=20 | eligible | ok\n"
        "S: q=4 | ineligible | ok\n"
    )
    empty = written_working_metrics(four_only)
    assert not empty["produced_working"]
    assert not empty["has_results_block"]
    bold = "working\n**RESULTS**\nP: q=0 | ineligible | x\nQ: q=12 | eligible | x\nR: q=20 | eligible | x\nS: q=4 | ineligible | x\n"
    assert written_working_metrics(bold)["has_results_block"]
    missing_q = (
        "P: 0 | ineligible | x\n"
        "Q: 12 | eligible | x\n"
        "R: 20 | eligible | x\n"
        "S: 4 | ineligible | x\n"
    )
    recovered = score_sender_message(missing_q, gold_q, gold_s)
    assert recovered["malformed"]
    assert recovered["n_missing_q_equals"] == 4
    assert recovered["all_four_correct_optional_q"]
    sent = handoff_from_written(text)
    assert "P hours" not in sent
    assert sent.startswith("P: q=0")
    assert "S: q=4" in sent


def test_b_plan_names_cheapest_and_must_match_answer():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    ex = examples[0]
    oracle = oracle_intermediate_message(ex)
    reason = b_reason_with_message_prompt(ex, oracle)
    answer = b_answer_after_message_plan_prompt(ex, oracle, "Choice: Q at $60")
    assert "Choice: <carrier> at $<price>" in reason
    assert "candidate or dropped" in reason
    assert "Answer: <letter>) <carrier>" not in reason
    assert "Choice carrier from those notes" in answer
    brief = b_reason_brief_prompt(ex, oracle)
    brief_ans = b_answer_after_brief_plan_prompt(ex, oracle, "notes")
    assert "at most a few sentences" in brief
    assert "Choice: <carrier>" not in brief
    assert "Stop after that line." not in brief
    assert "Choice carrier from those notes" not in brief_ans
    assert cheapest_named_in_plan("Q costs $60, R costs $65, and Q is cheapest") == "Q"
    assert cheapest_named_in_plan("the lowest-price carrier is S") == "S"
    assert cheapest_named_in_plan("The lowest price is Q.") == "Q"
    assert cheapest_named_in_plan("Choice: P at $55") == "P"
    missing = plan_extraction("no cheapest named here", "Q")
    assert missing["plan_extraction_status"] == "missing"
    assert missing["plan_answer_agrees"] is None
    disagree = plan_extraction("The lowest price is Q.", "R")
    assert disagree["plan_extraction_status"] == "disagrees"
    agree = plan_extraction("The lowest price is Q.", "Q")
    assert agree["plan_extraction_status"] == "agrees"
    none = b_reason_no_message_prompt(ex)
    assert "No previous agent sent an eligibility message." in none
    assert "Answer: <letter>) <carrier>" not in none
    assert not prompt_contains_a_private(none, ex)
    assert "Choice carrier from those notes" in b_answer_no_message_prompt(ex, "notes")
    kv_plan = b_reason_kv_prompt(ex)
    kv_ans = b_answer_kv_prompt(ex, "Choice: Q at $60")
    assert "candidate or dropped" in kv_plan
    assert "from the relayed memory" in kv_plan
    assert "from the message" not in kv_plan
    assert "Answer: <letter>) <carrier>" not in kv_plan
    assert not prompt_contains_a_private(kv_plan, ex)
    assert "Choice carrier from those notes" in kv_ans
    assert not prompt_contains_a_private(kv_ans, ex)


def test_diag_subset_is_two_families_per_level():
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    selected = subset_plan4_diag(examples)
    assert len(PLAN4_DIAG_FAMILIES) == 6
    assert len(selected) == 18
    bases = [ex for ex in selected if ex.variant == "base"]
    assert {ex.family_id for ex in bases} == set(PLAN4_DIAG_FAMILIES)
    from collections import Counter

    assert Counter(ex.difficulty for ex in bases) == {"L1": 2, "L2": 2, "L3": 2}
    a_flip = next(ex for ex in selected if ex.example_id == "plan4_dev-fam-000-a_flip")
    assert a_flip.a_facts["P"]["H"] == 3
    assert a_flip.a_facts["P"]["tau"] == 2
    assert a_flip.a_facts["P"]["b"] == 1
    assert a_flip.a_facts["P"]["rate"] == 4
    assert a_flip.a_facts["P"]["vehicle_capacity"] == 20
    assert effective_capacity(a_flip.a_facts["P"], difficulty="L3") == 0


def test_cached_continuation_wraps_handoff_as_new_chat():
    from src.two_agent import TwoAgentInventory

    src = inspect.getsource(TwoAgentInventory.generate_from_cache)
    assert "_encode(prompt)" in src
    assert "new chat" in src.lower() or "add_generation_prompt" in inspect.getsource(TwoAgentInventory._encode)


def test_identical_token_cache_check_does_not_rewrap_chat():
    from src.stage_plan4_cachecheck import LEAD
    from src.two_agent import TwoAgentInventory

    src = inspect.getsource(TwoAgentInventory.compare_identical_token_prefix)
    assert "self.generate_from_cache" not in src
    assert "_prefill_prompt" in src
    assert "handoff_is_new_chat" in src
    assert "would be a cache bug" not in LEAD
    assert "would warrant investigation" in LEAD


def test_bprompt_and_cachecheck_preserve_prior_folders(monkeypatch):
    from src.stage_plan4_bprompt import PRESERVED as B_PRESERVED
    from src.stage_plan4_bprompt import _paths as b_paths
    from src.stage_plan4_bprompt import main as b_main
    from src.stage_plan4_cachecheck import PRESERVED as C_PRESERVED
    from src.stage_plan4_cachecheck import _paths as c_paths
    from src.stage_plan4_cachecheck import main as c_main

    assert "results_stage_plan4_cal" in B_PRESERVED and "results_stage_plan4_diag" in B_PRESERVED
    assert "results_stage_plan4_cal" in C_PRESERVED and "results_stage_plan4_diag" in C_PRESERVED
    assert "stage_plan4_bprompt" in str(b_paths()[1])
    assert "stage_plan4_cachecheck" in str(c_paths()[1])
    monkeypatch.setattr("sys.argv", ["stage_plan4_bprompt", "--split", "plan_dev"])
    assert b_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_cachecheck", "--split", "plan_test"])
    assert c_main() == 2


def test_chain_uses_written_a_and_frozen_b(monkeypatch):
    from src.plan4_task import mismatch_source_id
    from src.stage_plan4_chain import CONDS, PRESERVED, _paths as chain_paths, main as chain_main

    source = inspect.getsource(__import__("src.stage_plan4_chain", fromlist=["main"]))
    assert "make_compressor" not in source
    assert "b_reason_brief_prompt" not in source
    assert "b_reason_with_message_prompt" in source
    assert "b_reason_no_message_prompt" in source
    assert "handoff_from_written" in source
    assert CONDS == ("a_written", "collab", "oracle", "no_message", "mismatch")
    assert "kv_c0" not in CONDS
    assert "results_stage_plan4_bprompt" in PRESERVED
    assert "results_stage_plan4_cal" in PRESERVED
    data, out_root, _kaggle = chain_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_chain" in str(out_root)
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    donor = mismatch_source_id(examples[0], examples)
    assert donor.endswith("-base")
    assert donor.rsplit("-", 1)[0] != examples[0].family_id
    monkeypatch.setattr("sys.argv", ["stage_plan4_chain", "--split", "plan_dev"])
    assert chain_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_chain", "--split", "plan_test"])
    assert chain_main() == 2


def test_comm_is_text_vs_full_kv_of_written_a(monkeypatch):
    from src.stage_plan4_comm import CONDS, PRESERVED, _paths as comm_paths, main as comm_main
    from src.two_agent import TwoAgentInventory

    source = inspect.getsource(__import__("src.stage_plan4_comm", fromlist=["main"]))
    keep = inspect.getsource(TwoAgentInventory.generate_text_keep_cache)
    assert "make_compressor" not in source
    assert "continue_latent" not in source
    assert "compact_result_handoff" not in source
    assert "generate_text_keep_cache" in source
    assert "b_reason_with_message_prompt" in source
    assert "b_reason_kv_prompt" in source
    assert "not a latent rollout" in keep
    assert "cache_to_cpu" in keep
    assert CONDS == ("text", "kv", "mismatch_kv")
    assert "kv_c0" not in CONDS
    assert "headwise" not in CONDS
    assert "results_stage_plan4_chain" in PRESERVED
    assert "results_stage_plan4_bprompt" in PRESERVED
    data, out_root, _kaggle = comm_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_comm" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_comm", "--split", "plan_dev"])
    assert comm_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_comm", "--split", "plan_test"])
    assert comm_main() == 2


def test_compress_is_fuller_text_and_two_headwise_budgets(monkeypatch):
    from src.stage_plan4_compress import BUDGET_SCOPE, BUDGETS, CONDS, PRESERVED, SINK_SIZE
    from src.stage_plan4_compress import _paths as z_paths
    from src.stage_plan4_compress import main as z_main
    from src.two_agent import TwoAgentInventory

    source = inspect.getsource(__import__("src.stage_plan4_compress", fromlist=["main"]))
    compress = inspect.getsource(TwoAgentInventory.compress_sender)
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    ex = examples[0]
    written = "RESULTS\nP: q=8 | ineligible | x\nQ: q=12 | eligible | x\nR: q=20 | eligible | x\nS: q=4 | ineligible | x\n"
    fuller = fuller_text_handoff(ex, written)
    assert "Operational information:" in fuller
    assert "Written calculations:" in fuller
    assert ex.a_text.strip() in fuller
    assert "P: q=8" in fuller
    assert prompt_contains_a_private(fuller, ex)
    assert CONDS == ("text", "text_full", "kv", "kv_32", "kv_64")
    assert BUDGETS == {"kv_32": 32, "kv_64": 64}
    assert SINK_SIZE == 4
    assert "entire completed cache" in BUDGET_SCOPE
    assert "no reserved latent tail" in BUDGET_SCOPE
    assert "continue_latent" not in source
    assert "generate_text_keep_cache" in source
    assert "compress_sender" in source
    assert "fuller_text_handoff" in source
    assert "hobf" not in CONDS
    assert "entire completed cache" in compress
    assert "results_stage_plan4_comm" in PRESERVED
    data, out_root, _kaggle = z_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_compress" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_compress", "--split", "plan_dev"])
    assert z_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_compress", "--split", "plan_test"])
    assert z_main() == 2


def test_probe_scores_completed_cache_then_obf_same_budgets(monkeypatch):
    from src.stage_plan4_probe import (
        COMPRESS_CONDS,
        CONDS,
        PREDECLARED_LARGER_BUDGET,
        PRESERVED,
        SINK_SIZE,
        _process_time_s,
    )
    from src.stage_plan4_probe import _paths as p_paths
    from src.stage_plan4_probe import main as p_main
    from src.two_agent import TwoAgentInventory

    source = inspect.getsource(__import__("src.stage_plan4_probe", fromlist=["main"]))
    probe_fn = inspect.getsource(TwoAgentInventory.probe_completed_cache_attentions)
    compress = inspect.getsource(TwoAgentInventory.compress_sender)
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    ex = examples[0]
    probe = b_need_probe_prompt(ex)
    assert "effective capacity" in probe
    assert "eligibility" in probe
    assert ex.request_line in probe
    assert not prompt_contains_b_private(probe, ex)
    assert not prompt_contains_solver_labels(probe, ex)
    assert ex.b_text.replace(ex.request_line, "").strip() not in probe
    assert CONDS == ("kv", "probe_32", "probe_64", "obf_32", "obf_64")
    assert COMPRESS_CONDS == {
        "probe_32": ("headwise", 32),
        "probe_64": ("headwise", 64),
        "obf_32": ("hobf_fast", 32),
        "obf_64": ("hobf_fast", 64),
    }
    assert SINK_SIZE == 4
    assert PREDECLARED_LARGER_BUDGET == 256
    assert PREDECLARED_LARGER_BUDGET not in {budget for _relay, budget in COMPRESS_CONDS.values()}
    assert "last_query_attentions" not in source
    assert "probe_completed_cache_attentions" in source
    assert "b_need_probe_prompt" in source
    assert "generate_text_keep_cache" in source
    assert "continue_latent" not in source
    assert "results_stage_plan4_compress" in PRESERVED
    assert "from src.probe import cache_keys_attentions_to_steps" in probe_fn
    assert "from src.probe import probe_attentions_to_steps" not in probe_fn
    assert "n_kv_positions" in probe_fn
    assert "cache_len" in probe_fn
    assert "hobf_fast" in compress
    assert "not a replication" in compress
    charged = _process_time_s(
        {
            "sender_generation_time_s": 10.0,
            "plan_time_s": 2.0,
            "receiver_latency_s": 1.0,
            "probe_time_s": 0.4,
            "compression_time_s": 1.5,
        }
    )
    assert charged == 14.5
    assert "if args.rescore:" in source
    assert "return rescore()" in source
    data, out_root, _kaggle = p_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_probe" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_probe", "--split", "plan_dev"])
    assert p_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_probe", "--split", "plan_test"])
    assert p_main() == 2


def test_probe_summary_averages_recorded_process_time():
    from src.stage_plan4_probe import _probe_block

    items = [
        {
            "correct": True,
            "sender_generation_time_s": 10.0,
            "plan_time_s": 2.0,
            "receiver_latency_s": 1.0,
            "probe_time_s": 0.4,
            "compression_core_s": 0.3,
            "compression_time_s": 1.5,
            "process_time_s": 14.5,
            "message_bytes": 100,
            "n_kv_positions": 36,
            "n_kept": 36,
        },
        {
            "correct": False,
            "sender_generation_time_s": 10.0,
            "plan_time_s": 2.0,
            "receiver_latency_s": 1.0,
            "probe_time_s": 0.4,
            "compression_core_s": 0.3,
            "compression_time_s": 1.5,
            "process_time_s": 15.5,
            "message_bytes": 100,
            "n_kv_positions": 36,
            "n_kept": 36,
        },
    ]
    block = _probe_block(items)
    assert block["mean_process_time_s"] == 15.0
    assert block["mean_probe_time_s"] == 0.4
    assert block["mean_compression_core_s"] == 0.3


def test_budget_is_probe_256_without_obf(monkeypatch):
    from src.stage_plan4_budget import COMPRESS_CONDS, CONDS, KV_BUDGET, PRESERVED
    from src.stage_plan4_budget import _paths as b_paths
    from src.stage_plan4_budget import main as b_main
    from src.stage_plan4_compress import SINK_SIZE

    source = inspect.getsource(__import__("src.stage_plan4_budget", fromlist=["main"]))
    assert CONDS == ("kv", "probe_256")
    assert COMPRESS_CONDS == {"probe_256": ("headwise", 256)}
    assert KV_BUDGET == 256
    assert SINK_SIZE == 4
    assert "obf" not in CONDS
    assert "hobf" not in CONDS
    assert "probe_32" not in CONDS
    assert "last_query_attentions" not in source
    assert "probe_completed_cache_attentions" in source
    assert "b_need_probe_prompt" in source
    assert "hobf_fast" not in source
    assert "results_stage_plan4_probe" in PRESERVED
    data, out_root, _kaggle = b_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_budget" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_budget", "--split", "plan_dev"])
    assert b_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_budget", "--split", "plan_test"])
    assert b_main() == 2


def test_plan4_fresh_is_disjoint_same_shape():
    from src.paths import DATA_DIR

    fresh_path = DATA_DIR / "plan4_fresh.jsonl"
    dev_path = DATA_DIR / "plan4_dev.jsonl"
    fresh = load_plan4_jsonl(fresh_path)
    dev = load_plan4_jsonl(dev_path)
    assert len(fresh) == 90
    assert N_PLAN4_FRESH_FAMILIES == 30
    bases = [ex for ex in fresh if ex.variant == "base"]
    assert len(bases) == 30
    assert Counter(ex.difficulty for ex in bases) == {"L1": 10, "L2": 10, "L3": 10}
    assert all(ex.split == "plan4_fresh" for ex in fresh)
    for ex in fresh:
        assert_plan4_invariants(ex)
    banned = {sheet_fingerprint(ex) for ex in dev}
    assert not any(sheet_fingerprint(ex) in banned for ex in fresh)
    assert {ex.family_id for ex in fresh}.isdisjoint({ex.family_id for ex in dev})
    assert {ex.example_id for ex in fresh}.isdisjoint({ex.example_id for ex in dev})


def test_plan4_hold_is_disjoint_from_dev_and_fresh():
    from src.paths import DATA_DIR

    hold_path = DATA_DIR / "plan4_hold.jsonl"
    hold = load_plan4_jsonl(hold_path)
    prior = prior_plan4_examples()
    assert PLAN4_HOLD_SEED == 2070
    assert N_PLAN4_HOLD_FAMILIES == 30
    assert len(hold) == 90
    bases = [ex for ex in hold if ex.variant == "base"]
    assert len(bases) == 30
    assert Counter(ex.difficulty for ex in bases) == {"L1": 10, "L2": 10, "L3": 10}
    assert all(ex.split == "plan4_hold" for ex in hold)
    for ex in hold:
        assert_plan4_invariants(ex)
    report = overlap_with_prior(hold, prior)
    assert report["n_sheet_overlap"] == 0
    assert report["n_example_id_overlap"] == 0
    assert report["n_family_id_overlap"] == 0
    generated = generate_hold_disjoint(prior)
    assert overlap_with_prior(generated, prior)["n_sheet_overlap"] == 0
    assert {ex.family_id for ex in hold}.isdisjoint({ex.family_id for ex in prior})


def test_fresh_runner_is_frozen_four_way_plus_controls(monkeypatch):
    from src.stage_plan4_fresh import CONDS, CORE_CONDS, KV_BUDGET, PRESERVED
    from src.stage_plan4_fresh import _paths as f_paths
    from src.stage_plan4_fresh import main as f_main

    source = inspect.getsource(__import__("src.stage_plan4_fresh", fromlist=["main"]))
    assert CONDS == ("text", "text_full", "kv", "probe_256", "no_message", "mismatch")
    assert CORE_CONDS == ("text", "text_full", "kv", "probe_256")
    assert KV_BUDGET == 256
    assert "probe_32" not in CONDS
    assert "obf_32" not in CONDS
    assert "last_query_attentions" not in source
    assert "probe_completed_cache_attentions" in source
    assert "fuller_text_handoff" in source
    assert "hobf_fast" not in source
    assert "results_stage_plan4_budget" in PRESERVED
    data, out_root, _kaggle = f_paths()
    assert data.name == "plan4_fresh.jsonl"
    assert "stage_plan4_fresh" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_fresh", "--split", "plan4_dev"])
    assert f_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_fresh", "--split", "plan_dev"])
    assert f_main() == 2


def test_diag_runner_preserves_calibration_folder(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_diag", fromlist=["main"]))
    assert "results_stage_plan4_cal" in source
    assert "make_compressor" not in source
    from src.stage_plan4_diag import _paths as diag_paths, PRESERVED, main as diag_main

    data, out_root, _kaggle = diag_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_diag" in str(out_root)
    assert "stage_plan4_cal" not in str(out_root)
    assert "results_stage_plan4_cal" in PRESERVED
    monkeypatch.setattr("sys.argv", ["stage_plan4_diag", "--split", "plan_dev"])
    assert diag_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_diag", "--split", "plan_test"])
    assert diag_main() == 2


def test_calibration_is_sender_only_and_refuses_plan_dev(monkeypatch):
    source = inspect.getsource(__import__("src.stage_plan4_cal", fromlist=["main"]))
    assert "make_compressor" not in source
    assert "hop_retrieve" not in source
    assert "kv_c0" not in CONDS
    assert CONDS == A_CONDS + (B_COND,)
    assert LATENT_STEPS_C40 == 40
    assert compact_result_handoff() != written_result_handoff()
    assert "four lines" in compact_result_handoff()
    data, out_root, _kaggle = _paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_cal" in str(out_root)
    assert "stage_study1" not in str(out_root)
    from src.stage_plan4_cal import main

    monkeypatch.setattr("sys.argv", ["stage_plan4_cal", "--split", "plan_dev"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_cal", "--split", "plan_test"])
    assert main() == 2


def test_span_runner_matches_shared_positions_on_plan4_dev(monkeypatch):
    from src.stage_plan4_span import CONDS as SPAN_CONDS, PRESERVED
    from src.stage_plan4_span import _paths as span_paths
    from src.stage_plan4_span import main as span_main

    source = inspect.getsource(__import__("src.stage_plan4_span", fromlist=["main"]))
    two_agent = inspect.getsource(__import__("src.two_agent", fromlist=["TwoAgentInventory"]))
    assert SPAN_CONDS == ("kv", "kv_result", "text_result", "kv_probe", "text_probe")
    assert "keep_shared_positions" in source
    assert "keep_shared_positions" in two_agent
    assert "full_ids_cpu" in two_agent
    assert "shared_across_heads" in source
    assert "score_sender_message" in source
    assert "result_spans" in source
    assert "select_spans_by_score" in source
    assert "compress_sender" not in source
    assert "last_query_attentions" not in source
    assert "hobf_fast" not in source
    assert "probe_32" not in SPAN_CONDS
    assert "select_prompt_indices" not in source
    assert "results_stage_plan4_fresh" in PRESERVED
    assert "results_stage_plan4_budget" in PRESERVED
    data, out_root, _kaggle = span_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_span" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_span", "--split", "plan4_fresh"])
    assert span_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_span", "--split", "plan_dev"])
    assert span_main() == 2


def test_b_plan_scorer_uses_eligibility_not_capacity():
    plan = (
        "P: Eligible | $60 | candidate\n"
        "Q: Ineligible | $105 | dropped\n"
        "R: Ineligible | $35 | dropped\n"
        "S: Eligible | $80 | candidate\n"
        "Choice: P at $60"
    )
    scored = score_b_plan(
        plan,
        gold_status={"P": "eligible", "Q": "ineligible", "R": "ineligible", "S": "eligible"},
        prices={"P": 60, "Q": 105, "R": 35, "S": 80},
        budget=95,
        pred_carrier="R",
    )
    assert scored["all_status_correct"]
    assert scored["all_price_correct"]
    assert scored["all_flag_correct"]
    assert scored["choice"] == "P"
    assert scored["choice_matches_gold"]
    assert scored["plan_extraction_status"] == "disagrees"
    assert "n_q_correct" not in scored
    assert is_think_collapse("</think>\n\n</think>\n\n</think>")
    assert not is_think_collapse(plan)
    flags = selected_content_flags(
        "<|im_start|>user\nNow compute the result for Agent B.\n<|im_end|>\nRESULTS"
    )
    assert flags["has_chat_marker"]
    assert not flags["has_result_line"]


def test_kvdiag_is_three_families_and_preserves_span(monkeypatch):
    from src.plan4_task import SINK_INITIAL
    from src.stage_plan4_kvdiag import CONDS as DCONDS, PRESERVED
    from src.stage_plan4_kvdiag import _paths as d_paths
    from src.stage_plan4_kvdiag import main as d_main

    source = inspect.getsource(__import__("src.stage_plan4_kvdiag", fromlist=["main"]))
    assert DCONDS == ("kv", "kv_identity", "kv_result", "kv_result_sink", "text_result")
    assert PLAN4_KVDIAG_FAMILIES == (
        "plan4_dev-fam-006",
        "plan4_dev-fam-001",
        "plan4_dev-fam-000",
    )
    assert SINK_INITIAL == 4
    assert "keep_shared_positions" in source
    assert "with_initial_positions" in source
    assert "compress_sender" not in source
    assert "hobf_fast" not in source
    assert "probe_32" not in DCONDS
    assert "results_stage_plan4_span" in PRESERVED
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    selected = subset_plan4_kvdiag(examples)
    assert len(selected) == 9
    assert {ex.family_id for ex in selected} == set(PLAN4_KVDIAG_FAMILIES)
    assert {ex.difficulty for ex in selected if ex.variant == "base"} == {"L1", "L2", "L3"}
    data, out_root, _kaggle = d_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_kvdiag" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_kvdiag", "--split", "plan4_fresh"])
    assert d_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_kvdiag", "--split", "plan_dev"])
    assert d_main() == 2


def test_initctrl_adds_sink_only_and_preserves_kvdiag(monkeypatch):
    from src.plan4_task import SINK_INITIAL
    from src.stage_plan4_initctrl import CONDS as ICONDS, PRESERVED
    from src.stage_plan4_initctrl import _checked_bool, _cond_block
    from src.stage_plan4_initctrl import _paths as i_paths
    from src.stage_plan4_initctrl import main as i_main

    source = inspect.getsource(__import__("src.stage_plan4_initctrl", fromlist=["main"]))
    capture = inspect.getsource(__import__("src.two_agent", fromlist=["generate_text_keep_cache"]))
    assert ICONDS == ("kv", "kv_result", "kv_result_sink", "kv_sink", "text_result")
    assert "kv_identity" not in ICONDS
    assert "kv_sink" in ICONDS
    assert SINK_INITIAL == 4
    assert "keep_shared_positions" in source
    assert "with_initial_positions" in source
    assert "compress_sender" not in source
    assert "hobf_fast" not in source
    assert "probe_32" not in ICONDS
    assert "results_stage_plan4_kvdiag" in PRESERVED
    assert "results_stage_plan4_span" in PRESERVED
    assert "_capture_model_input_ids" in capture
    assert "generate_text_batch" in capture
    assert "model.generate =" not in capture
    counted = _checked_bool(
        [
            {"loop_eq_retokenize": None},
            {"loop_eq_retokenize": False},
            {"loop_eq_retokenize": True},
        ],
        "loop_eq_retokenize",
    )
    assert counted == {"n_true": 1, "n_false": 1, "n_not_checked": 1}
    block = _cond_block(
        [
            {
                "parse_ok": True,
                "think_collapse": False,
                "truncated": False,
                "plan_all_status_correct": False,
                "correct": True,
                "n_kept": 4,
                "n_result_kept": 0,
                "n_sink_kept": 4,
                "message_bytes": 1024,
                "ids_source": "decode_loop",
                "loop_eq_retokenize": None,
                "loop_len_eq_n_kv": True,
                "eos_counted_not_cached": True,
                "result_spans_from_loop_ids": True,
            }
        ]
    )
    assert "sequences_eq_retokenize" not in block
    assert block["loop_eq_retokenize"]["n_not_checked"] == 1
    assert block["correct"] == 1
    assert block["plan_all_status_correct"] == 0
    data, out_root, _kaggle = i_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_initctrl" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_initctrl", "--split", "plan4_fresh"])
    assert i_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_initctrl", "--split", "plan_dev"])
    assert i_main() == 2


def test_stable_is_all_dev_and_preserves_initctrl(monkeypatch):
    from src.plan4_task import SINK_INITIAL
    from src.stage_plan4_initctrl import CONDS as ICONDS
    from src.stage_plan4_stable import CONDS as SCONDS, PRESERVED, PAIRED
    from src.stage_plan4_stable import _paths as s_paths
    from src.stage_plan4_stable import main as s_main

    source = inspect.getsource(__import__("src.stage_plan4_stable", fromlist=["main"]))
    assert SCONDS == ICONDS
    assert SCONDS == ("kv", "kv_result", "kv_result_sink", "kv_sink", "text_result")
    assert SINK_INITIAL == 4
    assert "keep_shared_positions" in source
    assert "with_initial_positions" in source
    assert "_family_delta" in source
    assert "compress_sender" not in source
    assert "hobf_fast" not in source
    assert "probe_32" not in SCONDS
    assert "results_stage_plan4_initctrl" in PRESERVED
    assert "results_stage_plan4_kvdiag" in PRESERVED
    assert ("result_sink_vs_sink", "kv_sink", "kv_result_sink") in PAIRED
    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    assert len(examples) == 90
    data, out_root, _kaggle = s_paths()
    assert data.name == "plan4_dev.jsonl"
    assert "stage_plan4_stable" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_stable", "--split", "plan4_fresh"])
    assert s_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_stable", "--split", "plan_dev"])
    assert s_main() == 2


def test_stable_counts_ineligible_marked_eligible():
    from src.stage_plan4_stable import _cond_block, _plan_fields

    examples = generate_split("plan4_dev", N_PLAN4_FAMILIES, PLAN4_DEV_SEED)
    ex = next(e for e in examples if "ineligible" in gold_eligibility_statuses(e).values())
    gold = gold_eligibility_statuses(ex)
    prices = ex.b_facts["prices"]
    lines = [
        f"{c}: eligible | ${int(prices[c])} | candidate" for c in CARRIERS
    ]
    rec = {
        "plan_text": "\n".join(lines) + "\nChoice: P",
        "pred_carrier": "P",
        "raw_text": "Answer: A) P",
        "correct": True,
        "parse_ok": True,
        "think_collapse": False,
        "truncated": False,
        "n_kept": 80,
        "n_result_kept": 76,
        "n_sink_kept": 4,
        "message_bytes": 1000,
        "follows_trap": False,
        "process_time_s": 1.5,
        "ids_source": "decode_loop",
        "loop_eq_retokenize": True,
        "loop_len_eq_n_kv": True,
        "eos_counted_not_cached": True,
        "result_spans_from_loop_ids": True,
        "plan_answer_agrees": None,
    }
    _plan_fields(rec, ex)
    n_inelig = sum(gold[c] == "ineligible" for c in CARRIERS)
    assert rec["plan_ineligible_marked_eligible"] == n_inelig
    assert rec["plan_all_status_correct"] is False
    assert rec["correct_with_wrong_eligibility"] is True
    block = _cond_block([rec])
    assert block["plan_ineligible_marked_eligible"] == n_inelig
    assert block["correct_with_wrong_eligibility"] == 1
    assert block["mean_process_time_s"] == 1.5
    missing = dict(rec)
    missing["plan_answer_agrees"] = None
    assert _cond_block([missing])["plan_answer_agrees"]["n_not_checked"] == 1


def test_family_delta_keeps_variants_together():
    from src.stage_plan4_compress import _family_delta

    rows = []
    for fam in ("fam-0", "fam-1"):
        for variant in ("base", "a_flip", "b_flip"):
            eid = f"{fam}-{variant}"
            rows.append(
                {
                    "example_id": eid,
                    "family_id": fam,
                    "condition": "kv_sink",
                    "correct": False,
                    "plan_all_status_correct": False,
                }
            )
            rows.append(
                {
                    "example_id": eid,
                    "family_id": fam,
                    "condition": "kv_result_sink",
                    "correct": variant != "b_flip",
                    "plan_all_status_correct": False,
                }
            )
    acc = _family_delta(rows, "kv_sink", "kv_result_sink")
    elig = _family_delta(rows, "kv_sink", "kv_result_sink", key="plan_all_status_correct")
    assert acc["n_families"] == 2
    assert acc["key"] == "correct"
    assert acc["delta_b_minus_a"] == pytest.approx(4 / 6)
    assert elig["key"] == "plan_all_status_correct"
    assert elig["delta_b_minus_a"] == 0.0


def test_hold_runner_uses_new_split_and_preserves_stable(monkeypatch):
    from src.stage_plan4_initctrl import CONDS as ICONDS
    from src.stage_plan4_stable import CONDS as SCONDS
    from src.stage_plan4_hold import CONDS as HCONDS, PRESERVED
    from src.stage_plan4_hold import _paths as h_paths
    from src.stage_plan4_hold import main as h_main

    source = inspect.getsource(__import__("src.stage_plan4_hold", fromlist=["main"]))
    assert HCONDS == SCONDS == ICONDS
    assert "keep_shared_positions" in source
    assert "overlap_with_prior" in source
    assert "compress_sender" not in source
    assert "hobf_fast" not in source
    assert "probe_32" not in HCONDS
    assert "results_stage_plan4_stable" in PRESERVED
    assert "results_stage_plan4_fresh" in PRESERVED
    data, out_root, _kaggle = h_paths()
    assert data.name == "plan4_hold.jsonl"
    assert "stage_plan4_hold" in str(out_root)
    monkeypatch.setattr("sys.argv", ["stage_plan4_hold", "--split", "plan4_dev"])
    assert h_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_hold", "--split", "plan4_fresh"])
    assert h_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_hold", "--split", "plan_dev"])
    assert h_main() == 2


def test_span_rescore_preserves_original_files(tmp_path):
    from src.rescore_plan4_span import ORIGINAL_NAMES, rescore

    eval_path = tmp_path / "evaluations.jsonl"
    row = {
        "example_id": "plan4_dev-fam-000-base",
        "family_id": "plan4_dev-fam-000",
        "variant": "base",
        "difficulty": "L3",
        "condition": "kv",
        "correct": False,
        "kind": "trap",
        "parse_ok": True,
        "pred_carrier": "R",
        "gold_carrier": "P",
        "plan_text": (
            "P: Eligible | $60 | candidate\n"
            "Q: Ineligible | $105 | dropped\n"
            "R: Ineligible | $35 | dropped\n"
            "S: Eligible | $80 | candidate\n"
            "Choice: P at $60"
        ),
        "raw_text": "Answer: A) R",
        "selected_text": "P: q=10 | Eligible\nQ: q=14 | Ineligible\nR: q=12 | Ineligible\nS: q=16 | Eligible",
    }
    eval_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a_artifacts.jsonl").write_text("{}\n", encoding="utf-8")
    before = {name: (tmp_path / name).read_bytes() for name in ORIGINAL_NAMES if (tmp_path / name).is_file()}
    report = rescore(tmp_path)
    after = {name: (tmp_path / name).read_bytes() for name in ORIGINAL_NAMES if (tmp_path / name).is_file()}
    assert before == after
    assert (tmp_path / "plan_rescore.json").is_file()
    assert report["receiver"]["kv"]["all_status_correct"] == 1
    assert report["receiver"]["kv"]["final_correct"] == 0
    assert report["original_files_preserved"]
