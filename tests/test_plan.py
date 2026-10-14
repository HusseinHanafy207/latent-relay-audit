"""Study 0 delivery-planning task. No GPU."""

from __future__ import annotations

import inspect
from collections import Counter

from src.parse_plan import label_plan_answer, parse_plan_answer
from src.parse_plan import OLD_TEMPLATE_STATUSES, parse_eligibility_statuses
from src.plan_task import (
    N_COMBINE,
    N_EASY,
    N_FRESH_COMBINE,
    N_FRESH_EASY,
    N_PLAN_FAMILIES,
    N_PLAN_FRESH_FAMILIES,
    PLAN_DEV_SEED,
    PLAN_FRESH_SEED,
    QUESTION,
    assert_plan_invariants,
    evaluate,
    generate_split,
    gold_eligibility_statuses,
    mismatch_source_id,
    oracle_intermediate_message,
    oracle_sentence_message,
)
from src.prompts_plan import (
    ELIGIBILITY_CONSTRAINT,
    SHORT_ANSWER_RULE,
    a_alone_prompt,
    agent_a_message_prompt,
    b_alone_prompt,
    b_answer_after_message_plan_prompt,
    b_answer_after_plan_prompt,
    b_planning_prompt,
    b_reason_with_message_prompt,
    b_with_message_prompt,
    collab_b_prompt,
    oracle_to_b_prompt,
    pooled_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
)
from src.stage_plan0 import CONDS, _jobs, _paths, _process_time_s


def test_plan_dev_is_balanced_and_has_both_difficulties():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    assert len(examples) == 36
    bases = [ex for ex in examples if ex.variant == "base"]
    assert len(bases) == 12
    assert sum(ex.difficulty == "easy" for ex in bases) == N_EASY
    assert sum(ex.difficulty == "combine" for ex in bases) == N_COMBINE
    letters = Counter(ex.gold_letter for ex in bases)
    assert letters == {"A": 4, "B": 4, "C": 4}
    families = {ex.family_id for ex in examples}
    assert len(families) == 12
    for ex in examples:
        assert_plan_invariants(ex)
        assert ex.split == "plan_dev"
        assert ex.question == QUESTION
        assert "lowest-cost" in ex.question


def test_neither_agent_can_solve_base_from_private_information_alone():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    for ex in examples:
        if ex.variant == "a_flip":
            continue
        ev = evaluate(
            {"boxes": ex.boxes, "refrigerated": ex.refrigerated},
            ex.a_facts,
            ex.b_facts,
            difficulty=ex.difficulty,
        )
        assert len(ev.eligible) >= 2
        assert ev.trap_carrier not in ev.eligible
        assert ev.gold_carrier in ev.eligible
        assert ev.gold_carrier != ev.trap_carrier


def test_paired_interventions_change_only_one_agent_and_the_gold():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    by_family = {}
    for ex in examples:
        by_family.setdefault(ex.family_id, {})[ex.variant] = ex
    for family in by_family.values():
        base, a_flip, b_flip = family["base"], family["a_flip"], family["b_flip"]
        assert a_flip.b_text == base.b_text
        assert a_flip.b_facts == base.b_facts
        assert a_flip.a_text != base.a_text
        assert a_flip.gold_carrier != base.gold_carrier
        assert a_flip.gold_carrier == base.trap_carrier
        assert a_flip.choices == base.choices
        assert b_flip.a_text == base.a_text
        assert b_flip.a_facts == base.a_facts
        assert b_flip.b_text != base.b_text
        assert b_flip.gold_carrier != base.gold_carrier
        assert b_flip.gold_carrier == base.other_carrier


def test_oracle_is_eligibility_only_and_ordinary_prompts_do_not_insert_it():
    import src.prompts_plan as prompts_mod

    assert not hasattr(prompts_mod, "oracle_intermediate_message")
    assert "gold_carrier" not in inspect.signature(agent_a_message_prompt).parameters
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[:6]
    for ex in examples:
        oracle = oracle_intermediate_message(ex)
        assert "Budget" not in oracle
        assert "$" not in oracle
        assert "book " not in oracle.lower()
        assert "P: " in oracle and " | " in oracle
        a_msg = agent_a_message_prompt(ex)
        assert oracle not in a_msg
        assert "exactly three lines" in a_msg
        assert "P: eligible" not in a_msg
        assert "Q: ineligible" not in a_msg
        assert "R: eligible" not in a_msg
        assert "<status determined from P's operational facts>" in a_msg
        assert not prompt_contains_b_private(a_msg, ex)
        assert prompt_contains_a_private(a_msg, ex)
        assert not prompt_contains_b_private(a_alone_prompt(ex), ex)
        assert not prompt_contains_a_private(b_alone_prompt(ex), ex)
        assert QUESTION in a_alone_prompt(ex)
        assert QUESTION in b_alone_prompt(ex)
        assert not prompt_contains_a_private(b_planning_prompt(ex), ex)
        collab = collab_b_prompt(ex, "P: eligible | test")
        assert not prompt_contains_a_private(collab, ex)
        assert prompt_contains_b_private(collab, ex)
        assert oracle not in collab
        pooled = pooled_prompt(ex)
        assert prompt_contains_a_private(pooled, ex)
        assert prompt_contains_b_private(pooled, ex)
        ora_prompt = oracle_to_b_prompt(ex, oracle)
        assert oracle in ora_prompt
        assert not prompt_contains_a_private(ora_prompt, ex)


def test_collab_and_oracle_share_the_receiver_wrapper():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[:3]
    for ex in examples:
        body = "P: eligible | test\nQ: ineligible | test\nR: eligible | test"
        collab = collab_b_prompt(ex, body)
        oracle = oracle_to_b_prompt(ex, body)
        shared = b_with_message_prompt(ex, body)
        assert collab == oracle == shared
        assert ELIGIBILITY_CONSTRAINT in collab
        assert "if it helps" not in collab.lower()
        assert "diagnostic" not in collab.lower()
        assert QUESTION in collab


def test_extra_compute_is_a_planning_pass_then_an_answer_pass():
    ex = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[0]
    plan = b_planning_prompt(ex)
    assert "Stop after that line." not in plan
    assert SHORT_ANSWER_RULE not in plan
    assert "Do not write an Answer line yet." in plan
    assert not prompt_contains_a_private(plan, ex)
    answer = b_answer_after_plan_prompt(ex, "Cheapest under budget is Q, but I lack eligibility.")
    assert SHORT_ANSWER_RULE in answer
    assert "Cheapest under budget is Q, but I lack eligibility." in answer
    assert not prompt_contains_a_private(answer, ex)


def test_mismatch_comes_from_another_family_base():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    for ex in examples:
        src = mismatch_source_id(ex, examples)
        assert src.endswith("-base")
        src_family = src[: -len("-base")]
        assert src_family != ex.family_id


def test_combine_examples_require_more_than_a_cleared_flag():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    combine = [ex for ex in examples if ex.difficulty == "combine" and ex.variant == "base"]
    assert combine
    for ex in combine:
        assert "vehicle capacity" in ex.a_text
        assert "cleared to carry" not in ex.a_text
        easy = next(item for item in examples if item.difficulty == "easy")
        assert "cleared" in easy.a_text
        break
    easy_bases = [ex for ex in examples if ex.difficulty == "easy" and ex.variant == "base"]
    assert all("cleared" in ex.a_text for ex in easy_bases)


def test_parse_plan_prefers_last_answer_line():
    parsed = parse_plan_answer("A) P\nAnswer: C) R\n")
    labeled = label_plan_answer(
        parsed,
        gold_letter="C",
        gold_carrier="R",
        trap_letter="A",
        trap_carrier="Q",
    )
    assert labeled.parse_ok
    assert labeled.kind == "true"
    trap = label_plan_answer(
        parse_plan_answer("Answer: A) Q"),
        gold_letter="C",
        gold_carrier="R",
        trap_letter="A",
        trap_carrier="Q",
    )
    assert trap.kind == "trap"


def test_parse_plan_answer_mark_without_a_letter_is_unparsed():
    """B sometimes writes 'Answer:' with no A/B/C. That is incorrect, not a crash."""
    empty = parse_plan_answer("Answer:")
    assert empty.parse_ok is False
    assert empty.kind == "unparsed"
    assert empty.letter is None
    labeled = label_plan_answer(
        parse_plan_answer("Answer: the cheapest option"),
        gold_letter="C",
        gold_carrier="R",
        trap_letter="A",
        trap_carrier="Q",
    )
    assert labeled.kind == "unparsed"
    assert labeled.parse_ok is False
    carrier_only = parse_plan_answer("Answer: use P")
    assert carrier_only.letter is None
    assert carrier_only.carrier == "P"
    assert carrier_only.parse_ok is False


def test_study0_jobs_are_text_controls_without_a_compressor():
    assert [job["condition"] for job in _jobs()] == list(CONDS)
    assert "headwise" not in CONDS
    assert "full_kv" not in CONDS
    row = {"sender_generation_time_s": 1.2, "plan_time_s": 0.5, "receiver_latency_s": 0.4}
    assert abs(_process_time_s(row) - 2.1) < 1e-9
    data, out_root, _kaggle = _paths()
    assert data.name == "plan_dev.jsonl"
    assert "stage_plan0_corrected" in str(out_root)
    assert out_root.parts[-2] != "stage_plan0"


def test_a_template_is_not_an_answer_key():
    ex = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[0]
    prompt = agent_a_message_prompt(ex)
    for filled in ("P: eligible |", "Q: ineligible |", "R: eligible |"):
        assert filled not in prompt


def test_parse_eligibility_statuses_uses_last_label_per_carrier():
    text = "P: eligible | ok\nQ: ineligible | no\nR: eligible | ok\nP: ineligible | later"
    parsed = parse_eligibility_statuses(text)
    assert parsed == {"P": "ineligible", "Q": "ineligible", "R": "eligible"}
    assert parsed != OLD_TEMPLATE_STATUSES


def test_sentence_oracle_is_prose_and_matches_evaluator():
    examples = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[:6]
    for ex in examples:
        compact = oracle_intermediate_message(ex)
        sentences = oracle_sentence_message(ex)
        gold = gold_eligibility_statuses(ex)
        assert compact != sentences
        assert " | " not in sentences
        assert "$" not in sentences
        assert "Budget" not in sentences
        for carrier, status in gold.items():
            assert carrier in sentences
        if any(status == "ineligible" for status in gold.values()):
            assert "cannot be used" in sentences
        if any(status == "eligible" for status in gold.values()):
            assert "operational requirements" in sentences


def test_reason_with_message_sees_eligibility_unlike_extra_compute():
    ex = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)[0]
    message = oracle_sentence_message(ex)
    reason = b_reason_with_message_prompt(ex, message)
    extra = b_planning_prompt(ex)
    assert message in reason
    assert message not in extra
    assert SHORT_ANSWER_RULE not in reason
    assert prompt_contains_b_private(reason, ex)
    assert not prompt_contains_a_private(reason, ex)
    answer = b_answer_after_message_plan_prompt(ex, message, "Skip R; pick cheapest eligible.")
    assert SHORT_ANSWER_RULE in answer
    assert message in answer
    assert "Skip R; pick cheapest eligible." in answer


def test_diag_is_two_by_two_oracle_only():
    from src.stage_plan0_diag import B_CONDS, CONDS, _paths as diag_paths

    assert CONDS == ("a_status", "compact_direct", "compact_reason", "sentence_direct", "sentence_reason")
    assert B_CONDS == ("compact_direct", "compact_reason", "sentence_direct", "sentence_reason")
    _data, out_root, _kaggle = diag_paths()
    assert "stage_plan0_diag" in str(out_root)
    assert "stage_plan0_corrected" not in str(out_root)
    assert out_root.parts[-2] != "stage_plan0"


def test_stage_plan0_refuses_anything_but_plan_dev(monkeypatch):
    from src.stage_plan0 import main

    monkeypatch.setattr("sys.argv", ["stage_plan0", "--split", "plan_test"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan0", "--split", "test"])
    assert main() == 2


def test_stage_plan0_diag_refuses_anything_but_plan_dev(monkeypatch):
    from src.stage_plan0_diag import main as diag_main

    monkeypatch.setattr("sys.argv", ["stage_plan0_diag", "--split", "plan_test"])
    assert diag_main() == 2


def test_incorrect_old_template_copy_is_only_wrong_matches():
    from src.parse_plan import score_eligibility_message

    gold = {"P": "eligible", "Q": "ineligible", "R": "eligible"}
    ok = score_eligibility_message("P: eligible | a\nQ: ineligible | b\nR: eligible | c", gold)
    assert ok["copied_old_template"]
    assert not ok["incorrect_old_template_copy"]
    gold2 = {"P": "ineligible", "Q": "eligible", "R": "eligible"}
    bad = score_eligibility_message("P: eligible | a\nQ: ineligible | b\nR: eligible | c", gold2)
    assert bad["copied_old_template"]
    assert bad["incorrect_old_template_copy"]
    assert not bad["all_status_correct"]


def test_plan_fresh_is_small_balanced_and_disjoint_from_plan_dev():
    dev = generate_split("plan_dev", N_PLAN_FAMILIES, PLAN_DEV_SEED)
    fresh = generate_split(
        "plan_fresh",
        N_PLAN_FRESH_FAMILIES,
        PLAN_FRESH_SEED,
        n_easy=N_FRESH_EASY,
        n_combine=N_FRESH_COMBINE,
    )
    assert len(fresh) == 18
    bases = [ex for ex in fresh if ex.variant == "base"]
    assert len(bases) == 6
    assert sum(ex.difficulty == "easy" for ex in bases) == 3
    assert sum(ex.difficulty == "combine" for ex in bases) == 3
    letters = Counter(ex.gold_letter for ex in bases)
    assert letters == {"A": 2, "B": 2, "C": 2}
    for ex in fresh:
        assert_plan_invariants(ex)
        assert ex.split == "plan_fresh"
        assert ex.question == QUESTION
    dev_keys = {(ex.a_text, ex.b_text) for ex in dev}
    fresh_keys = {(ex.a_text, ex.b_text) for ex in fresh}
    assert dev_keys.isdisjoint(fresh_keys)


def test_e2e_sends_generated_a_messages_with_reason_then_answer():
    from src.stage_plan0_e2e import B_CONDS, CONDS, _paths as e2e_paths
    import src.stage_plan0_e2e as e2e_mod
    import inspect

    source = inspect.getsource(e2e_mod)
    assert "oracle_intermediate_message" not in source
    assert "oracle_sentence_message" not in source
    assert "b_reason_with_message_prompt" in source
    assert "b_planning_prompt" not in source
    assert CONDS == ("a_status", "collab_reason", "mismatch_reason")
    assert B_CONDS == ("collab_reason", "mismatch_reason")
    data, out_root, _kaggle = e2e_paths()
    assert data.name == "plan_fresh.jsonl"
    assert "stage_plan0_e2e" in str(out_root)
    assert "stage_plan0_diag" not in str(out_root)
    assert "stage_plan0_corrected" not in str(out_root)


def test_stage_plan0_e2e_refuses_plan_dev(monkeypatch):
    from src.stage_plan0_e2e import main as e2e_main

    monkeypatch.setattr("sys.argv", ["stage_plan0_e2e", "--split", "plan_dev"])
    assert e2e_main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan0_e2e", "--split", "plan_test"])
    assert e2e_main() == 2
