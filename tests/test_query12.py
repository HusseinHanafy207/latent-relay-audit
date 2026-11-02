"""Single-carrier label questions on the twelve-carrier message."""

from __future__ import annotations

import inspect
from pathlib import Path

from src.plan4_timing_task import gold_eligibility_statuses_n, load_timing12_jsonl
from src.prompts_plan4 import FRAMING_INSTRUCTION, prompt_contains_a_private
from src.prompts_plan4_query import (
    QUERY_MAX_NEW,
    b_query_kv_prompt,
    b_query_text_prompt,
    parse_single_label,
    single_label_question,
)
from src.stage_plan4_query12 import (
    CONDS,
    INTERPRETATION_GUIDE,
    PRESERVED,
    _line_labels,
    _summarize,
    main,
)
from src.stage_select import _load_rows

TWELVE_LINE = Path("results_stage_plan4_recon12/evaluations.jsonl")


def test_parser_never_reads_ineligible_as_eligible():
    assert parse_single_label("ineligible")["pred_label"] == "ineligible"
    assert parse_single_label("eligible")["pred_label"] == "eligible"
    assert parse_single_label("Ineligible.")["pred_label"] == "ineligible"
    assert parse_single_label("Ineligible.")["exact_raw"] is False
    assert parse_single_label("**eligible**")["pred_label"] == "eligible"
    assert parse_single_label("<think>\n\n</think>\n\nineligible")["exact_raw"] is True
    for bad in ("not eligible", "eligible or ineligible", "Carrier Q is ineligible", "ineligible, because", ""):
        assert parse_single_label(bad)["pred_label"] is None


def test_prompts_keep_b_facts_and_ask_one_carrier():
    ex = load_timing12_jsonl(Path("data/plan4_timing12.jsonl"))[0]
    kv = b_query_kv_prompt(ex, "Q")
    text = b_query_text_prompt(ex, "Q: q=5 | ineligible | x", "Q")
    for prompt in (kv, text):
        assert ex.b_text in prompt
        assert prompt.endswith(single_label_question("Q"))
        assert "Write exactly" not in prompt
        assert "Choice" not in prompt
        assert not prompt_contains_a_private(prompt, ex)
    assert FRAMING_INSTRUCTION in kv
    assert "Q: q=5" in text
    assert single_label_question("Q") == (
        "According to Agent A's results, is Carrier Q eligible or ineligible? "
        "Reply with exactly eligible or ineligible."
    )
    assert QUERY_MAX_NEW <= 16


def test_stage_is_frozen_and_uses_independent_copies(monkeypatch):
    assert CONDS == ("generic_r", "generic_fresh_r", "full_kv", "text_result")
    assert "results_stage_plan4_recon12" in PRESERVED
    assert "attention" in INTERPRETATION_GUIDE
    source = inspect.getsource(__import__("src.stage_plan4_query12", fromlist=["main"]))
    assert "clone_sender" in source
    assert "caches_equal" in source
    assert "generate_text_keep_cache" not in source
    monkeypatch.setattr("sys.argv", ["stage_plan4_query12", "--split", "plan4_dev"])
    assert main() == 2


def test_twelve_line_reference_matches_recon12_counts():
    line = _line_labels(_load_rows(TWELVE_LINE))
    counts = {}
    for (_eid, cond, _carrier), ref in line.items():
        counts.setdefault(cond, 0)
        counts[cond] += ref["pred"] == ref["gold"]
    assert counts["generic_r"] == 63
    assert counts["generic_fresh_r"] == 106
    assert counts["full_kv"] == 108
    assert counts["text_result"] == 107


def test_summary_groups_by_case_and_compares_with_twelve_line():
    examples = load_timing12_jsonl(Path("data/plan4_timing12.jsonl"))
    line_rows = _load_rows(TWELVE_LINE)
    line = _line_labels(line_rows)
    rows = []
    for ex in examples:
        gold = gold_eligibility_statuses_n(ex)
        for cond in CONDS:
            for idx, carrier in enumerate(ex.carriers):
                pred = gold[carrier] if cond != "generic_r" else "eligible"
                rows.append(
                    {
                        "example_id": ex.example_id,
                        "family_id": ex.family_id,
                        "difficulty": ex.difficulty,
                        "condition": cond,
                        "carrier": carrier,
                        "carrier_index": idx,
                        "gold_label": gold[carrier],
                        "pred_label": pred,
                        "parse_ok": True,
                        "exact_raw": True,
                        "truncated": False,
                        "correct": pred == gold[carrier],
                        "ineligible_marked_eligible": gold[carrier] == "ineligible" and pred == "eligible",
                        "eligible_marked_ineligible": gold[carrier] == "eligible" and pred == "ineligible",
                        "latency_s": 0.1,
                    }
                )
    summary = _summarize(rows, line_rows)
    assert summary["n_rows"] == summary["n_expected"] == 432
    assert summary["receiver"]["full_kv"]["correct"] == 108
    assert summary["receiver"]["full_kv"]["cases_all_twelve_correct"] == 9
    n_inelig = summary["receiver"]["generic_r"]["n_gold_ineligible"]
    assert summary["receiver"]["generic_r"]["ineligible_marked_eligible"] == n_inelig
    fresh_vs = summary["paired_by_case"]["generic_fresh_r_vs_generic_r"]["correct"]
    assert fresh_vs["n_cases"] == 9
    assert fresh_vs["total_delta"] == n_inelig
    tab = summary["query_vs_twelve_line"]["full_kv"]["crosstab"]
    assert tab["n"] == 108 and tab["both_correct"] == 108
    fresh_tab = summary["query_vs_twelve_line"]["generic_fresh_r"]["crosstab"]
    assert fresh_tab["query_only_correct"] == 108 - sum(
        ref["pred"] == ref["gold"] for (e, c, k), ref in line.items() if c == "generic_fresh_r"
    )
    assert "nine source cases" in summary["stats_note"]
