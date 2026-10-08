"""Shared answer extraction. No GPU."""

from __future__ import annotations

from src.parse import label_answer, parse_answer, parse_answer_legacy
from src.rescore import rescore_row, summarize_rescore


RECENCY_ECHO = (
    "(A) locker 46, (B) locker 60, (C) locker 65, (D) locker 41\n\n"
    "Answer: (C) locker 65"
)


def test_clean_answer_matches_legacy():
    text = "Answer: B) locker 7"
    current = parse_answer(text)
    legacy = parse_answer_legacy(text)
    assert current.letter == legacy.letter == "B"
    assert current.locker == legacy.locker == 7
    assert current.parse_ok and legacy.parse_ok


def test_choice_echo_uses_answer_line_not_first_option():
    current = parse_answer(RECENCY_ECHO)
    legacy = parse_answer_legacy(RECENCY_ECHO)
    assert current.letter == "C"
    assert current.locker == 65
    assert current.parse_ok
    assert legacy.letter == "A"
    assert legacy.locker == 46
    labeled = label_answer(
        current, gold_letter="C", gold_locker=65, donor_letter="A", donor_locker=46
    )
    assert labeled.kind == "true"
    labeled_legacy = label_answer(
        legacy, gold_letter="C", gold_locker=65, donor_letter="A", donor_locker=46
    )
    assert labeled_legacy.kind == "donor"


def test_same_rule_on_wrong_headwise_line():
    text = "Answer: B) locker 75"
    current = parse_answer(text)
    legacy = parse_answer_legacy(text)
    assert current.letter == legacy.letter == "B"
    assert current.locker == legacy.locker == 75
    labeled = label_answer(
        current, gold_letter="D", gold_locker=57, donor_letter="A", donor_locker=14
    )
    assert labeled.kind == "other"


def test_unlabeled_choice_list_stays_unparsed():
    text = "(A) locker 46, (B) locker 60, (C) locker 65, (D) locker 41"
    current = parse_answer(text)
    assert current.parse_ok is False
    assert current.letter is None
    assert current.locker is None


def test_single_unlabeled_pair_is_accepted():
    parsed = parse_answer("B) locker 7")
    assert parsed.letter == "B"
    assert parsed.locker == 7


def test_rescore_keeps_original_and_flags_the_echo():
    row = {
        "example_id": "validate-005",
        "condition": "recency_sender_only_filler",
        "gold_letter": "C",
        "gold_locker": 65,
        "donor_letter": "A",
        "donor_locker": 46,
        "kind": "other",
        "correct": False,
        "pred_letter": "A",
        "pred_locker": 46,
        "raw_text": RECENCY_ECHO,
    }
    scored = rescore_row(row)
    assert scored["correct_original"] is False
    assert scored["correct"] is True
    assert scored["correct_legacy"] is False
    assert scored["changed"] is True
    report = summarize_rescore([row])
    recency = report["accuracy"]["recency_sender_only_filler"]
    assert recency["correct_original"] == 0
    assert recency["correct"] == 1
