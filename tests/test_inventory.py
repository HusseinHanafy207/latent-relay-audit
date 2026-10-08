"""Inventory construction and answer parsing. No GPU."""

from __future__ import annotations

from src.inventory import assert_no_leak, generate_split
from src.parse import label_answer, parse_answer
from src.prompts_inventory import receiver_prompt, true_inventory_absent_from_receiver_prompt


def test_matched_pair_shares_surface_not_lockers():
    examples = generate_split("dev", 20, 2026)
    assert len(examples) == 20
    for ex in examples:
        assert ex.packages == ex.packages
        assert ex.true_locker != ex.alt_locker
        assert ex.true_locker in ex.choices
        assert ex.alt_locker in ex.choices
        assert len(set(ex.choices)) == 4
        assert ex.queried_package in ex.true_inventory_text
        assert f"locker {ex.true_locker}" in ex.true_inventory_text
        assert f"locker {ex.alt_locker}" in ex.alt_inventory_text
        assert ex.true_inventory_text != ex.alt_inventory_text
        assert ex.question == f"Which locker is package {ex.queried_package} in?"
        assert_no_leak(ex)


def test_test_split_size_and_balance():
    examples = generate_split("test", 80, 2027)
    assert len(examples) == 80
    letters = [ex.gold_letter for ex in examples]
    for letter in "ABCD":
        assert letters.count(letter) >= 8


def test_sender_only_prompt_does_not_leak_true_line():
    ex = generate_split("dev", 1, 2026)[0]
    prompt = receiver_prompt(ex, inventory_text=None, filler_text=ex.filler_text, has_relay=True)
    assert true_inventory_absent_from_receiver_prompt(prompt, ex)
    assert ex.queried_package in prompt
    assert f"Package {ex.queried_package} is in locker {ex.true_locker}." not in prompt


def test_parse_true_donor_other():
    parsed = parse_answer("Answer: B) locker 7")
    labeled = label_answer(
        parsed, gold_letter="B", gold_locker=7, donor_letter="A", donor_locker=3
    )
    assert labeled.kind == "true"
    labeled = label_answer(
        parse_answer("Answer: A) locker 3"),
        gold_letter="B",
        gold_locker=7,
        donor_letter="A",
        donor_locker=3,
    )
    assert labeled.kind == "donor"
    labeled = label_answer(
        parse_answer("I am not sure"),
        gold_letter="B",
        gold_locker=7,
        donor_letter="A",
        donor_locker=3,
    )
    assert labeled.kind == "unparsed"
    assert labeled.parse_ok is False
    partial = label_answer(
        parse_answer("Answer: B) locker 3"),
        gold_letter="B",
        gold_locker=7,
        donor_letter="A",
        donor_locker=3,
    )
    assert partial.kind == "true_partial"
    from src.parse import flags_from_kind

    flags = flags_from_kind("donor_partial")
    assert flags["follows_donor"] is False
    assert flags["donor_partial"] is True
    assert flags["correct"] is False
    flags = flags_from_kind("true")
    assert flags["correct"] is True
    assert flags["follows_donor"] is False
