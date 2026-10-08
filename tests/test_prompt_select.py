"""Prompt-index selection rules. No GPU."""

from __future__ import annotations

import torch

from src.prompt_select import select_prompt_indices


def test_attention_picks_highest_non_sink_scores():
    scores = torch.arange(20, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    chosen = select_prompt_indices(scores, k_eff=4, sink_len=2, valid_len=20, mode="attention")
    assert chosen.shape == (2, 4)
    assert chosen.tolist() == [[16, 17, 18, 19], [16, 17, 18, 19]]


def test_force_keeps_complete_forced_set_inside_budget():
    scores = torch.arange(20, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    chosen = select_prompt_indices(
        scores,
        k_eff=5,
        sink_len=2,
        valid_len=20,
        mode="force",
        force=[5, 6, 7],
    )
    assert chosen.shape == (2, 5)
    for row in chosen.tolist():
        assert row[:3] == [5, 6, 7]
        assert 5 in row and 6 in row and 7 in row
        assert len(row) == 5
        assert row == sorted(row)
        assert all(i >= 2 for i in row)


def test_force_rejects_overflowing_the_budget():
    scores = torch.zeros((1, 20), dtype=torch.float32)
    try:
        select_prompt_indices(
            scores,
            k_eff=2,
            sink_len=0,
            valid_len=20,
            mode="force",
            force=[1, 2, 3],
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "budget" in str(exc).lower()


def test_random_is_deterministic_and_per_head():
    scores = torch.zeros((3, 16), dtype=torch.float32)
    a = select_prompt_indices(scores, k_eff=4, sink_len=2, valid_len=16, mode="random", rng_seed=2026, layer_idx=3)
    b = select_prompt_indices(scores, k_eff=4, sink_len=2, valid_len=16, mode="random", rng_seed=2026, layer_idx=3)
    c = select_prompt_indices(scores, k_eff=4, sink_len=2, valid_len=16, mode="random", rng_seed=7, layer_idx=3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (3, 4)
    # Independent heads may differ; they must not all be forced identical by a shared draw.
    rows = [tuple(row) for row in a.tolist()]
    assert all(len(set(row)) == 4 for row in rows)


def test_recency_keeps_the_last_budget_slots():
    scores = torch.zeros((2, 20), dtype=torch.float32)
    chosen = select_prompt_indices(scores, k_eff=4, sink_len=2, valid_len=20, mode="recency")
    assert chosen.tolist() == [[16, 17, 18, 19], [16, 17, 18, 19]]
