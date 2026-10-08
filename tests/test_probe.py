"""Probe attention slicing. No GPU."""

from __future__ import annotations

import torch

from src.probe import probe_attentions_to_steps
from src.prompt_select import select_prompt_indices


def test_probe_keeps_only_original_prompt_keys():
    # B=1, H=2, Q=3 probe tokens, K=10 = 6 prompt + 2 latent + 2 probe
    attn = torch.zeros((1, 2, 3, 10), dtype=torch.float32)
    attn[0, 0, :, 4] = 1.0
    attn[0, 1, :, 5] = 2.0
    attn[:, :, :, 8:] = 9.0  # probe keys; must be discarded
    attn[:, :, :, 6:8] = 7.0  # latent keys; must be discarded
    steps = probe_attentions_to_steps([attn, attn], prompt_len=6)
    assert len(steps) == 1
    assert len(steps[0]) == 2
    scores = steps[0][0]
    assert scores.shape == (1, 2, 6)
    assert float(scores[0, 0, 4]) == 3.0
    assert float(scores[0, 1, 5]) == 6.0
    assert float(scores.max()) == 6.0
    chosen = select_prompt_indices(scores[0], k_eff=1, sink_len=0, valid_len=6, mode="attention")
    assert chosen.tolist() == [[4], [5]]
