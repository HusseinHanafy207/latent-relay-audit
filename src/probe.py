"""Turn a receiver-question forward pass into Headwise-style prompt scores."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def probe_attentions_to_steps(
    layer_attentions: Sequence[torch.Tensor],
    *,
    prompt_len: int,
) -> list[list[torch.Tensor]]:
    """Keep only original prompt keys; discard probe and latent key positions.

    Each layer tensor is (B, H, Q_probe, K_total). Returns one fake latent-style
    step: list over layers of (B, H, prompt_len), summed over probe queries.
    """
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    step = []
    for attn in layer_attentions:
        if attn.dim() != 4:
            raise ValueError(f"Expected attention (B, H, Q, K), got {tuple(attn.shape)}")
        if int(attn.shape[-1]) < prompt_len:
            raise ValueError(f"Attention key length {attn.shape[-1]} < prompt_len {prompt_len}")
        prompt_keys = attn[:, :, :, :prompt_len]
        step.append(prompt_keys.sum(dim=2).detach())
    return [step]
