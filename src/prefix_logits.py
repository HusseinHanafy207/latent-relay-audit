"""Compare teacher-forced continuation logits for prefix-cache reuse."""

from __future__ import annotations

from typing import Any

import torch


def logit_delta_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """left/right are (K, vocab) continuation logits."""
    if left.shape != right.shape:
        raise ValueError(f"Logit shapes {tuple(left.shape)} vs {tuple(right.shape)}")
    a = left.detach().float()
    b = right.detach().float()
    diff = (a - b).abs()
    argmax_eq = a.argmax(dim=-1) == b.argmax(dim=-1)
    return {
        "n_steps": int(a.shape[0]),
        "vocab": int(a.shape[1]),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "median_abs": float(diff.median().item()),
        "argmax_agree": float(argmax_eq.float().mean().item()),
        "n_argmax_agree": int(argmax_eq.sum().item()),
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def within_numerical_baseline(
    probe: dict[str, Any],
    *,
    repeat: dict[str, Any],
    ordinary: dict[str, Any],
) -> bool:
    """True if the gather-vs-reencode gap is not far above ordinary FP16 noise.

    This is a gate for the implementation check, not a task-accuracy claim.
    """
    if not probe.get("finite") or not repeat.get("finite") or not ordinary.get("finite"):
        return False
    cap = max(10.0 * float(repeat["max_abs"]), 3.0 * float(ordinary["max_abs"]), 2.0)
    return float(probe["max_abs"]) <= cap
