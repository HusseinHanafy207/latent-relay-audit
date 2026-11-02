"""Attention replay for matched two-row forwards (Qwen3, transformers 4.57.x).

Row `donor` and row `target` hold two caches with the same length, token IDs,
masks and positions. In "replace" mode, every attention layer gives the target
row the donor row's attention weights and applies them to the target row's own
values. In "off" mode the same FP32 eager computation runs without replacement.
Nothing is stored between layers or forwards.

The function is registered under its own name together with the eager mask
builder. Without the mask registration transformers creates no causal mask for
an unknown attention name.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Sequence

import torch

ATTN_NAME = "relay_replay"
MODES = ("off", "replace")


class ReplayController:
    def __init__(self) -> None:
        self.reset()

    def reset(self, mode: str = "off", *, donor: int = 0, target: int = 1) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown replay mode {mode!r}")
        self.mode = mode
        self.donor = int(donor)
        self.target = int(target)
        self.n_calls = 0
        self.n_replaced = 0
        self.layers_seen: set[int] = set()


CONTROLLER = ReplayController()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def relay_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
):
    if kwargs.get("sliding_window"):
        raise RuntimeError("Attention replay does not handle sliding-window layers.")
    groups = int(getattr(module, "num_key_value_groups", 1))
    k = repeat_kv(key, groups).float()
    v = repeat_kv(value, groups).float()
    scores = torch.matmul(query.float(), k.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        mask = attention_mask[:, :, :, : k.shape[-2]]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.float()
    weights = torch.softmax(scores, dim=-1)
    ctl = CONTROLLER
    ctl.n_calls += 1
    layer = getattr(module, "layer_idx", None)
    if layer is not None:
        ctl.layers_seen.add(int(layer))
    if ctl.mode == "replace":
        if weights.shape[0] <= max(ctl.donor, ctl.target):
            raise RuntimeError("Replay needs both donor and target rows in the batch.")
        weights[ctl.target] = weights[ctl.donor]
        ctl.n_replaced += 1
    out = torch.matmul(weights, v).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


def register() -> None:
    try:
        from transformers import AttentionInterface
    except ImportError:  # pragma: no cover
        from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask

    AttentionInterface.register(ATTN_NAME, relay_attention_forward)
    AttentionMaskInterface.register(ATTN_NAME, eager_mask)


def _configs(model) -> list[Any]:
    seen: dict[int, Any] = {}
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            seen[id(cfg)] = cfg
    return list(seen.values())


@contextmanager
def replay_attention(model, mode: str, *, donor: int = 0, target: int = 1):
    """Switch every attention layer to the replay function for the duration."""
    register()
    configs = _configs(model)
    for cfg in configs:
        types = getattr(cfg, "layer_types", None) or []
        if any(t != "full_attention" for t in types):
            raise RuntimeError("Attention replay supports full-attention layers only.")
    previous = [cfg._attn_implementation for cfg in configs]
    CONTROLLER.reset(mode, donor=donor, target=target)
    for cfg in configs:
        cfg._attn_implementation = ATTN_NAME
    try:
        yield CONTROLLER
    finally:
        for cfg, prev in zip(configs, previous):
            cfg._attn_implementation = prev
        CONTROLLER.mode = "off"


def token_index_at_char(offsets: Sequence[Sequence[int]], char_pos: int) -> int | None:
    for idx, (start, end) in enumerate(offsets):
        if int(start) <= int(char_pos) < int(end):
            return idx
    return None


def dist_stats(ref_logits: torch.Tensor, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row KL(ref || branch) and argmax agreement. Inputs (rows, vocab)."""
    ref_lp = torch.log_softmax(ref_logits.float(), dim=-1)
    lp = torch.log_softmax(logits.float(), dim=-1)
    kl = (ref_lp.exp() * (ref_lp - lp)).sum(dim=-1).clamp_min(0.0)
    agree = ref_logits.argmax(dim=-1) == logits.argmax(dim=-1)
    return {"kl": kl, "agree": agree, "logp": lp}


def label_decision(logits_row: torch.Tensor, tok_eligible: int, tok_ineligible: int) -> dict:
    """Binary decision between the two label first tokens at one position."""
    lp = torch.log_softmax(logits_row.float(), dim=-1)
    le = float(lp[int(tok_eligible)].item())
    li = float(lp[int(tok_ineligible)].item())
    return {
        "decision": "eligible" if le >= li else "ineligible",
        "logp_eligible": le,
        "logp_ineligible": li,
        "label_mass": float(torch.exp(torch.tensor(le)) + torch.exp(torch.tensor(li))),
    }


def within_tolerance(stats: dict, *, max_mean_kl: float, min_agree: float, labels_equal: bool) -> bool:
    return bool(labels_equal) and stats["mean_kl"] <= max_mean_kl and stats["argmax_agree"] >= min_agree
