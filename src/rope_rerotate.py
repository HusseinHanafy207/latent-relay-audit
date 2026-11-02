"""Rotary correction when selected KV entries are placed at new positions.

Stored keys already include R(p). Moving an entry from original position p
to assigned position q uses k_new = R(q) R(p)^T k_old. Values are unchanged.
The algebra runs in float32, then the keys return to the cache dtype.
"""

from __future__ import annotations

from typing import Sequence

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def invert_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Undo apply_rope at the same (cos, sin). rotate_half(rotate_half(x)) = -x."""
    return x * cos - rotate_half(x) * sin


def broadcast_cos_sin(cos: torch.Tensor, sin: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match RoPE tables to keys (B, H, S, D)."""
    while cos.dim() < keys.dim():
        if cos.shape[0] == keys.shape[0] and cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
    if cos.shape[-2] != keys.shape[-2]:
        raise ValueError(f"RoPE length {cos.shape[-2]} != key length {keys.shape[-2]}")
    rotary_dim = int(cos.shape[-1])
    if rotary_dim != int(keys.shape[-1]) and rotary_dim != int(keys.shape[-1]) // 2:
        if rotary_dim > int(keys.shape[-1]):
            raise ValueError(f"RoPE dim {rotary_dim} longer than key dim {keys.shape[-1]}")
    return cos, sin


def rerotate_keys(
    keys: torch.Tensor,
    cos_old: torch.Tensor,
    sin_old: torch.Tensor,
    cos_new: torch.Tensor,
    sin_new: torch.Tensor,
) -> torch.Tensor:
    """FP32 undo R(p), apply R(q), write back in the original dtype."""
    src_dtype = keys.dtype
    k = keys.float()
    c_old, s_old = broadcast_cos_sin(cos_old.float(), sin_old.float(), k)
    c_new, s_new = broadcast_cos_sin(cos_new.float(), sin_new.float(), k)
    rotary_dim = int(c_old.shape[-1])
    if rotary_dim == k.shape[-1]:
        unrot = invert_rope(k, c_old, s_old)
        out = apply_rope(unrot, c_new, s_new)
    else:
        rot, rest = k[..., :rotary_dim], k[..., rotary_dim:]
        unrot = invert_rope(rot, c_old, s_old)
        out = torch.cat([apply_rope(unrot, c_new, s_new), rest], dim=-1)
    if not torch.isfinite(out).all():
        raise RuntimeError("RoPE re-rotation produced nonfinite keys.")
    return out.to(dtype=src_dtype)


def dummy_cos_sin(positions: Sequence[int], dim: int, *, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard half-dim frequency table. Tests only; the live model uses its own rotary."""
    pos = torch.tensor([int(p) for p in positions], dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def find_rotary(model) -> torch.nn.Module:
    inner = getattr(model, "model", model)
    if hasattr(inner, "rotary_emb") and inner.rotary_emb is not None:
        return inner.rotary_emb
    layers = getattr(inner, "layers", None)
    if layers:
        attn = layers[0].self_attn
        if hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
            return attn.rotary_emb
    raise RuntimeError("Could not find rotary_emb on the loaded model.")


def rope_tables(
    rotary,
    positions: Sequence[int],
    ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for `positions` in float32, shaped (1, S, rotary_dim)."""
    device = ref.device
    pos = torch.tensor([int(p) for p in positions], device=device, dtype=torch.long).view(1, -1)
    dummy = ref.detach().float()[:1, :1, :1, :]
    out = rotary(dummy, pos)
    if not isinstance(out, tuple) or len(out) < 2:
        raise RuntimeError("rotary_emb did not return (cos, sin).")
    cos, sin = out[0].float(), out[1].float()
    if cos.dim() == 4:
        cos = cos[:, 0]
        sin = sin[:, 0]
    if cos.dim() != 3:
        raise RuntimeError(f"Unexpected RoPE table rank {tuple(cos.shape)}")
    return cos.contiguous(), sin.contiguous()


def rerotate_cache_keys(
    blob: dict,
    *,
    old_positions: Sequence[int],
    new_positions: Sequence[int],
    model,
) -> dict:
    """Rewrite packed keys from original positions to assigned positions. Values stay."""
    old = [int(p) for p in old_positions]
    new = [int(p) for p in new_positions]
    if len(old) != len(new):
        raise ValueError("old_positions and new_positions must have the same length")
    layers = blob.get("layers") or []
    if not layers:
        raise ValueError("Empty cache")
    n = int(layers[0][0].shape[-2])
    if len(old) != n:
        raise ValueError(f"Position list length {len(old)} != packed cache {n}")
    rotary = find_rotary(model)
    ref = layers[0][0]
    cos_old, sin_old = rope_tables(rotary, old, ref)
    cos_new, sin_new = rope_tables(rotary, new, ref)
    out_layers = []
    for key, value in layers:
        new_key = rerotate_keys(key, cos_old, sin_old, cos_new, sin_new)
        if not torch.isfinite(value).all():
            raise RuntimeError("Cache values were already nonfinite.")
        out_layers.append((new_key.contiguous().clone(), value.detach().contiguous().clone()))
    return {"layers": out_layers}


def _layer_abs_max(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    diff = (left.float() - right.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def cache_key_delta(left: dict, right: dict) -> dict:
    """Max/mean absolute key difference across layers. Values are not compared."""
    a = left.get("layers") or []
    b = right.get("layers") or []
    if len(a) != len(b):
        raise ValueError("Cache layer counts differ")
    max_abs = 0.0
    mean_acc = 0.0
    n = 0
    for (ak, _av), (bk, _bv) in zip(a, b):
        if ak.shape != bk.shape:
            raise ValueError(f"Key shapes {tuple(ak.shape)} vs {tuple(bk.shape)}")
        layer_max, layer_mean = _layer_abs_max(ak, bk)
        max_abs = max(max_abs, layer_max)
        mean_acc += layer_mean
        n += 1
    return {
        "n_layers": n,
        "max_abs": max_abs,
        "mean_abs": (mean_acc / n) if n else None,
        "finite": all(
            bool(torch.isfinite(ak).all().item()) and bool(torch.isfinite(bk).all().item())
            for (ak, _av), (bk, _bv) in zip(a, b)
        ),
    }


def identity_rerotate_stats(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> dict:
    """Re-rotating to the same positions should preserve keys."""
    same = rerotate_keys(keys, cos, sin, cos, sin)
    max_abs, mean_abs = _layer_abs_max(keys, same)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "finite": bool(torch.isfinite(same).all().item()),
    }


def rope_op_check(blob: dict, model, *, shift: int = 16) -> dict:
    """Identity and forward-and-back re-rotation on a packed cache.

    This checks the RoPE operation. It does not prove that a selected cache is
    semantically compatible with the receiver.
    """
    layers = blob.get("layers") or []
    if not layers:
        raise ValueError("Empty cache")
    n = int(layers[0][0].shape[-2])
    origin = list(range(n))
    mid = list(range(int(shift), int(shift) + n))
    ident = rerotate_cache_keys(blob, old_positions=origin, new_positions=origin, model=model)
    ident_stats = cache_key_delta(blob, ident)
    ident_again = rerotate_cache_keys(ident, old_positions=origin, new_positions=origin, model=model)
    repeat = cache_key_delta(ident, ident_again)
    moved = rerotate_cache_keys(blob, old_positions=origin, new_positions=mid, model=model)
    back = rerotate_cache_keys(moved, old_positions=mid, new_positions=origin, model=model)
    roundtrip = cache_key_delta(blob, back)
    cap = max(10.0 * float(repeat["max_abs"]), 2e-2)
    passed = (
        bool(ident_stats.get("finite"))
        and bool(roundtrip.get("finite"))
        and bool(repeat.get("finite"))
        and float(ident_stats["max_abs"]) <= cap
        and float(roundtrip["max_abs"]) <= cap
    )
    return {
        "n_positions": n,
        "shift": int(shift),
        "identity": ident_stats,
        "repeat": repeat,
        "roundtrip": roundtrip,
        "cap": cap,
        "passed": passed,
        "note": (
            "Identity and p→q→p re-rotation should stay within the measured numerical "
            "cap. Passing does not mean selected KV is usable by B."
        ),
    }
