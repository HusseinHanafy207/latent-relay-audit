"""CPU copies of KV caches so compression cannot mutate the stored full-cache reference."""

from __future__ import annotations

from typing import Any

import torch


def _layers(past: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if past is None:
        return []
    if hasattr(past, "key_cache"):
        return list(zip(past.key_cache, past.value_cache))
    return [(layer[0], layer[1]) for layer in past]


def cache_to_cpu(past: Any) -> dict[str, Any]:
    layers = []
    for key, value in _layers(past):
        layers.append((key.detach().cpu().contiguous().clone(), value.detach().cpu().contiguous().clone()))
    return {"layers": layers}


def clone_cache_blob(blob: dict[str, Any]) -> dict[str, Any]:
    """Independent CPU copy. Generating from one clone must not change another."""
    return {
        "layers": [
            (key.detach().contiguous().clone(), value.detach().contiguous().clone())
            for key, value in blob.get("layers", [])
        ]
    }


def _seq_gather_index(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    idx = torch.tensor(positions, dtype=torch.long)
    gather_idx = idx.view(1, 1, -1).expand(tensor.shape[0], tensor.shape[1], -1)
    return gather_idx.unsqueeze(-1).expand(-1, -1, -1, tensor.shape[-1])


def cache_slice_delta(left: dict[str, Any], right: dict[str, Any], positions: list[int]) -> dict[str, Any]:
    """Exact equality and max/mean abs difference on a shared position set.

    Sequence lengths may differ. Batch, head, and head-dim must match, and every
    requested index must exist in both caches.
    """
    ordered = sorted({int(p) for p in positions})
    if not ordered:
        raise ValueError("Empty position slice")
    a = left.get("layers") or []
    b = right.get("layers") or []
    if len(a) != len(b):
        raise ValueError("Cache layer counts differ")
    max_k = max_v = 0.0
    mean_k = mean_v = 0.0
    n = 0
    keys_equal = True
    values_equal = True
    for (ak, av), (bk, bv) in zip(a, b):
        if (
            ak.shape[0] != bk.shape[0]
            or ak.shape[1] != bk.shape[1]
            or ak.shape[-1] != bk.shape[-1]
            or av.shape[0] != bv.shape[0]
            or av.shape[1] != bv.shape[1]
            or av.shape[-1] != bv.shape[-1]
        ):
            raise ValueError("Cache shapes differ")
        if ordered[-1] >= ak.shape[-2] or ordered[-1] >= bk.shape[-2]:
            raise ValueError("Position out of range for cache slice")
        gi_a = _seq_gather_index(ak, ordered)
        gi_b = _seq_gather_index(bk, ordered)
        aks, avs = ak.gather(2, gi_a), av.gather(2, gi_a)
        bks, bvs = bk.gather(2, gi_b), bv.gather(2, gi_b)
        keys_equal = keys_equal and bool(torch.equal(aks, bks))
        values_equal = values_equal and bool(torch.equal(avs, bvs))
        dk = (aks.float() - bks.float()).abs()
        dv = (avs.float() - bvs.float()).abs()
        max_k = max(max_k, float(dk.max().item()))
        max_v = max(max_v, float(dv.max().item()))
        mean_k += float(dk.mean().item())
        mean_v += float(dv.mean().item())
        n += 1
    return {
        "n_positions": len(ordered),
        "n_layers": n,
        "keys_equal": keys_equal,
        "values_equal": values_equal,
        "equal": keys_equal and values_equal,
        "keys_max_abs": max_k,
        "values_max_abs": max_v,
        "keys_mean_abs": (mean_k / n) if n else None,
        "values_mean_abs": (mean_v / n) if n else None,
    }


def mix_cache_kv_at_positions(
    key_src: dict[str, Any],
    value_src: dict[str, Any],
    *,
    base: dict[str, Any],
    positions: list[int],
) -> dict[str, Any]:
    """Keep `base` except at `positions`, where keys come from `key_src` and values from `value_src`."""
    ordered = sorted({int(p) for p in positions})
    if not ordered:
        raise ValueError("Empty mix positions")
    base_layers = base.get("layers") or []
    key_layers = key_src.get("layers") or []
    value_layers = value_src.get("layers") or []
    if not (len(base_layers) == len(key_layers) == len(value_layers)):
        raise ValueError("Cache layer counts differ")
    out = []
    for (bk, bv), (kk, _kv), (_vk, vv) in zip(base_layers, key_layers, value_layers):
        if bk.shape != kk.shape or bv.shape != vv.shape:
            raise ValueError("Cannot mix mismatched cache shapes")
        gi = _seq_gather_index(bk, ordered)
        key = bk.detach().contiguous().clone()
        val = bv.detach().contiguous().clone()
        key.scatter_(2, gi, kk.gather(2, gi))
        val.scatter_(2, gi, vv.gather(2, gi))
        out.append((key.contiguous(), val.contiguous()))
    return {"layers": out}


def mix_cache_kv(key_src: dict[str, Any], value_src: dict[str, Any]) -> dict[str, Any]:
    """Packed cache with keys from `key_src` and values from `value_src`. Same positions."""
    a = key_src.get("layers") or []
    b = value_src.get("layers") or []
    if len(a) != len(b):
        raise ValueError("Cache layer counts differ")
    out = []
    for (ak, av), (bk, bv) in zip(a, b):
        if ak.shape != bk.shape or av.shape != bv.shape:
            raise ValueError(f"Cannot mix shapes {tuple(ak.shape)}/{tuple(av.shape)} with {tuple(bk.shape)}/{tuple(bv.shape)}")
        out.append((ak.detach().contiguous().clone(), bv.detach().contiguous().clone()))
    if not out:
        raise ValueError("Empty cache")
    return {"layers": out}


def concat_cache_blobs(blobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate packed caches along the sequence axis. Does not rerotate."""
    nonempty = [blob for blob in blobs if blob.get("layers")]
    if not nonempty:
        raise ValueError("No caches to concatenate")
    if len(nonempty) == 1:
        return clone_cache_blob(nonempty[0])
    n_layers = len(nonempty[0]["layers"])
    if any(len(blob["layers"]) != n_layers for blob in nonempty):
        raise ValueError("Cache layer counts differ")
    out = []
    for i in range(n_layers):
        keys = torch.cat([blob["layers"][i][0] for blob in nonempty], dim=-2)
        vals = torch.cat([blob["layers"][i][1] for blob in nonempty], dim=-2)
        out.append((keys.contiguous().clone(), vals.contiguous().clone()))
    return {"layers": out}


def n_kv_positions(blob: dict[str, Any]) -> int:
    layers = blob.get("layers") or []
    if not layers:
        return 0
    return int(layers[0][0].shape[-2])


def kv_blob_size_bytes(blob: dict[str, Any]) -> int:
    total = 0
    for key, value in blob.get("layers") or []:
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def cache_to_device(blob: dict[str, Any], device: torch.device):
    from transformers import DynamicCache

    layers = tuple((k.to(device), v.to(device)) for k, v in blob["layers"])
    return DynamicCache.from_legacy_cache(layers)


def attentions_to_cpu(all_steps: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
    return [[tensor.detach().cpu().contiguous().clone() for tensor in step] for step in all_steps]


def attentions_to_device(all_steps: list[list[torch.Tensor]], device: torch.device) -> list[list[torch.Tensor]]:
    return [[tensor.to(device) for tensor in step] for step in all_steps]


def kv_size_bytes(past: Any) -> int:
    total = 0
    for key, value in _layers(past):
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


def cache_is_finite(past: Any) -> bool:
    for key, value in _layers(past):
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            return False
    return True


def metadata_is_finite(meta: Any) -> bool:
    if meta is None:
        return True
    if isinstance(meta, dict):
        return all(metadata_is_finite(v) for v in meta.values())
    if torch.is_tensor(meta):
        return bool(torch.isfinite(meta).all().item())
    return True
