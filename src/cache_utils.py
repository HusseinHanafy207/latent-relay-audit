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
