"""Merged receiver for the frozen query adapter, and handoff timing helpers.

Merging folds the LoRA update into B's q_proj weights (W + scale * B @ A,
computed in fp32 and rounded once to fp16), so B runs one projection instead
of the base projection plus two fp32 adapter matmuls. The merged tensors live
beside the original weights and are swapped in only for B's own forwards; the
original weights are never written, so A, support construction and every
baseline still run on the exact original model.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import torch
from torch import nn

from src.qadapt import QueryAdapter, decoder_layers
from src.rope_rerotate import rerotate_cache_keys
from src.two_agent import _time_model_prefill_decode

MODES = ("unadapted", "unmerged", "merged")


# ---------------------------------------------------------------- merging


@torch.no_grad()
def merged_q_weights(adapter: QueryAdapter) -> list[nn.Parameter]:
    out = []
    for m in adapter.modules:
        w = m.base.weight
        delta = (m.lora_B.detach().float() @ m.lora_A.detach().float()) * m.scale
        out.append(nn.Parameter((w.detach().float() + delta).to(w.dtype), requires_grad=False))
    return out


@torch.no_grad()
def merge_stats(adapter: QueryAdapter, merged: Sequence[torch.Tensor]) -> dict:
    """How much of the fp32 update survives rounding the merged weight to the base dtype."""
    per = []
    for m, wm in zip(adapter.modules, merged):
        w = m.base.weight.detach().float()
        delta = (m.lora_B.detach().float() @ m.lora_A.detach().float()) * m.scale
        applied = wm.detach().float() - w
        err = applied - delta
        dn = float(delta.norm())
        per.append(
            {
                "delta_rel_norm": dn / max(float(w.norm()), 1e-30),
                "rounding_max_abs": float(err.abs().max()),
                "rounding_rel_to_delta": float(err.norm()) / max(dn, 1e-30),
                "unchanged_fraction": float((wm.detach() == m.base.weight.detach()).float().mean()),
                "applied_vs_delta_cos": float(
                    torch.nn.functional.cosine_similarity(applied.flatten(), delta.flatten(), dim=0)
                )
                if dn > 0
                else 1.0,
            }
        )
    worst = {k: max(p[k] for p in per) for k in ("delta_rel_norm", "rounding_max_abs", "rounding_rel_to_delta", "unchanged_fraction")}
    worst["min_applied_vs_delta_cos"] = min(p["applied_vs_delta_cos"] for p in per)
    return {"n_layers": len(per), "worst": worst, "layers": per}


def q_weight_digest(adapter: QueryAdapter) -> str:
    """SHA-256 over the original q_proj weights, byte for byte."""
    h = hashlib.sha256()
    for m in adapter.modules:
        t = m.base.weight.detach().contiguous().cpu()
        h.update(str(tuple(t.shape)).encode() + str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def attached_as_built(adapter: QueryAdapter, originals: Sequence[torch.Tensor]) -> bool:
    """Every q_proj is the adapter wrapper again and holds its original weight object."""
    layers = decoder_layers(adapter.model)
    return all(
        layer.self_attn.q_proj is m and m.base.weight is w for layer, m, w in zip(layers, adapter.modules, originals)
    )


@contextmanager
def receiver_weights(adapter: QueryAdapter, mode: str, merged: Sequence[nn.Parameter] | None = None) -> Iterator[None]:
    """B's weights for one forward block.

    unadapted: bare original q_proj (no wrapper in the path).
    unmerged:  wrapper with the fp32 adapter path switched on (the evaluated candidate).
    merged:    bare q_proj holding the merged weight.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown receiver mode {mode}")
    if mode == "unmerged":
        with adapter.on():
            yield
        return
    if mode == "merged" and (merged is None or len(merged) != len(adapter.modules)):
        raise ValueError("Merged mode needs one merged weight per adapted layer.")
    layers = decoder_layers(adapter.model)
    originals = [m.base.weight for m in adapter.modules]
    if any(layer.self_attn.q_proj is not m for layer, m in zip(layers, adapter.modules)):
        raise RuntimeError("Adapter wrappers are not attached as built.")
    try:
        for i, (layer, m) in enumerate(zip(layers, adapter.modules)):
            layer.self_attn.q_proj = m.base
            if mode == "merged":
                m.base.weight = merged[i]
        yield
    finally:
        for layer, m, w in zip(layers, adapter.modules, originals):
            m.base.weight = w
            layer.self_attn.q_proj = m


# ---------------------------------------------------------------- handoff pieces


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def lcp(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def gather_on_device(layers, positions: Sequence[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Gather sequence positions on whatever device the cache lives on."""
    if not layers:
        raise ValueError("Empty cache")
    idx = torch.tensor([int(p) for p in positions], dtype=torch.long, device=layers[0][0].device)
    return [(k.index_select(-2, idx), v.index_select(-2, idx)) for k, v in layers]


def compact_sender(layers, src_pos: Sequence[int], new_pos: Sequence[int], model) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A's side of the compact handoff: select its result states and re-rotate them to packed positions."""
    blob = {"layers": gather_on_device(layers, src_pos)}
    return rerotate_cache_keys(blob, old_positions=src_pos, new_positions=new_pos, model=model)["layers"]


def cat_layers(head, tail) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.cat([hk, tk], dim=-2), torch.cat([hv, tv], dim=-2)) for (hk, hv), (tk, tv) in zip(head, tail)]


def to_device(layers, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(k.to(device), v.to(device)) for k, v in layers]


def clone_layers(layers) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(k.clone(), v.clone()) for k, v in layers]


def cache_len(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2]) if len(cache) else 0


def cache_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "to_legacy_cache"):
        return [(k, v) for k, v in cache.to_legacy_cache()]
    return [(layer.keys, layer.values) for layer in cache.layers]


@torch.no_grad()
def extend(model, cache, ids, *, n_past: int, cursor: int, device) -> None:
    """Prefill tokens onto a cache in place (no logits needed). `ids` may be a list or a 1-D tensor."""
    if len(ids) == 0:
        return
    if isinstance(ids, torch.Tensor):
        x = ids.view(1, -1).to(device=device, dtype=torch.long)
    else:
        x = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=device)
    mask = torch.ones((1, int(n_past) + x.size(1)), dtype=torch.long, device=device)
    pos = (int(cursor) + torch.arange(x.size(1), device=device, dtype=torch.long)).view(1, -1)
    model.model(input_ids=x, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, return_dict=True)


def _suffix(wrapper, cache, ids: Sequence[int], *, n_past: int, cursor: int, max_new: int, device) -> dict:
    x = torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=device)
    attn = torch.ones_like(x)
    with _time_model_prefill_decode(wrapper.model) as timed:
        t0 = now()
        texts, past, _pos, lens = wrapper.generate_text_batch(
            x,
            attn,
            pos_cursor=torch.tensor([int(cursor)], dtype=torch.long, device=device),
            max_new_tokens=int(max_new),
            temperature=0.0,
            top_p=1.0,
            past_key_values=cache,
            past_attention_mask=torch.ones((1, int(n_past)), dtype=torch.long, device=device),
        )
        total = now() - t0
    n_new = int(lens[0]) if lens else 0
    return {
        "text": texts[0],
        "n_new": n_new,
        "truncated": n_new >= int(max_new),
        "prefill_s": float(timed["prefill_s"]),
        "decode_s": float(timed["decode_s"]),
        "total_s": float(total),
        "n_prefill_tokens": len(ids),
        "past": past,
    }


def two_pass(
    wrapper,
    *,
    cache,
    n_past: int,
    cursor: int,
    plan_ids: Sequence[int],
    start: int,
    shared: int,
    answer_of: Callable[[str], tuple[list[int], str]],
    plan_max: int,
    answer_max: int,
    device,
) -> dict:
    """B's plan and answer passes; the tokens both prompts share are prefilled once.

    `cache` already covers plan_ids[:start] (text: the public header; KV: none of
    the prompt). plan_ids[start:shared] is prefilled once, then each pass continues
    from that point and the cache is cropped back between passes.
    """
    plan_ids = [int(t) for t in plan_ids]
    if not (0 <= start <= shared < len(plan_ids)):
        raise ValueError(f"Bad shared prefix: start={start} shared={shared} n={len(plan_ids)}")
    if cache_len(cache) != int(n_past):
        raise RuntimeError("Cache length does not match n_past.")
    t0 = now()
    extend(wrapper.model, cache, plan_ids[start:shared], n_past=n_past, cursor=cursor, device=device)
    shared_s = now() - t0
    n_sh, c_sh = int(n_past) + shared - start, int(cursor) + shared - start
    if cache_len(cache) != n_sh:
        raise RuntimeError("Shared prefill did not extend the cache as expected.")
    plan = _suffix(wrapper, cache, plan_ids[shared:], n_past=n_sh, cursor=c_sh, max_new=plan_max, device=device)
    t0 = now()
    cache = plan.pop("past")
    cache.crop(n_sh)
    crop_s = now() - t0
    if cache_len(cache) != n_sh:
        raise RuntimeError("Crop did not restore the shared prefix.")
    t0 = time.perf_counter()
    answer_ids, answer_prompt = answer_of(plan["text"])
    answer_encode_s = time.perf_counter() - t0
    answer_ids = [int(t) for t in answer_ids]
    if answer_ids[:shared] != plan_ids[:shared] or len(answer_ids) <= shared:
        raise RuntimeError("The answer prompt does not continue the prefilled shared prefix.")
    ans = _suffix(wrapper, cache, answer_ids[shared:], n_past=n_sh, cursor=c_sh, max_new=answer_max, device=device)
    ans.pop("past")
    return {
        "plan": plan,
        "answer": ans,
        "answer_prompt": answer_prompt,
        "shared_prefill_s": float(shared_s),
        "n_shared_tokens": shared - start,
        "crop_s": float(crop_s),
        "answer_encode_s": float(answer_encode_s),
    }
