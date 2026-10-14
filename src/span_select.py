"""Shared-position complete-span selection from a completed cache.

Every head keeps the same token indices. Text decode uses those indices, not
the union of headwise selections. Spans are newline-delimited lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from src.cache_utils import kv_blob_size_bytes
from src.parse_plan4 import RESULT_LINE

DecodeOne = Callable[[Sequence[int]], str]


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    text: str

    @property
    def n_tokens(self) -> int:
        return max(0, int(self.end) - int(self.start))


def contiguous_runs(positions: Sequence[int]) -> list[tuple[int, int]]:
    ordered = sorted({int(p) for p in positions})
    if not ordered:
        return []
    runs = []
    start = prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev + 1))
        start = prev = pos
    runs.append((start, prev + 1))
    return runs


def positions_from_spans(spans: Sequence[TokenSpan]) -> list[int]:
    out: list[int] = []
    for span in spans:
        out.extend(range(int(span.start), int(span.end)))
    return out


def line_spans(
    token_ids: Sequence[int],
    *,
    start: int,
    end: int,
    decode: DecodeOne,
) -> list[TokenSpan]:
    """Complete lines in [start, end), split where a decoded token contains a newline."""
    start = int(start)
    end = int(end)
    if end <= start:
        return []
    spans: list[TokenSpan] = []
    line_start = start
    for idx in range(start, end):
        piece = decode([int(token_ids[idx])])
        if "\n" in piece or idx == end - 1:
            text = decode([int(t) for t in token_ids[line_start : idx + 1]])
            spans.append(TokenSpan(line_start, idx + 1, text))
            line_start = idx + 1
    return [span for span in spans if span.n_tokens > 0]


def result_spans(spans: Sequence[TokenSpan]) -> list[TokenSpan]:
    """Keep complete lines that state a carrier q/eligibility result."""
    found: dict[str, TokenSpan] = {}
    for span in spans:
        match = RESULT_LINE.search(span.text or "")
        if match:
            found[match.group("carrier").upper()] = span
    return [found[c] for c in ("P", "Q", "R", "S") if c in found]


def mean_span_score(scores: Sequence[float], span: TokenSpan) -> float:
    if span.n_tokens <= 0:
        return float("-inf")
    total = 0.0
    for idx in range(span.start, span.end):
        total += float(scores[idx])
    return total / span.n_tokens


def select_spans_by_score(
    spans: Sequence[TokenSpan],
    scores: Sequence[float],
    *,
    target_n: int,
) -> list[TokenSpan]:
    """Take highest-scoring complete lines until at least target_n tokens, no truncation."""
    usable = [span for span in spans if span.n_tokens > 0]
    if not usable:
        return []
    ranked = sorted(usable, key=lambda span: mean_span_score(scores, span), reverse=True)
    chosen: list[TokenSpan] = []
    n = 0
    want = max(1, int(target_n))
    for span in ranked:
        if chosen and n >= want:
            break
        chosen.append(span)
        n += span.n_tokens
    return sorted(chosen, key=lambda span: span.start)


def shared_position_scores(layer_steps: Sequence[Sequence[torch.Tensor]]) -> list[float]:
    """Mean over layers and heads. layer_steps[step][layer] is (B, H, L)."""
    acc = None
    n = 0
    for step in layer_steps:
        for layer in step:
            if layer.dim() != 3:
                raise ValueError(f"Expected attention scores (B, H, L), got {tuple(layer.shape)}")
            head_mean = layer[0].detach().float().mean(dim=0)
            acc = head_mean if acc is None else acc + head_mean
            n += 1
    if acc is None or n <= 0:
        raise ValueError("No attention scores to average")
    return (acc / n).tolist()


def decode_positions(
    token_ids: Sequence[int],
    positions: Sequence[int],
    decode: DecodeOne,
) -> str:
    parts = []
    for start, end in contiguous_runs(positions):
        piece = decode([int(t) for t in token_ids[start:end]]).strip()
        if piece:
            parts.append(piece)
    return "\n".join(parts)


def gather_cache_positions(blob: dict, positions: Sequence[int]) -> dict:
    """Keep the same indices on every head and layer. Does not mutate `blob`."""
    ordered = sorted({int(p) for p in positions})
    layers = blob.get("layers") or []
    if not layers:
        raise ValueError("Empty cache")
    n_full = int(layers[0][0].shape[-2])
    if not ordered:
        raise ValueError("Shared selection is empty")
    if ordered[0] < 0 or ordered[-1] >= n_full:
        raise ValueError(f"Position out of range for cache length {n_full}")
    idx = torch.tensor(ordered, dtype=torch.long)
    new_layers = []
    for key, value in layers:
        gather_idx = idx.view(1, 1, -1).expand(key.shape[0], key.shape[1], -1)
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
        new_layers.append(
            (
                key.gather(2, gather_idx).contiguous().clone(),
                value.gather(2, gather_idx).contiguous().clone(),
            )
        )
    out = {"layers": new_layers}
    if kv_blob_size_bytes(out) <= 0:
        raise RuntimeError("Gathered cache is empty")
    return out


def caches_equal(left: dict, right: dict) -> bool:
    """Exact CPU tensor equality. Used to check identity gather."""
    a = left.get("layers") or []
    b = right.get("layers") or []
    if len(a) != len(b):
        return False
    for (ak, av), (bk, bv) in zip(a, b):
        if ak.shape != bk.shape or av.shape != bv.shape:
            return False
        if not torch.equal(ak, bk) or not torch.equal(av, bv):
            return False
    return True


def with_initial_positions(
    positions: Sequence[int],
    *,
    n_full: int,
    n_initial: int = 4,
) -> tuple[list[int], list[int]]:
    """Keep the original first n_initial cache slots plus `positions`. Count extras."""
    kept = {int(p) for p in positions if 0 <= int(p) < int(n_full)}
    extra = [i for i in range(min(int(n_initial), int(n_full))) if i not in kept]
    ordered = sorted(kept.union(extra))
    return ordered, extra
