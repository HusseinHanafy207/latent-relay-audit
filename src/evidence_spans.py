"""Locate inventory-line, package-name, and locker-value tokens in A's prompt.

Token presence in the full cache is not the audit. The audit asks whether a
compressed *head* kept the evidence-line name tokens and the locker-value
tokens on that same line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.inventory import InventoryExample
from src.prompts_inventory import sender_prompt


def inventory_line_text(ex: InventoryExample, record_index: int) -> str:
    pkg = ex.packages[record_index]
    loc = ex.true_lockers[record_index]
    return f"{record_index + 1}. Package {pkg} is in locker {loc}."


def evidence_line_text(ex: InventoryExample) -> str:
    return inventory_line_text(ex, ex.queried_index)


def unrelated_record_index(ex: InventoryExample) -> int:
    n = len(ex.packages)
    if n < 2:
        raise ValueError("Need at least two inventory records for the unrelated-line control")
    other = (ex.queried_index + max(1, n // 2)) % n
    if other == ex.queried_index:
        other = (ex.queried_index + 1) % n
    return other


def unrelated_line_text(ex: InventoryExample) -> str:
    return inventory_line_text(ex, unrelated_record_index(ex))


def _unique_span(haystack: str, needle: str) -> tuple[int, int]:
    start = haystack.find(needle)
    if start < 0:
        raise ValueError(f"Could not find {needle!r} in prompt")
    if haystack.find(needle, start + 1) >= 0:
        raise ValueError(f"Needle is not unique: {needle!r}")
    return start, start + len(needle)


def _span_inside(haystack: str, needle: str, lo: int, hi: int) -> tuple[int, int]:
    region = haystack[lo:hi]
    start = region.find(needle)
    if start < 0:
        raise ValueError(f"Could not find {needle!r} inside the inventory line")
    abs_start = lo + start
    return abs_start, abs_start + len(needle)


@dataclass(frozen=True)
class CharSpans:
    line: tuple[int, int]
    package: tuple[int, int]
    locker: tuple[int, int]
    line_text: str
    package_text: str
    locker_text: str
    record_index: int


def char_spans_for_record(prompt: str, ex: InventoryExample, record_index: int) -> CharSpans:
    line = inventory_line_text(ex, record_index)
    line_span = _unique_span(prompt, line)
    package = ex.packages[record_index]
    locker = str(ex.true_lockers[record_index])
    pkg_span = _span_inside(prompt, package, line_span[0], line_span[1])
    locker_span = _span_inside(prompt, f"locker {locker}.", line_span[0], line_span[1])
    number_start = locker_span[0] + len("locker ")
    number_end = number_start + len(locker)
    if prompt[number_start:number_end] != locker:
        raise ValueError(f"Locker digits misaligned for record {record_index}")
    return CharSpans(
        line=line_span,
        package=pkg_span,
        locker=(number_start, number_end),
        line_text=line,
        package_text=package,
        locker_text=locker,
        record_index=record_index,
    )


def overlapping_token_indices(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    lo, hi = span
    out = []
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if end > lo and start < hi:
            out.append(i)
    if not out:
        raise ValueError(f"No tokens overlap character span {span}")
    return out


def _flat_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def _flat_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        value
        and isinstance(value[0], (list, tuple))
        and value[0]
        and isinstance(value[0][0], (list, tuple))
    ):
        value = value[0]
    return [(int(pair[0]), int(pair[1])) for pair in value]


def token_offsets(tokenizer: Any, text: str, input_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Map each prompt token to a character span in the rendered chat string."""
    ids = [int(x) for x in input_ids]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    enc_ids = _flat_ids(encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded.input_ids)
    offsets = _flat_offsets(encoded["offset_mapping"] if hasattr(encoded, "__getitem__") else encoded.offset_mapping)
    if enc_ids != ids:
        raise ValueError(
            f"Tokenizer offsets do not match sender input_ids (len {len(enc_ids)} vs {len(ids)})"
        )
    if len(offsets) != len(ids):
        raise ValueError("offset_mapping length does not match input_ids")
    return offsets


@dataclass
class TokenSpans:
    record_index: int
    line_text: str
    line: list[int]
    package: list[int]
    locker: list[int]
    n_line: int = field(init=False)
    n_package: int = field(init=False)
    n_locker: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_line = len(self.line)
        self.n_package = len(self.package)
        self.n_locker = len(self.locker)

    def budget_tokens(self, sink_len: int) -> list[int]:
        """Line tokens that consume the kv_budget (not already kept as sink)."""
        return [i for i in self.line if i >= sink_len]


def token_spans_for_line_text(
    tokenizer: Any,
    rendered: str,
    input_ids: Sequence[int],
    line_text: str,
    package_name: str,
    locker_text: str,
    record_index: int = -1,
) -> TokenSpans:
    """Map a concrete inventory line onto token ids. Does not use a gold index."""
    line_span = _unique_span(rendered, line_text)
    pkg_span = _span_inside(rendered, package_name, line_span[0], line_span[1])
    locker_span = _span_inside(rendered, f"locker {locker_text}.", line_span[0], line_span[1])
    number_start = locker_span[0] + len("locker ")
    number_end = number_start + len(locker_text)
    if rendered[number_start:number_end] != locker_text:
        raise ValueError("Locker digits misaligned in matched line")
    offsets = token_offsets(tokenizer, rendered, input_ids)
    return TokenSpans(
        record_index=record_index,
        line_text=line_text,
        line=overlapping_token_indices(offsets, line_span),
        package=overlapping_token_indices(offsets, pkg_span),
        locker=overlapping_token_indices(offsets, (number_start, number_end)),
    )


def token_spans_for_record(
    tokenizer: Any,
    rendered: str,
    input_ids: Sequence[int],
    ex: InventoryExample,
    record_index: int,
) -> TokenSpans:
    chars = char_spans_for_record(rendered, ex, record_index)
    return token_spans_for_line_text(
        tokenizer,
        rendered,
        input_ids,
        chars.line_text,
        chars.package_text,
        chars.locker_text,
        record_index=record_index,
    )


def sender_token_spans(
    tokenizer: Any,
    rendered: str,
    input_ids: Sequence[int],
    ex: InventoryExample,
) -> tuple[TokenSpans, TokenSpans]:
    evidence = token_spans_for_record(tokenizer, rendered, input_ids, ex, ex.queried_index)
    unrelated = token_spans_for_record(tokenizer, rendered, input_ids, ex, unrelated_record_index(ex))
    return evidence, unrelated


def match_slot_count(source: TokenSpans, n_slots: int, sink_len: int) -> list[int]:
    """Reserve exactly n_slots from source's non-sink line tokens when possible.

    Prefers a window that still covers that record's package name and locker value.
    """
    available = source.budget_tokens(sink_len)
    if n_slots <= 0:
        return []
    if len(available) <= n_slots:
        return list(available)
    must = [i for i in (source.package + source.locker) if i in set(available)]
    if not must:
        return available[:n_slots]
    lo, hi = min(must), max(must)
    i0 = next(i for i, t in enumerate(available) if t >= lo)
    i1 = max(i for i, t in enumerate(available) if t <= hi)
    while (i1 - i0 + 1) < n_slots:
        if i0 > 0:
            i0 -= 1
        elif i1 + 1 < len(available):
            i1 += 1
        else:
            break
    while (i1 - i0 + 1) > n_slots:
        left_excess = available[i0] < lo
        right_excess = available[i1] > hi
        if right_excess:
            i1 -= 1
        elif left_excess:
            i0 += 1
        else:
            i1 -= 1
    return available[i0 : i0 + n_slots]


def prompt_text_for_spans(ex: InventoryExample) -> str:
    """Unrendered sender body. Use the rendered chat string for token mapping."""
    return sender_prompt(ex.true_inventory_text, ex)


def kept_flags(selected: Iterable[int], sink_len: int, spans: TokenSpans) -> dict[str, Any]:
    kept = set(int(x) for x in selected) | set(range(sink_len))
    n_pkg = sum(t in kept for t in spans.package)
    n_loc = sum(t in kept for t in spans.locker)
    n_line = sum(t in kept for t in spans.line)
    return {
        "kept_package": n_pkg == spans.n_package and spans.n_package > 0,
        "kept_locker": n_loc == spans.n_locker and spans.n_locker > 0,
        "kept_both": (n_pkg == spans.n_package and n_loc == spans.n_locker and spans.n_package > 0 and spans.n_locker > 0),
        "kept_line": n_line == spans.n_line and spans.n_line > 0,
        "n_package_kept": n_pkg,
        "n_package": spans.n_package,
        "n_locker_kept": n_loc,
        "n_locker": spans.n_locker,
        "n_line_kept": n_line,
        "n_line": spans.n_line,
    }


def retention_grid(
    selected_by_layer_head: Sequence[Sequence[Sequence[int]]],
    sink_len: int,
    spans: TokenSpans,
) -> dict[str, Any]:
    """selected_by_layer_head[layer][head] -> prompt indices (excluding sink)."""
    n_layers = len(selected_by_layer_head)
    n_heads = len(selected_by_layer_head[0]) if n_layers else 0
    kept_package = []
    kept_locker = []
    kept_both = []
    kept_line = []
    n_both = n_pkg = n_loc = n_line = 0
    n_cells = n_layers * n_heads
    for layer in selected_by_layer_head:
        row_pkg, row_loc, row_both, row_line = [], [], [], []
        for head_idx in layer:
            flags = kept_flags(head_idx, sink_len, spans)
            row_pkg.append(int(flags["kept_package"]))
            row_loc.append(int(flags["kept_locker"]))
            row_both.append(int(flags["kept_both"]))
            row_line.append(int(flags["kept_line"]))
            n_pkg += flags["kept_package"]
            n_loc += flags["kept_locker"]
            n_both += flags["kept_both"]
            n_line += flags["kept_line"]
        kept_package.append(row_pkg)
        kept_locker.append(row_loc)
        kept_both.append(row_both)
        kept_line.append(row_line)
    return {
        "n_layers": n_layers,
        "n_kv_heads": n_heads,
        "n_cells": n_cells,
        "n_kept_package": n_pkg,
        "n_kept_locker": n_loc,
        "n_kept_both": n_both,
        "n_kept_line": n_line,
        "frac_kept_package": n_pkg / n_cells if n_cells else None,
        "frac_kept_locker": n_loc / n_cells if n_cells else None,
        "frac_kept_both": n_both / n_cells if n_cells else None,
        "frac_kept_line": n_line / n_cells if n_cells else None,
        "kept_package": kept_package,
        "kept_locker": kept_locker,
        "kept_both": kept_both,
        "kept_line": kept_line,
        "note": "A cell counts only if that layer/head kept every token of the span. Tokens kept by some other head do not count.",
    }
