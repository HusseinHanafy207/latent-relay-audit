"""Compact message-boundary layouts for frozen cons_instr_sink token IDs.

Result-line IDs stay as recorded. Added tokens are only a newline separator
between the initial four tokens and those result lines, and/or a proper
end-of-turn close before B's separately rendered user turn.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.plan4_task import SINK_INITIAL

LAYOUTS = ("current", "sep", "close", "both")
SEP_KIND = "separator"
CLOSE_KIND = "turn_close"
EXTRA_KIND = "extra"
EXTRA_SEP_KIND = "extra_separator"


def layout_flags(name: str) -> dict:
    if name not in LAYOUTS:
        raise ValueError(f"Unknown boundary layout {name}")
    return {
        "separator": name in {"sep", "both"},
        "turn_close": name in {"close", "both"},
    }


def split_sink_positions(sink_pos: Sequence[int], *, n_initial: int = SINK_INITIAL) -> dict:
    ordered = [int(p) for p in sink_pos]
    initial = list(range(int(n_initial)))
    if ordered[: int(n_initial)] != initial:
        raise RuntimeError(f"Sink positions do not start with the first {n_initial} slots: {ordered[:8]}")
    result = [p for p in ordered if p >= int(n_initial)]
    if sorted(initial + result) != ordered:
        raise RuntimeError("Sink positions are not initial slots plus later result slots.")
    return {"initial_pos": initial, "result_pos": result}


def newline_ids(tokenizer) -> list[int]:
    ids = [int(t) for t in tokenizer.encode("\n", add_special_tokens=False)]
    if not ids:
        raise RuntimeError("Tokenizer produced no newline token.")
    return ids


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    n = list(needle)
    h = list(haystack)
    if not n:
        return 0
    for i in range(0, len(h) - len(n) + 1):
        if h[i : i + len(n)] == n:
            return i
    return -1


def turn_close_ids(tokenizer) -> list[int]:
    """Tokens the chat template places after user content, before the next turn."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        rendered = tokenizer.decode(list(rendered), skip_special_tokens=False)
    full = [int(t) for t in tokenizer.encode(rendered, add_special_tokens=False)]
    x_ids = [int(t) for t in tokenizer.encode("X", add_special_tokens=False)]
    at = _find_subsequence(full, x_ids)
    if at < 0:
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if end_id is None or int(end_id) < 0:
            end_id = tokenizer.eos_token_id
        if end_id is None:
            raise RuntimeError("Could not find a turn-close token.")
        return [int(end_id)] + newline_ids(tokenizer)
    close = full[at + len(x_ids) :]
    if not close:
        raise RuntimeError("Chat template did not close the user turn after content.")
    return close


def compact_slot_ranges(composed: dict) -> dict:
    """Packed-cache indices for prefix, optional extra block, and result-line pieces."""
    prefix: list[int] = []
    extra: list[int] = []
    extra_sep: list[int] = []
    result: list[int] = []
    cursor = 0
    for piece in composed.get("pieces") or []:
        n = int(piece.get("n") or 0)
        slots = list(range(cursor, cursor + n))
        kind = piece.get("kind")
        if kind == "result":
            result.extend(slots)
        elif kind == EXTRA_KIND:
            extra.extend(slots)
        elif kind == EXTRA_SEP_KIND:
            extra_sep.extend(slots)
        else:
            prefix.extend(slots)
        cursor += n
    if cursor != int(composed.get("n") or 0):
        raise RuntimeError("Compact piece lengths do not match the composed sequence.")
    if not prefix or not result:
        raise RuntimeError("Compact layout needs both prefix slots and result slots.")
    return {
        "prefix": prefix,
        "extra": extra,
        "extra_sep": extra_sep,
        "result": result,
        "n_prefix": len(prefix),
        "n_extra": len(extra),
        "n_extra_sep": len(extra_sep),
        "n_result": len(result),
        "result_start": result[0],
    }


def compose_compact_ids(
    initial_ids: Sequence[int],
    result_ids: Sequence[int],
    *,
    sep_ids: Sequence[int],
    close_ids: Sequence[int],
    layout: str,
) -> dict:
    """Build one compact prefix. Result IDs are copied, not retokenized."""
    flags = layout_flags(layout)
    initial = [int(t) for t in initial_ids]
    result = [int(t) for t in result_ids]
    if not initial or not result:
        raise RuntimeError("Compact layout needs initial tokens and result tokens.")
    pieces = [{"kind": "initial", "ids": initial, "n": len(initial)}]
    if flags["separator"]:
        sep = [int(t) for t in sep_ids]
        if not sep:
            raise RuntimeError("Separator layout needs newline token IDs.")
        pieces.append({"kind": SEP_KIND, "ids": sep, "n": len(sep)})
    pieces.append({"kind": "result", "ids": result, "n": len(result)})
    if flags["turn_close"]:
        close = [int(t) for t in close_ids]
        if not close:
            raise RuntimeError("Turn-close layout needs end-of-turn token IDs.")
        pieces.append({"kind": CLOSE_KIND, "ids": close, "n": len(close)})
    ids = [t for piece in pieces for t in piece["ids"]]
    return {
        "layout": layout,
        "ids": ids,
        "pieces": pieces,
        "n": len(ids),
        "n_added": sum(piece["n"] for piece in pieces if piece["kind"] in {SEP_KIND, CLOSE_KIND}),
        "n_initial": len(initial),
        "n_result": len(result),
        "result_ids": result,
        **flags,
    }


def compose_compact_with_extra(
    initial_ids: Sequence[int],
    extra_ids: Sequence[int],
    result_ids: Sequence[int],
    *,
    sep_ids: Sequence[int],
    extra_pos: Sequence[int],
    result_pos: Sequence[int],
    extra_sep: bool = True,
) -> dict:
    """Prefix, then one extra block, then an optional newline, then result lines.

    Frozen before the instruction comparison. Equal-size random controls use the
    same piece order, extra length, and extra-separator. Result IDs are copied.
    """
    initial = [int(t) for t in initial_ids]
    extra = [int(t) for t in extra_ids]
    result = [int(t) for t in result_ids]
    sep = [int(t) for t in sep_ids]
    src_extra = [int(p) for p in extra_pos]
    src_result = [int(p) for p in result_pos]
    if not initial or not extra or not result or not sep:
        raise RuntimeError("Extra compact layout needs initial tokens, extra tokens, result tokens, and a newline.")
    if len(src_extra) != len(extra):
        raise RuntimeError("Extra token IDs and source positions have different lengths.")
    if len(src_result) != len(result):
        raise RuntimeError("Result token IDs and source positions have different lengths.")
    pieces = [
        {"kind": "initial", "ids": initial, "n": len(initial), "source": "reconstruct"},
        {"kind": SEP_KIND, "ids": sep, "n": len(sep), "source": "forward"},
        {"kind": EXTRA_KIND, "ids": extra, "n": len(extra), "source": "gather", "source_pos": src_extra},
    ]
    if extra_sep:
        pieces.append({"kind": EXTRA_SEP_KIND, "ids": sep, "n": len(sep), "source": "forward"})
    pieces.append({"kind": "result", "ids": result, "n": len(result), "source": "gather", "source_pos": src_result})
    ids = [t for piece in pieces for t in piece["ids"]]
    n_extra_sep = len(sep) if extra_sep else 0
    return {
        "layout": "sep_extra",
        "ids": ids,
        "pieces": pieces,
        "n": len(ids),
        "n_added": len(sep) + n_extra_sep,
        "n_initial": len(initial),
        "n_extra": len(extra),
        "n_extra_sep": n_extra_sep,
        "n_result": len(result),
        "n_prefix": len(initial) + len(sep),
        "result_start": len(initial) + len(sep) + len(extra) + n_extra_sep,
        "result_ids": result,
        "extra_ids": extra,
        "separator": True,
        "turn_close": False,
        "extra_separator": extra_sep,
    }


def junction_flags(prefix_text: str) -> dict:
    text = prefix_text or ""
    return {
        "has_youp": bool(re.search(r"YouP\s*:", text)),
        "has_you_newline_p": bool(re.search(r"You\nP\s*:", text)),
        "has_im_end": "<|im_end|>" in text,
        "starts_im_start_user": text.startswith("<|im_start|>user"),
    }


def render_handoff_view(tokenizer, prefix_ids: Sequence[int], b_turn_ids: Sequence[int]) -> dict:
    """Complete prefix plus B's encoded user turn, including special tokens."""
    prefix = [int(t) for t in prefix_ids]
    b_turn = [int(t) for t in b_turn_ids]
    full = prefix + b_turn
    prefix_text = tokenizer.decode(prefix, skip_special_tokens=False)
    b_text = tokenizer.decode(b_turn, skip_special_tokens=False)
    full_text = tokenizer.decode(full, skip_special_tokens=False)
    flags = junction_flags(prefix_text)
    flags["im_end_before_b_turn"] = bool(flags["has_im_end"])
    return {
        "prefix_ids": prefix,
        "b_turn_ids": b_turn,
        "full_ids": full,
        "prefix_text": prefix_text,
        "b_turn_text": b_text,
        "full_text": full_text,
        "n_prefix": len(prefix),
        "n_b_turn": len(b_turn),
        "n_full": len(full),
        "junction": flags,
        "prefix_head": prefix_text[:120],
        "prefix_tail": prefix_text[-80:],
    }
