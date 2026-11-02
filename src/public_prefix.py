"""Complete public task prefix and its matched neutral controls (twelve carriers).

The public prefix is A's recorded token sequence from the chat header through
the end of the eligibility instruction: role, local task and capacity rules,
no shipment request and no carrier facts. The neutral controls are the frozen,
boundary-checked woodland and library sequences from the nbound stage; within a
template they keep the length, chat header, instruction IDs and positions, and
change only the ordinary context before the instruction.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from src.neutral_select import CONTEXT_NEUTRAL_1, CONTEXT_NEUTRAL_2
from src.plan4_timing_task import CARRIERS_12

NEUTRAL_KEYS = {"woodland": "instr_neutral_1", "library": "instr_neutral_2"}
NEUTRAL_PASSAGE_IDS = {"woodland": CONTEXT_NEUTRAL_1, "library": CONTEXT_NEUTRAL_2}
CONTEXTS = ("task", "woodland", "library")
PRIVATE_MARKERS = ("Shipment request:", "Shipment:")
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def private_prefix_hits(text: str, examples: Sequence = ()) -> list[str]:
    """Private-information markers found in a public prefix. Empty means clean."""
    hits = [m for m in PRIVATE_MARKERS if m in text]
    for c in CARRIERS_12:
        if re.search(rf"\bCarrier {c}\b", text):
            hits.append(f"Carrier {c}")
        if re.search(rf"(?m)^\s*{c}\s*[:|]", text):
            hits.append(f"line {c}:")
    for ex in examples:
        if ex.request_line and ex.request_line in text:
            hits.append(f"request line of {ex.example_id}")
        for doc, name in ((ex.a_text, "A"), (ex.b_text, "B")):
            for line in doc.splitlines():
                line = line.strip()
                if line and line != ex.request_line and len(line) >= 8 and line in text:
                    hits.append(f"{name} line of {ex.example_id}: {line[:40]}")
    if IM_END in text:
        hits.append(IM_END)
    if text.count(IM_START) != 1 or not text.startswith(IM_START):
        hits.append("chat header count")
    return hits


def boundary_record(prefix_ids: Sequence[int], a_ids: Sequence[int], decode: Callable, *, n_look: int = 8) -> dict:
    """How the public prefix meets the rest of A's recorded sequence."""
    prefix = [int(t) for t in prefix_ids]
    a = [int(t) for t in a_ids]
    n = len(prefix)
    nxt = a[n : n + n_look]
    joined = decode(a[: n + n_look])
    return {
        "n_prefix": n,
        "prefix_matches_a": a[:n] == prefix,
        "tail_tokens": [[t, decode([t])] for t in prefix[-n_look:]],
        "next_a_tokens": [[t, decode([t])] for t in nxt],
        "ends_with_paragraph_break": decode(prefix).endswith("\n\n"),
        "no_merge_across_boundary": decode(prefix) + decode(nxt) == joined,
        "next_a_text": decode(nxt),
    }


def neutral_prefixes(nbound_contexts: dict, task_ids_by_template: dict) -> dict:
    """Frozen woodland/library IDs per template, verified against the twelve-carrier task prefix."""
    out = {}
    for template, task_ids in task_ids_by_template.items():
        recs = nbound_contexts[template]
        task_ids = [int(t) for t in task_ids]
        saved_task = [int(t) for t in recs["instr_generic"]["ids"]]
        if saved_task != task_ids:
            raise RuntimeError(f"Saved task prefix for {template} differs from the twelve-carrier prefix.")
        entry = {"task": {"ids": task_ids}}
        for name, key in NEUTRAL_KEYS.items():
            rec = recs[key]
            ids = [int(t) for t in rec["ids"]]
            check = matched_control_check(
                task_ids,
                ids,
                header_len=int(rec["header_len"]),
                instruction_start=int(rec["instruction_start"]),
                instruction_end=int(rec["instruction_end"]),
            )
            if not check["ok"]:
                raise RuntimeError(f"Neutral {name} for {template} is not a matched control: {check}")
            if rec.get("passage_id") != NEUTRAL_PASSAGE_IDS[name]:
                raise RuntimeError(f"Saved {key} for {template} is not the {name} passage.")
            entry[name] = {
                "ids": ids,
                "header_len": int(rec["header_len"]),
                "instruction_start": int(rec["instruction_start"]),
                "instruction_end": int(rec["instruction_end"]),
                "decoded_prefix": rec.get("decoded_prefix"),
                "decoded_join": rec.get("decoded_join"),
                "matched": check,
            }
        entry["task"].update(
            {
                "header_len": entry["woodland"]["header_len"],
                "instruction_start": entry["woodland"]["instruction_start"],
                "instruction_end": entry["woodland"]["instruction_end"],
            }
        )
        out[template] = entry
    return out


def matched_control_check(
    task_ids: Sequence[int],
    other_ids: Sequence[int],
    *,
    header_len: int,
    instruction_start: int,
    instruction_end: int,
) -> dict:
    task = [int(t) for t in task_ids]
    other = [int(t) for t in other_ids]
    row = {
        "length_equal": len(task) == len(other),
        "header_equal": task[:header_len] == other[:header_len],
        "instruction_equal": task[instruction_start:instruction_end] == other[instruction_start:instruction_end],
        "instruction_is_suffix": instruction_end == len(task),
        "context_differs": task[header_len:instruction_start] != other[header_len:instruction_start],
    }
    row["ok"] = all(row.values())
    return row


def packed_result_slots(n_head: int, n_result: int) -> list[int]:
    return list(range(int(n_head), int(n_head) + int(n_result)))
