"""Causal prefixes that end at the eligibility instruction, before private facts.

These prefixes are shared task context. They do not include shipment facts,
carrier sheets, or the calculation block.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from src.prompts_plan4 import ELIGIBILITY_RULE

CARRIER_HEADING = ("Carrier P", "Carrier Q", "Carrier R", "Carrier S")
N_EXPECTED_TEMPLATES = 3


def causal_prefix_from_ids(
    full_ids: Sequence[int],
    instruction: dict,
    *,
    decode=None,
) -> dict:
    """Recorded tokens from the start of A's sequence through the instruction span."""
    ids = [int(t) for t in full_ids]
    start = int(instruction["start"])
    end = int(instruction["end"])
    if not (0 <= start < end <= len(ids)):
        raise RuntimeError(f"Instruction span [{start}, {end}) falls outside {len(ids)} tokens.")
    prefix_ids = ids[:end]
    instr_pos = list(range(start, end))
    rec = {
        "ids": prefix_ids,
        "n": len(prefix_ids),
        "instruction_start": start,
        "instruction_end": end,
        "instruction_pos": instr_pos,
        "instruction_ids": [ids[p] for p in instr_pos],
        "template_key": tuple(prefix_ids),
        "contains_rule": None,
        "contains_shipment": None,
        "contains_carrier_heading": None,
        "private_facts": False,
    }
    if decode is not None:
        text = decode(prefix_ids)
        rec["text"] = text
        rec["contains_rule"] = ELIGIBILITY_RULE in text
        rec["contains_shipment"] = "Shipment:" in text
        rec["contains_carrier_heading"] = any(name in text for name in CARRIER_HEADING)
        rec["private_facts"] = bool(rec["contains_shipment"] or rec["contains_carrier_heading"])
        if not rec["contains_rule"]:
            raise RuntimeError("Causal prefix does not contain the eligibility rule.")
        if rec["private_facts"]:
            raise RuntimeError("Causal prefix includes shipment facts or carrier data.")
    return rec


def group_generic_templates(records: Sequence[dict]) -> dict[str, dict]:
    """Group sheets that share an identical causal prefix."""
    grouped: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[tuple(rec["template_key"])].append(rec)
    templates = {}
    for i, (key, rows) in enumerate(sorted(grouped.items(), key=lambda item: (len(item[0]), item[0][:8]))):
        name = f"generic-{i}"
        difficulties = sorted({str(r.get("difficulty") or "") for r in rows if r.get("difficulty")})
        if len(difficulties) == 1:
            name = difficulties[0]
        instr_ids = {tuple(r["instruction_ids"]) for r in rows}
        starts = {int(r["instruction_start"]) for r in rows}
        if len(instr_ids) != 1 or len(starts) != 1:
            raise RuntimeError(f"Template {name} does not share one instruction span.")
        templates[name] = {
            "template_id": name,
            "n": len(rows),
            "n_prefix": len(key),
            "ids": list(key),
            "instruction_start": rows[0]["instruction_start"],
            "instruction_end": rows[0]["instruction_end"],
            "instruction_pos": list(rows[0]["instruction_pos"]),
            "instruction_ids": list(rows[0]["instruction_ids"]),
            "a_state_keys": [r["a_state_key"] for r in rows],
            "difficulties": difficulties,
        }
    return templates


def template_id_for(prefix: dict, templates: dict[str, dict]) -> str:
    key = tuple(prefix["ids"])
    for name, rec in templates.items():
        if tuple(rec["ids"]) == key:
            return name
    raise KeyError("Causal prefix is not one of the frozen generic templates.")
