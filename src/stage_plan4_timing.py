"""4- vs 12-carrier timing of the frozen generic+R candidate.

Keep the candidate frozen. Accuracy stays the 90-example hold confirmation:
generic+R 83/90 labels vs full KV 88/90 and result text 87/90, so recovery
within five points of both references is unestablished. This stage reports
that gap alongside costs.

Measure separately: message preparation, receiver prefill, and decoding of
both B calls. Prefill generic support once per template and reuse it.
Cache mebibytes are packed-state sizes, not measured network traffic.

4-carrier: previously evaluated plan4_hold base families, frozen A artifacts.
12-carrier: new timing-only split, generated here. Do not rewrite plan4_dev,
plan4_fresh, or plan4_hold. Do not train an adapter. Do not add calculations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import n_kv_positions
from src.generic_select import (
    N_EXPECTED_TEMPLATES,
    causal_prefix_from_ids,
    group_generic_templates,
    template_id_for,
)
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse_plan4 import RESULT_LINE, is_think_collapse, score_sender_message, written_working_metrics
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    CARRIERS,
    LEVELS,
    SINK_INITIAL,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
)
from src.plan4_timing_task import (
    CARRIERS_12,
    N_CARRIERS_12,
    N_TIMING_FAMILIES,
    assert_timing12_invariants,
    ensure_timing12_jsonl,
    gold_eligibility_statuses_n,
    gold_quantities_n,
    select_timing4_examples,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    ELIGIBILITY_RULE,
    FRAMING_INSTRUCTION,
    b_answer_after_message_plan_prompt,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
    b_reason_with_message_prompt,
    ordinary_sender_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
)
from src.prompts_plan4_timing import (
    b_answer_after_message_plan_prompt_n,
    b_answer_kv_instruct_prompt_n,
    b_reason_kv_instruct_prompt_n,
    b_reason_with_message_prompt_n,
    encode_prompt_for,
    written_result_handoff_n,
)
from src.span_select import decode_positions, line_spans, positions_from_spans, with_initial_positions
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_boundary import PAYLOAD_NOTE, _added_token_ids, _annotate_plan, _build_recon, _build_transfer, _layouts_for_spans
from src.stage_plan4_cal import _b_row
from src.stage_plan4_comm import _a_extra
from src.stage_plan4_confirm import _attach_baselines, _copy_hold_artifacts
from src.stage_plan4_generic import _collect_templates, _prefill_templates
from src.stage_plan4_instr import EXTRA_SEP, _build_extra_bundle
from src.stage_plan4_kvresult import PREFIX_SOURCE, _rebuild_pair
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_rdep import _append_r, _build_generic_head, _bundle_ids, _pair_state
from src.stage_plan4_recon import _rebuild_a
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block
from src.stage_select import _load_rows
from src.support_select import find_first_text_span, results_marker_span
from src.two_agent import TwoAgentInventory

A_WRITTEN_MAX_NEW_12 = 2048
B_PLAN_MAX_NEW_12 = 384
CONDS = ("generic_r", "result_kv", "full_kv", "text_result")
SPEC = {
    "generic_r": {"kind": "kv", "cache_source": "generic_instr_same_R", "framing": "instruct"},
    "result_kv": {"kind": "kv", "cache_source": "mix_result_a_k_a_v", "framing": "instruct"},
    "full_kv": {"kind": "kv", "cache_source": "rebuilt_full", "framing": "instruct"},
    "text_result": {"kind": "text", "cache_source": "text", "framing": "baseline"},
}
PRESERVED = (
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
    "results_stage_plan4_rdep",
    "results_stage_plan4_nbound",
    "results_stage_plan4_generic",
    "results_stage_plan4_recon",
    "results_stage_plan4_iface",
)
CONFIRM_ACCURACY = {
    "n": 90,
    "source": "results_stage_plan4_confirm on previously evaluated plan4_hold families",
    "labels": {"result_kv": 19, "generic_r": 83, "full_kv": 88, "text_result": 87},
    "choices": {"result_kv": 43, "generic_r": 77, "full_kv": 79, "text_result": 74},
    "generic_r_minus_full_kv_pp": round(100 * (83 - 88) / 90, 2),
    "generic_r_minus_text_pp": round(100 * (83 - 87) / 90, 2),
    "paired_ci_vs_full_kv": [-13.3, 2.2],
    "paired_ci_vs_text": [-11.1, 1.1],
    "note": (
        "Generic support substantially improved selected-state reuse (19/90 to 83/90). "
        "Complete recovery to the reference methods is unestablished: the candidate "
        "is 5.56 points below full KV and 4.44 below text, and both paired intervals "
        "include gaps larger than five points."
    ),
}
INTERPRETATION_GUIDE = (
    "Report the frozen confirmation accuracy gap alongside 4- and 12-carrier costs. "
    "Separate message preparation, receiver prefill, and decoding of both B calls. "
    "Generic support is prepared once per template and reused; do not treat cache "
    "mebibytes as measured network traffic, especially if that support is local. "
    "This timing split is not an independent accuracy confirmation. Do not retune "
    "the candidate from these times."
)
RESULT_LINE_N = re.compile(
    r"(?P<carrier>[A-Z]{1,2})\s*:\s*q\s*=\s*(?P<q>-?\d+)\s*\|\s*(?P<status>eligible|ineligible)\b",
    re.IGNORECASE,
)
B_PLAN_LINE_N = re.compile(
    r"(?P<carrier>[A-Z]{1,2})\s*:\s*(?P<status>eligible|ineligible)\b"
    r"(?:\s*[|,]\s*\$?\s*(?P<price>-?\d+))?"
    r"(?:\s*\|\s*(?P<flag>candidate|dropped))?",
    re.IGNORECASE,
)
ANSWER_N = re.compile(
    r"answer\s*:\s*\(?(?P<letter>[A-L])\)\s*(?:carrier\s*)?(?P<carrier>[A-Z]{1,2})\b",
    re.IGNORECASE,
)
CHOICE_N = re.compile(
    r"(?im)^(?:choice|pick|selected)\s*:\s*(?:carrier\s*)?\*{0,2}([A-Z]{1,2})\*{0,2}\b",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    hold = DATA_DIR / "plan4_hold.jsonl"
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_timing" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_timing")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return hold, timing12, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in (
            "check_summary.json",
            "construction.json",
            "a_artifacts_12.jsonl",
            "timing_summary.json",
        ):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def carriers_of(ex) -> tuple[str, ...]:
    got = getattr(ex, "carriers", None)
    if got:
        return tuple(got)
    return CARRIERS


def gold_status_of(ex) -> dict[str, str]:
    if getattr(ex, "n_carriers", 4) == N_CARRIERS_12:
        return gold_eligibility_statuses_n(ex)
    return gold_eligibility_statuses(ex)


def _process_time_s(row: dict) -> float:
    return float(
        (row.get("rebuild_time_s") or 0.0)
        + (row.get("construct_s") or 0.0)
        + (row.get("plan_time_s") or 0.0)
        + (row.get("receiver_latency_s") or 0.0)
    )


def result_spans_n(spans, carriers: Sequence[str]):
    want = {str(c).upper() for c in carriers}
    found = {}
    for span in spans:
        match = RESULT_LINE_N.search(span.text or "")
        if match:
            name = match.group("carrier").upper()
            if name in want:
                found[name] = span
    return [found[c] for c in carriers if c in found]


def compact_spans_n(full_ids, *, n_prompt: int, n_full: int, decode, carriers: Sequence[str]) -> dict:
    written = line_spans(full_ids, start=n_prompt, end=n_full, decode=decode)
    selected = result_spans_n(written, carriers)
    if len(selected) != len(carriers):
        raise RuntimeError(
            f"Expected {len(carriers)} result lines, found {len(selected)} "
            f"{[s.text[:40] for s in selected]}"
        )
    result_pos = positions_from_spans(selected)
    sink_pos, extra = with_initial_positions(result_pos, n_full=n_full, n_initial=SINK_INITIAL)
    carriers_found = []
    for span in selected:
        match = RESULT_LINE_N.search(span.text or "")
        if match:
            carriers_found.append(match.group("carrier").upper())
    return {
        "result_pos": result_pos,
        "sink_pos": sink_pos,
        "n_sink_extra": len(extra),
        "selected_ids": [int(full_ids[p]) for p in sink_pos],
        "result_text": decode_positions(full_ids, result_pos, decode),
        "n_full": n_full,
        "n_prompt": n_prompt,
        "result_carriers": carriers_found,
        "result_fallback": False,
    }


def extract_timing_pack(
    token_ids,
    *,
    n_prompt: int,
    n_full: int,
    decode,
    a_state_key: str,
    carriers: Sequence[str],
) -> dict:
    ids = [int(t) for t in token_ids]
    failures = []
    instruction = find_first_text_span(ids, ELIGIBILITY_RULE, decode, start=0, end=n_prompt)
    if instruction is None:
        instruction = find_first_text_span(ids, ELIGIBILITY_RULE, decode, start=0, end=n_full)
    if instruction is None:
        failures.append("eligibility instruction paragraph not found")
    written = line_spans(ids, start=n_prompt, end=n_full, decode=decode)
    marker = results_marker_span(written)
    after = [span for span in written if marker is not None and span.start >= marker.end]
    selected = result_spans_n(after if marker is not None else written, carriers)
    if len(selected) != len(carriers):
        failures.append(f"expected {len(carriers)} result lines, found {len(selected)}")
    result_pos = positions_from_spans(selected)
    carriers_found = []
    for span in selected:
        match = RESULT_LINE_N.search(span.text or "")
        if match:
            carriers_found.append(match.group("carrier").upper())
    instr_pos = list(range(instruction.start, instruction.end)) if instruction is not None else []
    prompt_text = decode(ids[: int(n_prompt)]) if int(n_prompt) > 0 else ""
    instruction_before_facts = bool(instr_pos) and instr_pos[-1] < int(n_prompt)
    ship_at = prompt_text.find("Shipment:")
    rule_at = prompt_text.find(ELIGIBILITY_RULE)
    if ship_at >= 0 and rule_at >= 0:
        instruction_before_facts = instruction_before_facts and rule_at < ship_at
    return {
        "ok": not failures,
        "failures": failures,
        "a_state_key": a_state_key,
        "n_full": n_full,
        "n_prompt": n_prompt,
        "instruction": None
        if instruction is None
        else {"start": instruction.start, "end": instruction.end, "text": instruction.text, "n_tokens": instruction.n_tokens},
        "instruction_pos": instr_pos,
        "n_instruction": len(instr_pos),
        "instruction_before_facts": instruction_before_facts,
        "instruction_contains_rule": ELIGIBILITY_RULE in ((instruction.text if instruction is not None else "") or ""),
        "result_pos": result_pos,
        "result_text": decode_positions(ids, result_pos, decode) if result_pos else "",
        "result_carriers": carriers_found,
    }


def prompt_contains_solver_labels_n(prompt: str, ex) -> bool:
    return any(
        f"{c}: q={int(ex.intermediates[c]['q'])} |" in prompt for c in carriers_of(ex)
    )


def _rebuild_a_any(runner, rec, src_ex):
    ordinary = rec.get("ordinary_prompt") or ""
    if prompt_contains_b_private(ordinary, src_ex):
        raise RuntimeError("Recorded A prompt leaked B's prices or budget.")
    if prompt_contains_solver_labels_n(ordinary, src_ex):
        raise RuntimeError("Recorded A prompt leaked solver labels.")
    bundle, rebuild_s = runner.rebuild_cache_from_ids(
        rec["full_ids"],
        prompt=ordinary,
        n_prompt=int(rec["n_prompt_positions"]),
        pos_cursor=int(rec["pos_cursor"]),
        source="rebuild",
        token_align=rec.get("token_align"),
        sender_latency_s=float(rec.get("written_s") or 0.0),
    )
    if n_kv_positions(bundle.cache_cpu) != len(rec["full_ids"]):
        raise RuntimeError("Rebuilt cache length does not match recorded token IDs.")
    return bundle, rebuild_s


def _rebuild_pair_n(runner, rec, src, decode, sep_ids, close_ids, carriers):
    a_bundle, rebuild_s = _rebuild_a_any(runner, rec, src)
    spans = compact_spans_n(
        rec["full_ids"],
        n_prompt=int(rec["n_prompt_positions"]),
        n_full=int(rec["n_kv_positions"]),
        decode=decode,
        carriers=carriers,
    )
    pack = _layouts_for_spans(spans, sep_ids=sep_ids, close_ids=close_ids)
    reconstruct, recon_s, composed = _build_recon(runner, pack, LAYOUT)
    transfer, trans_s, info, _c = _build_transfer(runner, a_bundle, pack, LAYOUT)
    recon_ids = [int(t) for t in reconstruct.full_ids_cpu.view(-1).tolist()]
    trans_ids = [int(t) for t in transfer.full_ids_cpu.view(-1).tolist()]
    if recon_ids != list(composed["ids"]) or trans_ids != list(composed["ids"]):
        raise RuntimeError("Sep-layout token IDs drifted from the compact sequence.")
    from src.boundary_layout import compact_slot_ranges

    slots = compact_slot_ranges(composed)
    return {
        "bundle": a_bundle,
        "meta": rec,
        "rebuild_time_s": rebuild_s,
        "spans": spans,
        "pack": pack,
        "composed": composed,
        "slots": slots,
        "info": info,
        "transfer": transfer,
        "reconstruct": reconstruct,
        "times": {"a_k_a_v": trans_s + recon_s, "reconstruct": recon_s},
    }


def _collect_templates_n(examples, artifacts, decode) -> tuple[dict, dict]:
    records = []
    spans_by_key = {}
    seen = set()
    unique = []
    for ex in examples:
        key = a_state_key(ex)
        if key not in seen:
            unique.append(ex)
            seen.add(key)
    for ex in unique:
        key = a_state_key(ex)
        rec = artifacts[key]
        spans = extract_timing_pack(
            rec["full_ids"],
            n_prompt=int(rec["n_prompt_positions"]),
            n_full=int(rec.get("n_kv_positions") or len(rec["full_ids"])),
            decode=decode,
            a_state_key=key,
            carriers=carriers_of(ex),
        )
        if spans.get("instruction") is None:
            raise RuntimeError(f"No eligibility instruction span for {key}.")
        prefix = causal_prefix_from_ids(rec["full_ids"], spans["instruction"], decode=decode)
        prefix["a_state_key"] = key
        prefix["difficulty"] = ex.difficulty
        records.append(prefix)
        spans["causal_prefix"] = {
            k: prefix[k]
            for k in (
                "n",
                "instruction_start",
                "instruction_end",
                "instruction_pos",
                "instruction_ids",
                "contains_rule",
                "contains_shipment",
                "private_facts",
            )
        }
        spans_by_key[key] = spans
    templates = group_generic_templates(records)
    if len(templates) != N_EXPECTED_TEMPLATES:
        raise RuntimeError(f"Expected {N_EXPECTED_TEMPLATES} generic templates, found {len(templates)}.")
    prefix_by_key = {r["a_state_key"]: r for r in records}
    for key, spans in spans_by_key.items():
        spans["template_id"] = template_id_for(prefix_by_key[key], templates)
    return templates, spans_by_key


def _pair_state_n(runner, rec, src, decode, sep_ids, close_ids, spans, generics):
    carriers = tuple(spans.get("result_carriers") or carriers_of(src))
    pair = _rebuild_pair_n(runner, rec, src, decode, sep_ids, close_ids, carriers)
    if list(spans.get("result_pos") or []) != list(pair["spans"]["result_pos"]):
        raise RuntimeError("Timing-pack result positions drifted from compact R.")
    extra_pos = list(spans["instruction_pos"])
    transferred, trans_s, composed, slots, trans_info = _build_extra_bundle(
        runner, pair, extra_pos, source="transfer_instr_kv"
    )
    generic_rec = generics[spans["template_id"]]
    head, head_s, head_info = _build_generic_head(
        runner, pair, transferred, composed, slots, generic_rec, extra_pos
    )
    generic_r, r_held = _append_r(runner, head, transferred, slots["result"], "generic_instr_same_R")
    ids_equal = _bundle_ids(generic_r) == list(composed["ids"])
    if not ids_equal:
        raise RuntimeError("Generic+R token IDs drifted from the transferred layout.")
    pair.update(
        {
            "instr": spans,
            "composed_extra": composed,
            "slots_extra": slots,
            "transferred": transferred,
            "head": head,
            "mixed": {"generic_r": generic_r},
            "times_by": {"generic_r": trans_s + head_s},
            "info_by": {"generic_r": {**head_info, "result_held_from_transferred": r_held}},
            "trans_info": trans_info,
            "generic_construct_s": float(trans_s + head_s),
            "generic_prefill_s": float(generic_rec.get("prefill_s") or 0.0),
            "generic_reused": bool(generic_rec.get("reused")),
        }
    )
    _attach_baselines(runner, pair)
    pair["times_by"]["result_kv"] = float((pair.get("times") or {}).get("a_k_a_v") or 0.0)
    pair["times_by"]["full_kv"] = float(pair.get("rebuild_time_s") or 0.0)
    pair["times_by"]["text_result"] = 0.0
    return pair


def label_direction_counts(pred_status: dict, gold_status: dict) -> dict:
    """Count eligibility confusions. Independent of how many carriers there are."""
    ie = 0
    ei = 0
    for carrier, gold in gold_status.items():
        pred = pred_status.get(carrier)
        if gold == "ineligible" and pred == "eligible":
            ie += 1
        if gold == "eligible" and pred == "ineligible":
            ei += 1
    return {
        "gold_status": dict(gold_status),
        "plan_ineligible_marked_eligible": ie,
        "plan_eligible_marked_ineligible": ei,
    }


def _score_b_n(text: str, plan_text: str, ex) -> dict:
    carriers = carriers_of(ex)
    gold_status = gold_status_of(ex)
    prices = {c: int(ex.b_facts["prices"][c]) for c in carriers}
    budget = int(ex.b_facts["budget"])
    pred_plan = {c: {"status": None} for c in carriers}
    for match in B_PLAN_LINE_N.finditer(plan_text or ""):
        name = match.group("carrier").upper()
        if name in pred_plan:
            pred_plan[name]["status"] = match.group("status").lower()
    n_status = sum(pred_plan[c]["status"] == gold_status[c] for c in carriers)
    parsed = list(ANSWER_N.finditer(text or ""))
    pred_carrier = parsed[-1].group("carrier").upper() if parsed else None
    if pred_carrier not in carriers:
        pred_carrier = None
    pred_letter = parsed[-1].group("letter").upper() if parsed else None
    choice = None
    found_choice = list(CHOICE_N.finditer(plan_text or ""))
    if found_choice:
        maybe = found_choice[-1].group(1).upper()
        if maybe in carriers:
            choice = maybe
    correct = pred_carrier == ex.gold_carrier
    pred_status = {c: pred_plan[c]["status"] for c in carriers}
    return {
        "parse_ok": pred_carrier is not None and pred_letter is not None,
        "pred_letter": pred_letter,
        "pred_carrier": pred_carrier,
        "correct": correct,
        "follows_trap": pred_carrier == ex.trap_carrier,
        "plan_all_status_correct": n_status == len(carriers),
        "plan_n_status_correct": n_status,
        "plan_pred_status": pred_status,
        "plan_choice": choice,
        "plan_choice_matches_gold": choice == ex.gold_carrier,
        "think_collapse": is_think_collapse(text) or is_think_collapse(plan_text),
        "correct_with_wrong_eligibility": bool(correct) and n_status != len(carriers),
        "kind": "true" if correct else "false",
        "raw_text": text,
        **label_direction_counts(pred_status, gold_status),
    }


def _row_n(ex, *, prompt: str, text: str, extra: dict) -> dict:
    scored = _score_b_n(text, str(extra.get("plan_text") or ""), ex)
    return {
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "split": ex.split,
        "variant": ex.variant,
        "difficulty": ex.difficulty,
        "n_carriers": int(getattr(ex, "n_carriers", 4)),
        "gold_letter": ex.gold_letter,
        "gold_carrier": ex.gold_carrier,
        "trap_letter": ex.trap_letter,
        "trap_carrier": ex.trap_carrier,
        "prompt": prompt,
        **scored,
        **extra,
    }


def _prompts_for(ex):
    if getattr(ex, "n_carriers", 4) == N_CARRIERS_12:
        return {
            "kv_plan": b_reason_kv_instruct_prompt_n(ex),
            "kv_answer": b_answer_kv_instruct_prompt_n,
            "text_plan": b_reason_with_message_prompt_n,
            "text_answer": b_answer_after_message_plan_prompt_n,
        }
    return {
        "kv_plan": b_reason_kv_instruct_prompt(ex),
        "kv_answer": b_answer_kv_instruct_prompt,
        "text_plan": b_reason_with_message_prompt,
        "text_answer": b_answer_after_message_plan_prompt,
    }


def _apply_split(extra: dict, plan: dict, ans: dict) -> None:
    extra["plan_time_s"] = float(plan["total_s"])
    extra["plan_prefill_s"] = float(plan["prefill_s"])
    extra["plan_decode_s"] = float(plan["decode_s"])
    extra["plan_encode_s"] = float(plan["encode_s"])
    extra["n_plan_tokens"] = int(plan["n_new"])
    extra["plan_truncated"] = bool(plan["truncated"])
    extra["plan_text"] = plan["text"]
    extra["receiver_latency_s"] = float(ans["total_s"])
    extra["answer_prefill_s"] = float(ans["prefill_s"])
    extra["answer_decode_s"] = float(ans["decode_s"])
    extra["answer_encode_s"] = float(ans["encode_s"])
    extra["n_new_tokens"] = int(ans["n_new"])
    extra["truncated"] = bool(ans["truncated"])
    extra["b_prefill_s"] = float(plan["prefill_s"]) + float(ans["prefill_s"])
    extra["b_decode_s"] = float(plan["decode_s"]) + float(ans["decode_s"])
    extra["b_encode_s"] = float(plan["encode_s"]) + float(ans["encode_s"])


def _run_kv_b_timed(runner, ex, bundle, *, plan_max_new, answer_max_new):
    prompts = _prompts_for(ex)
    plan_prompt = prompts["kv_plan"]
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("KV B leaked A's raw sheet.")
    plan = runner.generate_from_cache_split(bundle, plan_prompt, max_new_tokens=plan_max_new)
    prompt = prompts["kv_answer"](ex, plan["text"])
    if prompt_contains_a_private(prompt, ex):
        raise RuntimeError("KV answer pass leaked A's raw sheet.")
    ans = runner.generate_from_cache_split(bundle, prompt, max_new_tokens=answer_max_new)
    extra = {
        "channel": "kv",
        "message_text": "",
        "message_bytes": int(bundle.full_relay_bytes),
        "n_kv_positions": n_kv_positions(bundle.cache_cpu),
        "plan_max_new": plan_max_new,
        "max_new_tokens": answer_max_new,
        "peak_gpu_gb": peak_gpu_memory_gb(),
    }
    _apply_split(extra, plan, ans)
    if getattr(ex, "n_carriers", 4) == N_CARRIERS_12:
        return _row_n(ex, prompt=prompt, text=ans["text"], extra=extra)
    rec = _b_row(ex, prompt=prompt, text=ans["text"], extra=extra)
    _annotate_plan(rec, ex)
    return rec


def _run_text_b_timed(runner, ex, *, message: str, plan_max_new, answer_max_new):
    prompts = _prompts_for(ex)
    plan_prompt = prompts["text_plan"](ex, message)
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("Text B leaked A's raw sheet.")
    plan = runner.generate_text_message_split(plan_prompt, max_new_tokens=plan_max_new)
    prompt = prompts["text_answer"](ex, message, plan["text"])
    ans = runner.generate_text_message_split(prompt, max_new_tokens=answer_max_new)
    extra = {
        "channel": "text",
        "message_text": message,
        "message_bytes": len(message.encode("utf-8")),
        "n_kv_positions": 0,
        "plan_max_new": plan_max_new,
        "max_new_tokens": answer_max_new,
        "peak_gpu_gb": peak_gpu_memory_gb(),
        "construct_s": 0.0,
    }
    _apply_split(extra, plan, ans)
    if getattr(ex, "n_carriers", 4) == N_CARRIERS_12:
        return _row_n(ex, prompt=prompt, text=ans["text"], extra=extra)
    rec = _b_row(ex, prompt=prompt, text=ans["text"], extra=extra)
    _annotate_plan(rec, ex)
    return rec


def _timing_block(items: list[dict]) -> dict:
    block = _cond_block(items)
    n = len(items)
    if not n:
        return block

    def mean(key: str) -> float:
        return sum(float(item.get(key) or 0.0) for item in items) / n

    block.update(
        {
            "mean_construct_s": mean("construct_s"),
            "mean_prep_s": mean("rebuild_time_s") + mean("construct_s"),
            "mean_plan_prefill_s": mean("plan_prefill_s"),
            "mean_plan_decode_s": mean("plan_decode_s"),
            "mean_answer_prefill_s": mean("answer_prefill_s"),
            "mean_answer_decode_s": mean("answer_decode_s"),
            "mean_b_prefill_s": mean("b_prefill_s"),
            "mean_b_decode_s": mean("b_decode_s"),
            "mean_generic_prefill_s": mean("generic_prefill_s"),
            "mean_process_time_with_unamortized_generic_s": mean("process_time_with_unamortized_generic_s"),
            "mean_process_time_amortized_s": mean("process_time_amortized_s"),
            "mean_sender_generation_time_s": mean("sender_generation_time_s"),
        }
    )
    return block


def _summarize(rows: list[dict], *, check: dict | None = None, generic_once: dict | None = None) -> dict:
    by_width: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_width[int(row.get("n_carriers") or 4)].append(row)
    widths = {}
    for n_car, items in sorted(by_width.items()):
        by_cond: dict[str, list[dict]] = defaultdict(list)
        for row in items:
            by_cond[row["condition"]].append(row)
        receiver = {cond: _timing_block(by_cond.get(cond, [])) for cond in CONDS}
        full_mb = (receiver.get("full_kv") or {}).get("mean_relay_mb") or 0.0
        cand_mb = (receiver.get("generic_r") or {}).get("mean_relay_mb") or 0.0
        text_bytes = (receiver.get("text_result") or {}).get("mean_message_bytes")
        reduction = ((full_mb - cand_mb) / full_mb) if full_mb else None
        widths[str(n_car)] = {
            "n_carriers": n_car,
            "n_examples": len({row["example_id"] for row in items}),
            "n_rows": len(items),
            "receiver": receiver,
            "cache_mib": {
                "generic_r": cand_mb,
                "full_kv": full_mb,
                "result_kv": (receiver.get("result_kv") or {}).get("mean_relay_mb"),
                "text_result_bytes": text_bytes,
                "generic_r_vs_full_reduction": reduction,
                "note": (
                    "Packed-cache mebibytes, not measured network traffic. "
                    "Generic support can be prepared locally and reused."
                ),
            },
        }
    return {
        "n_rows": len(rows),
        "n_examples": len({row["example_id"] for row in rows}),
        "conditions": list(CONDS),
        "layout": LAYOUT,
        "prefix_source": PREFIX_SOURCE,
        "extra_separator": EXTRA_SEP,
        "confirm_accuracy": CONFIRM_ACCURACY,
        "by_n_carriers": widths,
        "generic_prepare_once": generic_once or {},
        "check": None
        if check is None
        else {"all_passed": check.get("all_passed"), "n": check.get("n")},
        "twelve_carrier_label_error_note": (
            "Twelve-carrier directional label errors were missing from the original "
            "scorer and previously printed as zero. They are filled from saved "
            "plan_pred_status. generic+R is 44 ineligible-marked-eligible and 1 "
            "eligible-marked-ineligible. This reporting issue does not explain the "
            "underlying accuracy failure."
        ),
        "lead": (
            "Frozen generic+R vs result-only KV, full KV, and result text at 4 and "
            "12 carriers. Accuracy gap is the hold confirmation, not this timing subset."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "payload_note": PAYLOAD_NOTE,
        "after_this_run": (
            "If the reliability gain over result-only KV has practical value versus "
            "text after these costs, stop. Independent confirmation would need "
            "genuinely new families with the method and criteria fixed beforehand. "
            "Do not train an adapter. Do not add the calculation block."
        ),
    }


def _prepare_runner(args, out_root: Path, extra: dict):
    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-timing",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_dev.jsonl", "data/plan4_fresh.jsonl"],
            "do_not_overwrite": [
                "results_stage_plan4_hold",
                "results_stage_plan4_confirm",
                "data/plan4_hold.jsonl",
            ],
            "layout": LAYOUT,
            "conditions": list(CONDS),
            "confirm_accuracy": CONFIRM_ACCURACY,
            **extra,
        },
    )
    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=0,
        kv_budget=32,
        sink_size=4,
        pca_rank=2,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha") and not extra.get("skip_pin"):
        pin_model_revision(revision["sha"])
    print("warmup generate", flush=True)
    runner.generate_text_message("Reply with the word ok.", max_new_tokens=4)
    return policy, runner, revision


def _fill_common(rec, ex, extra_a, state, cond, *, n_using_template: int):
    spec = SPEC[cond]
    rec.update(extra_a)
    rec["condition"] = cond
    rec["n_carriers"] = int(getattr(ex, "n_carriers", 4))
    rec["cache_source"] = spec["cache_source"]
    rec["framing"] = spec["framing"]
    rec["framing_instruction"] = FRAMING_INSTRUCTION if spec["framing"] == "instruct" else None
    rec["layout"] = LAYOUT if cond != "full_kv" and cond != "text_result" else None
    rec["prefix_source"] = PREFIX_SOURCE if cond not in ("full_kv", "text_result") else cond
    rec["extra_separator"] = EXTRA_SEP
    rec["payload_note"] = PAYLOAD_NOTE
    rec["template_id"] = (state.get("instr") or {}).get("template_id")
    rec["generic_prefill_s"] = float(state.get("generic_prefill_s") or 0.0) if cond == "generic_r" else 0.0
    rec["generic_reused"] = bool(state.get("generic_reused")) if cond == "generic_r" else None
    rec["process_time_s"] = _process_time_s(rec)
    charged = rec["generic_prefill_s"] if cond == "generic_r" else 0.0
    rec["process_time_with_unamortized_generic_s"] = rec["process_time_s"] + charged
    rec["process_time_amortized_s"] = rec["process_time_s"] + (
        charged / n_using_template if n_using_template else charged
    )
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    return rec


def _run_one(runner, ex, extra_a, state, cond, *, plan_max_new, answer_max_new, n_using_template: int):
    spans = state["instr"]
    if cond == "text_result":
        rec = _run_text_b_timed(
            runner,
            ex,
            message=spans.get("result_text") or "",
            plan_max_new=plan_max_new,
            answer_max_new=answer_max_new,
        )
        rec["construct_s"] = 0.0
        rec["n_kept"] = 0
        rec["n_result_kept"] = len(spans.get("result_pos") or [])
        rec["rebuild_time_s"] = float(extra_a.get("rebuild_time_s") or 0.0)
        return _fill_common(rec, ex, extra_a, state, cond, n_using_template=n_using_template)

    if cond == "generic_r":
        bundle = runner.clone_sender(state["mixed"][cond])
        construct_s = float(state.get("generic_construct_s") or state["times_by"].get(cond) or 0.0)
        slots = state["slots_extra"]
    elif cond == "result_kv":
        bundle = runner.clone_sender(state["mixed"][cond])
        construct_s = float(state["times_by"].get(cond) or 0.0)
        slots = state["slots"]
    else:
        bundle = runner.clone_sender(state["mixed"][cond])
        construct_s = 0.0
        slots = state.get("slots") or {}
    rec = _run_kv_b_timed(
        runner, ex, bundle, plan_max_new=plan_max_new, answer_max_new=answer_max_new
    )
    rec["construct_s"] = construct_s
    rec["selection_time_s"] = construct_s
    rec["n_kept"] = n_kv_positions(bundle.cache_cpu)
    rec["n_result_kept"] = int(slots.get("n_result") or len(spans.get("result_pos") or []))
    rec["n_kv_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["message_bytes"] = int(bundle.full_relay_bytes)
    rec["rebuild_time_s"] = float(extra_a.get("rebuild_time_s") or 0.0)
    return _fill_common(rec, ex, extra_a, state, cond, n_using_template=n_using_template)


def _run_check4(args, out_root, kaggle_dest, examples, runner, artifacts) -> tuple[int, dict]:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    templates, spans_by_key = _collect_templates(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    generic_once = {
        "width": 4,
        "n_templates": len(generics),
        "by_template": {
            name: {"prefill_s": float(rec.get("prefill_s") or 0.0), "n_prefix": rec["n_prefix"], "n": rec["n"]}
            for name, rec in generics.items()
        },
        "sum_prefill_s": sum(float(rec.get("prefill_s") or 0.0) for rec in generics.values()),
    }
    rows = []
    by_id = {ex.example_id: ex for ex in examples}
    print(f"Timing check4 n={len(examples)} templates={len(templates)}", flush=True)
    for i, ex in enumerate(examples):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        state = _pair_state(runner, rec, src, decode, sep_ids, close_ids, spans_by_key[key], generics)
        _attach_baselines(runner, state)
        state["generic_construct_s"] = float(state["times_by"]["generic_r"])
        state["generic_prefill_s"] = float(generics[spans_by_key[key]["template_id"]].get("prefill_s") or 0.0)
        state["generic_reused"] = True
        ok = (
            bool(state["partial_check"]["ids_equal"])
            and bool(state["partial_check"]["result_equal"])
            and bool(state["result_kv_prefix_held"])
            and int(state["slots_extra"]["n_prefix"]) == 5
        )
        row = {
            "a_state_key": key,
            "ok": bool(ok),
            "n_carriers": 4,
            "n_result": int(state["slots_extra"]["n_result"]),
            "n_instruction": int(spans_by_key[key].get("n_instruction") or 0),
            "template_id": spans_by_key[key].get("template_id"),
            "span_extract_ok": bool(spans_by_key[key].get("ok")),
            "span_failures": list(spans_by_key[key].get("failures") or []),
        }
        rows.append(row)
        print(f"  [{i + 1}/{len(examples)}] {key} ok={ok} n_R={row['n_result']}", flush=True)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_passed = bool(rows) and all(r["ok"] for r in rows)
    summary = {
        "n": len(rows),
        "all_passed": all_passed,
        "width": 4,
        "rows": rows,
        "generic_prepare_once": generic_once,
        "lead": "4-carrier construction on previously evaluated holdout base families.",
    }
    _write_json(out_root / "check_summary.json", summary)
    _write_json(out_root / "generic_once_4.json", generic_once)
    _persist(out_root, kaggle_dest)
    return (0 if all_passed else 1), summary


def _write_artifacts12(path: Path, artifacts: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in artifacts.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _score_sender_n(text: str, ex) -> dict:
    gold_q = gold_quantities_n(ex)
    gold_status = gold_eligibility_statuses_n(ex)
    found = {c: {"q": None, "status": None} for c in ex.carriers}
    for match in RESULT_LINE_N.finditer(text or ""):
        name = match.group("carrier").upper()
        if name in found:
            found[name] = {"q": int(match.group("q")), "status": match.group("status").lower()}
    n_q = sum(found[c]["q"] == gold_q[c] for c in ex.carriers)
    n_status = sum(found[c]["status"] == gold_status[c] for c in ex.carriers)
    n_parsed = sum(found[c]["status"] is not None for c in ex.carriers)
    return {
        "n_q_correct": n_q,
        "n_status_correct": n_status,
        "n_carriers_parsed": n_parsed,
        "all_status_correct": n_status == len(ex.carriers),
        "all_four_correct": n_q == len(ex.carriers) and n_status == len(ex.carriers),
        "malformed": n_parsed != len(ex.carriers),
    }


def _run_write12(args, out_root, kaggle_dest, examples, runner) -> int:
    artifacts_path = out_root / "a_artifacts_12.jsonl"
    artifacts = _load_messages(artifacts_path) if artifacts_path.is_file() else {}
    print(f"Timing write12 n={len(examples)} existing={len(artifacts)}", flush=True)
    for i, ex in enumerate(examples):
        key = a_state_key(ex)
        if key in artifacts and artifacts[key].get("full_ids"):
            print(f"  [{i + 1}/{len(examples)}] {key} already written", flush=True)
            continue
        encode_prompt = encode_prompt_for(ex)
        written = written_result_handoff_n(ex.carriers)
        ordinary = ordinary_sender_prompt(encode_prompt, written)
        if prompt_contains_b_private(ordinary, ex) or prompt_contains_solver_labels_n(ordinary, ex):
            raise RuntimeError(f"12-carrier A prompt leaked private fields for {key}.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text, n_new, truncated, gen_s, bundle = runner.generate_text_keep_cache(
            ordinary, max_new_tokens=args.written_max_new_12, source="written"
        )
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
        rec = {
            "example_id": ex.example_id,
            "family_id": ex.family_id,
            "variant": ex.variant,
            "difficulty": ex.difficulty,
            "a_state_key": key,
            "n_carriers": N_CARRIERS_12,
            "ordinary_prompt": ordinary,
            "written": text,
            "written_n_new": n_new,
            "written_truncated": truncated,
            "written_s": gen_s,
            "n_kv_positions": n_full,
            "n_prompt_positions": n_prompt,
            "full_relay_bytes": int(bundle.full_relay_bytes),
            "latent_steps": 0,
            "ids_source": bundle.ids_source,
            "token_align": dict(bundle.token_align or {}),
            "full_ids": bundle.full_ids_cpu.tolist(),
            "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
            "handoff": "\n".join(m.group(0).strip() for m in RESULT_LINE_N.finditer(text or "")),
            "written_score": _score_sender_n(text, ex),
            "written_working": written_working_metrics(text),
        }
        artifacts[key] = rec
        _write_artifacts12(artifacts_path, artifacts)
        _persist(out_root, kaggle_dest)
        print(
            f"  [{i + 1}/{len(examples)}] {key} trunc={truncated} n_new={n_new} "
            f"status={rec['written_score']['n_status_correct']}/{N_CARRIERS_12} s={gen_s:.1f}",
            flush=True,
        )
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    decode = _decode_fn(runner)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    n_ok = sum(bool(spans_by_key[k]["ok"]) for k in spans_by_key)
    summary = {
        "n": len(spans_by_key),
        "n_ok": n_ok,
        "all_passed": n_ok == len(spans_by_key),
        "width": 12,
        "templates": {
            name: {"n": rec["n"], "n_prefix": rec["n_prefix"]} for name, rec in templates.items()
        },
        "rows": [
            {
                "a_state_key": key,
                "ok": bool(spans["ok"]),
                "failures": spans.get("failures"),
                "n_result": len(spans.get("result_pos") or []),
                "n_instruction": spans.get("n_instruction"),
                "template_id": spans.get("template_id"),
            }
            for key, spans in spans_by_key.items()
        ],
    }
    _write_json(out_root / "check12_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"n_ok": n_ok, "n": summary["n"], "all_passed": summary["all_passed"]}, indent=2))
    return 0 if summary["all_passed"] else 1


def _rewrite_label_export(out_root: Path, examples12, extra_dest: Path | None = None, *, persist: bool = True) -> int:
    path = out_root / "evaluations.jsonl"
    if not path.is_file():
        print(f"Missing {path}. Cannot rewrite twelve-carrier label errors.")
        return 2
    by_id = {ex.example_id: ex for ex in examples12}
    rows = _load_rows(path)
    n12 = 0
    counts = {}
    for row in rows:
        if int(row.get("n_carriers") or 4) != 12:
            continue
        ex = by_id.get(row["example_id"])
        if ex is None:
            raise RuntimeError(f"No timing-12 example for {row['example_id']}.")
        extra = label_direction_counts(row.get("plan_pred_status") or {}, gold_eligibility_statuses_n(ex))
        row.update(extra)
        n12 += 1
        cond = row["condition"]
        bucket = counts.setdefault(cond, {"n": 0, "plan_ineligible_marked_eligible": 0, "plan_eligible_marked_ineligible": 0})
        bucket["n"] += 1
        bucket["plan_ineligible_marked_eligible"] += int(extra["plan_ineligible_marked_eligible"])
        bucket["plan_eligible_marked_ineligible"] += int(extra["plan_eligible_marked_ineligible"])
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    check = None
    cpath = out_root / "check_summary.json"
    if cpath.is_file():
        check = json.loads(cpath.read_text(encoding="utf-8"))
    summary = _summarize(rows, check=check, generic_once=_load_generic_once(out_root))
    _write_json(out_root / "summary.json", summary)
    _write_json(out_root / "timing_summary.json", summary)
    if persist:
        _persist(out_root, extra_dest)
    print(json.dumps({"n_twelve_carrier_rows": n12, "directional": counts}, indent=2))
    return 0 if n12 else 2


def _load_generic_once(out_root: Path) -> dict:
    path = out_root / "generic_once_4.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _run_compare(args, out_root, kaggle_dest, hold_ex, ex12, runner, artifacts4, artifacts12) -> int:
    check4 = None
    cpath = out_root / "check_summary.json"
    if cpath.is_file():
        check4 = json.loads(cpath.read_text(encoding="utf-8"))
    if (check4 is None or not check4.get("all_passed")) and not args.force_compare:
        print("Run --phase check first.")
        return 2
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"], int(row.get("n_carriers") or 4)) for row in rows}
    generic_once = dict(_load_generic_once(out_root))

    def run_width(examples, artifacts, n_carriers: int, collect, pair_fn, plan_max_new: int):
        templates, spans_by_key = collect(examples, artifacts, decode)
        generics = {}
        for name, rec in templates.items():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bundle, wall = runner.prefill_token_sequence(rec["ids"], source=f"generic_{name}_{n_carriers}")
            generics[name] = {**rec, "bundle": bundle, "prefill_s": float(wall), "reused": True}
            print(
                f"  generic {n_carriers} {name} prefill_s={wall:.3f} "
                "(prepared once, reused per example)",
                flush=True,
            )
        once = {
            "width": n_carriers,
            "n_templates": len(generics),
            "n_reused_from_other_width": 0,
            "by_template": {
                name: {
                    "prefill_s": float(rec.get("prefill_s") or 0.0),
                    "n_prefix": rec["n_prefix"],
                    "n": rec["n"],
                    "reused": bool(rec.get("reused")),
                }
                for name, rec in generics.items()
            },
            "sum_prefill_s": sum(float(rec.get("prefill_s") or 0.0) for rec in generics.values()),
        }
        generic_once[f"width_{n_carriers}"] = once
        counts = defaultdict(int)
        for ex in examples:
            counts[spans_by_key[a_state_key(ex)]["template_id"]] += 1
        print(f"Timing compare n_carriers={n_carriers} n={len(examples)}", flush=True)
        by_id = {ex.example_id: ex for ex in examples}
        for i, ex in enumerate(examples):
            remaining = [c for c in CONDS if (ex.example_id, c, n_carriers) not in done]
            if not remaining:
                print(f"  [{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
                continue
            key = a_state_key(ex)
            src = by_id.get(key) or ex
            rec_a = artifacts[key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            state = pair_fn(
                runner, rec_a, src, decode, sep_ids, close_ids, spans_by_key[key], generics
            )
            extra_a = _a_extra(state)
            extra_a["rebuild_time_s"] = float(state["rebuild_time_s"])
            extra_a["n_carriers"] = n_carriers
            extra_a["sender_generation_time_s"] = float(rec_a.get("written_s") or 0.0) if n_carriers == 12 else 0.0
            n_using = int(counts[spans_by_key[key]["template_id"]])
            reset_peak_memory()
            for cond in remaining:
                rec_out = _run_one(
                    runner,
                    ex,
                    extra_a,
                    state,
                    cond,
                    plan_max_new=plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    n_using_template=n_using,
                )
                rows.append(rec_out)
                done.add((ex.example_id, cond, n_carriers))
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                print(
                    f"    {ex.example_id} {cond}: elig={rec_out.get('plan_all_status_correct')} "
                    f"prep={rec_out.get('rebuild_time_s' ,0)+rec_out.get('construct_s',0):.3f} "
                    f"prefill={rec_out.get('b_prefill_s'):.3f} decode={rec_out.get('b_decode_s'):.3f} "
                    f"mb={(rec_out.get('message_bytes') or 0)/(1024*1024):.2f}",
                    flush=True,
                )
                _persist(out_root, kaggle_dest)
            del state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return once

    run_width(
        hold_ex,
        artifacts4,
        4,
        _collect_templates,
        _pair_state_4,
        args.plan_max_new,
    )
    if artifacts12:
        run_width(
            ex12,
            artifacts12,
            12,
            _collect_templates_n,
            _pair_state_n,
            args.plan_max_new_12,
        )
    summary = _summarize(rows, check=check4, generic_once=generic_once)
    _write_json(out_root / "summary.json", summary)
    _write_json(out_root / "timing_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary.get("confirm_accuracy"), indent=2))
    print(json.dumps(summary.get("by_n_carriers"), indent=2)[:16000])
    return 0


def _pair_state_4(runner, rec, src, decode, sep_ids, close_ids, spans, generics):
    state = _pair_state(runner, rec, src, decode, sep_ids, close_ids, spans, generics)
    _attach_baselines(runner, state)
    state["generic_construct_s"] = float(state["times_by"]["generic_r"])
    state["generic_prefill_s"] = float(generics[spans["template_id"]].get("prefill_s") or 0.0)
    state["generic_reused"] = True
    return state


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="4- vs 12-carrier timing of frozen generic+R.")
    parser.add_argument("--split", default="plan4_timing")
    parser.add_argument(
        "--phase",
        choices=("check", "write12", "compare", "all", "rescore", "rewrite_labels"),
        default="check",
    )
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--written_max_new_12", type=int, default=A_WRITTEN_MAX_NEW_12)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args()
    if args.split not in {"plan4_timing", "plan4_hold"}:
        print("Timing uses plan4_hold bases plus the timing-12 split. Do not load plan4_dev or plan4_fresh.")
        return 2

    hold_path, timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_timing" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_timing.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "check_summary.json",
            "check12_summary.json",
            "a_artifacts.jsonl",
            "a_artifacts_12.jsonl",
            "manifest.json",
        ):
            src = kaggle_dest / name
            dest = out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    hold_all = load_plan4_jsonl(hold_path)
    for ex in hold_all:
        if ex.split != "plan4_hold":
            print(f"Refusing {ex.example_id}.")
            return 2
        assert_plan4_invariants(ex)
    hold_ex = select_timing4_examples(hold_all)
    if len(hold_ex) != N_TIMING_FAMILIES:
        print(f"Expected {N_TIMING_FAMILIES} hold base families, got {len(hold_ex)}.")
        return 2
    ex12 = ensure_timing12_jsonl(timing12_path)
    for ex in ex12:
        assert_timing12_invariants(ex)

    if args.phase == "rewrite_labels":
        shared = REPO_ROOT / "results_stage_plan4_timing"
        seen = set()
        rc = 2
        for path, do_persist in (
            (out_root, True),
            (out_root.parent / "latest", False),
            (shared, False),
            (kaggle_dest, False),
        ):
            if path is None or not (path / "evaluations.jsonl").is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            print(f"Rewriting twelve-carrier directional labels in {path}", flush=True)
            rc = _rewrite_label_export(path, ex12, kaggle_dest if do_persist else None, persist=do_persist)
        if not seen:
            print("No evaluations.jsonl found to rewrite.")
            return 2
        return rc

    if args.phase == "rescore":
        check = None
        cpath = out_root / "check_summary.json"
        if cpath.is_file():
            check = json.loads(cpath.read_text(encoding="utf-8"))
        eval_path = out_root / "evaluations.jsonl"
        shared = REPO_ROOT / "results_stage_plan4_timing"
        if not eval_path.is_file() and (shared / "evaluations.jsonl").is_file():
            eval_path = shared / "evaluations.jsonl"
            out_root = shared
        summary = _summarize(_load_rows(eval_path), check=check, generic_once=_load_generic_once(out_root))
        _write_json(out_root / "summary.json", summary)
        _write_json(out_root / "timing_summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps(summary.get("by_n_carriers"), indent=2)[:12000])
        return 0

    artifacts4_path = _copy_hold_artifacts(out_root, explicit=args.artifacts)
    artifacts4 = _load_messages(artifacts4_path)
    needed4 = {a_state_key(ex) for ex in hold_ex}
    if any(key not in artifacts4 for key in needed4):
        print("Frozen hold artifacts are missing selected timing-4 sheets. Do not regenerate A.")
        return 2

    policy, runner, revision = _prepare_runner(
        args, out_root, extra={"phase": args.phase, "n_timing4": len(hold_ex), "n_timing12": len(ex12)}
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n4={len(hold_ex)} n12={len(ex12)}")

    def do_check():
        return _run_check4(args, out_root, kaggle_dest, hold_ex, runner, artifacts4)[0]

    def do_write12():
        return _run_write12(args, out_root, kaggle_dest, ex12, runner)

    def do_compare():
        artifacts12_path = out_root / "a_artifacts_12.jsonl"
        artifacts12 = _load_messages(artifacts12_path) if artifacts12_path.is_file() else {}
        return _run_compare(
            args, out_root, kaggle_dest, hold_ex, ex12, runner, artifacts4, artifacts12
        )

    if args.phase == "check":
        return do_check()
    if args.phase == "write12":
        return do_write12()
    if args.phase == "compare":
        return do_compare()
    rc = do_check()
    if rc != 0 and not args.force_compare:
        print("4-carrier construction failed; stopping.")
        return rc
    rc12 = do_write12()
    if rc12 != 0 and not args.force_compare:
        print("12-carrier construction failed; stopping.")
        return rc12
    return do_compare()


if __name__ == "__main__":
    raise SystemExit(main())
