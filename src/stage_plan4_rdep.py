"""Does the repaired generic handoff still depend on A's example-specific R?

Keep the independently prepared generic task-context instruction. Compare three
uses of the same support on plan4_dev, with B's prompt and prices held fixed:

- generic_r: this example's transferred result states
- generic_mismatch: the paired A-flip (or base) result states
- generic_no_r: the generic support alone, with no result slots

Do not search more neutral passages. Do not add calculations. Do not train an
adapter. Do not load plan4_hold or plan4_fresh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import cache_slice_delta, n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    CARRIERS,
    LEVELS,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    load_plan4_jsonl,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    ELIGIBILITY_RULE,
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_boundary import (
    PAYLOAD_NOTE,
    _added_token_ids,
    _annotate_plan,
    _carrier_block,
    candidate_qualifies,
)
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta
from src.stage_plan4_generic import _collect_templates, _prefill_templates
from src.stage_plan4_ifresh import _extra_sep_ids
from src.stage_plan4_instr import EXTRA_SEP, _build_extra_bundle
from src.stage_plan4_kvresult import PREFIX_SOURCE, _rebuild_pair
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _copy_frozen_artifacts
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("generic_r", "generic_mismatch", "generic_no_r")
SPEC = {
    "generic_r": {"r_source": "this", "cache_source": "generic_instr_same_R"},
    "generic_mismatch": {"r_source": "paired_a_flip", "cache_source": "generic_instr_mismatch_R"},
    "generic_no_r": {"r_source": "none", "cache_source": "generic_instr_no_R"},
}
PAIRED = (
    ("generic_r_vs_generic_mismatch", "generic_mismatch", "generic_r"),
    ("generic_r_vs_generic_no_r", "generic_no_r", "generic_r"),
    ("generic_mismatch_vs_generic_no_r", "generic_no_r", "generic_mismatch"),
)
PRESERVED = (
    "results_stage_plan4_nbound",
    "results_stage_plan4_neutral",
    "results_stage_plan4_generic",
    "results_stage_plan4_ifresh",
    "results_stage_plan4_instr",
    "results_stage_plan4_kvresult",
    "results_stage_plan4_kvswap",
    "results_stage_plan4_boundary",
    "results_stage_plan4_boundary1",
    "results_stage_plan4_recon",
    "results_stage_plan4_iface",
    "results_stage_plan4_support",
    "results_stage_plan4_hold",
    "results_stage_plan4_stable",
    "results_stage_plan4_fresh",
)
FROZEN_PLACEMENT = (
    "All three keep the independently prepared generic task-context instruction "
    "on the reconstructed five-token prefix plus one newline. generic_r appends "
    "this example's transferred result states. generic_mismatch appends the paired "
    "A-flip (or base) result states at the same packed start, keeping this example's "
    "B prompt. generic_no_r stops at the instruction newline. Do not crop, pad, or "
    "search new context passages."
)
INTERPRETATION_GUIDE = (
    "If correct R recovers labels, removing R harms recovery, and substituting R "
    "shifts B toward the substituted labels, the reusable support helps B use A's "
    "example-specific result states. If no-R stays high, the support alone can "
    "supply the labels. If mismatch does not move toward the substituted labels, "
    "R is not carrying those example-specific states into B's behavior. "
    "These remain development-family comparisons. Do not load plan4_hold. "
    "Do not train an adapter. Do not add the calculation block."
)


def mismatch_a_key(ex) -> str:
    """Paired A sheet: a_flip if this example used base, otherwise base."""
    if ex.variant == "a_flip":
        return f"{ex.family_id}-base"
    return f"{ex.family_id}-a_flip"


def partner_a_key(key: str) -> str:
    if key.endswith("-a_flip"):
        return key[: -len("-a_flip")] + "-base"
    if key.endswith("-base"):
        return key[: -len("-base")] + "-a_flip"
    raise ValueError(f"Cannot find an A-flip partner for {key}")


def n_status_agree(pred: dict | None, gold: dict | None) -> int:
    if not pred or not gold:
        return 0
    return sum(pred.get(carrier) == gold.get(carrier) for carrier in CARRIERS)


def sheet_example(examples, key: str):
    for ex in examples:
        if a_state_key(ex) == key and ex.variant != "b_flip":
            return ex
    raise RuntimeError(f"No unique A sheet example for {key}.")


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_rdep" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_rdep")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("check_summary.json", "construction.json"):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _frozen_refs() -> dict:
    generic_r = {
        "n": 90,
        "plan_all_status_correct": 87,
        "correct": 69,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_nbound instr_generic",
    }
    reconstruct = {
        "n": 90,
        "plan_all_status_correct": 89,
        "correct": 74,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_instr reconstruct",
    }
    for folder, generic_name, recon_name in (
        (REPO_ROOT / "results_stage_plan4_nbound" / "summary.json", "instr_generic", None),
        (Path("/kaggle/working/results_stage_plan4_nbound/summary.json"), "instr_generic", None),
        (REPO_ROOT / "results_stage_plan4_generic" / "summary.json", "instr_generic", None),
        (REPO_ROOT / "results_stage_plan4_instr" / "summary.json", None, "reconstruct"),
    ):
        if not folder.is_file():
            continue
        data = json.loads(folder.read_text(encoding="utf-8"))
        rec = data.get("receiver") or {}
        if generic_name:
            block = rec.get(generic_name) or {}
            if int(block.get("n") or 0) == 90:
                generic_r = {
                    "n": 90,
                    "plan_all_status_correct": block.get("plan_all_status_correct"),
                    "correct": block.get("correct"),
                    "parse_ok": block.get("parse_ok"),
                    "truncated": block.get("truncated"),
                    "source": str(folder).replace("\\", "/"),
                }
        if recon_name:
            block = rec.get(recon_name) or {}
            if int(block.get("n") or 0) == 90:
                reconstruct = {
                    "n": 90,
                    "plan_all_status_correct": block.get("plan_all_status_correct"),
                    "correct": block.get("correct"),
                    "parse_ok": block.get("parse_ok"),
                    "truncated": block.get("truncated"),
                    "source": str(folder).replace("\\", "/"),
                }
    return {"generic_r": generic_r, "reconstruct": reconstruct}


def _bundle_ids(bundle) -> list[int]:
    return [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]


def _build_generic_head(runner, pair, transferred, composed, slots, generic, extra_pos):
    prefix = runner.prefix_bundle(pair["reconstruct"], slots["n_prefix"])
    extra_ids = [int(t) for t in composed["extra_ids"]]
    if extra_ids != [int(t) for t in generic["instruction_ids"]]:
        raise RuntimeError("Generic instruction token IDs do not match this example's instruction tokens.")
    mid = runner.concat_gathered_block(
        prefix,
        generic["bundle"],
        src_pos=extra_pos,
        token_ids=extra_ids,
        source="generic_instr",
    )
    head, wall = runner.extend_cache_with_tokens(mid, _extra_sep_ids(composed), source="generic_instr")
    n_head = n_kv_positions(head.cache_cpu)
    if n_head != int(slots["result_start"]):
        raise RuntimeError(
            f"Generic instruction+newline length {n_head} != result start {slots['result_start']}."
        )
    prefix_held = cache_slice_delta(head.cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
    extra_vs_kv = cache_slice_delta(head.cache_cpu, transferred.cache_cpu, slots["extra"])
    if not prefix_held:
        raise RuntimeError("Generic instruction cache did not keep the reconstructed prefix.")
    cursor = int(head.pos_cursor_cpu.view(-1)[0].item())
    if cursor != int(slots["result_start"]):
        raise RuntimeError("B starting position drifted after generic instruction+newline.")
    info = {
        "prefix_held_from_reconstruct": True,
        "extra_equal": extra_vs_kv["equal"],
        "extra_keys_max_abs": extra_vs_kv["keys_max_abs"],
        "extra_values_max_abs": extra_vs_kv["values_max_abs"],
        "template_id": generic["template_id"],
        "result_start": int(slots["result_start"]),
        "pos_cursor": cursor,
        "n_kept": n_head,
        "extend_time_s": float(wall),
        "generic_prefill_s": float(generic.get("prefill_s") or 0.0),
    }
    return head, wall, info


def _append_r(runner, head, transferred, result_slots, source: str):
    bundle = runner.append_packed_slots(
        head, transferred, tail_positions=result_slots, source=source
    )
    held = cache_slice_delta(bundle.cache_cpu, transferred.cache_cpu, result_slots)["equal"]
    if not held:
        raise RuntimeError(f"{source} did not keep the selected R tensors.")
    return bundle, held


def _pair_state(runner, rec, src, decode, sep_ids, close_ids, spans, generics):
    pair = _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids)
    if list(spans.get("result_pos") or []) != list(pair["spans"]["result_pos"]):
        raise RuntimeError("Instruction-pack result positions drifted from compact R.")
    extra_pos = list(spans["instruction_pos"])
    transferred, trans_s, composed, slots, trans_info = _build_extra_bundle(
        runner, pair, extra_pos, source="transfer_instr_kv"
    )
    template_id = spans["template_id"]
    generic_rec = generics[template_id]
    head, head_s, head_info = _build_generic_head(
        runner, pair, transferred, composed, slots, generic_rec, extra_pos
    )
    generic_r, r_held = _append_r(
        runner, head, transferred, slots["result"], "generic_instr_same_R"
    )
    r_ids = _bundle_ids(generic_r)
    ids_equal = r_ids == list(composed["ids"])
    if not ids_equal:
        raise RuntimeError("Generic+R token IDs drifted from the transferred layout.")
    prefix_eq = cache_slice_delta(generic_r.cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
    cursor_eq = int(generic_r.pos_cursor_cpu.view(-1)[0].item()) == int(composed["n"])
    no_r_n = n_kv_positions(head.cache_cpu)
    no_r_cursor = int(head.pos_cursor_cpu.view(-1)[0].item())
    pair.update(
        {
            "instr": spans,
            "composed_extra": composed,
            "slots_extra": slots,
            "transferred": transferred,
            "head": head,
            "mixed": {"generic_r": generic_r, "generic_no_r": head},
            "times_by": {
                "generic_r": trans_s + head_s,
                "generic_no_r": trans_s + head_s,
                "generic_mismatch": trans_s + head_s,
            },
            "info_by": {
                "generic_r": {**head_info, "result_held_from_transferred": r_held},
                "generic_no_r": {**head_info, "result_held_from_transferred": False},
            },
            "trans_info": trans_info,
            "partial_check": {
                "a_state_key": spans.get("a_state_key"),
                "template_id": template_id,
                "ids_equal": ids_equal,
                "result_equal": r_held,
                "prefix_equal": prefix_eq,
                "cursor_equal": cursor_eq,
                "no_r_n": no_r_n,
                "no_r_cursor": no_r_cursor,
                "result_start": int(slots["result_start"]),
                "n_result": int(slots["n_result"]),
                "n_instruction": int(spans.get("n_instruction") or 0),
                "private_facts": bool(spans.get("causal_prefix", {}).get("private_facts")),
            },
        }
    )
    return pair


def _attach_mismatch(runner, state: dict, other: dict) -> None:
    this_slots = state["slots_extra"]
    other_slots = other["slots_extra"]
    this_start = int(this_slots["result_start"])
    other_start = int(other_slots["result_start"])
    if this_start != other_start:
        raise RuntimeError(
            f"Paired result starts differ: {state['partial_check']['a_state_key']} "
            f"{this_start} vs {other['partial_check']['a_state_key']} {other_start}."
        )
    bundle, held = _append_r(
        runner,
        state["head"],
        other["transferred"],
        other_slots["result"],
        "generic_instr_mismatch_R",
    )
    prefix_eq = cache_slice_delta(
        bundle.cache_cpu, state["reconstruct"].cache_cpu, this_slots["prefix"]
    )["equal"]
    if not prefix_eq:
        raise RuntimeError("Mismatch R splice overwrote the reconstructed prefix.")
    cursor = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    expected = this_start + int(other_slots["n_result"])
    if cursor != expected:
        raise RuntimeError(f"Mismatch cursor {cursor} != expected {expected}.")
    state["mixed"]["generic_mismatch"] = bundle
    state["partner_key"] = other["partial_check"]["a_state_key"]
    state["n_result_partner"] = int(other_slots["n_result"])
    state["mismatch_result_equal"] = held
    state["mismatch_prefix_equal"] = prefix_eq
    state["mismatch_cursor"] = cursor
    state["info_by"]["generic_mismatch"] = {
        **(state["info_by"].get("generic_r") or {}),
        "result_held_from_transferred": held,
        "r_source_key": state["partner_key"],
        "n_result_src": int(other_slots["n_result"]),
    }


def instruction_construction_ready(spans: dict) -> bool:
    """Generic+R construction needs the instruction span and four result lines.

    Support-pack extras such as 'Carrier P' calculation headings are recorded,
    but they are not required for this handoff.
    """
    carriers = [str(c).upper() for c in (spans.get("result_carriers") or [])]
    return (
        int(spans.get("n_instruction") or 0) > 0
        and bool(spans.get("instruction_before_facts"))
        and bool(spans.get("instruction_contains_rule"))
        and not bool((spans.get("causal_prefix") or {}).get("private_facts"))
        and sorted(carriers) == ["P", "Q", "R", "S"]
    )


def construction_ok_from_row(row: dict) -> bool:
    """Rebuild per-sheet ok from saved construction flags. Does not use span_extract_ok."""
    result_kv = row.get("result_kv_prefix_held")
    return bool(
        coerce_sheet_ok(row.get("ids_equal"))
        and coerce_sheet_ok(row.get("result_equal"))
        and coerce_sheet_ok(row.get("prefix_equal"))
        and coerce_sheet_ok(row.get("cursor_equal"))
        and coerce_sheet_ok(row.get("no_r_ok"))
        and coerce_sheet_ok(row.get("mismatch_ok"))
        and (result_kv is None or coerce_sheet_ok(result_kv))
        and coerce_sheet_ok(row.get("instruction_before_facts"))
        and coerce_sheet_ok(row.get("instruction_contains_rule"))
        and bool(row.get("partner_key"))
        and int(row.get("n_instruction") or 0) > 0
        and int(row.get("n_result") or 0) > 0
        and not coerce_sheet_ok(row.get("private_facts"))
    )


def _check_row(state: dict) -> dict:
    part = state["partial_check"]
    spans = state["instr"]
    no_r_ok = part["no_r_n"] == part["result_start"] and part["no_r_cursor"] == part["result_start"]
    mismatch_ok = bool(state.get("mismatch_result_equal")) and bool(state.get("mismatch_prefix_equal"))
    instr_ready = instruction_construction_ready(spans)
    ok = (
        instr_ready
        and bool(part["ids_equal"])
        and bool(part["result_equal"])
        and bool(part["prefix_equal"])
        and bool(part["cursor_equal"])
        and no_r_ok
        and mismatch_ok
        and int(state["slots_extra"]["n_prefix"]) == 5
        and EXTRA_SEP
        and not bool(part["private_facts"])
        and bool(state.get("partner_key"))
    )
    return {
        "a_state_key": part["a_state_key"],
        "ok": bool(ok),
        "template_id": part["template_id"],
        "ids_equal": part["ids_equal"],
        "result_equal": part["result_equal"],
        "prefix_equal": part["prefix_equal"],
        "cursor_equal": part["cursor_equal"],
        "no_r_ok": no_r_ok,
        "mismatch_ok": mismatch_ok,
        "mismatch_result_equal": bool(state.get("mismatch_result_equal")),
        "partner_key": state.get("partner_key"),
        "result_start": part["result_start"],
        "n_result": part["n_result"],
        "n_result_partner": state.get("n_result_partner"),
        "result_len_differ": int(part["n_result"]) != int(state.get("n_result_partner") or part["n_result"]),
        "n_instruction": part["n_instruction"],
        "private_facts": part["private_facts"],
        "instruction_before_facts": bool(spans.get("instruction_before_facts")),
        "instruction_contains_rule": ELIGIBILITY_RULE
        in ((spans.get("instruction") or {}).get("text") or ""),
        "instr_ready": instr_ready,
        "span_extract_ok": bool(spans.get("ok")),
        "span_failures": list(spans.get("failures") or []),
    }


def coerce_sheet_ok(value) -> bool:
    """Per-sheet ok must be a Boolean. A leftover partner-id string means passed."""
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _rewrite_ok_fields(obj):
    if isinstance(obj, dict):
        if "ok" in obj and "a_state_key" in obj and "partner_key" in obj:
            obj["ok"] = coerce_sheet_ok(obj.get("ok"))
        for value in obj.values():
            _rewrite_ok_fields(value)
    elif isinstance(obj, list):
        for item in obj:
            _rewrite_ok_fields(item)


def _rewrite_ok_export(out_root: Path, extra_dest: Path | None = None, *, persist: bool = True) -> int:
    found = False
    for name in ("check_summary.json", "construction.json", "summary.json"):
        path = out_root / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        _rewrite_ok_fields(data)
        _write_json(path, data)
        found = True
        n_true = None
        rows = data.get("rows") if isinstance(data, dict) else None
        if isinstance(rows, list):
            n_true = sum(coerce_sheet_ok(row.get("ok")) and isinstance(row.get("ok"), bool) for row in rows)
            print(f"{path}: n_rows={len(rows)} n_ok_bool_true={n_true}")
    if not found:
        print(f"No rdep JSON under {out_root} to rewrite.")
        return 2
    if persist:
        _persist(out_root, extra_dest)
    return 0


def _summarize_check(rows: list[dict], templates: dict) -> dict:
    n = len(rows)
    return {
        "n_unique_a_sheets": n,
        "n_ok": sum(coerce_sheet_ok(r.get("ok")) for r in rows),
        "all_passed": bool(n) and all(coerce_sheet_ok(r.get("ok")) for r in rows),
        "n_templates": len(templates),
        "templates": {
            name: {
                "n": rec["n"],
                "n_prefix": rec["n_prefix"],
                "instruction_start": rec["instruction_start"],
                "instruction_end": rec["instruction_end"],
                "difficulties": rec["difficulties"],
            }
            for name, rec in templates.items()
        },
        "n_ids_equal": sum(bool(r.get("ids_equal")) for r in rows),
        "n_result_equal": sum(bool(r.get("result_equal")) for r in rows),
        "n_prefix_equal": sum(bool(r.get("prefix_equal")) for r in rows),
        "n_no_r_ok": sum(bool(r.get("no_r_ok")) for r in rows),
        "n_mismatch_ok": sum(bool(r.get("mismatch_ok")) for r in rows),
        "n_result_len_differ": sum(bool(r.get("result_len_differ")) for r in rows),
        "mean_n_instruction": (sum(int(r.get("n_instruction") or 0) for r in rows) / n) if n else None,
        "note": (
            "Generic support is reused from the three causal-prefix templates. "
            "generic_r must keep this sheet's transferred R. generic_mismatch must "
            "keep the paired sheet's R at the same packed start. generic_no_r must "
            "stop at the instruction newline. B's prompt is not applied during check."
        ),
    }


def _shift_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    this_vals = [int(item.get("n_status_match_this") or 0) for item in items]
    src_vals = [
        int(item["n_status_match_src"])
        for item in items
        if item.get("n_status_match_src") is not None
    ]
    return {
        "n": n,
        "parse_ok": sum(bool(item.get("parse_ok")) for item in items),
        "truncated": sum(bool(item.get("truncated") or item.get("plan_truncated")) for item in items),
        "mean_n_match_this": sum(this_vals) / n,
        "mean_n_match_src": (sum(src_vals) / len(src_vals)) if src_vals else None,
        "n_all_match_this": sum(bool(item.get("plan_all_status_correct")) for item in items),
        "n_all_match_src": sum(bool(item.get("all_match_src")) for item in items),
        "n_match_src_better": sum(bool(item.get("match_src_better")) for item in items),
        "n_result_len_differ": sum(bool(item.get("result_len_differ")) for item in items),
        "correct_with_wrong_eligibility": sum(
            bool(item.get("correct_with_wrong_eligibility")) for item in items
        ),
    }


def _summarize(rows: list[dict], *, check: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("generic_r", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    refs = _frozen_refs()
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "n_evals_expected": 90 * len(CONDS),
        "layout": LAYOUT,
        "prefix_source": PREFIX_SOURCE,
        "extra_separator": EXTRA_SEP,
        "baseline": "independently prepared generic task-context instruction plus this example's transferred R",
        "unit_note": (
            "plan4_dev only. Same reconstructed prefix, generic instruction tokens, "
            "newline, B prompt, and decoding. generic_r keeps this example's R. "
            "generic_mismatch substitutes the paired A-flip R. generic_no_r omits R."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
        "carrier_label_errors": {cond: _carrier_block(by_cond.get(cond, [])) for cond in CONDS},
        "label_shift": {cond: _shift_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "by_variant": {},
        "paired_receiver": {name: _paired(grouped_plain, a, b) for name, a, b in PAIRED},
        "family_clustered_delta": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED) for name, a, b in PAIRED
        },
        "family_clustered_eligibility": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED, key="plan_all_status_correct")
            for name, a, b in PAIRED
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "frozen_generic_r": refs["generic_r"],
        "frozen_reconstruct": refs["reconstruct"],
        "qualifies_vs_reconstruct": {
            cond: candidate_qualifies(receiver.get(cond) or {}, refs["reconstruct"]) for cond in CONDS
        },
        "spec": SPEC,
        "check": None
        if check is None
        else {
            "all_passed": check.get("all_passed"),
            "n": check.get("n"),
            **(check.get("construction") or {}),
        },
        "framing_change": FRAMING_INSTRUCTION,
        "frozen_placement": FROZEN_PLACEMENT,
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels. Ask whether the repaired "
            "generic support still depends on A's example-specific result states."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "Interpret R-dependence before fresh confirmation or timing. "
            "Do not load plan4_hold. Do not train an adapter. Do not add the "
            "calculation block. Do not search additional neutral passages."
        ),
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _cond_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            },
            "label_shift": {
                cond: _shift_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            },
        }
    for variant in ("base", "a_flip", "b_flip"):
        out["by_variant"][variant] = {
            "receiver": {
                cond: _cond_block(_subset(by_cond.get(cond, []), variant=variant)) for cond in CONDS
            },
            "label_shift": {
                cond: _shift_block(_subset(by_cond.get(cond, []), variant=variant)) for cond in CONDS
            },
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _prepare_runner(args, out_root: Path, extra: dict) -> tuple:
    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-rdep",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "layout": LAYOUT,
            "prefix_source": PREFIX_SOURCE,
            "extra_separator": EXTRA_SEP,
            "conditions": list(CONDS),
            "n_evals": 90 * len(CONDS),
            "framing_change": FRAMING_INSTRUCTION,
            "frozen_placement": FROZEN_PLACEMENT,
            "payload_note": PAYLOAD_NOTE,
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
    return policy, runner, revision


def _annotate_rdep(rec: dict, ex, src_ex) -> None:
    pred = rec.get("plan_pred_status") or {}
    gold_this = rec.get("gold_status") or gold_eligibility_statuses(ex)
    rec["n_status_match_this"] = n_status_agree(pred, gold_this)
    if src_ex is None:
        rec["gold_status_src"] = None
        rec["r_source_key"] = None
        rec["n_status_match_src"] = None
        rec["all_match_src"] = False
        rec["match_src_better"] = False
        return
    gold_src = gold_eligibility_statuses(src_ex)
    rec["gold_status_src"] = gold_src
    rec["r_source_key"] = a_state_key(src_ex)
    rec["n_status_match_src"] = n_status_agree(pred, gold_src)
    rec["all_match_src"] = rec["n_status_match_src"] == len(CARRIERS)
    rec["match_src_better"] = rec["n_status_match_src"] > rec["n_status_match_this"]


def _run_one(
    runner,
    ex,
    extra_a,
    spans,
    composed,
    slots,
    cond,
    args,
    *,
    bundle,
    wall,
    extra_meta,
    src_ex,
):
    rec = _run_kv_b(
        runner,
        ex,
        bundle,
        plan_max_new=args.plan_max_new,
        answer_max_new=args.max_new_tokens,
        plan_prompt=b_reason_kv_instruct_prompt(ex),
        make_answer_prompt=b_answer_kv_instruct_prompt,
    )
    rec.update(extra_a)
    rec.update(extra_meta)
    rec["condition"] = cond
    rec["r_source"] = SPEC[cond]["r_source"]
    rec["prefix_source"] = PREFIX_SOURCE
    rec["layout"] = composed.get("layout")
    rec["layout_separator"] = True
    rec["layout_turn_close"] = False
    rec["extra_separator"] = EXTRA_SEP
    rec["n_prefix"] = int(slots["n_prefix"])
    rec["n_extra"] = int(slots["n_extra"])
    rec["n_extra_sep"] = int(slots["n_extra_sep"])
    rec["n_result_this"] = int(slots["n_result"])
    rec["n_result_slots"] = int(extra_meta.get("n_result_slots") or 0)
    rec["result_start"] = int(slots["result_start"])
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = n_kv_positions(bundle.cache_cpu)
    rec["n_result_kept"] = int(extra_meta.get("n_result_slots") or 0)
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["n_added"] = int(composed.get("n_added") or 0)
    rec["n_full"] = int(spans["n_full"])
    rec["n_causal_prefix"] = int(spans.get("causal_prefix", {}).get("n") or 0)
    rec["template_id"] = spans.get("template_id")
    rec["selected_text"] = extra_meta.get("selected_text")
    rec["selected_token_ids"] = extra_meta.get("selected_token_ids") or _bundle_ids(bundle)
    rec["instruction_paragraph"] = (spans.get("instruction") or {}).get("text")
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
    rec["message_bytes"] = int(bundle.full_relay_bytes) + rec["framing_bytes"]
    rec["cache_source"] = SPEC[cond]["cache_source"]
    rec["shared_across_heads"] = True
    rec["n_kv_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["plan_prompt"] = b_reason_kv_instruct_prompt(ex)
    rec["payload_note"] = PAYLOAD_NOTE
    rec["process_time_s"] = _process_time_s(rec)
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    _annotate_plan(rec, ex)
    _annotate_rdep(rec, ex, src_ex)
    return rec


def _family_members(examples) -> dict[str, list]:
    by_fam: dict[str, list] = defaultdict(list)
    for ex in examples:
        by_fam[ex.family_id].append(ex)
    return dict(by_fam)


def _rebuild_family(
    runner,
    family_id,
    members,
    artifacts,
    by_id,
    decode,
    sep_ids,
    close_ids,
    spans_by_key,
    generics,
    *,
    require_ok: bool = True,
):
    keys = [f"{family_id}-base", f"{family_id}-a_flip"]
    states = {}
    for key in keys:
        rec = artifacts[key]
        src = by_id.get(key) or sheet_example(members, key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        states[key] = _pair_state(
            runner, rec, src, decode, sep_ids, close_ids, spans_by_key[key], generics
        )
    for key in keys:
        _attach_mismatch(runner, states[key], states[partner_a_key(key)])
        states[key]["check_row"] = _check_row(states[key])
        if require_ok and not states[key]["check_row"]["ok"]:
            raise RuntimeError(f"R-dependence construction failed for {key}.")
    return states


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    templates, spans_by_key = _collect_templates(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    families = _family_members(examples)
    rows = []
    print(
        f"R-dependence check on {len(artifacts)} unique A sheets / {len(families)} families, layout={LAYOUT}",
        flush=True,
    )
    for i, family_id in enumerate(sorted(families)):
        members = families[family_id]
        states = _rebuild_family(
            runner,
            family_id,
            members,
            artifacts,
            by_id,
            decode,
            sep_ids,
            close_ids,
            spans_by_key,
            generics,
            require_ok=False,
        )
        for key in (f"{family_id}-base", f"{family_id}-a_flip"):
            row = states[key]["check_row"]
            rows.append(row)
            print(
                f"[{i + 1}/{len(families)}] {key} ok={row['ok']} partner={row['partner_key']} "
                f"n_R={row['n_result']} n_R_src={row['n_result_partner']} "
                f"start={row['result_start']} mismatch={row['mismatch_ok']} no_r={row['no_r_ok']}",
                flush=True,
            )
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _persist(out_root, kaggle_dest)

    construction = _summarize_check(rows, templates)
    slim_spans = {}
    for key, spans in spans_by_key.items():
        slim_spans[key] = {
            "ok": spans.get("ok"),
            "failures": spans.get("failures"),
            "n_instruction": spans.get("n_instruction"),
            "instruction": spans.get("instruction"),
            "instruction_pos": spans.get("instruction_pos"),
            "result_pos": spans.get("result_pos"),
            "result_text": spans.get("result_text"),
            "n_full": spans.get("n_full"),
            "n_prompt": spans.get("n_prompt"),
            "n_sink_extra": spans.get("n_sink_extra"),
            "instruction_before_facts": spans.get("instruction_before_facts"),
            "causal_prefix": spans.get("causal_prefix"),
            "template_id": spans.get("template_id"),
            "a_state_key": spans.get("a_state_key"),
        }
    summary = {
        "n": construction["n_unique_a_sheets"],
        "all_passed": construction["all_passed"],
        "layout": LAYOUT,
        "construction": construction,
        "rows": rows,
        "lead": (
            "If all_passed, run --phase compare: generic support with correct R, "
            "paired A-flip R, and no R, holding B's prompt fixed."
        ),
        "frozen_placement": FROZEN_PLACEMENT,
    }
    template_meta = {
        name: {k: v for k, v in rec.items() if k != "bundle"}
        for name, rec in generics.items()
    }
    _write_json(
        out_root / "construction.json",
        {
            **construction,
            "by_a_state": slim_spans,
            "templates": template_meta,
            "pairs": [
                {
                    "a_state_key": row["a_state_key"],
                    "partner_key": row["partner_key"],
                    "n_result": row["n_result"],
                    "n_result_partner": row["n_result_partner"],
                    "result_start": row["result_start"],
                    "result_len_differ": row["result_len_differ"],
                }
                for row in rows
            ],
            "all_ok": construction["all_passed"],
        },
    )
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(
        json.dumps(
            {
                k: construction[k]
                for k in (
                    "n_unique_a_sheets",
                    "all_passed",
                    "n_templates",
                    "n_result_equal",
                    "n_mismatch_ok",
                    "n_no_r_ok",
                    "n_result_len_differ",
                )
            },
            indent=2,
        )
    )
    return 0 if construction["all_passed"] else 1


def _selected_meta(state: dict, cond: str, other: dict | None) -> dict:
    spans = state["instr"]
    composed = state["composed_extra"]
    slots = state["slots_extra"]
    if cond == "generic_no_r":
        bundle = state["mixed"][cond]
        return {
            "n_result_slots": 0,
            "n_result_src": 0,
            "result_len_differ": False,
            "selected_text": "",
            "selected_token_ids": _bundle_ids(bundle),
            "result_held_from_transferred": False,
        }
    if cond == "generic_mismatch":
        if other is None:
            raise RuntimeError("Mismatch condition needs the paired A sheet.")
        bundle = state["mixed"][cond]
        n_src = int(other["slots_extra"]["n_result"])
        return {
            "n_result_slots": n_src,
            "n_result_src": n_src,
            "result_len_differ": n_src != int(slots["n_result"]),
            "selected_text": (other["instr"].get("result_text") or ""),
            "selected_token_ids": _bundle_ids(bundle),
            "result_held_from_transferred": True,
        }
    bundle = state["mixed"][cond]
    return {
        "n_result_slots": int(slots["n_result"]),
        "n_result_src": int(slots["n_result"]),
        "result_len_differ": False,
        "selected_text": spans.get("result_text") or "",
        "selected_token_ids": list(composed["ids"]),
        "result_held_from_transferred": True,
    }


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    check_path = out_root / "check_summary.json"
    check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.is_file() else None
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first. Pass --force-compare only to skip that gate.")
        return 2
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    templates, spans_by_key = _collect_templates(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    families = _family_members(examples)
    print(f"R-dependence compare layout={LAYOUT} n_families={len(families)}", flush=True)

    for i, family_id in enumerate(sorted(families)):
        members = families[family_id]
        remaining_any = any((ex.example_id, c) not in done for ex in members for c in CONDS)
        if not remaining_any:
            print(f"[{i + 1}/{len(families)}] {family_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(families)}] {family_id} rebuilding paired A sheets", flush=True)
        states = _rebuild_family(
            runner, family_id, members, artifacts, by_id, decode, sep_ids, close_ids, spans_by_key, generics
        )
        for ex in members:
            remaining = [c for c in CONDS if (ex.example_id, c) not in done]
            if not remaining:
                print(f"  {ex.example_id} already complete", flush=True)
                continue
            print(f"  {ex.example_id} remaining={len(remaining)}", flush=True)
            reset_peak_memory()
            key = a_state_key(ex)
            state = states[key]
            other = states[partner_a_key(key)]
            extra_a = _a_extra(state)
            extra_a["rebuild_time_s"] = float(state["rebuild_time_s"])
            extra_a["ids_source"] = state["meta"].get("ids_source")
            extra_a["reused_a_from"] = (
                state["meta"]["a_state_key"] if state["meta"]["a_state_key"] != ex.example_id else None
            )
            spans = state["instr"]
            src_by_cond = {
                "generic_r": sheet_example(members, key),
                "generic_mismatch": sheet_example(members, partner_a_key(key)),
                "generic_no_r": None,
            }
            for cond in remaining:
                selected = _selected_meta(state, cond, other)
                extra_meta = {
                    "span_extract_ok": bool(spans.get("ok")),
                    "n_instruction": int(spans.get("n_instruction") or 0),
                    "instruction_before_facts": bool(spans.get("instruction_before_facts")),
                    "instruction_contains_rule": ELIGIBILITY_RULE
                    in ((spans.get("instruction") or {}).get("text") or ""),
                    "template_id": spans.get("template_id"),
                    "partner_key": state.get("partner_key"),
                    "n_result_src": selected["n_result_src"],
                    "result_len_differ": selected["result_len_differ"],
                    "result_held_from_transferred": selected["result_held_from_transferred"],
                    "selected_text": selected["selected_text"],
                    "selected_token_ids": selected["selected_token_ids"],
                    "n_result_slots": selected["n_result_slots"],
                    "extra_equal": (state["info_by"].get("generic_r") or {}).get("extra_equal"),
                    "extra_keys_max_abs": (state["info_by"].get("generic_r") or {}).get("extra_keys_max_abs"),
                    "extra_values_max_abs": (state["info_by"].get("generic_r") or {}).get("extra_values_max_abs"),
                }
                rec_out = _run_one(
                    runner,
                    ex,
                    extra_a,
                    spans,
                    state["composed_extra"],
                    state["slots_extra"],
                    cond,
                    args,
                    bundle=runner.clone_sender(state["mixed"][cond]),
                    wall=state["times_by"].get(cond, 0.0),
                    extra_meta=extra_meta,
                    src_ex=src_by_cond[cond],
                )
                rows.append(rec_out)
                done.add((ex.example_id, cond))
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                print(
                    f"    {cond}: pred={rec_out.get('pred_carrier')} correct={rec_out.get('correct')} "
                    f"elig={rec_out.get('plan_all_status_correct')} "
                    f"match_this={rec_out.get('n_status_match_this')} "
                    f"match_src={rec_out.get('n_status_match_src')} parse={rec_out.get('parse_ok')}",
                    flush=True,
                )
                _persist(out_root, kaggle_dest)
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = _summarize(rows, check=check)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:12000])
    print(json.dumps(summary.get("label_shift"), indent=2)[:8000])
    print(json.dumps(summary.get("family_clustered_eligibility"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="Generic support with correct R, paired A-flip R, or no R."
    )
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--phase", choices=("check", "compare", "all", "rescore", "rewrite_ok"), default="check")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("R-dependence uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_rdep" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_rdep.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "check_summary.json",
            "construction.json",
            "a_artifacts.jsonl",
            "manifest.json",
        ):
            src = kaggle_dest / name
            dest = out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    if args.phase == "rewrite_ok":
        shared = REPO_ROOT / "results_stage_plan4_rdep"
        seen = set()
        for path, do_persist in (
            (out_root, True),
            (out_root.parent / "latest", False),
            (shared, False),
            (kaggle_dest, False),
        ):
            if path is None or not path.exists():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            if not any((path / name).is_file() for name in ("check_summary.json", "construction.json", "summary.json")):
                continue
            seen.add(resolved)
            print(f"Rewriting ok fields in {path}", flush=True)
            rc = _rewrite_ok_export(path, None, persist=do_persist)
            if rc != 0:
                return rc
        if not seen:
            print("No rdep JSON found to rewrite.")
            return 2
        return 0

    examples = load_plan4_jsonl(data)
    for ex in examples:
        assert_plan4_invariants(ex)
    if len(examples) != 90:
        print(f"Expected 90 plan4_dev examples, got {len(examples)}.")
        return 2

    artifacts_path = _copy_frozen_artifacts(out_root, explicit=args.artifacts)
    artifacts = _load_messages(artifacts_path)
    if len(artifacts) != 60:
        print(f"Expected 60 unique frozen A sheets, got {len(artifacts)}.")
        return 2
    for rec in artifacts.values():
        if not rec.get("full_ids"):
            print("Frozen artifacts are missing token IDs. Do not regenerate A.")
            return 2

    if args.phase == "rescore":
        check = None
        cpath = out_root / "check_summary.json"
        if cpath.is_file():
            check = json.loads(cpath.read_text(encoding="utf-8"))
        summary = _summarize(_load_rows(out_root / "evaluations.jsonl"), check=check)
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:8000])
        print(json.dumps(summary.get("label_shift"), indent=2)[:8000])
        return 0

    policy, runner, revision = _prepare_runner(args, out_root, extra={"phase": args.phase})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} layout={LAYOUT} n={len(examples)}")
    if args.phase == "check":
        return _run_check(args, out_root, kaggle_dest, examples, runner, artifacts)
    if args.phase == "compare":
        return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)
    rc = _run_check(args, out_root, kaggle_dest, examples, runner, artifacts)
    if rc != 0 and not args.force_compare:
        print("Check failed; stopping.")
        return rc
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
