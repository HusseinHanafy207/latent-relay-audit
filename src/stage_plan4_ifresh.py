"""Transferred instruction KV vs a fresh forward of the same instruction tokens.

The previous instruction-only run recovered labels with transferred instruction
states plus transferred R. This comparison asks whether B needs those cached
instruction states, or whether freshly processing the identical tokens in the
same packed location is enough.

Both conditions keep the same reconstructed prefix, the same transferred R
tensors at the same packed slots, the same B prompt, and the same decoding.
Do not add calculations. Do not train an adapter. plan4_hold stays out.
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

from src.boundary_layout import EXTRA_SEP_KIND
from src.cache_utils import cache_slice_delta, n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    a_state_key,
    assert_plan4_invariants,
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
from src.stage_plan4_instr import EXTRA_SEP, _build_extra_bundle
from src.stage_plan4_kvresult import PREFIX_SOURCE, _rebuild_pair
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _copy_frozen_artifacts
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.support_select import extract_instruction_pack
from src.two_agent import TwoAgentInventory

LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("instr_kv", "instr_fresh")
SPEC = {
    "instr_kv": {"instruction": "transfer", "cache_source": "transfer_instr_kv"},
    "instr_fresh": {"instruction": "fresh", "cache_source": "fresh_instr_same_R"},
}
PAIRED = (("instr_fresh_vs_instr_kv", "instr_kv", "instr_fresh"),)
PRESERVED = (
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
    "instr_kv: reconstructed prefix, transferred instruction KV, newline, transferred R. "
    "instr_fresh: reconstructed prefix, freshly forwarded identical instruction tokens "
    "and newline, then the same transferred R tensors appended at the same packed slots. "
    "B's prompt and decoding stay unchanged."
)
INTERPRETATION_GUIDE = (
    "If both perform similarly, a freshly built general-instruction prefix may suffice; "
    "B does not need A's cached instruction states at this location. "
    "If transferred instruction states remain better, their conditioning on A's "
    "preceding context becomes worth examining. "
    "This comparison holds R's keys, values, and packed positions fixed, so a gap "
    "is about the instruction block, not a second copy of the result states. "
    "It does not isolate every property of those instruction states. "
    "Do not expand to calculations or start training an adapter from this file."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_ifresh" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_ifresh")
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
    instr_kv = {
        "n": 90,
        "plan_all_status_correct": 87,
        "correct": 70,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_instr instr_kv",
    }
    reconstruct = {
        "n": 90,
        "plan_all_status_correct": 89,
        "correct": 74,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_instr reconstruct",
    }
    for folder in (
        REPO_ROOT / "results_stage_plan4_instr" / "summary.json",
        Path("/kaggle/working/results_stage_plan4_instr/summary.json"),
    ):
        if not folder.is_file():
            continue
        data = json.loads(folder.read_text(encoding="utf-8"))
        rec = data.get("receiver") or {}
        i_block = rec.get("instr_kv") or {}
        r_block = rec.get("reconstruct") or {}
        if int(i_block.get("n") or 0) == 90:
            instr_kv = {
                "n": 90,
                "plan_all_status_correct": i_block.get("plan_all_status_correct"),
                "correct": i_block.get("correct"),
                "parse_ok": i_block.get("parse_ok"),
                "truncated": i_block.get("truncated"),
                "source": str(folder).replace("\\", "/"),
            }
        if int(r_block.get("n") or 0) == 90:
            reconstruct = {
                "n": 90,
                "plan_all_status_correct": r_block.get("plan_all_status_correct"),
                "correct": r_block.get("correct"),
                "parse_ok": r_block.get("parse_ok"),
                "truncated": r_block.get("truncated"),
                "source": str(folder).replace("\\", "/"),
            }
        break
    return {"instr_kv": instr_kv, "reconstruct": reconstruct}


def _extra_sep_ids(composed: dict) -> list[int]:
    for piece in composed.get("pieces") or []:
        if piece.get("kind") == EXTRA_SEP_KIND:
            return [int(t) for t in piece["ids"]]
    return []


def _build_fresh_from_transferred(runner, pair, transferred, composed, slots):
    prefix = runner.prefix_bundle(pair["reconstruct"], slots["n_prefix"])
    extra_ids = [int(t) for t in composed["extra_ids"]]
    sep_ids = _extra_sep_ids(composed)
    head, wall = runner.extend_cache_with_tokens(prefix, extra_ids + sep_ids, source="fresh_instr")
    if n_kv_positions(head.cache_cpu) != int(slots["result_start"]):
        raise RuntimeError(
            f"Fresh instruction+newline length {n_kv_positions(head.cache_cpu)} "
            f"!= result start {slots['result_start']}."
        )
    bundle = runner.append_packed_slots(
        head, transferred, tail_positions=slots["result"], source="fresh_instr_same_R"
    )
    got = [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]
    if got != list(composed["ids"]):
        raise RuntimeError("Fresh instruction cache token IDs drifted from the transferred layout.")
    prefix_held = cache_slice_delta(bundle.cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
    r_held = cache_slice_delta(bundle.cache_cpu, transferred.cache_cpu, slots["result"])["equal"]
    extra_delta = cache_slice_delta(bundle.cache_cpu, transferred.cache_cpu, slots["extra"])
    if not prefix_held:
        raise RuntimeError("Fresh instruction cache did not keep the reconstructed prefix.")
    if not r_held:
        raise RuntimeError("Fresh instruction cache did not keep the transferred R tensors.")
    if int(bundle.pos_cursor_cpu.view(-1)[0].item()) != int(composed["n"]):
        raise RuntimeError("B starting position drifted after splicing R.")
    info = {
        "prefix_held_from_reconstruct": True,
        "result_held_from_transferred": True,
        "extra_equal": extra_delta["equal"],
        "extra_keys_max_abs": extra_delta["keys_max_abs"],
        "extra_values_max_abs": extra_delta["values_max_abs"],
        "result_start": slots["result_start"],
        "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
        "n_kept": n_kv_positions(bundle.cache_cpu),
        "extend_time_s": float(wall),
    }
    return bundle, wall, info


def _summarize_check(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_unique_a_sheets": n,
        "n_ok": sum(bool(r.get("ok")) for r in rows),
        "all_passed": bool(n) and all(bool(r.get("ok")) for r in rows),
        "n_ids_equal": sum(bool(r.get("ids_equal")) for r in rows),
        "n_result_equal": sum(bool(r.get("result_equal")) for r in rows),
        "n_prefix_equal": sum(bool(r.get("prefix_equal")) for r in rows),
        "n_cursor_equal": sum(bool(r.get("cursor_equal")) for r in rows),
        "n_extra_equal": sum(bool(r.get("extra_equal")) for r in rows),
        "max_extra_keys_max_abs": max((float(r.get("extra_keys_max_abs") or 0) for r in rows), default=None),
        "max_extra_values_max_abs": max((float(r.get("extra_values_max_abs") or 0) for r in rows), default=None),
        "mean_n_instruction": (sum(int(r.get("n_instruction") or 0) for r in rows) / n) if n else None,
        "note": (
            "Both caches must share token IDs, reconstructed prefix, transferred R, "
            "and B's starting position. Extra-block states may differ."
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
    kv_rows = by_cond.get("instr_kv", [])
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
        "baseline": "transferred instruction KV plus the same transferred R",
        "unit_note": (
            "plan4_dev only. Same reconstructed prefix, instruction token IDs, newline, "
            "transferred R tensors and packed slots, B prompt, and decoding. "
            "The two conditions differ only in whether the instruction block is A's "
            "cached states or a fresh forward on that prefix."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
        "carrier_label_errors": {cond: _carrier_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {name: _paired(grouped_plain, a, b) for name, a, b in PAIRED},
        "family_clustered_delta": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED) for name, a, b in PAIRED
        },
        "family_clustered_eligibility": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED, key="plan_all_status_correct")
            for name, a, b in PAIRED
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "frozen_instr_kv": refs["instr_kv"],
        "frozen_reconstruct": refs["reconstruct"],
        "qualifies_vs_reconstruct": {
            cond: candidate_qualifies(receiver.get(cond) or {}, refs["reconstruct"]) for cond in CONDS
        },
        "spec": SPEC,
        "check": None if check is None else {"all_passed": check.get("all_passed"), "n": check.get("n"), **(check.get("construction") or {})},
        "framing_change": FRAMING_INSTRUCTION,
        "frozen_placement": FROZEN_PLACEMENT,
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels. Compare transferred "
            "instruction KV with a fresh forward of the same instruction tokens, "
            "holding R fixed."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "Interpret this instruction-source comparison before freezing a candidate "
            "or taking the professor's timing measurements. Do not load plan4_hold. "
            "Do not train an adapter. Do not add the calculation block yet."
        ),
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _cond_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
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
            "stage": "plan4-ifresh",
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


def _extract_spans(rec, decode, key: str) -> dict:
    return extract_instruction_pack(
        rec["full_ids"],
        n_prompt=int(rec["n_prompt_positions"]),
        n_full=int(rec.get("n_kv_positions") or len(rec["full_ids"])),
        decode=decode,
        a_state_key=key,
        a_error_recorded=bool(rec.get("a_error_recorded"))
        or not bool((rec.get("written_score") or {}).get("all_four_correct")),
    )


def _pair_with_both(runner, rec, src, decode, sep_ids, close_ids, spans):
    pair = _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids)
    if list(spans.get("result_pos") or []) != list(pair["spans"]["result_pos"]):
        raise RuntimeError("Instruction-pack result positions drifted from compact R.")
    extra_pos = list(spans["instruction_pos"])
    transferred, trans_s, composed, slots, trans_info = _build_extra_bundle(
        runner, pair, extra_pos, source="transfer_instr_kv"
    )
    fresh, fresh_s, fresh_info = _build_fresh_from_transferred(
        runner, pair, transferred, composed, slots
    )
    trans_ids = [int(t) for t in transferred.full_ids_cpu.view(-1).tolist()]
    fresh_ids = [int(t) for t in fresh.full_ids_cpu.view(-1).tolist()]
    result_eq = cache_slice_delta(fresh.cache_cpu, transferred.cache_cpu, slots["result"])["equal"]
    prefix_eq = cache_slice_delta(fresh.cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
    cursor_eq = int(fresh.pos_cursor_cpu.view(-1)[0].item()) == int(transferred.pos_cursor_cpu.view(-1)[0].item())
    ok = (
        bool(spans.get("ok"))
        and trans_ids == fresh_ids == list(composed["ids"])
        and result_eq
        and prefix_eq
        and cursor_eq
        and int(slots["n_prefix"]) == 5
        and EXTRA_SEP
    )
    pair.update(
        {
            "instr": spans,
            "composed_extra": composed,
            "slots_extra": slots,
            "mixed": {"instr_kv": transferred, "instr_fresh": fresh},
            "times_by": {"instr_kv": trans_s, "instr_fresh": trans_s + fresh_s},
            "info_by": {"instr_kv": trans_info, "instr_fresh": fresh_info},
            "check_row": {
                "a_state_key": spans.get("a_state_key"),
                "ok": ok,
                "ids_equal": trans_ids == fresh_ids,
                "result_equal": result_eq,
                "prefix_equal": prefix_eq,
                "cursor_equal": cursor_eq,
                "extra_equal": fresh_info.get("extra_equal"),
                "extra_keys_max_abs": fresh_info.get("extra_keys_max_abs"),
                "extra_values_max_abs": fresh_info.get("extra_values_max_abs"),
                "n_instruction": int(spans.get("n_instruction") or 0),
                "result_start": int(slots["result_start"]),
                "pos_cursor": int(fresh.pos_cursor_cpu.view(-1)[0].item()),
                "instruction_before_facts": bool(spans.get("instruction_before_facts")),
                "instruction_contains_rule": ELIGIBILITY_RULE
                in ((spans.get("instruction") or {}).get("text") or ""),
            },
        }
    )
    return pair


def _run_one(runner, ex, extra_a, spans, composed, slots, cond, args, *, bundle, wall, extra_meta):
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
    rec["instruction_source"] = SPEC[cond]["instruction"]
    rec["prefix_source"] = PREFIX_SOURCE
    rec["layout"] = composed.get("layout")
    rec["layout_separator"] = True
    rec["layout_turn_close"] = False
    rec["extra_separator"] = EXTRA_SEP
    rec["n_prefix"] = int(slots["n_prefix"])
    rec["n_extra"] = int(slots["n_extra"])
    rec["n_extra_sep"] = int(slots["n_extra_sep"])
    rec["n_result_slots"] = int(slots["n_result"])
    rec["result_start"] = int(slots["result_start"])
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = n_kv_positions(bundle.cache_cpu)
    rec["n_result_kept"] = len(spans["result_pos"])
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["n_added"] = int(composed.get("n_added") or 0)
    rec["n_full"] = int(spans["n_full"])
    rec["selected_text"] = spans["result_text"]
    rec["selected_token_ids"] = list(composed["ids"])
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
    return rec


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    unique = []
    seen = set()
    for ex in examples:
        key = a_state_key(ex)
        if key not in seen:
            unique.append(ex)
            seen.add(key)
    rows = []
    slim = {}
    print(f"Instruction-source check on {len(unique)} unique A sheets, layout={LAYOUT}", flush=True)
    for i, ex in enumerate(unique):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        spans = _extract_spans(rec, decode, key)
        pair = _pair_with_both(runner, rec, src, decode, sep_ids, close_ids, spans)
        row = pair["check_row"]
        rows.append(row)
        slim[key] = spans
        print(
            f"[{i + 1}/{len(unique)}] {key} ok={row['ok']} n_instr={row['n_instruction']} "
            f"R_eq={row['result_equal']} extra_eq={row['extra_equal']} "
            f"extra_kmax={row['extra_keys_max_abs']:.4g}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    construction = _summarize_check(rows)
    summary = {
        "n": construction["n_unique_a_sheets"],
        "all_passed": construction["all_passed"],
        "layout": LAYOUT,
        "construction": construction,
        "rows": rows,
        "lead": (
            "If all_passed, run --phase compare: transferred instruction KV vs a "
            "fresh forward of the same tokens, with R held fixed."
        ),
        "frozen_placement": FROZEN_PLACEMENT,
    }
    _write_json(out_root / "construction.json", {**construction, "by_a_state": slim, "all_ok": construction["all_passed"]})
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: construction[k] for k in ("n_unique_a_sheets", "all_passed", "n_result_equal", "n_extra_equal")}, indent=2))
    return 0 if construction["all_passed"] else 1


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    check_path = out_root / "check_summary.json"
    check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.is_file() else None
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first. Pass --force-compare only to skip that gate.")
        return 2
    construction_path = out_root / "construction.json"
    construction = json.loads(construction_path.read_text(encoding="utf-8")) if construction_path.is_file() else {}
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in CONDS)}
    print(f"Instruction-source compare layout={LAYOUT} rebuilding {len(needed)} unique A caches", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        spans = (construction.get("by_a_state") or {}).get(key) or _extract_spans(rec, decode, key)
        pair = _pair_with_both(runner, rec, src, decode, sep_ids, close_ids, spans)
        if not pair["check_row"]["ok"]:
            raise RuntimeError(f"Instruction-source construction failed for {key}.")
        states[key] = pair
        print(
            f"  rebuilt {key} n_instr={spans['n_instruction']} result_start={pair['slots_extra']['result_start']}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [c for c in CONDS if (ex.example_id, c) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        state = states[a_state_key(ex)]
        extra_a = _a_extra(state)
        extra_a["rebuild_time_s"] = float(state["rebuild_time_s"])
        extra_a["ids_source"] = state["meta"].get("ids_source")
        extra_a["reused_a_from"] = (
            state["meta"]["a_state_key"] if state["meta"]["a_state_key"] != ex.example_id else None
        )
        spans = state["instr"]
        extra_info = state["info_by"].get("instr_fresh") or {}
        for cond in remaining:
            extra_meta = {
                "span_extract_ok": bool(spans.get("ok")),
                "n_instruction": int(spans.get("n_instruction") or 0),
                "instruction_before_facts": bool(spans.get("instruction_before_facts")),
                "instruction_contains_rule": ELIGIBILITY_RULE
                in ((spans.get("instruction") or {}).get("text") or ""),
                "result_held_from_transferred": True,
                "extra_equal": extra_info.get("extra_equal"),
                "extra_keys_max_abs": extra_info.get("extra_keys_max_abs"),
                "extra_values_max_abs": extra_info.get("extra_values_max_abs"),
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
            )
            rows.append(rec_out)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec_out.get('pred_carrier')} correct={rec_out.get('correct')} "
                f"elig={rec_out.get('plan_all_status_correct')} parse={rec_out.get('parse_ok')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    summary = _summarize(rows, check=check)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:12000])
    print(json.dumps(summary.get("family_clustered_eligibility"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="Transferred instruction KV vs a fresh forward of the same tokens, R held fixed."
    )
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--phase", choices=("check", "compare", "all", "rescore"), default="check")
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
        print("Instruction-source comparison uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_ifresh" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_ifresh.")
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
