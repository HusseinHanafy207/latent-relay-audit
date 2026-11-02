"""Result-line-only key/value mix, after comparing the packed prefix.

The previous mix replaced the whole compact cache, including the first four
tokens and the newline. This stage:
1. Compares those prefix positions between transferred and reconstructed caches.
2. If they are not exactly identical, repeats the four conditions with one
   shared reconstructed prefix, replacing only result-line keys or values.

A is not regenerated. plan4_hold stays out. Do not resume kvswap.
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

from src.boundary_layout import compact_slot_ranges
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
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
)
from src.span_select import caches_equal
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_boundary import (
    PAYLOAD_NOTE,
    _added_token_ids,
    _annotate_plan,
    _build_recon,
    _build_transfer,
    _carrier_block,
    _layouts_for_spans,
    candidate_qualifies,
)
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta
from src.stage_plan4_kvswap import LAYOUT, _frozen_refs
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _compact_spans, _copy_frozen_artifacts, _rebuild_a
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
PREFIX_SOURCE = "reconstruct"
CONDS = ("a_k_a_v", "recon_k_a_v", "a_k_recon_v", "reconstruct")
SPEC = {
    "a_k_a_v": {"keys": "A", "values": "A", "cache_source": "mix_result_a_k_a_v"},
    "recon_k_a_v": {"keys": "reconstruct", "values": "A", "cache_source": "mix_result_recon_k_a_v"},
    "a_k_recon_v": {"keys": "A", "values": "reconstruct", "cache_source": "mix_result_a_k_recon_v"},
    "reconstruct": {"keys": "reconstruct", "values": "reconstruct", "cache_source": "reconstruct_prefill"},
}
PAIRED = (
    ("recon_k_a_v_vs_a_k_a_v", "a_k_a_v", "recon_k_a_v"),
    ("a_k_recon_v_vs_a_k_a_v", "a_k_a_v", "a_k_recon_v"),
    ("reconstruct_vs_a_k_a_v", "a_k_a_v", "reconstruct"),
    ("recon_k_a_v_vs_reconstruct", "reconstruct", "recon_k_a_v"),
    ("a_k_recon_v_vs_reconstruct", "reconstruct", "a_k_recon_v"),
    ("recon_k_a_v_vs_a_k_recon_v", "a_k_recon_v", "recon_k_a_v"),
)
PRESERVED = (
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
INTERPRETATION_GUIDE = (
    "If the packed prefix is exactly identical, the previous whole-cache mix "
    "already replaced result-line components. "
    "If the prefix differs, these four conditions keep one reconstructed prefix "
    "and replace only result-line keys or values. "
    "Replacing result keys recovering labels supports an attention-side problem "
    "on the selected result states. Replacing result values recovering labels "
    "supports a representation-side problem. Only replacing both recovering "
    "labels supports compatibility. Mixed failures alone do not settle that."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_kvresult" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_kvresult")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("check_summary.json", "prefix_summary.json"):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _summarize_prefix(rows: list[dict]) -> dict:
    n = len(rows)
    n_equal = sum(bool(r.get("equal")) for r in rows)
    n_keys = sum(bool(r.get("keys_equal")) for r in rows)
    n_vals = sum(bool(r.get("values_equal")) for r in rows)
    return {
        "n_unique_a_sheets": n,
        "n_prefix_equal": n_equal,
        "n_prefix_keys_equal": n_keys,
        "n_prefix_values_equal": n_vals,
        "prefix_identical": bool(n) and n_equal == n,
        "max_keys_max_abs": max((float(r.get("keys_max_abs") or 0) for r in rows), default=None),
        "max_values_max_abs": max((float(r.get("values_max_abs") or 0) for r in rows), default=None),
        "mean_n_prefix": (sum(int(r.get("n_prefix") or 0) for r in rows) / n) if n else None,
        "mean_n_result": (sum(int(r.get("n_result") or 0) for r in rows) / n) if n else None,
        "note": (
            "Exact tensor equality on packed prefix slots (initial four plus the "
            "newline). If every sheet matches, the earlier whole-cache mix was "
            "already a result-line mix."
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
    kv_rows = by_cond.get("reconstruct", [])
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
        "mix_scope": "result_slots",
        "baseline": "reconstructed prefix; A's keys and values on result slots",
        "unit_note": (
            "plan4_dev only. Frozen sep layout. Shared reconstructed prefix "
            "(first four tokens plus newline). Result-line keys and values are "
            "taken from A or from reconstruction. Tokens, positions, framing, "
            "and decoding stay unchanged."
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
        "frozen_reconstruct_sep": refs["reconstruct_sep"],
        "frozen_text_result": refs["text_result"],
        "qualifies_vs_reconstruct": {
            cond: candidate_qualifies(receiver.get(cond) or {}, refs["reconstruct_sep"]) for cond in CONDS
        },
        "spec": SPEC,
        "prefix": None if check is None else check.get("prefix"),
        "check": None if check is None else {"all_passed": check.get("all_passed"), "n": check.get("n")},
        "framing_change": FRAMING_INSTRUCTION,
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels after replacing only "
            "result-line keys or values on a shared reconstructed prefix."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": "These are development diagnostics. Do not load plan4_hold. Do not train an adapter yet.",
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
            "stage": "plan4-kvresult",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "layout": LAYOUT,
            "prefix_source": PREFIX_SOURCE,
            "mix_scope": "result_slots",
            "conditions": list(CONDS),
            "framing_change": FRAMING_INSTRUCTION,
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


def _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids):
    a_bundle, rebuild_s = _rebuild_a(runner, rec, src)
    spans = _compact_spans(
        rec["full_ids"],
        n_prompt=int(rec["n_prompt_positions"]),
        n_full=int(rec["n_kv_positions"]),
        decode=decode,
    )
    pack = _layouts_for_spans(spans, sep_ids=sep_ids, close_ids=close_ids)
    reconstruct, recon_s, composed = _build_recon(runner, pack, LAYOUT)
    transfer, trans_s, info, _c = _build_transfer(runner, a_bundle, pack, LAYOUT)
    recon_ids = [int(t) for t in reconstruct.full_ids_cpu.view(-1).tolist()]
    trans_ids = [int(t) for t in transfer.full_ids_cpu.view(-1).tolist()]
    if recon_ids != list(composed["ids"]) or trans_ids != list(composed["ids"]):
        raise RuntimeError("Sep-layout token IDs drifted from the frozen compact sequence.")
    slots = compact_slot_ranges(composed)
    prefix_delta = cache_slice_delta(transfer.cache_cpu, reconstruct.cache_cpu, slots["prefix"])
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
        "times": {
            "a_k_a_v": trans_s + recon_s,
            "recon_k_a_v": trans_s + recon_s,
            "a_k_recon_v": trans_s + recon_s,
            "reconstruct": recon_s,
        },
        "prefix_delta": prefix_delta,
    }


def _result_mix_pack(runner, transfer, reconstruct, result_pos: list[int]) -> dict:
    identity_r = runner.mix_sender_kv_at(
        reconstruct, reconstruct, base=reconstruct, positions=result_pos, source="mix_result_recon_k_recon_v"
    )
    a_k_a_v = runner.mix_sender_kv_at(
        transfer, transfer, base=reconstruct, positions=result_pos, source="mix_result_a_k_a_v"
    )
    recon_k_a_v = runner.mix_sender_kv_at(
        reconstruct, transfer, base=reconstruct, positions=result_pos, source="mix_result_recon_k_a_v"
    )
    a_k_recon_v = runner.mix_sender_kv_at(
        transfer, reconstruct, base=reconstruct, positions=result_pos, source="mix_result_a_k_recon_v"
    )
    prefix = list(range(min(result_pos))) if result_pos else []
    prefix_held = True
    if prefix:
        prefix_held = cache_slice_delta(a_k_a_v.cache_cpu, reconstruct.cache_cpu, prefix)["equal"]
        prefix_held = prefix_held and cache_slice_delta(recon_k_a_v.cache_cpu, reconstruct.cache_cpu, prefix)["equal"]
        prefix_held = prefix_held and cache_slice_delta(a_k_recon_v.cache_cpu, reconstruct.cache_cpu, prefix)["equal"]
    return {
        "a_k_a_v": a_k_a_v,
        "recon_k_a_v": recon_k_a_v,
        "a_k_recon_v": a_k_recon_v,
        "reconstruct": reconstruct,
        "identity_reconstruct_eq": caches_equal(identity_r.cache_cpu, reconstruct.cache_cpu),
        "prefix_held_from_reconstruct": prefix_held,
    }


def _run_one(runner, ex, extra_a, spans, pack, composed, slots, cond, args, *, bundle, wall):
    spec = SPEC[cond]
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
    rec["condition"] = cond
    rec["keys_from"] = spec["keys"]
    rec["values_from"] = spec["values"]
    rec["prefix_source"] = PREFIX_SOURCE
    rec["mix_scope"] = "result_slots"
    rec["n_prefix"] = int(slots["n_prefix"])
    rec["n_result_slots"] = int(slots["n_result"])
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = int(composed["n"])
    rec["n_result_kept"] = len(spans["result_pos"])
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["n_added"] = int(composed["n_added"])
    rec["n_full"] = int(spans["n_full"])
    rec["selected_text"] = spans["result_text"]
    rec["selected_token_ids"] = list(composed["ids"])
    rec["layout"] = LAYOUT
    rec["layout_separator"] = True
    rec["layout_turn_close"] = False
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
    rec["message_bytes"] = int(bundle.full_relay_bytes) + rec["framing_bytes"]
    rec["cache_source"] = spec["cache_source"]
    rec["shared_across_heads"] = True
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
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
    prefix_rows = []
    construction = []
    print(f"Prefix comparison on {len(unique)} unique A sheets, layout={LAYOUT}", flush=True)
    for i, ex in enumerate(unique):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pair = _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids)
        slots = pair["slots"]
        delta = dict(pair["prefix_delta"])
        delta.update(
            {
                "a_state_key": key,
                "n_prefix": slots["n_prefix"],
                "n_result": slots["n_result"],
                "prefix_positions": slots["prefix"],
            }
        )
        prefix_rows.append(delta)
        mixed = _result_mix_pack(runner, pair["transfer"], pair["reconstruct"], slots["result"])
        ok = (
            mixed["identity_reconstruct_eq"]
            and mixed["prefix_held_from_reconstruct"]
            and slots["n_prefix"]
            == int(pair["composed"]["n_initial"]) + int(pair["composed"]["n_added"])
            and pair["composed"]["separator"]
            and not pair["composed"]["turn_close"]
        )
        construction.append(
            {
                "a_state_key": key,
                "ok": ok,
                "n": pair["composed"]["n"],
                "n_prefix": slots["n_prefix"],
                "n_result": slots["n_result"],
                "prefix_equal": delta["equal"],
                "prefix_keys_equal": delta["keys_equal"],
                "prefix_values_equal": delta["values_equal"],
                "prefix_keys_max_abs": delta["keys_max_abs"],
                "prefix_values_max_abs": delta["values_max_abs"],
                "identity_reconstruct_eq": mixed["identity_reconstruct_eq"],
                "prefix_held_from_reconstruct": mixed["prefix_held_from_reconstruct"],
            }
        )
        print(
            f"[{i + 1}/{len(unique)}] {key} n_prefix={slots['n_prefix']} "
            f"equal={delta['equal']} k_max={delta['keys_max_abs']:.4g} "
            f"v_max={delta['values_max_abs']:.4g} mix_ok={ok}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    prefix = _summarize_prefix(prefix_rows)
    construct_ok = bool(construction) and all(item["ok"] for item in construction)
    summary = {
        "n": len(construction),
        "layout": LAYOUT,
        "all_passed": construct_ok,
        "prefix_identical": prefix["prefix_identical"],
        "prefix": prefix,
        "construction": construction,
        "lead": (
            "If prefix_identical is true, stop: the previous mix already replaced "
            "result-line components. If it is false, run --phase compare: shared "
            "reconstructed prefix, result-line keys/values only."
        ),
    }
    _write_json(out_root / "prefix_summary.json", {"prefix": prefix, "rows": prefix_rows})
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("n", "all_passed", "prefix_identical")}, indent=2))
    print(json.dumps(prefix, indent=2))
    if prefix["prefix_identical"]:
        print("Packed prefixes are exactly identical. Do not run the 90-case result-slot mix.")
    return 0 if construct_ok else 1


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    check_path = out_root / "check_summary.json"
    check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.is_file() else None
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first. Pass --force-compare only to skip that gate.")
        return 2
    if check is not None and check.get("prefix_identical") and not args.force_compare:
        print("Prefix tensors are exactly identical. The previous mix already replaced result-line components.")
        return 0
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in CONDS)}
    print(f"Result-slot mix layout={LAYOUT} rebuilding {len(needed)} unique A caches", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pair = _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids)
        mixed = _result_mix_pack(runner, pair["transfer"], pair["reconstruct"], pair["slots"]["result"])
        if not mixed["identity_reconstruct_eq"] or not mixed["prefix_held_from_reconstruct"]:
            raise RuntimeError(f"Result-slot mix construction failed for {key}.")
        pair["mixed"] = mixed
        states[key] = pair
        print(f"  rebuilt {key} n={pair['composed']['n']} n_result={pair['slots']['n_result']}", flush=True)
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
        for cond in remaining:
            rec_out = _run_one(
                runner,
                ex,
                extra_a,
                state["spans"],
                state["pack"],
                state["composed"],
                state["slots"],
                cond,
                args,
                bundle=runner.clone_sender(state["mixed"][cond]),
                wall=state["times"].get(cond, 0.0),
            )
            rows.append(rec_out)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec_out.get('pred_carrier')} correct={rec_out.get('correct')} "
                f"elig={rec_out.get('plan_all_status_correct')}",
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
        description="Compare packed prefixes, then mix only result-line keys/values if they differ."
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
        print("Result-slot mix uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_kvresult" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_kvresult.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "check_summary.json",
            "prefix_summary.json",
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
