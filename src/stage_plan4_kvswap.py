"""Key/value diagnostic on the frozen sep compact layout.

For the same selected tokens, first four positions, newline, assignments, and
B procedure, compare:
- transfer: A's keys and A's values
- recon_k_a_v: reconstructed keys with A's values
- a_k_recon_v: A's keys with reconstructed values
- reconstruct: reconstructed keys and values

The mix covers the packed compact cache so identity mixes recover the two
references. This is a diagnostic, not a communication method.

A is not regenerated. plan4_hold stays out. Do not resume boundary or recon.
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

from src.cache_utils import mix_cache_kv, n_kv_positions
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
    _wanted_examples,
    candidate_qualifies,
)
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _compact_spans, _copy_frozen_artifacts, _rebuild_a
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

LAYOUT = "sep"
LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("transfer", "recon_k_a_v", "a_k_recon_v", "reconstruct")
SPEC = {
    "transfer": {"keys": "A", "values": "A", "cache_source": "transfer_gather_rerotate_extend"},
    "recon_k_a_v": {"keys": "reconstruct", "values": "A", "cache_source": "mix_recon_k_a_v"},
    "a_k_recon_v": {"keys": "A", "values": "reconstruct", "cache_source": "mix_a_k_recon_v"},
    "reconstruct": {"keys": "reconstruct", "values": "reconstruct", "cache_source": "reconstruct_prefill"},
}
PAIRED = (
    ("recon_k_a_v_vs_transfer", "transfer", "recon_k_a_v"),
    ("a_k_recon_v_vs_transfer", "transfer", "a_k_recon_v"),
    ("reconstruct_vs_transfer", "transfer", "reconstruct"),
    ("recon_k_a_v_vs_reconstruct", "reconstruct", "recon_k_a_v"),
    ("a_k_recon_v_vs_reconstruct", "reconstruct", "a_k_recon_v"),
    ("recon_k_a_v_vs_a_k_recon_v", "a_k_recon_v", "recon_k_a_v"),
)
PRESERVED = (
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
    "If replacing keys recovers labels, that motivates how B attends to the message. "
    "If replacing values recovers labels, that motivates the representations being read. "
    "If only replacing both recovers labels, key/value compatibility may matter. "
    "Mixed-state failures alone do not settle those explanations. "
    "This diagnostic requires recomputation; it is not an efficient method."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_kvswap" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_kvswap")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("check_summary.json", "check_evaluations.jsonl"):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _frozen_refs() -> dict:
    recon = {
        "n": 90,
        "plan_all_status_correct": 89,
        "correct": 74,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_boundary transfer reconstruct/sep",
    }
    text = {
        "n": 90,
        "plan_all_status_correct": 90,
        "correct": 79,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_recon text_result",
    }
    for folder in (
        REPO_ROOT / "results_stage_plan4_boundary1" / "transfer_summary.json",
        REPO_ROOT / "results_stage_plan4_boundary" / "transfer_summary.json",
    ):
        if folder.is_file():
            data = json.loads(folder.read_text(encoding="utf-8"))
            if data.get("layout") == LAYOUT:
                block = (data.get("receiver") or {}).get("reconstruct") or {}
                if int(block.get("n") or 0) == 90:
                    recon = {
                        "n": 90,
                        "plan_all_status_correct": block.get("plan_all_status_correct"),
                        "correct": block.get("correct"),
                        "parse_ok": block.get("parse_ok"),
                        "truncated": block.get("truncated"),
                        "source": str(folder).replace("\\", "/"),
                    }
                    break
    text_path = REPO_ROOT / "results_stage_plan4_recon" / "summary.json"
    if text_path.is_file():
        data = json.loads(text_path.read_text(encoding="utf-8"))
        block = (data.get("receiver") or {}).get("text_result") or {}
        if int(block.get("n") or 0) == 90:
            text = {
                "n": 90,
                "plan_all_status_correct": block.get("plan_all_status_correct"),
                "correct": block.get("correct"),
                "parse_ok": block.get("parse_ok"),
                "truncated": block.get("truncated"),
                "source": "results_stage_plan4_recon/summary.json",
            }
    return {"reconstruct_sep": recon, "text_result": text}


def _summarize(rows: list[dict], *, check: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("transfer", [])
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
        "baseline": "sep compact: transferred A's keys and values",
        "unit_note": (
            "plan4_dev only. Frozen sep layout. Same selected result tokens, first "
            "four positions, newline, assignments, framing, and decoding. Conditions "
            "differ only in whether packed keys and values come from A or from a "
            "fresh prefill of those tokens."
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
        "framing_change": FRAMING_INSTRUCTION,
        "check": None if check is None else {"all_passed": check.get("all_passed"), "n": check.get("n")},
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels. Ask which packed "
            "components must change before B can use the selected message."
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
            "stage": "plan4-kvswap",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "layout": LAYOUT,
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


def _mixed_pack(runner, transfer, reconstruct) -> dict:
    identity_t = runner.mix_sender_kv(transfer, transfer, source="mix_a_k_a_v")
    identity_r = runner.mix_sender_kv(reconstruct, reconstruct, source="mix_recon_k_recon_v")
    recon_k_a_v = runner.mix_sender_kv(reconstruct, transfer, source="mix_recon_k_a_v")
    a_k_recon_v = runner.mix_sender_kv(transfer, reconstruct, source="mix_a_k_recon_v")
    return {
        "transfer": transfer,
        "recon_k_a_v": recon_k_a_v,
        "a_k_recon_v": a_k_recon_v,
        "reconstruct": reconstruct,
        "identity_transfer_eq": caches_equal(identity_t.cache_cpu, transfer.cache_cpu),
        "identity_reconstruct_eq": caches_equal(identity_r.cache_cpu, reconstruct.cache_cpu),
        "recon_k_a_v_keys_from_recon": caches_equal(
            mix_cache_kv(reconstruct.cache_cpu, reconstruct.cache_cpu),
            mix_cache_kv(recon_k_a_v.cache_cpu, reconstruct.cache_cpu),
        ),
        "recon_k_a_v_values_from_a": caches_equal(
            mix_cache_kv(transfer.cache_cpu, transfer.cache_cpu),
            mix_cache_kv(transfer.cache_cpu, recon_k_a_v.cache_cpu),
        ),
        "a_k_recon_v_keys_from_a": caches_equal(
            mix_cache_kv(transfer.cache_cpu, transfer.cache_cpu),
            mix_cache_kv(a_k_recon_v.cache_cpu, transfer.cache_cpu),
        ),
        "a_k_recon_v_values_from_recon": caches_equal(
            mix_cache_kv(reconstruct.cache_cpu, reconstruct.cache_cpu),
            mix_cache_kv(reconstruct.cache_cpu, a_k_recon_v.cache_cpu),
        ),
    }


def _run_one(runner, ex, extra_a, spans, pack, composed, cond, args, *, bundle, wall):
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
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = int(composed["n"])
    rec["n_result_kept"] = len(spans["result_pos"])
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["n_added"] = int(composed["n_added"])
    rec["n_full"] = int(spans["n_full"])
    rec["selected_text"] = spans["result_text"]
    rec["selected_token_ids"] = list(composed["ids"])
    rec["result_token_ids"] = list(pack["result_ids"])
    rec["layout"] = LAYOUT
    rec["layout_separator"] = True
    rec["layout_turn_close"] = False
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
    rec["message_bytes"] = int(bundle.full_relay_bytes) + rec["framing_bytes"]
    rec["cache_source"] = spec["cache_source"]
    rec["result_fallback"] = bool(spans.get("result_fallback"))
    rec["shared_across_heads"] = True
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["n_kv_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["n_prompt_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["plan_prompt"] = b_reason_kv_instruct_prompt(ex)
    rec["payload_note"] = PAYLOAD_NOTE
    rec["process_time_s"] = _process_time_s(rec)
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    _annotate_plan(rec, ex)
    return rec


def _rebuild_state(runner, rec, src, decode, sep_ids, close_ids):
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
    mixed = _mixed_pack(runner, transfer, reconstruct)
    return {
        "bundle": a_bundle,
        "meta": rec,
        "rebuild_time_s": rebuild_s,
        "spans": spans,
        "pack": pack,
        "composed": composed,
        "info": info,
        "times": {"transfer": trans_s, "reconstruct": recon_s, "recon_k_a_v": trans_s + recon_s, "a_k_recon_v": trans_s + recon_s},
        "mixed": mixed,
    }


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    wanted = _wanted_examples(examples)
    jsonl_path = out_root / "check_evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    construction = []
    print(f"KV-swap check on {len(wanted)} unique A sheets, layout={LAYOUT}", flush=True)
    for i, ex in enumerate(wanted):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state = _rebuild_state(runner, rec, src, decode, sep_ids, close_ids)
        mixed = state["mixed"]
        composed = state["composed"]
        ok = (
            mixed["identity_transfer_eq"]
            and mixed["identity_reconstruct_eq"]
            and mixed["recon_k_a_v_keys_from_recon"]
            and mixed["recon_k_a_v_values_from_a"]
            and mixed["a_k_recon_v_keys_from_a"]
            and mixed["a_k_recon_v_values_from_recon"]
            and int(mixed["transfer"].pos_cursor_cpu.view(-1)[0].item()) == composed["n"]
            and int(mixed["reconstruct"].pos_cursor_cpu.view(-1)[0].item()) == composed["n"]
            and n_kv_positions(mixed["recon_k_a_v"].cache_cpu) == composed["n"]
            and n_kv_positions(mixed["a_k_recon_v"].cache_cpu) == composed["n"]
            and composed["separator"]
            and not composed["turn_close"]
            and composed["n_added"] == 1
        )
        construction.append(
            {
                "a_state_key": key,
                "ok": ok,
                "n": composed["n"],
                "n_added": composed["n_added"],
                "identity_transfer_eq": mixed["identity_transfer_eq"],
                "identity_reconstruct_eq": mixed["identity_reconstruct_eq"],
                "recon_k_a_v_keys_from_recon": mixed["recon_k_a_v_keys_from_recon"],
                "recon_k_a_v_values_from_a": mixed["recon_k_a_v_values_from_a"],
                "a_k_recon_v_keys_from_a": mixed["a_k_recon_v_keys_from_a"],
                "a_k_recon_v_values_from_recon": mixed["a_k_recon_v_values_from_recon"],
                "caches_equal_pair": caches_equal(mixed["transfer"].cache_cpu, mixed["reconstruct"].cache_cpu),
            }
        )
        extra_a = _a_extra(state)
        extra_a["rebuild_time_s"] = float(state["rebuild_time_s"])
        extra_a["ids_source"] = rec.get("ids_source")
        remaining = [c for c in CONDS if (ex.example_id, c) not in done]
        reset_peak_memory()
        for cond in remaining:
            rec_out = _run_one(
                runner,
                ex,
                extra_a,
                state["spans"],
                state["pack"],
                composed,
                cond,
                args,
                bundle=runner.clone_sender(mixed[cond]),
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
        print(
            f"[{i + 1}/{len(wanted)}] {key} n={composed['n']} "
            f"idT={mixed['identity_transfer_eq']} idR={mixed['identity_reconstruct_eq']} ok={ok}",
            flush=True,
        )

    all_ok = bool(construction) and all(item["ok"] for item in construction)
    summary = {
        "n": len(construction),
        "layout": LAYOUT,
        "all_passed": all_ok,
        "construction": construction,
        "peek_receiver": {cond: _cond_block([r for r in rows if r["condition"] == cond]) for cond in CONDS},
        "lead": (
            "Identity mixes must recover transfer and reconstruct caches. Token IDs, "
            "lengths, and cursors stay matched. Cache equality of transfer vs "
            "reconstruct is not required."
        ),
    }
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("n", "all_passed", "layout")}, indent=2))
    print(json.dumps(summary.get("peek_receiver"), indent=2)[:8000])
    return 0 if all_ok else 1


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
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in CONDS)}
    print(f"KV-swap compare layout={LAYOUT} rebuilding {len(needed)} unique A caches", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        states[key] = _rebuild_state(runner, rec, src, decode, sep_ids, close_ids)
        mixed = states[key]["mixed"]
        if not mixed["identity_transfer_eq"] or not mixed["identity_reconstruct_eq"]:
            raise RuntimeError(f"Identity mix failed for {key}.")
        print(f"  rebuilt {key} n={states[key]['composed']['n']}", flush=True)
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
                f"elig={rec_out.get('plan_all_status_correct')} parse={rec_out.get('parse_ok')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    summary = _summarize(rows, check=check)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:12000])
    print(json.dumps(summary.get("family_clustered_eligibility"), indent=2))
    print(json.dumps(summary.get("qualifies_vs_reconstruct"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="plan4_dev key/value mix diagnostic on frozen sep compact KV."
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
        print("KV-swap uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_kvswap" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_kvswap.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "check_summary.json",
            "check_evaluations.jsonl",
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
        print("Check failed; stopping before the 90-case comparison.")
        return rc
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
