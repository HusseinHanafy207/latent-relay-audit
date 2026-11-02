"""Matched reconstruction reference for frozen cons_instr_sink on plan4_dev.

Transferred KV gathers A's cached entries and applies consecutive RoPE.
Reconstructed KV prefills the exact same compact token IDs afresh.
Both use the same receiver prompts, framing, positions, masks, and decoding.
Ordinary result text is an additional reference with a different message path.

A is not regenerated. plan4_hold stays out. Do not resume iface or support.
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

from src.cache_utils import n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse_plan4 import RESULT_LINE
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    PLAN4_KVDIAG_FAMILIES,
    SINK_INITIAL,
    a_state_key,
    assert_plan4_invariants,
    load_plan4_jsonl,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
)
from src.rope_rerotate import cache_key_delta, rope_op_check
from src.span_select import (
    caches_equal,
    decode_positions,
    line_spans,
    positions_from_spans,
    result_spans,
    with_initial_positions,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta, _run_text_b
from src.stage_plan4_hold import _plan_fields
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

CONDS = ("transfer", "reconstruct", "text_result")
SPEC = {
    "transfer": {
        "kind": "kv",
        "layout": "consecutive",
        "framing": "instruct",
        "positions": "result_sink",
        "cache_source": "transfer_gather_rerotate",
    },
    "reconstruct": {
        "kind": "kv",
        "layout": "consecutive",
        "framing": "instruct",
        "positions": "result_sink",
        "cache_source": "reconstruct_prefill",
    },
    "text_result": {
        "kind": "text",
        "layout": None,
        "framing": "baseline",
        "positions": "result",
        "cache_source": "text",
    },
}
PAIRED = (
    ("reconstruct_vs_transfer", "transfer", "reconstruct"),
    ("text_vs_transfer", "transfer", "text_result"),
    ("text_vs_reconstruct", "reconstruct", "text_result"),
)
N_BOOT = 2000
BOOT_SEED = 2026
PRESERVED = (
    "results_stage_plan4_iface",
    "results_stage_plan4_support",
    "results_stage_plan4_hold",
    "results_stage_plan4_stable",
    "results_stage_plan4_fresh",
)
INTERPRETATION_GUIDE = (
    "Reconstruction succeeds while transferred states fail: this supports a problem "
    "with reusing representations computed in A's original context. "
    "Both fail, but ordinary result text succeeds: the compact message layout or "
    "boundary remains a plausible problem. Resolve that before training a repair. "
    "Both succeed: investigate what differs from the earlier failing path."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_recon" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_recon")
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


def _artifact_candidates(out_root: Path, explicit: str | None = None) -> list[Path]:
    paths = []
    if explicit:
        paths.append(Path(explicit))
    paths.extend(
        [
            out_root / "a_artifacts.jsonl",
            REPO_ROOT / "results_stage_plan4_iface" / "a_artifacts.jsonl",
            REPO_ROOT / "results_stage_plan4_support" / "a_artifacts.jsonl",
            RESULTS_DIR / "stage_plan4_iface" / "run" / "a_artifacts.jsonl",
            Path("/kaggle/working/results_stage_plan4_iface/a_artifacts.jsonl"),
        ]
    )
    return paths


def _copy_frozen_artifacts(out_root: Path, *, explicit: str | None = None) -> Path:
    dest = out_root / "a_artifacts.jsonl"
    if dest.is_file():
        return dest
    for src in _artifact_candidates(out_root, explicit):
        if src.is_file() and src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
            print(f"Copied frozen A artifacts from {src}")
            return dest
    raise FileNotFoundError(
        "Need results_stage_plan4_iface/a_artifacts.jsonl with recorded token IDs. "
        "Do not regenerate A's answers."
    )


def _compact_spans(full_ids: list[int], *, n_prompt: int, n_full: int, decode) -> dict:
    written = line_spans(full_ids, start=n_prompt, end=n_full, decode=decode)
    selected = result_spans(written)
    fallback = False
    if not selected:
        selected = written[-4:] if len(written) >= 4 else list(written)
        fallback = True
    result_pos = positions_from_spans(selected)
    if not result_pos:
        raise RuntimeError("No complete spans in the written region.")
    sink_pos, extra = with_initial_positions(result_pos, n_full=n_full, n_initial=SINK_INITIAL)
    selected_ids = [int(full_ids[p]) for p in sink_pos]
    carriers = []
    for span in selected:
        match = RESULT_LINE.search(span.text or "")
        if match:
            carriers.append(match.group("carrier").upper())
    if not fallback and sorted(carriers) != ["P", "Q", "R", "S"]:
        raise RuntimeError(f"Result carriers {carriers} are incomplete.")
    return {
        "result_pos": result_pos,
        "sink_pos": sink_pos,
        "n_sink_extra": len(extra),
        "selected_ids": selected_ids,
        "result_text": decode_positions(full_ids, result_pos, decode),
        "n_full": n_full,
        "n_prompt": n_prompt,
        "result_fallback": fallback,
    }


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
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "n_evals_expected": 90 * len(CONDS),
        "baseline": "cons_instr_sink",
        "unit_note": (
            "plan4_dev only. Frozen cons_instr_sink token IDs. Transfer gathers A's "
            "cached states with consecutive RoPE. Reconstruct prefills those exact IDs. "
            "Both KV conditions use the instruct framing. Result text is a separate path."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
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
        "framing_change": FRAMING_INSTRUCTION,
        "check": None
        if check is None
        else {
            "all_passed": check.get("all_passed"),
            "n": check.get("n"),
            "rope_passed": check.get("rope_passed"),
        },
        "lead": (
            "Primary measure: all four eligibility labels. Compare transferred KV, "
            "reconstructed KV of the same compact IDs, and ordinary result text."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "cacheback_note": (
            "Reconstructed states follow CacheBack Appendix C.3: receiver caches "
            "are built by a fresh prefill of the selected token IDs."
        ),
        "after_this_run": "These are development results. Do not load plan4_hold.",
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
            "stage": "plan4-recon",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "baseline": "cons_instr_sink",
            "framing_change": FRAMING_INSTRUCTION,
            "sink_initial": SINK_INITIAL,
            "cacheback_note": (
                "Reconstructed states follow CacheBack Appendix C.3: receiver caches "
                "are built by a fresh prefill of the selected token IDs, not by "
                "evaluating direct transfer of selected KV entries."
            ),
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


def _rebuild_a(runner, rec, src_ex):
    ordinary = rec.get("ordinary_prompt") or ""
    if prompt_contains_b_private(ordinary, src_ex):
        raise RuntimeError("Recorded A prompt leaked B's prices or budget.")
    if prompt_contains_solver_labels(ordinary, src_ex):
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


def _align_compact_ids(bundle, selected_ids: list[int]) -> None:
    """Receiver forwards use cache, mask, and pos_cursor. Keep token fields matched too."""
    n_kept = len(selected_ids)
    ids_t = torch.tensor(selected_ids, dtype=torch.long).view(1, -1)
    bundle.input_ids_cpu = ids_t.contiguous().clone()
    bundle.prompt_mask_cpu = torch.ones((1, n_kept), dtype=torch.long)
    bundle.past_mask_cpu = torch.ones((1, n_kept), dtype=torch.long)
    bundle.full_ids_cpu = ids_t.view(-1).contiguous().clone()


def _matched_caches(runner, bundle, spans: dict):
    cloned = runner.clone_sender(bundle)
    transferred, wall, info = runner.keep_shared_positions(
        cloned, spans["sink_pos"], layout="consecutive"
    )
    selected_ids = list(info.get("selected_token_ids") or spans["selected_ids"])
    if selected_ids != list(spans["selected_ids"]):
        raise RuntimeError("Transfer selected token IDs do not match the frozen compact sequence.")
    _align_compact_ids(transferred, selected_ids)
    reconstructed, recon_s = runner.prefill_token_sequence(
        selected_ids, pos_cursor=len(selected_ids), source="reconstruct"
    )
    _align_compact_ids(reconstructed, selected_ids)
    if int(reconstructed.pos_cursor_cpu.view(-1)[0].item()) != int(info["pos_cursor"]):
        raise RuntimeError("Reconstruct pos_cursor does not match transfer.")
    if n_kv_positions(reconstructed.cache_cpu) != int(info["n_kept"]):
        raise RuntimeError("Reconstruct cache length does not match transfer.")
    if int(transferred.past_mask_cpu.size(1)) != int(reconstructed.past_mask_cpu.size(1)):
        raise RuntimeError("Receiver masks differ between transfer and reconstruct.")
    delta = cache_key_delta(transferred.cache_cpu, reconstructed.cache_cpu)
    return {
        "transfer": transferred,
        "reconstruct": reconstructed,
        "info": info,
        "selected_ids": selected_ids,
        "selection_time_s": float(wall),
        "reconstruct_time_s": float(recon_s),
        "key_delta": delta,
        "caches_equal": caches_equal(transferred.cache_cpu, reconstructed.cache_cpu),
    }


def _run_one(runner, ex, extra_a, spans, matched, cond, args):
    if cond == "text_result":
        rec = _run_text_b(
            runner,
            ex,
            message=spans["result_text"],
            plan_max_new=args.plan_max_new,
            answer_max_new=args.max_new_tokens,
            allow_a_source=True,
        )
        rec.update(extra_a)
        rec["n_kept"] = 0
        rec["n_kv_positions"] = 0
        rec["n_result_kept"] = len(spans["result_pos"])
        rec["n_sink_kept"] = 0
        rec["positions"] = list(spans["result_pos"])
        rec["assigned_positions"] = list(spans["result_pos"])
        rec["selected_text"] = spans["result_text"]
        rec["selected_token_ids"] = list(spans["selected_ids"])
        rec["selection_time_s"] = 0.0
        rec["layout"] = None
        rec["framing"] = "baseline"
        rec["framing_instruction"] = None
        rec["message_bytes"] = len(spans["result_text"].encode("utf-8"))
        rec["cache_source"] = "text"
        rec["result_fallback"] = bool(spans.get("result_fallback"))
        rec["plan_prompt"] = None
        return rec

    if cond == "transfer":
        bundle = runner.clone_sender(matched["transfer"])
        cache_source = "transfer_gather_rerotate"
        wall = float(matched["selection_time_s"])
        info = matched["info"]
    else:
        bundle = runner.clone_sender(matched["reconstruct"])
        cache_source = "reconstruct_prefill"
        wall = float(matched["reconstruct_time_s"])
        info = {
            "n_kept": n_kv_positions(bundle.cache_cpu),
            "positions": list(matched["info"].get("positions") or spans["sink_pos"]),
            "assigned_positions": list(range(n_kv_positions(bundle.cache_cpu))),
            "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
            "cache_finite": True,
            "layout": "consecutive",
            "selected_token_ids": list(matched["selected_ids"]),
        }
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
    rec.update(info)
    rec["selection_time_s"] = wall
    rec["compression_time_s"] = wall
    rec["n_kept"] = int(info.get("n_kept") or 0)
    rec["n_result_kept"] = len(spans["result_pos"])
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["selected_text"] = spans["result_text"]
    rec["selected_token_ids"] = list(matched["selected_ids"])
    rec["layout"] = "consecutive"
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
    rec["message_bytes"] = int(bundle.full_relay_bytes) + rec["framing_bytes"]
    rec["cache_source"] = cache_source
    rec["n_prompt_positions"] = int(info.get("n_kept") or 0)
    rec["result_fallback"] = bool(spans.get("result_fallback"))
    rec["reconstruct_time_s"] = float(matched["reconstruct_time_s"])
    rec["key_delta_vs_pair"] = matched["key_delta"]
    rec["caches_equal_pair"] = matched["caches_equal"]
    rec["plan_prompt"] = b_reason_kv_instruct_prompt(ex)
    return rec


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    by_id = {ex.example_id: ex for ex in examples}
    wanted = []
    seen = set()
    for ex in examples:
        if ex.family_id in PLAN4_KVDIAG_FAMILIES and a_state_key(ex) not in seen:
            wanted.append(ex)
            seen.add(a_state_key(ex))
    jsonl_path = out_root / "check_evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    states = {}
    rope_reports = []
    construction = []
    print(f"Matched reconstruction check on {len(wanted)} unique A sheets", flush=True)
    for i, ex in enumerate(wanted):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        if key not in states:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bundle, rebuild_s = _rebuild_a(runner, rec, src)
            spans = _compact_spans(
                rec["full_ids"],
                n_prompt=int(rec["n_prompt_positions"]),
                n_full=int(rec["n_kv_positions"]),
                decode=decode,
            )
            matched = _matched_caches(runner, bundle, spans)
            rope = rope_op_check(matched["transfer"].cache_cpu, runner.wrapper.model)
            rope_reports.append({"a_state_key": key, **{k: rope[k] for k in ("passed", "cap", "identity", "repeat", "roundtrip")}})
            construction.append(
                {
                    "a_state_key": key,
                    "n_kept": int(matched["info"]["n_kept"]),
                    "n_ids": len(matched["selected_ids"]),
                    "pos_cursor_transfer": int(matched["info"]["pos_cursor"]),
                    "pos_cursor_reconstruct": int(matched["reconstruct"].pos_cursor_cpu.view(-1)[0].item()),
                    "ids_match": matched["selected_ids"] == spans["selected_ids"],
                    "assigned": list(matched["info"].get("assigned_positions") or []),
                    "cache_finite_transfer": bool(matched["info"].get("cache_finite")),
                    "caches_equal": matched["caches_equal"],
                    "key_delta": matched["key_delta"],
                    "result_text": spans["result_text"],
                    "first8_ids": matched["selected_ids"][:8],
                    "rope_passed": rope["passed"],
                }
            )
            states[key] = {
                "bundle": bundle,
                "meta": rec,
                "rebuild_time_s": rebuild_s,
                "spans": spans,
                "matched": matched,
            }
            print(
                f"[{i + 1}/{len(wanted)}] {key} n_kept={matched['info']['n_kept']} "
                f"equal={matched['caches_equal']} key_max={matched['key_delta']['max_abs']:.4g} "
                f"rope={rope['passed']}",
                flush=True,
            )
        pack = states[key]
        extra_a = _a_extra(pack)
        extra_a["rebuild_time_s"] = float(pack["rebuild_time_s"])
        extra_a["cache_source"] = "prefill_recorded_ids"
        remaining = [c for c in CONDS if (ex.example_id, c) not in done]
        reset_peak_memory()
        for cond in remaining:
            rec_out = _run_one(runner, ex, extra_a, pack["spans"], pack["matched"], cond, args)
            rec_out["condition"] = cond
            rec_out["n_full"] = int(pack["spans"]["n_full"])
            rec_out["ids_source"] = pack["meta"].get("ids_source")
            rec_out["shared_across_heads"] = True
            rec_out["process_time_s"] = _process_time_s(rec_out)
            rec_out["peak_gpu_gb"] = peak_gpu_memory_gb()
            _plan_fields(rec_out, ex)
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

    rope_ok = all(item.get("passed") for item in rope_reports)
    construct_ok = all(
        item["ids_match"]
        and item["pos_cursor_transfer"] == item["pos_cursor_reconstruct"]
        and item["n_kept"] == item["n_ids"]
        and item["cache_finite_transfer"]
        and item["rope_passed"]
        for item in construction
    )
    summary = {
        "n": len(construction),
        "families": list(PLAN4_KVDIAG_FAMILIES),
        "all_passed": bool(construction) and rope_ok and construct_ok,
        "rope_passed": rope_ok,
        "construct_ok": construct_ok,
        "rope": rope_reports,
        "construction": construction,
        "peek_receiver": {cond: _cond_block([r for r in rows if r["condition"] == cond]) for cond in CONDS},
        "lead": (
            "Check construction and RoPE identity on a few unique A sheets, then a "
            "small B peek. Token IDs, lengths, and cursors must match. Cache equality "
            "is not required: transferred and reconstructed keys may differ."
        ),
    }
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("n", "all_passed", "rope_passed", "construct_ok")}, indent=2))
    print(json.dumps(summary.get("peek_receiver"), indent=2)[:8000])
    return 0 if summary["all_passed"] else 1


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    check_path = out_root / "check_summary.json"
    check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.is_file() else None
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first. Pass --force-compare only to skip that gate.")
        return 2
    decode = _decode_fn(runner)
    by_id = {ex.example_id: ex for ex in examples}
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in CONDS)}
    print(f"Rebuilding {len(needed)} unique A caches from recorded token IDs", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bundle, rebuild_s = _rebuild_a(runner, rec, src)
        spans = _compact_spans(
            rec["full_ids"],
            n_prompt=int(rec["n_prompt_positions"]),
            n_full=int(rec["n_kv_positions"]),
            decode=decode,
        )
        matched = _matched_caches(runner, bundle, spans)
        states[key] = {
            "bundle": bundle,
            "meta": rec,
            "rebuild_time_s": rebuild_s,
            "spans": spans,
            "matched": matched,
        }
        print(
            f"  rebuilt {key} n_kept={matched['info']['n_kept']} "
            f"key_max={matched['key_delta']['max_abs']:.4g}",
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
        pack = states[a_state_key(ex)]
        extra_a = _a_extra(pack)
        extra_a["rebuild_time_s"] = float(pack["rebuild_time_s"])
        for cond in remaining:
            rec = _run_one(runner, ex, extra_a, pack["spans"], pack["matched"], cond, args)
            rec["condition"] = cond
            rec["n_full"] = int(pack["spans"]["n_full"])
            rec["ids_source"] = pack["meta"].get("ids_source")
            rec["shared_across_heads"] = True
            rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            rec["process_time_s"] = _process_time_s(rec)
            rec["peak_gpu_gb"] = peak_gpu_memory_gb()
            _plan_fields(rec, ex)
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec.get('pred_carrier')} correct={rec.get('correct')} "
                f"elig={rec.get('plan_all_status_correct')} parse={rec.get('parse_ok')}",
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
        description="plan4_dev matched reconstruction vs transferred cons_instr_sink KV."
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
        print("Matched reconstruction uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_recon" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_recon.")
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

    policy, runner, revision = _prepare_runner(
        args, out_root, extra={"phase": args.phase, "conditions": list(CONDS)}
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    rc = 0
    if args.phase in {"check", "all"}:
        rc = _run_check(args, out_root, kaggle_dest, examples, runner, artifacts)
        if rc != 0 and args.phase == "all" and not args.force_compare:
            print("Check failed; stopping before the 90-case comparison.")
            return rc
        if args.phase == "check":
            return rc
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
