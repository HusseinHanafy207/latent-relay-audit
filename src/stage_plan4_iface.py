"""Interface checks on plan4_dev: prefix logits, then position x framing x sink.

plan4_hold stays out of tuning. A is written once per unique sheet. Conditions
branch from independent copies of that cache. No latent rollout, no learned
codec, no prompt retuning except one predeclared framing sentence.
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
from src.parse_plan4 import (
    RESULT_LINE,
    handoff_from_written,
    score_sender_message,
    written_working_metrics,
)
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    PLAN4_KVDIAG_FAMILIES,
    SINK_INITIAL,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    FRAMING_INSTRUCTION,
    a_local_task_prompt,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
    ordinary_sender_prompt,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.span_select import (
    decode_positions,
    line_spans,
    positions_from_spans,
    result_spans,
    with_initial_positions,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta, _run_text_b
from src.stage_plan4_hold import _cond_block, _plan_fields
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist, _process_time_s
from src.stage_plan4_span import _decode_fn
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

PREFIX_LEN = 100
CONTINUATION_LEN = 8
LABEL_MARGIN = 5.0

COMPACT = (
    "orig_base_result",
    "orig_base_sink",
    "orig_instr_result",
    "orig_instr_sink",
    "cons_base_result",
    "cons_base_sink",
    "cons_instr_result",
    "cons_instr_sink",
)
DIAG = ("orig_base_four",)
REFS = ("kv", "text_result")
CONDS = REFS + COMPACT + DIAG

SPEC = {
    "kv": {"kind": "full", "layout": "original", "framing": "baseline", "positions": "all"},
    "text_result": {"kind": "text", "layout": None, "framing": "baseline", "positions": "result"},
    "orig_base_result": {"kind": "kv", "layout": "original", "framing": "baseline", "positions": "result"},
    "orig_base_sink": {"kind": "kv", "layout": "original", "framing": "baseline", "positions": "result_sink"},
    "orig_instr_result": {"kind": "kv", "layout": "original", "framing": "instruct", "positions": "result"},
    "orig_instr_sink": {"kind": "kv", "layout": "original", "framing": "instruct", "positions": "result_sink"},
    "cons_base_result": {"kind": "kv", "layout": "consecutive", "framing": "baseline", "positions": "result"},
    "cons_base_sink": {"kind": "kv", "layout": "consecutive", "framing": "baseline", "positions": "result_sink"},
    "cons_instr_result": {"kind": "kv", "layout": "consecutive", "framing": "instruct", "positions": "result"},
    "cons_instr_sink": {"kind": "kv", "layout": "consecutive", "framing": "instruct", "positions": "result_sink"},
    "orig_base_four": {"kind": "kv", "layout": "original", "framing": "baseline", "positions": "sink"},
}

PAIRED = (
    ("orig_sink_vs_result", "orig_base_result", "orig_base_sink"),
    ("cons_sink_vs_result", "cons_base_result", "cons_base_sink"),
    ("instr_vs_base_orig_sink", "orig_base_sink", "orig_instr_sink"),
    ("cons_vs_orig_sink", "orig_base_sink", "cons_base_sink"),
    ("text_vs_orig_sink", "orig_base_sink", "text_result"),
    ("kv_vs_text", "kv", "text_result"),
)

PRESERVED = (
    "results_stage_plan0",
    "results_stage_plan0_corrected",
    "results_stage_plan0_diag",
    "results_stage_plan0_e2e",
    "results_stage_study1",
    "results_stage_plan4_cal",
    "results_stage_plan4_diag",
    "results_stage_plan4_cachecheck",
    "results_stage_plan4_bprompt",
    "results_stage_plan4_chain",
    "results_stage_plan4_comm",
    "results_stage_plan4_compress",
    "results_stage_plan4_probe",
    "results_stage_plan4_budget",
    "results_stage_plan4_fresh",
    "results_stage_plan4_span",
    "results_stage_plan4_kvdiag",
    "results_stage_plan4_initctrl",
    "results_stage_plan4_stable",
    "results_stage_plan4_hold",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_iface" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_iface")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def candidate_qualifies(block: dict, kv_block: dict, text_block: dict) -> bool:
    n = int(block.get("n") or 0)
    if n == 0 or int(kv_block.get("n") or 0) == 0 or int(text_block.get("n") or 0) == 0:
        return False
    if int(block.get("parse_ok") or 0) != n:
        return False
    if int(block.get("truncated") or 0) != 0:
        return False
    cand = 100.0 * int(block.get("plan_all_status_correct") or 0) / n
    kv = 100.0 * int(kv_block.get("plan_all_status_correct") or 0) / int(kv_block["n"])
    text = 100.0 * int(text_block.get("plan_all_status_correct") or 0) / int(text_block["n"])
    return cand + 1e-9 >= kv - LABEL_MARGIN and cand + 1e-9 >= text - LABEL_MARGIN


def choose_candidate(receiver: dict) -> dict:
    kv_block = receiver.get("kv") or {}
    text_block = receiver.get("text_result") or {}
    ranked = []
    for name in COMPACT:
        block = receiver.get(name) or {}
        ok = candidate_qualifies(block, kv_block, text_block)
        ranked.append(
            {
                "condition": name,
                "qualifies": ok,
                "mean_message_bytes": block.get("mean_message_bytes"),
                "mean_process_time_s": block.get("mean_process_time_s"),
                "plan_all_status_correct": block.get("plan_all_status_correct"),
                "parse_ok": block.get("parse_ok"),
                "truncated": block.get("truncated"),
            }
        )
    winners = [row for row in ranked if row["qualifies"]]
    winners.sort(
        key=lambda row: (
            float(row["mean_message_bytes"] or 1e18),
            float(row["mean_process_time_s"] or 1e18),
        )
    )
    return {
        "margin_points": LABEL_MARGIN,
        "candidates": ranked,
        "selected": None if not winners else winners[0]["condition"],
        "note": (
            "A qualifying candidate needs parseable untruncated outputs in every "
            "case and all-label accuracy within five points of full KV and result text. "
            "Smallest message wins; handoff time breaks ties. This selects a confirmation "
            "candidate; it does not itself establish recovery."
        ),
    }


def _prefix_cut(n_kv: int) -> tuple[int, int]:
    k = CONTINUATION_LEN
    cut = min(PREFIX_LEN, max(1, n_kv - k))
    if cut + k > n_kv:
        raise RuntimeError(f"Cache length {n_kv} is too short for a prefix check.")
    return cut, k


def _inspect_tokens(full_ids: list[int], tokenizer) -> dict:
    n = len(full_ids)
    first4 = full_ids[: min(SINK_INITIAL, n)]
    last8 = full_ids[max(0, n - 8) :]
    eos = tokenizer.eos_token_id
    return {
        "n_ids": n,
        "first4_ids": first4,
        "first4_text": tokenizer.decode(first4, skip_special_tokens=False),
        "last8_ids": last8,
        "last8_text": tokenizer.decode(last8, skip_special_tokens=False),
        "eos_token_id": eos,
        "eos_in_cache": bool(eos is not None and int(eos) in full_ids),
        "last_id_is_eos": bool(eos is not None and n and int(full_ids[-1]) == int(eos)),
    }


def _framing_prompts(ex, framing: str):
    if framing == "instruct":
        return b_reason_kv_instruct_prompt(ex), b_answer_kv_instruct_prompt
    return None, None


def _summarize_compare(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("kv", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "unit_note": (
            "plan4_dev only: 30 families with three versions each. "
            "plan4_hold is unused. Same written A on every branch. "
            "2 position conventions x 2 framing conditions x with/without the first four "
            "positions, plus full KV, result text, and four-only."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
        "by_difficulty": {},
        "paired_receiver": {name: _paired(grouped_plain, a, b) for name, a, b in PAIRED},
        "family_clustered_delta": {
            **{name: _family_delta(rows, a, b) for name, a, b in PAIRED},
            "orig_sink_vs_result_eligibility": _family_delta(
                rows, "orig_base_result", "orig_base_sink", key="plan_all_status_correct"
            ),
            "cons_vs_orig_sink_eligibility": _family_delta(
                rows, "orig_base_sink", "cons_base_sink", key="plan_all_status_correct"
            ),
            "instr_vs_base_orig_sink_eligibility": _family_delta(
                rows, "orig_base_sink", "orig_instr_sink", key="plan_all_status_correct"
            ),
            "text_vs_orig_sink_eligibility": _family_delta(
                rows, "orig_base_sink", "text_result", key="plan_all_status_correct"
            ),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "decision": choose_candidate(receiver),
        "framing_change": FRAMING_INSTRUCTION,
        "lead": (
            "Parseability, truncation, all-four eligibility, final-choice accuracy, "
            "and correct-choice-with-wrong-labels are reported separately. "
            "Malformed outputs stay in the denominator."
        ),
        "timing_note": (
            "mean_process_time_s includes written A, gather/re-rotation, and B. "
            "Network is not measured."
        ),
        "after_this_run": (
            "If a compact candidate qualifies, freeze it for confirmation and cost. "
            "If none qualifies, proceed to supporting context. Do not load plan4_hold."
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


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _prepare_runner(args, out_root: Path, extra: dict) -> tuple:
    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-iface",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "framing_change": FRAMING_INSTRUCTION,
            "sink_initial": SINK_INITIAL,
            "prefix_len": PREFIX_LEN,
            "continuation_len": CONTINUATION_LEN,
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


def _ensure_a_factory(runner, args, examples, artifacts, artifacts_path, states, decode, tokenizer):
    by_id = {ex.example_id: ex for ex in examples}

    def ensure_a(ex):
        key = a_state_key(ex)
        if key in states:
            return states[key]
        src_ex = by_id.get(key) or next(item for item in examples if a_state_key(item) == key)
        ordinary = ordinary_sender_prompt(a_local_task_prompt(src_ex), written_result_handoff())
        if prompt_contains_b_private(ordinary, src_ex):
            raise RuntimeError("A prompt leaked B's prices or budget.")
        if prompt_contains_solver_labels(ordinary, src_ex):
            raise RuntimeError("A prompt leaked solver labels.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text, n_new, truncated, gen_s, bundle = runner.generate_text_keep_cache(
            ordinary, max_new_tokens=args.written_max_new, source="written"
        )
        if int(bundle.latent_steps) != 0:
            raise RuntimeError("Written cache must have zero latent steps.")
        if bundle.full_ids_cpu is None:
            raise RuntimeError("Written cache is missing aligned token ids.")
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
        if int(bundle.full_ids_cpu.numel()) != n_full:
            raise RuntimeError("Aligned token ids do not match cache length.")
        token_align = dict(bundle.token_align or {})
        inspect = _inspect_tokens(bundle.full_ids_cpu.tolist(), tokenizer)
        rec = {
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "variant": src_ex.variant,
            "difficulty": src_ex.difficulty,
            "a_state_key": key,
            "ordinary_prompt": ordinary,
            "written": text,
            "written_n_new": n_new,
            "written_truncated": truncated,
            "written_s": gen_s,
            "handoff": handoff_from_written(text),
            "n_kv_positions": n_full,
            "n_prompt_positions": n_prompt,
            "full_relay_bytes": int(bundle.full_relay_bytes),
            "latent_steps": 0,
            "ids_source": bundle.ids_source,
            "token_align": token_align,
            "full_ids": bundle.full_ids_cpu.tolist(),
            "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
            "token_inspect": inspect,
            "written_score": score_sender_message(
                text, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_working": written_working_metrics(text),
            "a_error_recorded": not bool(
                score_sender_message(text, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)).get(
                    "all_status_correct"
                )
            ),
        }
        artifacts[key] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        states[key] = {"bundle": bundle, "meta": rec}
        return states[key]

    def ensure_spans(pack: dict, ex) -> None:
        if pack.get("result_pos") is not None:
            return
        bundle = pack["bundle"]
        full_ids = bundle.full_ids_cpu.tolist()
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
        written = line_spans(full_ids, start=n_prompt, end=n_full, decode=decode)
        selected_result = result_spans(written)
        fallback = False
        if not selected_result:
            selected_result = written[-4:] if len(written) >= 4 else list(written)
            fallback = True
        result_pos = positions_from_spans(selected_result)
        if not result_pos:
            raise RuntimeError(f"No complete spans in the written region for {ex.example_id}.")
        sink_only = list(range(min(SINK_INITIAL, n_full)))
        sink_pos, extra = with_initial_positions(result_pos, n_full=n_full, n_initial=SINK_INITIAL)
        pack["result_spans"] = selected_result
        pack["result_pos"] = result_pos
        pack["result_text"] = decode_positions(full_ids, result_pos, decode)
        pack["result_fallback"] = fallback
        pack["result_spans_from_loop_ids"] = (
            bool(RESULT_LINE.search(pack["result_text"] or "")) if pack["meta"].get("ids_source") == "decode_loop" else None
        )
        pack["sink_only_pos"] = sink_only
        pack["sink_pos"] = sink_pos
        pack["sink_extra"] = extra
        pack["sink_text"] = decode_positions(full_ids, sink_only, decode)
        pack["n_full"] = n_full
        pack["all_pos"] = list(range(n_full))
        pack["token_inspect"] = pack["meta"].get("token_inspect") or _inspect_tokens(full_ids, tokenizer)

    return ensure_a, ensure_spans


def _run_prefix(args, out_root, kaggle_dest, examples, runner, states: dict) -> int:
    jsonl_path = out_root / "prefix_evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {row["a_state_key"] for row in rows}
    tokenizer = runner.wrapper.tokenizer
    decode = _decode_fn(runner)
    artifacts_path = out_root / "a_artifacts.jsonl"
    artifacts = _load_messages(artifacts_path)
    ensure_a, _ensure_spans = _ensure_a_factory(
        runner, args, examples, artifacts, artifacts_path, states, decode, tokenizer
    )
    wanted = []
    seen = set()
    for ex in examples:
        if ex.family_id in PLAN4_KVDIAG_FAMILIES and a_state_key(ex) not in seen:
            wanted.append(ex)
            seen.add(a_state_key(ex))
    print(f"Prefix logit check on {len(wanted)} unique A sheets", flush=True)
    for i, ex in enumerate(wanted):
        key = a_state_key(ex)
        if key in done:
            print(f"[{i + 1}/{len(wanted)}] {key} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(wanted)}] {key}", flush=True)
        reset_peak_memory()
        pack = ensure_a(ex)
        bundle = pack["bundle"]
        n_full = n_kv_positions(bundle.cache_cpu)
        cut, k = _prefix_cut(n_full)
        rec = runner.compare_prefix_logits(
            bundle.full_ids_cpu,
            prefix_len=cut,
            continuation_len=k,
            completed_cache=bundle.cache_cpu,
            original_pos_cursor=int(bundle.pos_cursor_cpu.view(-1)[0].item()),
        )
        rec.update(
            {
                "example_id": ex.example_id,
                "family_id": ex.family_id,
                "variant": ex.variant,
                "difficulty": ex.difficulty,
                "a_state_key": key,
                "n_kv_positions": n_full,
                "ids_source": pack["meta"].get("ids_source"),
                "token_inspect": pack["meta"].get("token_inspect"),
                "a_error_recorded": pack["meta"].get("a_error_recorded"),
                "peak_gpu_gb": peak_gpu_memory_gb(),
                "condition": "prefix_check",
            }
        )
        rows.append(rec)
        done.add(key)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"  passed={rec.get('passed')} gather_max={rec['completed_gather_vs_reencode']['max_abs']:.4g} "
            f"ordinary_max={rec['ordinary_cached_vs_uncached']['max_abs']:.4g} "
            f"repeat_max={rec['repeat']['max_abs']:.4g}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    n_pass = sum(bool(r.get("passed")) for r in rows)
    summary = {
        "n": len(rows),
        "n_passed": n_pass,
        "all_passed": bool(rows) and n_pass == len(rows),
        "prefix_len": PREFIX_LEN,
        "continuation_len": CONTINUATION_LEN,
        "families": list(PLAN4_KVDIAG_FAMILIES),
        "mean_gather_max_abs": (
            sum(float(r["completed_gather_vs_reencode"]["max_abs"]) for r in rows) / len(rows) if rows else None
        ),
        "mean_ordinary_max_abs": (
            sum(float(r["ordinary_cached_vs_uncached"]["max_abs"]) for r in rows) / len(rows) if rows else None
        ),
        "mean_repeat_max_abs": (
            sum(float(r["repeat"]["max_abs"]) for r in rows) / len(rows) if rows else None
        ),
        "rows": [
            {
                "a_state_key": r["a_state_key"],
                "passed": r.get("passed"),
                "gather": r.get("completed_gather_vs_reencode"),
                "ordinary": r.get("ordinary_cached_vs_uncached"),
                "repeat": r.get("repeat"),
            }
            for r in rows
        ],
        "lead": (
            "If completed-cache prefix reuse exceeds ordinary cached-vs-uncached "
            "and repeat noise, fix the implementation before interpreting selected-cache "
            "results. Passing does not prove that removing middle positions is valid."
        ),
        "peak_gpu_gb_max_over_run": max((r.get("peak_gpu_gb") or 0) for r in rows) if rows else None,
    }
    _write_json(out_root / "prefix_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("n", "n_passed", "all_passed")}, indent=2))
    return 0 if summary["all_passed"] else 1


def _run_compare(args, out_root, kaggle_dest, examples, runner, states: dict) -> int:
    prefix_summary_path = out_root / "prefix_summary.json"
    if prefix_summary_path.is_file():
        prefix_summary = json.loads(prefix_summary_path.read_text(encoding="utf-8"))
        if not prefix_summary.get("all_passed") and not args.force_compare:
            print("Prefix check did not pass. Fix that before the factorial comparison, or pass --force-compare.")
            return 2
    elif not args.force_compare:
        print("Run --phase prefix first. Pass --force-compare only to skip that gate.")
        return 2

    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    tokenizer = runner.wrapper.tokenizer
    decode = _decode_fn(runner)
    artifacts_path = out_root / "a_artifacts.jsonl"
    artifacts = _load_messages(artifacts_path)
    ensure_a, ensure_spans = _ensure_a_factory(
        runner, args, examples, artifacts, artifacts_path, states, decode, tokenizer
    )
    by_id = {ex.example_id: ex for ex in examples}
    needed = set()
    for ex in examples:
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if remaining:
            needed.add(a_state_key(ex))
    print(f"Encoding written A caches for {len(needed)} unique operational sheets", flush=True)
    for eid in sorted(needed):
        src = by_id.get(eid) or next(ex for ex in examples if a_state_key(ex) == eid)
        ensure_a(src)
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        pack = ensure_a(ex)
        extra_a = _a_extra(pack)
        ensure_spans(pack, ex)
        token_align = dict(pack["meta"].get("token_align") or {})
        inspect = pack.get("token_inspect") or pack["meta"].get("token_inspect")

        for cond in remaining:
            spec = SPEC[cond]
            plan_prompt, make_answer = _framing_prompts(ex, spec["framing"] or "baseline")
            if spec["kind"] == "text":
                rec = _run_text_b(
                    runner,
                    ex,
                    message=pack["result_text"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    allow_a_source=True,
                )
                rec.update(extra_a)
                rec["n_kept"] = len(pack["result_pos"])
                rec["n_kv_positions"] = 0
                rec["n_result_kept"] = len(pack["result_pos"])
                rec["n_sink_kept"] = 0
                rec["positions"] = list(pack["result_pos"])
                rec["assigned_positions"] = list(pack["result_pos"])
                rec["selected_text"] = pack["result_text"]
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
                rec["layout"] = None
                rec["framing"] = "baseline"
                rec["framing_instruction"] = None
                rec["message_bytes"] = len(pack["result_text"].encode("utf-8"))
            elif spec["kind"] == "full":
                rec = _run_kv_b(
                    runner,
                    ex,
                    runner.clone_sender(pack["bundle"]),
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["positions"] = pack["all_pos"]
                rec["assigned_positions"] = pack["all_pos"]
                rec["n_kept"] = int(rec.get("n_kv_positions") or 0)
                rec["n_result_kept"] = 0
                rec["n_sink_kept"] = 0
                rec["selected_text"] = pack["meta"].get("written")
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
                rec["layout"] = "original"
                rec["framing"] = "baseline"
                rec["framing_instruction"] = None
            else:
                if spec["positions"] == "result":
                    positions = pack["result_pos"]
                    n_result = len(positions)
                    n_sink = 0
                    selected_text = pack["result_text"]
                elif spec["positions"] == "result_sink":
                    positions = pack["sink_pos"]
                    n_result = len(pack["result_pos"])
                    n_sink = len(pack["sink_extra"])
                    selected_text = pack["result_text"]
                else:
                    positions = pack["sink_only_pos"]
                    n_result = 0
                    n_sink = len(positions)
                    selected_text = pack["sink_text"]
                cloned = runner.clone_sender(pack["bundle"])
                compressed, wall, info = runner.keep_shared_positions(
                    cloned, positions, layout=spec["layout"]
                )
                rec = _run_kv_b(
                    runner,
                    ex,
                    compressed,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    plan_prompt=plan_prompt,
                    make_answer_prompt=make_answer,
                )
                rec.update(extra_a)
                rec["selection_time_s"] = float(wall)
                rec["compression_time_s"] = float(wall)
                rec.update(info)
                rec["n_kept"] = int(info["n_kept"])
                rec["n_result_kept"] = n_result
                rec["n_sink_kept"] = n_sink
                rec["n_sink_extra"] = n_sink if spec["positions"] == "result_sink" else 0
                rec["positions"] = list(info.get("positions") or positions)
                rec["assigned_positions"] = list(info.get("assigned_positions") or [])
                rec["selected_text"] = selected_text
                rec["layout"] = spec["layout"]
                rec["framing"] = spec["framing"]
                rec["framing_instruction"] = FRAMING_INSTRUCTION if spec["framing"] == "instruct" else None
                extra_framing = (
                    len(FRAMING_INSTRUCTION.encode("utf-8")) if spec["framing"] == "instruct" else 0
                )
                rec["framing_bytes"] = extra_framing
                rec["message_bytes"] = int(compressed.full_relay_bytes) + extra_framing

            rec["condition"] = cond
            rec["plan_prompt"] = plan_prompt
            rec["result_fallback"] = bool(pack.get("result_fallback"))
            rec["result_spans_from_loop_ids"] = pack.get("result_spans_from_loop_ids")
            rec["n_full"] = int(pack["n_full"])
            rec["ids_source"] = pack["meta"].get("ids_source")
            rec["loop_eq_retokenize"] = token_align.get("loop_eq_retokenize")
            rec["loop_len_eq_n_kv"] = token_align.get("loop_len_eq_n_kv")
            rec["eos_counted_not_cached"] = token_align.get("eos_counted_not_cached")
            rec["token_inspect"] = inspect
            rec["shared_across_heads"] = True
            rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            rec["process_time_s"] = _process_time_s(rec)
            _plan_fields(rec, ex)
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec.get('pred_carrier')} correct={rec.get('correct')} "
                f"elig={rec.get('plan_all_status_correct')} parse={rec.get('parse_ok')} "
                f"trunc={rec.get('truncated')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    summary = _summarize_compare(rows)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary.get("decision"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="plan4_dev prefix check then position/framing factorial.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--phase", choices=("prefix", "compare", "all"), default="prefix")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Interface checks use plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_iface" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_iface.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in ("evaluations.jsonl", "prefix_evaluations.jsonl", "summary.json", "prefix_summary.json", "a_artifacts.jsonl", "manifest.json"):
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

    policy, runner, revision = _prepare_runner(
        args,
        out_root,
        extra={"phase": args.phase, "conditions": list(CONDS)},
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")

    states: dict = {}
    rc = 0
    if args.phase in {"prefix", "all"}:
        rc = _run_prefix(args, out_root, kaggle_dest, examples, runner, states)
        if rc != 0 and args.phase == "all" and not args.force_compare:
            print("Prefix check failed; stopping before the factorial comparison.")
            return rc
    if args.phase in {"compare", "all"}:
        rc = _run_compare(args, out_root, kaggle_dest, examples, runner, states)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
