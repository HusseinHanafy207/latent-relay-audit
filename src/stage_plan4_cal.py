"""Sender calibration on the four-carrier task.

A produces intermediate results under: no extra rollout, written calculations,
and 40 latent steps then the same compact result request. Correct intermediates
are also given to B as a receiver diagnostic.

Development split only (plan4_dev). No compressor. No KV comparison. Do not
load plan_dev, hop_test, test.jsonl, or a plan_test file. Do not resume Study 1.
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

from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse_plan4 import (
    flags_from_plan_kind,
    label_plan4_answer,
    parse_plan4_answer,
    plan_extraction,
    score_sender_message,
    written_working_metrics,
)
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    assert_plan4_invariants,
    a_state_key,
    evaluate,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
    oracle_intermediate_message,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    b_answer_after_message_plan_prompt,
    b_reason_with_message_prompt,
    compact_result_handoff,
    prompt_contains_a_private,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, B_PLAN_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

A_CONDS = ("a_direct", "a_written", "a_latent40")
B_COND = "b_oracle"
CONDS = (*A_CONDS, B_COND)
A_COMPACT_MAX_NEW = 256
A_WRITTEN_MAX_NEW = 512
LATENT_STEPS_C40 = 40
PRESERVED = (
    "results_stage_plan0",
    "results_stage_plan0_corrected",
    "results_stage_plan0_diag",
    "results_stage_plan0_e2e",
    "results_stage_study1",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_cal" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_cal")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_core(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        src = out_root / "a_artifacts.jsonl"
        if src.is_file():
            (dest / "a_artifacts.jsonl").write_bytes(src.read_bytes())


def _process_time_s(row: dict) -> float:
    return float(
        (row.get("encode_c0_s") or 0.0)
        + (row.get("rollout_c40_s") or 0.0)
        + (row.get("sender_generation_time_s") or 0.0)
        + (row.get("plan_time_s") or 0.0)
        + (row.get("receiver_latency_s") or 0.0)
    )


def _a_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "all_four_correct": sum(bool(r.get("all_four_correct")) for r in items) / n,
        "all_q_correct": sum(bool(r.get("all_q_correct")) for r in items) / n,
        "all_status_correct": sum(bool(r.get("all_status_correct")) for r in items) / n,
        "mean_n_q_correct": sum(int(r.get("n_q_correct") or 0) for r in items) / n,
        "mean_n_status_correct": sum(int(r.get("n_status_correct") or 0) for r in items) / n,
        "malformed": sum(bool(r.get("malformed")) for r in items) / n,
        "truncated": sum(bool(r.get("truncated")) for r in items) / n,
        "mean_n_new_tokens": sum(int(r.get("n_new_tokens") or 0) for r in items) / n,
        "mean_message_bytes": sum(int(r.get("message_bytes") or 0) for r in items) / n,
        "mean_encode_c0_s": sum(r.get("encode_c0_s") or 0 for r in items) / n,
        "mean_rollout_c40_s": sum(r.get("rollout_c40_s") or 0 for r in items) / n,
        "mean_sender_generation_time_s": sum(r.get("sender_generation_time_s") or 0 for r in items) / n,
        "mean_process_time_s": sum(_process_time_s(r) for r in items) / n,
        "produced_working": sum(bool(r.get("produced_working")) for r in items) / n,
        "has_results_block": sum(bool(r.get("has_results_block")) for r in items) / n,
        "has_arithmetic": sum(bool(r.get("has_arithmetic")) for r in items) / n,
        "mean_n_working_lines": sum(int(r.get("n_working_lines") or 0) for r in items) / n,
        "all_four_correct_optional_q": sum(bool(r.get("all_four_correct_optional_q")) for r in items) / n,
        "mean_n_missing_q_equals": sum(int(r.get("n_missing_q_equals") or 0) for r in items) / n,
    }


def _b_block(items: list[dict]) -> dict:
    n = len(items)
    n_true = sum(bool(item.get("correct")) for item in items)
    return {
        "n": n,
        "correct": n_true,
        "accuracy": n_true / n if n else None,
        "follows_trap": sum(bool(item.get("follows_trap")) for item in items) / n if n else None,
        "parse_ok": sum(bool(item.get("parse_ok")) for item in items) / n if n else None,
        "truncated": sum(bool(item.get("truncated")) for item in items) / n if n else None,
        "answer_truncated": sum(bool(item.get("truncated")) for item in items) / n if n else None,
        "plan_truncated": sum(bool(item.get("plan_truncated")) for item in items) / n if n else None,
        "n_plan_truncated": sum(bool(item.get("plan_truncated")) for item in items),
        "feasible_affordable": (
            sum(bool(item.get("feasible_affordable")) for item in items) / n if n else None
        ),
        "mean_plan_time_s": (sum(item.get("plan_time_s") or 0 for item in items) / n if n else None),
        "mean_n_plan_tokens": (sum(item.get("n_plan_tokens") or 0 for item in items) / n if n else None),
        "mean_receiver_latency_s": (
            sum(item.get("receiver_latency_s") or 0 for item in items) / n if n else None
        ),
        "mean_process_time_s": (sum(_process_time_s(item) for item in items) / n if n else None),
        "plan_extraction_missing": sum(item.get("plan_extraction_status") == "missing" for item in items) / n
        if n
        else None,
        "plan_extraction_agrees": sum(item.get("plan_extraction_status") == "agrees" for item in items) / n
        if n
        else None,
        "plan_extraction_disagrees": sum(item.get("plan_extraction_status") == "disagrees" for item in items)
        / n
        if n
        else None,
        "n_plan_extracted": sum(item.get("plan_extraction_status") in {"agrees", "disagrees"} for item in items),
        "n_plan_agree_given_extracted": sum(item.get("plan_extraction_status") == "agrees" for item in items),
        "plan_named_cheapest": (
            sum(item.get("plan_named_cheapest") is not None for item in items) / n if n else None
        ),
    }


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    a_summary = {cond: _a_block(by_cond.get(cond, [])) for cond in A_CONDS}
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": 30,
        "unit_note": (
            "30 families with three versions each, not 90 independent scenarios. "
            "Base and B-change reuse the same A states. Keep variants together later."
        ),
        "sender": a_summary,
        "receiver_oracle": _b_block(by_cond.get(B_COND, [])),
        "by_difficulty": {},
        "paired_sender": {
            "written_vs_direct": _paired(grouped_plain, "a_direct", "a_written"),
            "latent40_vs_direct": _paired(grouped_plain, "a_direct", "a_latent40"),
            "written_vs_latent40": _paired(grouped_plain, "a_latent40", "a_written"),
        },
        "flip_tracking": {B_COND: _flip_tracking(rows, B_COND)},
        "lead": (
            "Four-carrier sender calibration on plan4_dev. "
            "A: no extra rollout, written calculations, and 40 latent steps, "
            "then the compact q/eligibility lines. "
            "B receives solver intermediates only (diagnostic target around 90%, not a validity proof). "
            "No compressor. No six-condition communication comparison. "
            "If latent steps do not help, keep that result."
        ),
        "timing_note": (
            "A process_time_s is encode_c0 + rollout_c40 (latent mode only) + message generation. "
            "Written generation is allowed 512 tokens; compact modes 256. "
            "These are not compute-matched. Network transmission is not measured."
        ),
        "deferred": {
            "six_condition": "Run only after interpreting which levels require additional computation.",
            "compression": "After full-relay behavior on this task is interpretable.",
        },
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "sender": {
                cond: _a_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in A_CONDS
            },
            "receiver_oracle": _b_block(_subset(by_cond.get(B_COND, []), difficulty=difficulty)),
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _score_b(text: str, ex, plan_text: str = "") -> dict:
    parsed = label_plan4_answer(
        parse_plan4_answer(text),
        gold_letter=ex.gold_letter,
        gold_carrier=ex.gold_carrier,
        trap_letter=ex.trap_letter,
        trap_carrier=ex.trap_carrier,
    )
    flags = flags_from_plan_kind(parsed.kind)
    ev = evaluate(
        {"boxes": ex.boxes, "refrigerated": ex.refrigerated},
        ex.a_facts,
        ex.b_facts,
        difficulty=ex.difficulty,
    )
    pred = parsed.carrier
    feasible = bool(pred) and pred in ev.affordable_eligible
    extracted = plan_extraction(plan_text, pred)
    return {
        "kind": parsed.kind,
        "parse_ok": parsed.parse_ok,
        "pred_letter": parsed.letter,
        "pred_carrier": parsed.carrier,
        "correct": flags["correct"],
        "follows_trap": flags["follows_trap"],
        "true_partial": flags["true_partial"],
        "trap_partial": flags["trap_partial"],
        "feasible_affordable": feasible,
        **extracted,
        "raw_text": text,
    }


def _a_row(ex, *, condition: str, prompt: str, text: str, extra: dict) -> dict:
    scored = score_sender_message(text, gold_quantities(ex), gold_eligibility_statuses(ex))
    working = written_working_metrics(text) if condition == "a_written" else {
        "n_nonempty_lines": len([ln for ln in (text or "").splitlines() if ln.strip()]),
        "n_working_lines": 0,
        "n_result_lines": scored["n_carriers_parsed"],
        "has_results_block": False,
        "has_arithmetic": False,
        "produced_working": False,
    }
    return {
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "split": ex.split,
        "variant": ex.variant,
        "difficulty": ex.difficulty,
        "condition": condition,
        "a_change": ex.a_change,
        "b_change": ex.b_change,
        "gold_letter": ex.gold_letter,
        "gold_carrier": ex.gold_carrier,
        "trap_letter": ex.trap_letter,
        "trap_carrier": ex.trap_carrier,
        "prompt": prompt,
        "kind": "a_status",
        "parse_ok": not scored["malformed"],
        "pred_letter": None,
        "pred_carrier": None,
        "follows_trap": False,
        "feasible_affordable": False,
        "raw_text": text,
        **scored,
        **working,
        **extra,
    }


def _b_row(ex, *, prompt: str, text: str, extra: dict) -> dict:
    scored = _score_b(text, ex, plan_text=str(extra.get("plan_text") or ""))
    return {
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "split": ex.split,
        "variant": ex.variant,
        "difficulty": ex.difficulty,
        "condition": B_COND,
        "a_change": ex.a_change,
        "b_change": ex.b_change,
        "gold_letter": ex.gold_letter,
        "gold_carrier": ex.gold_carrier,
        "trap_letter": ex.trap_letter,
        "trap_carrier": ex.trap_carrier,
        "prompt": prompt,
        **scored,
        **extra,
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Four-carrier sender calibration on plan4_dev.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--sender_text_max_new", type=int, default=A_COMPACT_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Calibration uses plan4_dev only. Do not load plan_dev or a plan_test file.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_cal" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_cal.")
        return 2
    if not path.is_file():
        print(f"Missing {path}. Run: python -m src.plan4_task --force")
        return 2
    all_examples = load_plan4_jsonl(path)
    examples = all_examples[: args.max_examples] if args.max_examples is not None else all_examples
    for ex in all_examples:
        if ex.split != "plan4_dev":
            print(f"Refusing {ex.example_id} on plan4 calibration.")
            return 2
        assert_plan4_invariants(ex)

    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = out_root.parent / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    artifacts_path = out_root / "a_artifacts.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json", "a_artifacts.jsonl"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    artifacts = _load_messages(artifacts_path)
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-cal",
            "split": "plan4_dev",
            "n_examples": len(examples),
            "n_families": len({ex.family_id for ex in examples}),
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "compressor": "none",
            "channel": "sender calibration plus oracle-to-B",
            "latent_steps_c40": LATENT_STEPS_C40,
            "sender_text_max_new": args.sender_text_max_new,
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "deferred": ["six-condition communication", "compressed KV"],
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=LATENT_STEPS_C40,
        kv_budget=32,
        sink_size=4,
        pca_rank=2,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha") and not pinned.get("model", {}).get("revision"):
        pin_model_revision(revision["sha"])
    extra_manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8")).get("extra", {})
    extra_manifest["model_revision"] = revision.get("sha")
    extra_manifest["model_revision_source"] = revision.get("source")
    write_manifest(out_root / "manifest.json", policy=policy, extra=extra_manifest)
    _persist(out_root, kaggle_dest)

    by_id = {ex.example_id: ex for ex in all_examples}
    states: dict[str, dict] = {}

    def _write_artifacts(key: str, rec: dict) -> None:
        artifacts[key] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def ensure_a(ex) -> dict:
        key = a_state_key(ex)
        if key in states:
            return states[key]
        if key in artifacts:
            states[key] = artifacts[key]
            return states[key]
        src_ex = by_id[key]
        encode_prompt = a_local_task_prompt(src_ex)
        if prompt_contains_b_private(encode_prompt, src_ex):
            raise RuntimeError("A local-task prompt leaked B's prices.")
        if prompt_contains_solver_labels(encode_prompt, src_ex):
            raise RuntimeError("A local-task prompt contained solver q/eligibility lines.")
        if "Now write" in encode_prompt or "Now compute" in encode_prompt:
            raise RuntimeError("C0 encode asked A to write the result.")
        compact = compact_result_handoff()
        written = written_result_handoff()
        if compact_result_handoff() != compact:
            raise RuntimeError("Direct and latent40 must share one compact handoff.")
        wall0 = time.time()
        c0 = runner.encode_prompt(encode_prompt, source="plan4_c0", latent_steps=0)
        c40 = runner.continue_latent(c0, LATENT_STEPS_C40)
        direct, n_d, d_trunc, d_s = runner.generate_from_cache(
            c0, compact, max_new_tokens=args.sender_text_max_new
        )
        lat, n_l, l_trunc, l_s = runner.generate_from_cache(
            c40, compact, max_new_tokens=args.sender_text_max_new
        )
        writ, n_w, w_trunc, w_s = runner.generate_from_cache(
            c0, written, max_new_tokens=args.written_max_new
        )
        encode_c0_s = c0.sender_latency_s
        rollout_c40_s = c40.sender_latency_s
        del c0, c40
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rec = {
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "variant": src_ex.variant,
            "a_state_key": key,
            "encode_prompt": encode_prompt,
            "compact_handoff": compact,
            "written_handoff": written,
            "encode_c0_s": encode_c0_s,
            "rollout_c40_s": rollout_c40_s,
            "cache_prep_s": time.time() - wall0,
            "direct": direct,
            "direct_n_new": n_d,
            "direct_truncated": d_trunc,
            "direct_s": d_s,
            "direct_bytes": len(direct.encode("utf-8")),
            "latent40": lat,
            "latent40_n_new": n_l,
            "latent40_truncated": l_trunc,
            "latent40_s": l_s,
            "latent40_bytes": len(lat.encode("utf-8")),
            "written": writ,
            "written_n_new": n_w,
            "written_truncated": w_trunc,
            "written_s": w_s,
            "written_bytes": len(writ.encode("utf-8")),
            "direct_score": score_sender_message(
                direct, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "latent40_score": score_sender_message(
                lat, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_score": score_sender_message(
                writ, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
        }
        _write_artifacts(key, rec)
        states[key] = rec
        return rec

    needed = set()
    for ex in examples:
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        kinds = {job["condition"] for job in remaining}
        if kinds & set(A_CONDS):
            needed.add(a_state_key(ex))
    print(f"Encoding A states for {len(needed)} unique operational sheets", flush=True)
    for eid in sorted(needed):
        ensure_a(by_id[eid] if eid in by_id else next(ex for ex in all_examples if a_state_key(ex) == eid))
        _persist(out_root, kaggle_dest)

    mode_field = {
        "a_direct": ("direct", "c0", 0.0),
        "a_written": ("written", "c0", 0.0),
        "a_latent40": ("latent40", "c40", None),
    }

    for i, ex in enumerate(examples):
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        for job in remaining:
            cond = job["condition"]
            extra = {
                "encode_c0_s": 0.0,
                "rollout_c40_s": 0.0,
                "sender_generation_time_s": 0.0,
                "sender_truncated": False,
                "n_sender_new_tokens": 0,
                "message_bytes": 0,
                "message_text": "",
                "reused_a_from": None,
                "cache_point": None,
                "plan_time_s": 0.0,
                "n_plan_tokens": 0,
                "plan_truncated": False,
                "plan_text": "",
                "max_new_tokens": args.max_new_tokens,
            }
            if cond in A_CONDS:
                pack = ensure_a(ex)
                field, cache_point, rollout = mode_field[cond]
                text = pack[field]
                extra["reused_a_from"] = pack["a_state_key"] if pack["a_state_key"] != ex.example_id else None
                extra["cache_point"] = cache_point
                extra["encode_c0_s"] = pack["encode_c0_s"]
                extra["rollout_c40_s"] = pack["rollout_c40_s"] if cache_point == "c40" else 0.0
                extra["sender_generation_time_s"] = pack[f"{field}_s"]
                extra["sender_truncated"] = pack[f"{field}_truncated"]
                extra["n_sender_new_tokens"] = pack[f"{field}_n_new"]
                extra["n_new_tokens"] = pack[f"{field}_n_new"]
                extra["truncated"] = pack[f"{field}_truncated"]
                extra["message_bytes"] = pack[f"{field}_bytes"]
                extra["message_text"] = text
                extra["cache_prep_s"] = pack["cache_prep_s"]
                extra["receiver_latency_s"] = 0.0
                extra["peak_gpu_gb"] = peak_gpu_memory_gb()
                extra["process_time_s"] = _process_time_s(extra)
                prompt = pack["written_handoff"] if cond == "a_written" else pack["compact_handoff"]
                rec = _a_row(ex, condition=cond, prompt=prompt, text=text, extra=extra)
            elif cond == B_COND:
                oracle = oracle_intermediate_message(ex)
                plan_prompt = b_reason_with_message_prompt(ex, oracle)
                if prompt_contains_a_private(plan_prompt, ex):
                    raise RuntimeError("Oracle B leaked A's raw sheet.")
                if "Answer: <letter>) <carrier>" in plan_prompt:
                    raise RuntimeError("Oracle reasoning pass still asks for a one-line final answer.")
                plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
                    plan_prompt, max_new_tokens=args.plan_max_new
                )
                prompt = b_answer_after_message_plan_prompt(ex, oracle, plan_text)
                extra["message_text"] = oracle
                extra["message_bytes"] = len(oracle.encode("utf-8"))
                extra["plan_time_s"] = plan_time
                extra["n_plan_tokens"] = n_plan
                extra["plan_truncated"] = plan_trunc
                extra["plan_text"] = plan_text
                extra["plan_max_new"] = args.plan_max_new
                text, n_new, truncated, latency = runner.generate_text_message(
                    prompt, max_new_tokens=args.max_new_tokens
                )
                extra["truncated"] = truncated
                extra["n_new_tokens"] = n_new
                extra["receiver_latency_s"] = latency
                extra["peak_gpu_gb"] = peak_gpu_memory_gb()
                extra["process_time_s"] = _process_time_s({**extra, "receiver_latency_s": latency})
                rec = _b_row(ex, prompt=prompt, text=text, extra=extra)
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if cond in A_CONDS:
                print(
                    f"  {cond}: four={rec['all_four_correct']} q={rec['n_q_correct']} "
                    f"status={rec['n_status_correct']} mal={rec['malformed']} "
                    f"tok={rec['n_new_tokens']} trunc={rec['truncated']}",
                    flush=True,
                )
            else:
                print(
                    f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                    f"correct={rec['correct']} trap={rec['follows_trap']}",
                    flush=True,
                )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Interpret by difficulty before the six-condition communication study.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
