"""Study 1 core comparison: does extra sender processing help, and how is it sent?

Immediate milestone: text from C0, text from C40, full KV from C0, full KV
from C40, plus source-evidence text, B without a message, and mismatched
C40 text/KV. No compressor. No length-control diagnostics.

Development split only (plan_dev). Do not load data/test.jsonl, hop_test, or
a plan_test file. Do not resume Study 0 result folders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse_plan import score_eligibility_message
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan_task import (
    assert_plan_invariants,
    gold_eligibility_statuses,
    load_plan_jsonl,
    mismatch_source_id,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan import (
    b_answer_after_message_plan_prompt,
    b_answer_after_plan_prompt,
    b_planning_prompt,
    b_reason_with_message_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
)
from src.prompts_study1 import (
    INTERMEDIATE_HANDOFF,
    a_local_task_prompt,
    b_kv_answer_prompt,
    b_kv_reason_prompt,
    intermediate_handoff_prompt,
    source_evidence_handoff_prompt,
    source_fact_coverage,
)
from src.stage_plan0 import (
    A_MESSAGE_MAX_NEW,
    B_ANSWER_MAX_NEW,
    B_PLAN_MAX_NEW,
    _flip_tracking,
    _load_messages,
    _persist as _persist_core,
    _record,
    _subset,
)
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

CORE_CONDS = ("text_c0", "text_c40", "kv_c0", "kv_c40")
BASELINE_CONDS = ("source_evidence", "no_message", "mismatch_text_c40", "mismatch_kv_c40")
CONDS = CORE_CONDS + BASELINE_CONDS
TEXT_CONDS = ("text_c0", "text_c40", "source_evidence", "mismatch_text_c40")
KV_CONDS = ("kv_c0", "kv_c40", "mismatch_kv_c40")
LATENT_STEPS_C40 = 40
PRESERVED = (
    "results_stage_plan0",
    "results_stage_plan0_corrected",
    "results_stage_plan0_diag",
    "results_stage_plan0_e2e",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _process_time_s(row: dict) -> float:
    return float(
        (row.get("encode_c0_s") or 0.0)
        + (row.get("rollout_c40_s") or 0.0)
        + (row.get("sender_generation_time_s") or 0.0)
        + (row.get("compression_time_s") or 0.0)
        + (row.get("plan_time_s") or 0.0)
        + (row.get("receiver_latency_s") or 0.0)
    )


def _paths():
    data = DATA_DIR / "plan_dev.jsonl"
    out_root = RESULTS_DIR / "stage_study1" / "run"
    kaggle = Path("/kaggle/working/results_stage_study1")
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


def _cond_block(items: list[dict]) -> dict:
    n = len(items)
    n_true = sum(bool(item.get("correct")) for item in items)
    n_trap = sum(bool(item.get("follows_trap")) for item in items)
    n_parse = sum(bool(item.get("parse_ok")) for item in items)
    sizes = [int(item.get("message_bytes") or 0) for item in items]
    kv_pos = [int(item.get("n_kv_positions") or 0) for item in items]
    latent_pos = [int(item.get("n_latent_positions") or 0) for item in items]
    return {
        "n": n,
        "correct": n_true,
        "accuracy": n_true / n if n else None,
        "follows_trap": n_trap / n if n else None,
        "parse_ok": n_parse / n if n else None,
        "truncated": sum(bool(item.get("truncated")) for item in items) / n if n else None,
        "sender_truncated": sum(bool(item.get("sender_truncated")) for item in items) / n if n else None,
        "feasible_affordable": (
            sum(bool(item.get("feasible_affordable")) for item in items) / n if n else None
        ),
        "mean_message_bytes": (sum(sizes) / n) if n else None,
        "mean_relay_mb": (sum(sizes) / n / (1024 * 1024)) if n else None,
        "mean_encode_c0_s": (
            sum(item.get("encode_c0_s") or 0 for item in items) / n if n else None
        ),
        "mean_rollout_c40_s": (
            sum(item.get("rollout_c40_s") or 0 for item in items) / n if n else None
        ),
        "mean_sender_generation_time_s": (
            sum(item.get("sender_generation_time_s") or 0 for item in items) / n if n else None
        ),
        "mean_compression_time_s": (
            sum(item.get("compression_time_s") or 0 for item in items) / n if n else None
        ),
        "mean_plan_time_s": (sum(item.get("plan_time_s") or 0 for item in items) / n if n else None),
        "mean_n_plan_tokens": (sum(item.get("n_plan_tokens") or 0 for item in items) / n if n else None),
        "mean_receiver_latency_s": (
            sum(item.get("receiver_latency_s") or 0 for item in items) / n if n else None
        ),
        "mean_process_time_s": (sum(_process_time_s(item) for item in items) / n if n else None),
        "mean_n_kv_positions": (sum(kv_pos) / n) if n else None,
        "mean_n_latent_positions": (sum(latent_pos) / n) if n else None,
        "mean_n_carriers_complete": (
            sum(int(item.get("n_carriers_complete") or 0) for item in items) / n if n else None
        ),
    }


def _delta_block(paired: dict) -> dict:
    return {
        **paired,
        "delta": paired.get("delta_b_minus_a"),
        "interpretation": (
            "Effect of adding the 40-step rollout under this procedure. "
            "C0 already contains computation performed while reading the input. "
            "This is additional latent processing versus no additional rollout, "
            "not reasoning versus no reasoning. A positive KV delta does not "
            "establish that B reused a stored calculation: extra cache positions "
            "and retained source information can also change B's context."
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
    out: dict = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": 12,
        "unit_note": (
            "12 families with three versions each, not 36 independent scenarios. "
            "Keep variants together if estimating uncertainty later."
        ),
        "accuracy": {cond: _cond_block(items) for cond, items in by_cond.items()},
        "by_variant": {},
        "by_difficulty": {},
        "delta_kv": _delta_block(_paired(grouped_plain, "kv_c0", "kv_c40")),
        "delta_text": _delta_block(_paired(grouped_plain, "text_c0", "text_c40")),
        "paired": {
            "text_c40_vs_text_c0": _paired(grouped_plain, "text_c0", "text_c40"),
            "kv_c40_vs_kv_c0": _paired(grouped_plain, "kv_c0", "kv_c40"),
            "kv_c40_vs_text_c40": _paired(grouped_plain, "text_c40", "kv_c40"),
            "text_c40_vs_source_evidence": _paired(grouped_plain, "source_evidence", "text_c40"),
            "text_c0_vs_source_evidence": _paired(grouped_plain, "source_evidence", "text_c0"),
            "kv_c40_vs_no_message": _paired(grouped_plain, "no_message", "kv_c40"),
            "text_c40_vs_no_message": _paired(grouped_plain, "no_message", "text_c40"),
            "mismatch_text_vs_text_c40": _paired(grouped_plain, "text_c40", "mismatch_text_c40"),
            "mismatch_kv_vs_kv_c40": _paired(grouped_plain, "kv_c40", "mismatch_kv_c40"),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "timing_note": (
            "process_time_s is encode_c0 + rollout_c40 (if used) + A's message "
            "generation (if text) + compression (0 in this run) + B plan + B answer. "
            "n_kv_positions counts every retained cache position, including the 40 "
            "latent positions on C40. mean_relay_mb is bytes/1024^2 of the payload "
            "sent to B (UTF-8 text or full KV bytes), not the whole receiver prompt."
        ),
        "lead": (
            "Study 1 development comparison on plan_dev. Four core conditions "
            "(text C0, text C40, full KV C0, full KV C40), plus source-evidence "
            "text from C0, B without a message, and mismatched C40 text/KV. "
            "No compressor. No length-control diagnostics. "
            "Matched intermediate-result handoff for the two text conditions. "
            "Do not drop individual failures. Do not retune prompts for a single error."
        ),
        "a_flip_note": (
            "On a_flip the gold carrier becomes B's cheapest-under-budget option. "
            "Read prediction_changed, not a_flip accuracy alone."
        ),
        "deferred": {
            "compression": "Add two predefined budgets only after uncompressed full-relay behavior is understood.",
            "length_controls": (
                "Repeated or norm-matched random latent inputs are optional "
                "diagnostics, not a clean same-length zero-computation condition."
            ),
        },
    }
    for variant in ("base", "a_flip", "b_flip"):
        out["by_variant"][variant] = {
            cond: _cond_block(_subset(items, variant=variant)) for cond, items in by_cond.items()
        }
    for difficulty in ("easy", "combine"):
        out["by_difficulty"][difficulty] = {
            cond: _cond_block(_subset(items, difficulty=difficulty)) for cond, items in by_cond.items()
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _bundle_meta(bundle, *, encode_c0_s: float, rollout_c40_s: float) -> dict:
    n_prompt = int(bundle.prompt_mask_cpu.sum().item())
    n_kv = n_kv_positions(bundle.cache_cpu)
    return {
        "n_prompt_positions": n_prompt,
        "n_kv_positions": n_kv,
        "n_latent_positions": int(bundle.latent_steps),
        "message_bytes": int(bundle.full_relay_bytes),
        "encode_c0_s": encode_c0_s,
        "rollout_c40_s": rollout_c40_s,
        "latent_steps": int(bundle.latent_steps),
    }


def _empty_extra(*, max_new_tokens: int) -> dict:
    return {
        "encode_c0_s": 0.0,
        "rollout_c40_s": 0.0,
        "sender_generation_time_s": 0.0,
        "sender_truncated": False,
        "n_sender_new_tokens": 0,
        "compression_time_s": 0.0,
        "message_bytes": 0,
        "message_text": "",
        "mismatch_source_id": None,
        "channel": None,
        "cache_point": None,
        "handoff": None,
        "max_new_tokens": max_new_tokens,
        "plan_time_s": 0.0,
        "n_plan_tokens": 0,
        "plan_truncated": False,
        "plan_text": "",
        "n_prompt_positions": 0,
        "n_kv_positions": 0,
        "n_latent_positions": 0,
        "n_carriers_complete": None,
        "all_carriers_complete": None,
        "wrote_eligibility_label": None,
        "a_all_status_correct": None,
        "a_n_status_correct": None,
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Study 1 four-condition comparison on plan_dev.")
    parser.add_argument("--split", default="plan_dev")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--sender_text_max_new", type=int, default=A_MESSAGE_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan_dev":
        print("Study 1 development freeze uses plan_dev only. No plan_test. No plan_fresh in this run.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_study1" not in str(out_root):
        print("Refusing to write outside results_stage_study1.")
        return 2
    for name in PRESERVED:
        if name in str(out_root) and name != "results_stage_study1":
            print(f"Refusing to write over preserved {name}.")
            return 2
    if not path.is_file():
        print(f"Missing {path}. Run: python -m src.plan_task --force")
        return 2
    all_examples = load_plan_jsonl(path)
    examples = all_examples[: args.max_examples] if args.max_examples is not None else all_examples
    for ex in all_examples:
        if ex.split != "plan_dev":
            print(f"Refusing {ex.example_id} on Study 1.")
            return 2
        assert_plan_invariants(ex)

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
            "stage": "study1-plan_dev",
            "split": "plan_dev",
            "n_examples": len(examples),
            "n_families": len({ex.family_id for ex in examples}),
            "n_examples_in_file": len(all_examples),
            "n_receiver_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "core_conditions": list(CORE_CONDS),
            "compressor": "none",
            "compression_time_s": 0,
            "latent_steps_c0": 0,
            "latent_steps_c40": LATENT_STEPS_C40,
            "sender_text_max_new": args.sender_text_max_new,
            "plan_max_new": args.plan_max_new,
            "matched_text_handoff": True,
            "source_evidence": "A-generated from C0; not hop retriever",
            "b_procedure": "reason-then-answer, frozen after Study 0 diagnostic",
            "held_out": "data/test.jsonl, hop_test, plan_fresh, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "deferred": ["compressed KV relay", "length-control diagnostics"],
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

    def _write_artifacts(rec: dict) -> None:
        artifacts[rec["example_id"]] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def ensure_states(ex) -> dict:
        if ex.example_id in states:
            return states[ex.example_id]
        encode_prompt = a_local_task_prompt(ex)
        if prompt_contains_b_private(encode_prompt, ex):
            raise RuntimeError("A local-task prompt leaked B's prices.")
        if INTERMEDIATE_HANDOFF in encode_prompt:
            raise RuntimeError("C0 encode included the message handoff.")
        if "Now write a message" in encode_prompt:
            raise RuntimeError("C0 encode asked A to write a message.")
        handoff = intermediate_handoff_prompt()
        source_handoff = source_evidence_handoff_prompt()
        if handoff != intermediate_handoff_prompt():
            raise RuntimeError("Text conditions must share one handoff string.")
        c0 = runner.encode_prompt(encode_prompt, source="plan_a_c0", latent_steps=0)
        n_c0 = n_kv_positions(c0.cache_cpu)
        if int(c0.latent_steps) != 0:
            raise RuntimeError("C0 must have zero extra latent steps.")
        c40 = runner.continue_latent(runner.clone_sender(c0), LATENT_STEPS_C40)
        if int(c40.latent_steps) != LATENT_STEPS_C40:
            raise RuntimeError("C40 must add exactly 40 latent steps.")
        if n_kv_positions(c0.cache_cpu) != n_c0:
            raise RuntimeError("C40 continuation mutated the stored C0 cache.")
        if n_kv_positions(c40.cache_cpu) != n_c0 + LATENT_STEPS_C40:
            raise RuntimeError("C40 did not add 40 cache positions.")

        text_c0, n_t0, t0_trunc, t0_s = runner.generate_from_cache(
            c0, handoff, max_new_tokens=args.sender_text_max_new
        )
        text_c40, n_t40, t40_trunc, t40_s = runner.generate_from_cache(
            c40, handoff, max_new_tokens=args.sender_text_max_new
        )
        source_text, n_src, src_trunc, src_s = runner.generate_from_cache(
            c0, source_handoff, max_new_tokens=args.sender_text_max_new
        )
        gold = gold_eligibility_statuses(ex)
        coverage = source_fact_coverage(source_text, ex)
        rec = {
            "example_id": ex.example_id,
            "family_id": ex.family_id,
            "variant": ex.variant,
            "encode_prompt": encode_prompt,
            "handoff": handoff,
            "source_handoff": source_handoff,
            "encode_c0_s": c0.sender_latency_s,
            "rollout_c40_s": c40.sender_latency_s,
            "n_kv_c0": n_kv_positions(c0.cache_cpu),
            "n_kv_c40": n_kv_positions(c40.cache_cpu),
            "n_prompt_positions": int(c0.prompt_mask_cpu.sum().item()),
            "n_latent_c40": LATENT_STEPS_C40,
            "full_relay_bytes_c0": c0.full_relay_bytes,
            "full_relay_bytes_c40": c40.full_relay_bytes,
            "text_c0": text_c0,
            "text_c0_n_new": n_t0,
            "text_c0_truncated": t0_trunc,
            "text_c0_s": t0_s,
            "text_c0_bytes": len(text_c0.encode("utf-8")),
            "text_c40": text_c40,
            "text_c40_n_new": n_t40,
            "text_c40_truncated": t40_trunc,
            "text_c40_s": t40_s,
            "text_c40_bytes": len(text_c40.encode("utf-8")),
            "source_evidence": source_text,
            "source_evidence_n_new": n_src,
            "source_evidence_truncated": src_trunc,
            "source_evidence_s": src_s,
            "source_evidence_bytes": len(source_text.encode("utf-8")),
            "text_c0_status": score_eligibility_message(text_c0, gold),
            "text_c40_status": score_eligibility_message(text_c40, gold),
            "source_coverage": coverage,
        }
        _write_artifacts(rec)
        states[ex.example_id] = {"c0": c0, "c40": c40, "meta": rec}
        return states[ex.example_id]

    needed = set()
    for ex in examples:
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        kinds = {job["condition"] for job in remaining}
        if kinds - {"no_message"}:
            needed.add(ex.example_id)
        if kinds & {"mismatch_text_c40", "mismatch_kv_c40"}:
            needed.add(mismatch_source_id(ex, all_examples))
    print(f"Encoding A states for {len(needed)} examples", flush=True)
    for eid in sorted(needed):
        if eid not in by_id:
            print(f"Missing mismatch source {eid}")
            return 2
        ensure_states(by_id[eid])
        _persist(out_root, kaggle_dest)

    def _b_text(ex, message: str, extra: dict) -> tuple[str, str]:
        plan_prompt = b_reason_with_message_prompt(ex, message)
        if "Answer: <letter>) <carrier>" in plan_prompt or "Stop after that line." in plan_prompt:
            raise RuntimeError("Reasoning pass still asks for a one-line final answer.")
        if message not in plan_prompt:
            raise RuntimeError("Reasoning pass omitted A's message.")
        if prompt_contains_a_private(plan_prompt, ex):
            raise RuntimeError("Reasoning pass leaked A's raw sheet.")
        plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
            plan_prompt, max_new_tokens=args.plan_max_new
        )
        prompt = b_answer_after_message_plan_prompt(ex, message, plan_text)
        extra["plan_time_s"] = plan_time
        extra["n_plan_tokens"] = n_plan
        extra["plan_truncated"] = plan_trunc
        extra["plan_text"] = plan_text
        extra["plan_max_new"] = args.plan_max_new
        return prompt, plan_prompt

    def _b_kv(ex, bundle, extra: dict) -> str:
        plan_prompt = b_kv_reason_prompt(ex)
        if "Answer: <letter>) <carrier>" in plan_prompt or "Stop after that line." in plan_prompt:
            raise RuntimeError("KV reasoning pass still asks for a one-line final answer.")
        if prompt_contains_a_private(plan_prompt, ex):
            raise RuntimeError("KV reasoning pass leaked A's raw sheet.")
        plan_text, n_plan, plan_trunc, plan_time = runner.generate_from_cache(
            bundle, plan_prompt, max_new_tokens=args.plan_max_new
        )
        prompt = b_kv_answer_prompt(ex, plan_text)
        extra["plan_time_s"] = plan_time
        extra["n_plan_tokens"] = n_plan
        extra["plan_truncated"] = plan_trunc
        extra["plan_text"] = plan_text
        extra["plan_max_new"] = args.plan_max_new
        return prompt

    for i, ex in enumerate(examples):
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        for job in remaining:
            cond = job["condition"]
            extra = _empty_extra(max_new_tokens=args.max_new_tokens)
            extra["compression_time_s"] = 0.0
            if cond == "no_message":
                extra["channel"] = "none"
                extra["cache_point"] = None
                extra["handoff"] = None
                plan_prompt = b_planning_prompt(ex)
                if prompt_contains_a_private(plan_prompt, ex):
                    raise RuntimeError("no_message leaked A's operations.")
                if "Answer: <letter>) <carrier>" in plan_prompt or "Stop after that line." in plan_prompt:
                    raise RuntimeError("no_message planning pass still asks for a one-line final answer.")
                plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
                    plan_prompt, max_new_tokens=args.plan_max_new
                )
                prompt = b_answer_after_plan_prompt(ex, plan_text)
                extra["plan_time_s"] = plan_time
                extra["n_plan_tokens"] = n_plan
                extra["plan_truncated"] = plan_trunc
                extra["plan_text"] = plan_text
                extra["plan_max_new"] = args.plan_max_new
                text, n_new, truncated, latency = runner.generate_text_message(
                    prompt, max_new_tokens=args.max_new_tokens
                )
            elif cond in TEXT_CONDS:
                extra["channel"] = "text"
                extra["handoff"] = "intermediate" if cond != "source_evidence" else "source_evidence"
                if cond == "mismatch_text_c40":
                    src_id = mismatch_source_id(ex, all_examples)
                    pack = ensure_states(by_id[src_id])
                    extra["mismatch_source_id"] = src_id
                    extra["cache_point"] = "c40"
                    message = pack["meta"]["text_c40"]
                    extra["encode_c0_s"] = pack["meta"]["encode_c0_s"]
                    extra["rollout_c40_s"] = pack["meta"]["rollout_c40_s"]
                    extra["sender_generation_time_s"] = pack["meta"]["text_c40_s"]
                    extra["sender_truncated"] = pack["meta"]["text_c40_truncated"]
                    extra["n_sender_new_tokens"] = pack["meta"]["text_c40_n_new"]
                    extra["message_bytes"] = pack["meta"]["text_c40_bytes"]
                    extra["a_all_status_correct"] = pack["meta"]["text_c40_status"]["all_status_correct"]
                    extra["a_n_status_correct"] = pack["meta"]["text_c40_status"]["n_status_correct"]
                else:
                    pack = ensure_states(ex)
                    extra["encode_c0_s"] = pack["meta"]["encode_c0_s"]
                    if cond == "text_c0":
                        extra["cache_point"] = "c0"
                        extra["rollout_c40_s"] = 0.0
                        message = pack["meta"]["text_c0"]
                        extra["sender_generation_time_s"] = pack["meta"]["text_c0_s"]
                        extra["sender_truncated"] = pack["meta"]["text_c0_truncated"]
                        extra["n_sender_new_tokens"] = pack["meta"]["text_c0_n_new"]
                        extra["message_bytes"] = pack["meta"]["text_c0_bytes"]
                        extra["a_all_status_correct"] = pack["meta"]["text_c0_status"]["all_status_correct"]
                        extra["a_n_status_correct"] = pack["meta"]["text_c0_status"]["n_status_correct"]
                    elif cond == "text_c40":
                        extra["cache_point"] = "c40"
                        extra["rollout_c40_s"] = pack["meta"]["rollout_c40_s"]
                        message = pack["meta"]["text_c40"]
                        extra["sender_generation_time_s"] = pack["meta"]["text_c40_s"]
                        extra["sender_truncated"] = pack["meta"]["text_c40_truncated"]
                        extra["n_sender_new_tokens"] = pack["meta"]["text_c40_n_new"]
                        extra["message_bytes"] = pack["meta"]["text_c40_bytes"]
                        extra["a_all_status_correct"] = pack["meta"]["text_c40_status"]["all_status_correct"]
                        extra["a_n_status_correct"] = pack["meta"]["text_c40_status"]["n_status_correct"]
                    else:
                        extra["cache_point"] = "c0"
                        extra["rollout_c40_s"] = 0.0
                        message = pack["meta"]["source_evidence"]
                        extra["sender_generation_time_s"] = pack["meta"]["source_evidence_s"]
                        extra["sender_truncated"] = pack["meta"]["source_evidence_truncated"]
                        extra["n_sender_new_tokens"] = pack["meta"]["source_evidence_n_new"]
                        extra["message_bytes"] = pack["meta"]["source_evidence_bytes"]
                        cov = pack["meta"]["source_coverage"]
                        extra["n_carriers_complete"] = cov["n_carriers_complete"]
                        extra["all_carriers_complete"] = cov["all_carriers_complete"]
                        extra["wrote_eligibility_label"] = cov["wrote_eligibility_label"]
                extra["message_text"] = message
                extra["n_kv_positions"] = 0
                extra["n_latent_positions"] = 0
                prompt, _plan_prompt = _b_text(ex, message, extra)
                text, n_new, truncated, latency = runner.generate_text_message(
                    prompt, max_new_tokens=args.max_new_tokens
                )
            elif cond in KV_CONDS:
                extra["channel"] = "kv"
                extra["handoff"] = "full_kv"
                extra["sender_generation_time_s"] = 0.0
                extra["message_text"] = ""
                if cond == "mismatch_kv_c40":
                    src_id = mismatch_source_id(ex, all_examples)
                    pack = ensure_states(by_id[src_id])
                    bundle = pack["c40"]
                    extra["mismatch_source_id"] = src_id
                    extra["cache_point"] = "c40"
                    extra["encode_c0_s"] = pack["meta"]["encode_c0_s"]
                    extra["rollout_c40_s"] = pack["meta"]["rollout_c40_s"]
                elif cond == "kv_c0":
                    pack = ensure_states(ex)
                    bundle = pack["c0"]
                    extra["cache_point"] = "c0"
                    extra["encode_c0_s"] = pack["meta"]["encode_c0_s"]
                    extra["rollout_c40_s"] = 0.0
                else:
                    pack = ensure_states(ex)
                    bundle = pack["c40"]
                    extra["cache_point"] = "c40"
                    extra["encode_c0_s"] = pack["meta"]["encode_c0_s"]
                    extra["rollout_c40_s"] = pack["meta"]["rollout_c40_s"]
                extra.update(
                    _bundle_meta(
                        bundle,
                        encode_c0_s=extra["encode_c0_s"],
                        rollout_c40_s=extra["rollout_c40_s"],
                    )
                )
                prompt = _b_kv(ex, bundle, extra)
                if prompt_contains_a_private(prompt, ex):
                    raise RuntimeError(f"{cond} leaked A's raw operational sheet.")
                text, n_new, truncated, latency = runner.generate_from_cache(
                    bundle, prompt, max_new_tokens=args.max_new_tokens
                )
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            extra["truncated"] = truncated
            extra["n_new_tokens"] = n_new
            extra["receiver_latency_s"] = latency
            extra["peak_gpu_gb"] = peak_gpu_memory_gb()
            extra["process_time_s"] = _process_time_s({**extra, "receiver_latency_s": latency})
            rec = _record(ex, condition=cond, prompt=prompt, text=text, extra=extra)
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                f"correct={rec['correct']} trap={rec['follows_trap']} "
                f"bytes={extra['message_bytes']} kv={extra['n_kv_positions']} "
                f"encode={extra['encode_c0_s']:.3f} rollout={extra['rollout_c40_s']:.3f} "
                f"agen={extra['sender_generation_time_s']:.3f} "
                f"plan={extra['plan_time_s']:.3f} recv={latency:.3f}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print(
        "Do not drop individual failures. Interpret delta_kv and delta_text as "
        "additional rollout versus no additional rollout. Compression and length "
        "controls are not in this run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
