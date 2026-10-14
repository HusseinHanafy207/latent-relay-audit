"""Full-development chain: A writes calculations, B uses the frozen line plan.

A: ordinary one-turn written working plus RESULTS lines.
B (frozen line-by-line procedure): A's actual result lines, solver oracle,
no message, and a mismatched A message from another family.

plan4_dev only. No compressor. No six-condition KV comparison.
Do not resume plan4_cal, plan4_diag, cachecheck, or bprompt.
Do not retune B's prompt.
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
from src.parse_plan4 import handoff_from_written, score_sender_message, written_working_metrics
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
    mismatch_source_id,
    oracle_intermediate_message,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    b_answer_after_message_plan_prompt,
    b_answer_no_message_prompt,
    b_reason_no_message_prompt,
    b_reason_with_message_prompt,
    ordinary_sender_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_cal import _a_block, _a_row, _b_block, _b_row, _process_time_s
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

A_COND = "a_written"
B_CONDS = ("collab", "oracle", "no_message", "mismatch")
CONDS = (A_COND, *B_CONDS)
A_WRITTEN_MAX_NEW = 768
B_PLAN_MAX_NEW = 128
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
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_chain" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_chain")
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


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    unique_a = [r for r in by_cond.get(A_COND, []) if r.get("variant") != "b_flip"]
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len(unique_a),
        "unit_note": (
            "30 families with three versions each, not 90 independent scenarios. "
            "Base and B-change reuse the same A states. "
            "A writes calculations; B receives the result lines, not the scratchpad. "
            "Oracle is program-generated intermediates. 80/90 oracle-B lines is a "
            "reference, not a ceiling for every communication method."
        ),
        "sender": _a_block(by_cond.get(A_COND, [])),
        "sender_unique_sheets": _a_block(unique_a),
        "receiver": {cond: _b_block(by_cond.get(cond, [])) for cond in B_CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "collab_vs_oracle": _paired(grouped_plain, "oracle", "collab"),
            "collab_vs_no_message": _paired(grouped_plain, "no_message", "collab"),
            "collab_vs_mismatch": _paired(grouped_plain, "mismatch", "collab"),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in B_CONDS},
        "lead": (
            "Full plan4_dev chain. A: written calculations. "
            "B: frozen line-by-line plan on A's result lines, solver oracle, "
            "no message, and a mismatched A message. "
            "No compressor. No six-condition KV comparison. Do not retune B."
        ),
        "timing_note": (
            "A process_time_s is ordinary written generation (max 768). "
            "B plan max 128. These are not compute-matched with latent-40. "
            "Network transmission is not measured."
        ),
        "deferred": {
            "six_condition": "text vs full KV vs two compressed KV budgets after this chain is interpretable",
            "compression": "After full-relay behavior on this task is interpretable.",
        },
        "b_prompt": "frozen line-by-line after plan4_bprompt (80/90 oracle reference)",
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "sender": _a_block(_subset(by_cond.get(A_COND, []), difficulty=difficulty)),
            "receiver": {
                cond: _b_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in B_CONDS
            },
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _run_b(runner, ex, *, message: str | None, plan_max_new: int, answer_max_new: int):
    if message is None:
        plan_prompt = b_reason_no_message_prompt(ex)
        if prompt_contains_a_private(plan_prompt, ex):
            raise RuntimeError("No-message B leaked A's sheet.")
        plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
            plan_prompt, max_new_tokens=plan_max_new
        )
        prompt = b_answer_no_message_prompt(ex, plan_text)
        sent = ""
    else:
        plan_prompt = b_reason_with_message_prompt(ex, message)
        if prompt_contains_a_private(plan_prompt, ex):
            raise RuntimeError("B leaked A's raw sheet.")
        if "Answer: <letter>) <carrier>" in plan_prompt:
            raise RuntimeError("Reasoning pass still asks for a one-line final answer.")
        plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
            plan_prompt, max_new_tokens=plan_max_new
        )
        prompt = b_answer_after_message_plan_prompt(ex, message, plan_text)
        sent = message
    text, n_new, truncated, latency = runner.generate_text_message(
        prompt, max_new_tokens=answer_max_new
    )
    extra = {
        "encode_c0_s": 0.0,
        "rollout_c40_s": 0.0,
        "sender_generation_time_s": 0.0,
        "message_text": sent,
        "message_bytes": len(sent.encode("utf-8")),
        "plan_time_s": plan_time,
        "n_plan_tokens": n_plan,
        "plan_truncated": plan_trunc,
        "plan_text": plan_text,
        "plan_max_new": plan_max_new,
        "truncated": truncated,
        "n_new_tokens": n_new,
        "receiver_latency_s": latency,
        "max_new_tokens": answer_max_new,
        "peak_gpu_gb": peak_gpu_memory_gb(),
    }
    extra["process_time_s"] = _process_time_s(extra)
    rec = _b_row(ex, prompt=prompt, text=text, extra=extra)
    return rec


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="plan4_dev chain: written A to frozen B.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Chain uses plan4_dev only. Do not load plan_dev or a plan_test file.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_chain" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_chain.")
        return 2
    if not path.is_file():
        print(f"Missing {path}.")
        return 2
    examples = load_plan4_jsonl(path)
    by_id = {ex.example_id: ex for ex in examples}
    for ex in examples:
        if ex.split != "plan4_dev":
            print(f"Refusing {ex.example_id}.")
            return 2
        assert_plan4_invariants(ex)

    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
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
            "stage": "plan4-chain",
            "split": "plan4_dev",
            "n_examples": len(examples),
            "n_families": 30,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "compressor": "none",
            "a_path": "ordinary written calculations; B receives result lines only",
            "b_prompt": "frozen line-by-line after plan4_bprompt",
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "deferred": ["six-condition text vs full KV vs two compressed budgets"],
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

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
    if revision.get("sha") and not pinned.get("model", {}).get("revision"):
        pin_model_revision(revision["sha"])
    extra_manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8")).get("extra", {})
    extra_manifest["model_revision"] = revision.get("sha")
    extra_manifest["model_revision_source"] = revision.get("source")
    write_manifest(out_root / "manifest.json", policy=policy, extra=extra_manifest)
    _persist(out_root, kaggle_dest)

    states: dict[str, dict] = {}

    def ensure_a(ex) -> dict:
        key = a_state_key(ex)
        if key in states:
            return states[key]
        if key in artifacts:
            states[key] = artifacts[key]
            return states[key]
        src_ex = by_id[key]
        encode_prompt = a_local_task_prompt(src_ex)
        written = written_result_handoff()
        ordinary = ordinary_sender_prompt(encode_prompt, written)
        if prompt_contains_b_private(ordinary, src_ex):
            raise RuntimeError("A prompt leaked B's prices.")
        if prompt_contains_solver_labels(ordinary, src_ex):
            raise RuntimeError("A prompt contained solver q/eligibility lines.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wall0 = time.time()
        text, n_new, truncated, gen_s = runner.generate_text_message(
            ordinary, max_new_tokens=args.written_max_new
        )
        rec = {
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "variant": src_ex.variant,
            "difficulty": src_ex.difficulty,
            "a_state_key": key,
            "encode_prompt": encode_prompt,
            "written_handoff": written,
            "ordinary_prompt": ordinary,
            "written": text,
            "written_n_new": n_new,
            "written_truncated": truncated,
            "written_s": gen_s,
            "written_bytes": len(text.encode("utf-8")),
            "handoff": handoff_from_written(text),
            "encode_c0_s": 0.0,
            "rollout_c40_s": 0.0,
            "cache_prep_s": time.time() - wall0,
            "written_score": score_sender_message(
                text, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_working": written_working_metrics(text),
        }
        artifacts[key] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        states[key] = rec
        return rec

    needed = set()
    for ex in examples:
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if A_COND in remaining or "collab" in remaining or "mismatch" in remaining:
            needed.add(a_state_key(ex))
        if "mismatch" in remaining:
            needed.add(mismatch_source_id(ex, examples))
    print(f"Encoding A written states for {len(needed)} unique operational sheets", flush=True)
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
        for cond in remaining:
            if cond == A_COND:
                extra = {
                    "encode_c0_s": 0.0,
                    "rollout_c40_s": 0.0,
                    "sender_generation_time_s": pack["written_s"],
                    "sender_truncated": pack["written_truncated"],
                    "n_sender_new_tokens": pack["written_n_new"],
                    "n_new_tokens": pack["written_n_new"],
                    "truncated": pack["written_truncated"],
                    "message_bytes": pack["written_bytes"],
                    "message_text": pack["written"],
                    "handoff_text": pack["handoff"],
                    "reused_a_from": pack["a_state_key"] if pack["a_state_key"] != ex.example_id else None,
                    "cache_point": "ordinary",
                    "cache_prep_s": pack["cache_prep_s"],
                    "receiver_latency_s": 0.0,
                    "plan_time_s": 0.0,
                    "peak_gpu_gb": peak_gpu_memory_gb(),
                    "max_new_tokens": args.written_max_new,
                }
                extra["process_time_s"] = _process_time_s(extra)
                rec = _a_row(ex, condition=A_COND, prompt=pack["written_handoff"], text=pack["written"], extra=extra)
            elif cond == "oracle":
                rec = _run_b(
                    runner,
                    ex,
                    message=oracle_intermediate_message(ex),
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec["condition"] = cond
            elif cond == "collab":
                rec = _run_b(
                    runner,
                    ex,
                    message=pack["handoff"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec["condition"] = cond
                rec["reused_a_from"] = pack["a_state_key"] if pack["a_state_key"] != ex.example_id else None
            elif cond == "no_message":
                rec = _run_b(
                    runner,
                    ex,
                    message=None,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec["condition"] = cond
            elif cond == "mismatch":
                donor_id = mismatch_source_id(ex, examples)
                donor = by_id[donor_id]
                donor_pack = ensure_a(donor)
                rec = _run_b(
                    runner,
                    ex,
                    message=donor_pack["handoff"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec["condition"] = cond
                rec["mismatch_source_id"] = donor_id
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if cond == A_COND:
                print(
                    f"  {cond}: four={rec['all_four_correct']} q={rec['n_q_correct']} "
                    f"work={rec.get('produced_working')} tok={rec['n_new_tokens']} "
                    f"trunc={rec['truncated']}",
                    flush=True,
                )
            else:
                print(
                    f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                    f"correct={rec['correct']} trap={rec['follows_trap']} "
                    f"plan_trunc={rec.get('plan_truncated')}",
                    flush=True,
                )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Interpret collaboration vs oracle vs no-message vs mismatch before KV comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
