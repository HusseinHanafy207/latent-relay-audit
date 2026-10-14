"""Text vs full KV of the same written calculation.

A writes calculations once (ordinary one-turn). Text sends the RESULTS lines.
KV sends the cache after that generation. Same frozen line-B procedure.
mismatch_kv uses another family's completed cache.

plan4_dev only. No compressor. No latent-40 sender computation.
Do not resume plan4_cal, diag, cachecheck, bprompt, or chain.
Do not retune A or B prompts.
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
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    b_answer_after_message_plan_prompt,
    b_answer_kv_prompt,
    b_reason_kv_prompt,
    b_reason_with_message_prompt,
    ordinary_sender_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_cal import _b_block, _b_row, _process_time_s
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

CONDS = ("text", "kv", "mismatch_kv")
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
    "results_stage_plan4_chain",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_comm" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_comm")
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


def _comm_block(items: list[dict]) -> dict:
    out = _b_block(items)
    n = len(items)
    if not n:
        return out
    bytes_ = [int(item.get("message_bytes") or 0) for item in items]
    out["mean_message_bytes"] = sum(bytes_) / n
    out["mean_relay_mb"] = (sum(bytes_) / n) / (1024 * 1024)
    out["mean_n_kv_positions"] = sum(int(item.get("n_kv_positions") or 0) for item in items) / n
    out["mean_sender_generation_time_s"] = (
        sum(item.get("sender_generation_time_s") or 0 for item in items) / n
    )
    out["mean_b_only_s"] = (
        sum((item.get("plan_time_s") or 0) + (item.get("receiver_latency_s") or 0) for item in items) / n
    )
    return out


def _sender_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "all_four_correct": sum(bool(r.get("a_all_four_correct")) for r in items) / n,
        "all_status_correct": sum(bool(r.get("a_all_status_correct")) for r in items) / n,
        "produced_working": sum(bool(r.get("a_produced_working")) for r in items) / n,
        "truncated": sum(bool(r.get("a_truncated")) for r in items) / n,
        "mean_n_new_tokens": sum(int(r.get("a_n_new_tokens") or 0) for r in items) / n,
        "mean_sender_generation_time_s": (
            sum(r.get("sender_generation_time_s") or 0 for r in items) / n
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
    text_rows = by_cond.get("text", [])
    unique_a = [r for r in text_rows if r.get("variant") != "b_flip"]
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in text_rows if r.get("a_state_key")}),
        "unit_note": (
            "30 families with three versions each, not 90 independent scenarios. "
            "Base and B-change reuse the same written A states. "
            "Text sends the four RESULT lines. Full KV is the cache after the same "
            "written generation, including working and A's sheet. "
            "A's computation is textual; the KV handoff is still latent. "
            "Chain collab 77/90 is the text-channel reference."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": {cond: _comm_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "kv_vs_text": _paired(grouped_plain, "text", "kv"),
            "kv_vs_mismatch_kv": _paired(grouped_plain, "mismatch_kv", "kv"),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "lead": (
            "Text vs full KV of the same written calculation. "
            "Frozen line-B. mismatch_kv uses another family's completed cache. "
            "No compressor. No latent-40 sender path. Do not retune prompts."
        ),
        "timing_note": (
            "process_time_s charges the shared written generation plus B's plan and answer. "
            "mean_b_only_s is B alone. Network transfer is not measured. "
            "mean_relay_mb is mebibytes (bytes/1024²): UTF-8 of RESULT lines for text, "
            "full KV bytes for kv/mismatch_kv."
        ),
        "deferred": {
            "compression": "Two predefined budgets only after this full relay is interpretable.",
            "fresh_split": "After the communication procedure is settled.",
        },
        "b_prompt": "frozen line-by-line; KV uses the same procedure over relayed memory",
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _comm_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _a_extra(pack: dict) -> dict:
    meta = pack["meta"]
    scored = meta["written_score"]
    working = meta["written_working"]
    return {
        "a_state_key": meta["a_state_key"],
        "a_all_four_correct": scored["all_four_correct"],
        "a_all_status_correct": scored["all_status_correct"],
        "a_n_q_correct": scored["n_q_correct"],
        "a_produced_working": working["produced_working"],
        "a_has_results_block": working["has_results_block"],
        "a_truncated": meta["written_truncated"],
        "a_n_new_tokens": meta["written_n_new"],
        "sender_generation_time_s": meta["written_s"],
        "sender_truncated": meta["written_truncated"],
        "n_sender_new_tokens": meta["written_n_new"],
        "handoff_text": meta["handoff"],
        "encode_c0_s": 0.0,
        "rollout_c40_s": 0.0,
        "compression_time_s": 0.0,
        "cache_point": "written",
        "latent_steps": 0,
    }


def _run_text_b(runner, ex, *, message: str, plan_max_new: int, answer_max_new: int):
    plan_prompt = b_reason_with_message_prompt(ex, message)
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("Text B leaked A's raw sheet.")
    if "Answer: <letter>) <carrier>" in plan_prompt:
        raise RuntimeError("Text reasoning pass still asks for a one-line final answer.")
    plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
        plan_prompt, max_new_tokens=plan_max_new
    )
    prompt = b_answer_after_message_plan_prompt(ex, message, plan_text)
    text, n_new, truncated, latency = runner.generate_text_message(
        prompt, max_new_tokens=answer_max_new
    )
    extra = {
        "channel": "text",
        "handoff": "results_lines",
        "message_text": message,
        "message_bytes": len(message.encode("utf-8")),
        "n_kv_positions": 0,
        "n_latent_positions": 0,
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
    rec = _b_row(ex, prompt=prompt, text=text, extra=extra)
    return rec


def _run_kv_b(runner, ex, bundle, *, plan_max_new: int, answer_max_new: int):
    plan_prompt = b_reason_kv_prompt(ex)
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("KV B leaked A's raw sheet.")
    if "Answer: <letter>) <carrier>" in plan_prompt:
        raise RuntimeError("KV reasoning pass still asks for a one-line final answer.")
    plan_text, n_plan, plan_trunc, plan_time = runner.generate_from_cache(
        bundle, plan_prompt, max_new_tokens=plan_max_new
    )
    prompt = b_answer_kv_prompt(ex, plan_text)
    if prompt_contains_a_private(prompt, ex):
        raise RuntimeError("KV answer pass leaked A's raw sheet.")
    text, n_new, truncated, latency = runner.generate_from_cache(
        bundle, prompt, max_new_tokens=answer_max_new
    )
    extra = {
        "channel": "kv",
        "handoff": "full_kv",
        "message_text": "",
        "message_bytes": int(bundle.full_relay_bytes),
        "n_kv_positions": n_kv_positions(bundle.cache_cpu),
        "n_latent_positions": int(bundle.latent_steps),
        "n_prompt_positions": int(bundle.prompt_mask_cpu.sum().item()),
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
    rec = _b_row(ex, prompt=prompt, text=text, extra=extra)
    return rec


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="plan4_dev text vs full KV of the same written A.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Communication comparison uses plan4_dev only. Do not load plan_dev or a plan_test file.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_comm" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_comm.")
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
            "stage": "plan4-comm",
            "split": "plan4_dev",
            "n_examples": len(examples),
            "n_families": 30,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "compressor": "none",
            "a_path": "ordinary written calculations; text gets RESULT lines; KV gets that cache",
            "b_prompt": "frozen line-by-line; KV uses relayed-memory wording",
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "deferred": ["two compressed KV budgets after this full relay is interpretable"],
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
        text, n_new, truncated, gen_s, bundle = runner.generate_text_keep_cache(
            ordinary, max_new_tokens=args.written_max_new, source="written"
        )
        if int(bundle.latent_steps) != 0:
            raise RuntimeError("Written cache must have zero latent steps.")
        if n_kv_positions(bundle.cache_cpu) < int(bundle.prompt_mask_cpu.sum().item()):
            raise RuntimeError("Written cache is shorter than the prompt.")
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
            "n_kv_positions": n_kv_positions(bundle.cache_cpu),
            "n_prompt_positions": int(bundle.prompt_mask_cpu.sum().item()),
            "full_relay_bytes": int(bundle.full_relay_bytes),
            "latent_steps": 0,
            "written_score": score_sender_message(
                text, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_working": written_working_metrics(text),
        }
        artifacts[key] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        states[key] = {"bundle": bundle, "meta": rec}
        return states[key]

    needed = set()
    for ex in examples:
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if remaining:
            needed.add(a_state_key(ex))
        if "mismatch_kv" in remaining:
            needed.add(mismatch_source_id(ex, examples))
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
        for cond in remaining:
            extra_a = _a_extra(pack)
            if cond == "text":
                rec = _run_text_b(
                    runner,
                    ex,
                    message=pack["meta"]["handoff"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            elif cond == "kv":
                rec = _run_kv_b(
                    runner,
                    ex,
                    pack["bundle"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            elif cond == "mismatch_kv":
                donor_id = mismatch_source_id(ex, examples)
                donor = by_id[donor_id]
                donor_pack = ensure_a(donor)
                rec = _run_kv_b(
                    runner,
                    ex,
                    donor_pack["bundle"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(_a_extra(donor_pack))
                rec["condition"] = cond
                rec["mismatch_source_id"] = donor_id
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rec["process_time_s"] = _process_time_s(rec)
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                f"correct={rec['correct']} trap={rec['follows_trap']} "
                f"bytes={rec.get('message_bytes')} kv={rec.get('n_kv_positions')} "
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
    print("Interpret text vs full KV before compression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
