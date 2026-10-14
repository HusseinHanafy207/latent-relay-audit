"""Study 0 end-to-end check on a small fresh development split.

A's generated compact message is sent to B. B uses the frozen reason-then-answer
procedure. A mismatched A message is the control. Prompts are not retuned.
Does not resume earlier Study 0 folders. Does not load plan_dev, hop_test, or a
plan_test file.
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

from src.manifest import load_pinned, write_manifest
from src.model_revision import resolve_model_revision
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
    agent_a_message_prompt,
    b_answer_after_message_plan_prompt,
    b_reason_with_message_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
)
from src.stage_plan0 import (
    A_MESSAGE_MAX_NEW,
    B_ANSWER_MAX_NEW,
    B_PLAN_MAX_NEW,
    _cond_block,
    _flip_tracking,
    _load_messages,
    _persist,
    _process_time_s,
    _record,
    _subset,
)
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

A_COND = "a_status"
B_CONDS = ("collab_reason", "mismatch_reason")
CONDS = (A_COND, *B_CONDS)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan_fresh.jsonl"
    out_root = RESULTS_DIR / "stage_plan0_e2e" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan0_e2e")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _a_row(ex, msg: dict, extra: dict) -> dict:
    return {
        **extra,
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "split": ex.split,
        "variant": ex.variant,
        "difficulty": ex.difficulty,
        "condition": A_COND,
        "gold_letter": ex.gold_letter,
        "gold_carrier": ex.gold_carrier,
        "trap_letter": ex.trap_letter,
        "trap_carrier": ex.trap_carrier,
        "prompt": msg["prompt"],
        "kind": "a_status",
        "parse_ok": msg["n_status_parsed"] == 3,
        "pred_letter": None,
        "pred_carrier": None,
        "follows_trap": False,
        "true_partial": False,
        "trap_partial": False,
        "feasible_affordable": False,
        "raw_text": msg["message_text"],
        "truncated": msg["truncated"],
        "n_new_tokens": msg["n_new_tokens"],
        "peak_gpu_gb": peak_gpu_memory_gb(),
        "receiver_latency_s": 0.0,
        "process_time_s": msg["sender_generation_time_s"],
        "gold_statuses": msg["gold_statuses"],
        "pred_statuses": msg["pred_statuses"],
        "n_status_correct": msg["n_status_correct"],
        "n_status_parsed": msg["n_status_parsed"],
        "all_status_correct": msg["all_status_correct"],
        "copied_old_template": msg["copied_old_template"],
        "incorrect_old_template_copy": msg["incorrect_old_template_copy"],
        "correct": msg["correct"],
        "message_text": msg["message_text"],
        "message_bytes": msg["message_bytes"],
        "sender_generation_time_s": msg["sender_generation_time_s"],
        "sender_truncated": msg["truncated"],
        "n_sender_new_tokens": msg["n_new_tokens"],
    }


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    a_rows = by_cond.get(A_COND, [])
    n_a = len(a_rows)
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": 6,
        "unit_note": "6 fresh families with three versions each, not 18 independent scenarios.",
        "a_status": {
            "n": n_a,
            "all_three_correct": sum(bool(r.get("all_status_correct")) for r in a_rows) if n_a else None,
            "mean_n_status_correct": (
                sum(int(r.get("n_status_correct") or 0) for r in a_rows) / n_a if n_a else None
            ),
            "copied_old_template": (
                sum(bool(r.get("copied_old_template")) for r in a_rows) / n_a if n_a else None
            ),
            "incorrect_old_template_copy": (
                sum(bool(r.get("incorrect_old_template_copy")) for r in a_rows) / n_a if n_a else None
            ),
            "truncated": sum(bool(r.get("truncated")) for r in a_rows) / n_a if n_a else None,
        },
        "accuracy": {cond: _cond_block(items) for cond, items in by_cond.items() if cond != A_COND},
        "by_variant": {},
        "by_difficulty": {},
        "paired": {
            "collab_reason_vs_mismatch_reason": _paired(grouped_plain, "mismatch_reason", "collab_reason"),
        },
        "flip_tracking": {
            cond: _flip_tracking(rows, cond) for cond in B_CONDS
        },
        "timing_note": (
            "B uses the frozen reason-then-answer procedure on A's generated compact "
            "message. Timings include that planning pass. This is not b_extra_compute "
            "and not an oracle message."
        ),
        "lead": (
            "Study 0 end-to-end on plan_fresh. Frozen prompts. A's actual message to B "
            "with reason-then-answer, plus a mismatched-message control. "
            "Not Study 1. Not held-out confirmation. Do not drop individual failures. "
            "Do not retune prompts for a single error."
        ),
        "a_flip_note": (
            "On a_flip the gold carrier becomes B's cheapest-under-budget option. "
            "Read prediction_changed, not a_flip accuracy alone."
        ),
    }
    for variant in ("base", "a_flip", "b_flip"):
        out["by_variant"][variant] = {
            cond: _cond_block(_subset(items, variant=variant))
            for cond, items in by_cond.items()
            if cond != A_COND
        }
    for difficulty in ("easy", "combine"):
        out["by_difficulty"][difficulty] = {
            cond: _cond_block(_subset(items, difficulty=difficulty))
            for cond, items in by_cond.items()
            if cond != A_COND
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Study 0 end-to-end check on plan_fresh.")
    parser.add_argument("--split", default="plan_fresh")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--sender_text_max_new", type=int, default=A_MESSAGE_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan_fresh":
        print("End-to-end check runs only on plan_fresh. plan_dev was used for earlier development.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan0_e2e" not in str(out_root):
        print("Refusing to write over preserved Study 0 runs.")
        return 2
    if not path.is_file():
        print(f"Missing {path}. Run: python -m src.plan_task --fresh")
        return 2
    all_examples = load_plan_jsonl(path)
    examples = all_examples[: args.max_examples] if args.max_examples is not None else all_examples
    for ex in all_examples:
        if ex.split != "plan_fresh":
            print(f"Refusing {ex.example_id} on the end-to-end check.")
            return 2
        assert_plan_invariants(ex)

    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = out_root.parent / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    messages_path = out_root / "a_messages.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json", "a_messages.jsonl"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    messages = _load_messages(messages_path)
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan0-e2e",
            "split": "plan_fresh",
            "n_examples": len(examples),
            "n_families": len({ex.family_id for ex in examples}),
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "channel": "text",
            "compressor": "none",
            "message": "A generated compact eligibility; not the evaluator oracle",
            "procedure": "reason-then-answer, frozen after the diagnostic",
            "do_not_resume": [
                "results_stage_plan0",
                "results_stage_plan0_corrected",
                "results_stage_plan0_diag",
            ],
            "held_out": "data/plan_dev.jsonl, data/test.jsonl, hop_test, and plan_test are not loaded",
            "prompt_tuning": "frozen; do not retune for a single development error",
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=40,
        kv_budget=32,
        sink_size=4,
        pca_rank=2,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    extra_manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8")).get("extra", {})
    extra_manifest["model_revision"] = revision.get("sha")
    extra_manifest["model_revision_source"] = revision.get("source")
    write_manifest(out_root / "manifest.json", policy=policy, extra=extra_manifest)
    _persist(out_root, kaggle_dest)

    by_id = {ex.example_id: ex for ex in all_examples}

    def ensure_a_message(ex) -> dict:
        if ex.example_id in messages:
            return messages[ex.example_id]
        prompt = agent_a_message_prompt(ex)
        if "P: eligible" in prompt or "Q: ineligible" in prompt or "R: eligible" in prompt:
            raise RuntimeError("A prompt still contains filled eligibility examples.")
        if prompt_contains_b_private(prompt, ex):
            raise RuntimeError("A's message prompt leaked B's prices.")
        text, n_new, truncated, latency = runner.generate_text_message(
            prompt, max_new_tokens=args.sender_text_max_new
        )
        rec = {
            "example_id": ex.example_id,
            "family_id": ex.family_id,
            "variant": ex.variant,
            "prompt": prompt,
            "message_text": text,
            "n_new_tokens": n_new,
            "truncated": truncated,
            "sender_generation_time_s": latency,
            "message_bytes": len(text.encode("utf-8")),
            **score_eligibility_message(text, gold_eligibility_statuses(ex)),
        }
        messages[ex.example_id] = rec
        with messages_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    needed = set()
    for ex in examples:
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        kinds = {job["condition"] for job in remaining}
        if kinds:
            needed.add(ex.example_id)
        if "mismatch_reason" in kinds:
            needed.add(mismatch_source_id(ex, all_examples))
    print(f"Generating A messages for {len(needed)} examples", flush=True)
    for eid in sorted(needed):
        if eid not in by_id:
            print(f"Missing mismatch source {eid}")
            return 2
        ensure_a_message(by_id[eid])
        _persist(out_root, kaggle_dest)

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
                "sender_generation_time_s": 0.0,
                "sender_truncated": False,
                "n_sender_new_tokens": 0,
                "message_bytes": 0,
                "message_text": "",
                "mismatch_source_id": None,
                "message_source": None,
                "procedure": "reason",
                "max_new_tokens": args.max_new_tokens,
                "plan_time_s": 0.0,
                "n_plan_tokens": 0,
                "plan_truncated": False,
                "plan_text": "",
            }
            if cond == A_COND:
                rec = _a_row(ex, ensure_a_message(ex), extra)
            else:
                if cond == "collab_reason":
                    msg = ensure_a_message(ex)
                    extra["message_source"] = "generated_a"
                else:
                    src_id = mismatch_source_id(ex, all_examples)
                    msg = ensure_a_message(by_id[src_id])
                    extra["mismatch_source_id"] = src_id
                    extra["message_source"] = "mismatched_a"
                message = msg["message_text"]
                extra["message_text"] = message
                extra["message_bytes"] = msg["message_bytes"]
                extra["sender_generation_time_s"] = msg["sender_generation_time_s"]
                extra["sender_truncated"] = msg["truncated"]
                extra["n_sender_new_tokens"] = msg["n_new_tokens"]
                extra["a_all_status_correct"] = msg["all_status_correct"]
                extra["a_n_status_correct"] = msg["n_status_correct"]
                extra["a_incorrect_old_template_copy"] = msg["incorrect_old_template_copy"]
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
                if prompt_contains_a_private(prompt, ex):
                    raise RuntimeError(f"{cond} leaked A's raw operational sheet.")
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
                rec = _record(ex, condition=cond, prompt=prompt, text=text, extra=extra)

            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: correct={rec.get('correct')} trap={rec.get('follows_trap')} "
                f"status={rec.get('n_status_correct')} "
                f"bad_template={rec.get('incorrect_old_template_copy')} "
                f"plan={extra['plan_time_s']:.3f} n_plan={extra['n_plan_tokens']}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Do not drop individual failures. Do not retune prompts. This is not Study 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
