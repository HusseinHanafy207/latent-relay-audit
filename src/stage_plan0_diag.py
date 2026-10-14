"""Study 0 diagnostic: A status scoring and B oracle format × procedure.

Does not resume results_stage_plan0 or results_stage_plan0_corrected.
Same 36 development examples. Oracle messages only for B. No compressor.
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
    oracle_intermediate_message,
    oracle_sentence_message,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan import (
    agent_a_message_prompt,
    b_answer_after_message_plan_prompt,
    b_reason_with_message_prompt,
    b_with_message_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
)
from src.stage_plan0 import (
    A_MESSAGE_MAX_NEW,
    B_ANSWER_MAX_NEW,
    B_PLAN_MAX_NEW,
    _cond_block,
    _load_messages,
    _persist,
    _process_time_s,
    _record,
    _subset,
)
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

A_COND = "a_status"
B_CONDS = (
    "compact_direct",
    "compact_reason",
    "sentence_direct",
    "sentence_reason",
)
CONDS = (A_COND, *B_CONDS)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan0_diag" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan0_diag")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _a_status_fields(text: str, ex) -> dict:
    return score_eligibility_message(text, gold_eligibility_statuses(ex))


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
        "n_families": 12,
        "unit_note": "12 families with three versions each, not 36 independent scenarios.",
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
            "mean_n_new_tokens": (
                sum(int(r.get("n_new_tokens") or 0) for r in a_rows) / n_a if n_a else None
            ),
        },
        "accuracy": {cond: _cond_block(items) for cond, items in by_cond.items() if cond != A_COND},
        "by_variant": {},
        "by_difficulty": {},
        "paired": {
            "sentence_direct_vs_compact_direct": _paired(grouped_plain, "compact_direct", "sentence_direct"),
            "compact_reason_vs_compact_direct": _paired(grouped_plain, "compact_direct", "compact_reason"),
            "sentence_reason_vs_sentence_direct": _paired(grouped_plain, "sentence_direct", "sentence_reason"),
            "sentence_reason_vs_compact_reason": _paired(grouped_plain, "compact_reason", "sentence_reason"),
        },
        "timing_note": (
            "Reported timings cover local message preparation and receiver processing. "
            "Reason conditions include a planning pass that sees the oracle message and prices. "
            "That is not b_extra_compute, which has no eligibility message."
        ),
        "lead": (
            "Study 0 diagnostic, not Study 1. Neutral A template; score A's three "
            "eligibility labels. B sees oracle messages only: compact vs sentences, "
            "each with direct answer vs reason-then-answer. Preserve results_stage_plan0 "
            "and results_stage_plan0_corrected. Do not drop individual failures."
        ),
    }
    for variant in ("base", "a_flip", "b_flip"):
        out["by_variant"][variant] = {
            "a_status_all_three_correct": (
                sum(bool(r.get("all_status_correct")) for r in _subset(a_rows, variant=variant))
                / max(1, len(_subset(a_rows, variant=variant)))
                if a_rows
                else None
            ),
            **{
                cond: _cond_block(_subset(items, variant=variant))
                for cond, items in by_cond.items()
                if cond != A_COND
            },
        }
    for difficulty in ("easy", "combine"):
        out["by_difficulty"][difficulty] = {
            "a_status_all_three_correct": (
                sum(bool(r.get("all_status_correct")) for r in _subset(a_rows, difficulty=difficulty))
                / max(1, len(_subset(a_rows, difficulty=difficulty)))
                if a_rows
                else None
            ),
            **{
                cond: _cond_block(_subset(items, difficulty=difficulty))
                for cond, items in by_cond.items()
                if cond != A_COND
            },
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Study 0 A/B diagnostic on plan_dev.")
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
        print("This diagnostic runs only on plan_dev.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan0_diag" not in str(out_root):
        print("Refusing to write over preserved Study 0 runs.")
        return 2
    if not path.is_file():
        print(f"Missing {path}.")
        return 2
    all_examples = load_plan_jsonl(path)
    examples = all_examples[: args.max_examples] if args.max_examples is not None else all_examples
    for ex in all_examples:
        if ex.split != "plan_dev":
            print(f"Refusing {ex.example_id}.")
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
            "stage": "plan0-diag",
            "split": "plan_dev",
            "n_examples": len(examples),
            "n_families": len({ex.family_id for ex in examples}),
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "b_design": "2 message formats x 2 receiver procedures, oracle messages only",
            "channel": "text",
            "compressor": "none",
            "sender_text_max_new": args.sender_text_max_new,
            "plan_max_new": args.plan_max_new,
            "do_not_resume": ["results_stage_plan0", "results_stage_plan0_corrected"],
            "held_out": "data/test.jsonl, hop_test, and plan_test are not loaded",
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
            **_a_status_fields(text, ex),
        }
        messages[ex.example_id] = rec
        with messages_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    for i, ex in enumerate(examples):
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        compact = oracle_intermediate_message(ex)
        sentences = oracle_sentence_message(ex)
        if compact == sentences:
            raise RuntimeError("Compact and sentence oracles must differ.")
        for job in remaining:
            cond = job["condition"]
            extra = {
                "sender_generation_time_s": 0.0,
                "sender_truncated": False,
                "n_sender_new_tokens": 0,
                "message_bytes": 0,
                "message_text": "",
                "message_format": None,
                "procedure": None,
                "max_new_tokens": args.max_new_tokens,
                "plan_time_s": 0.0,
                "n_plan_tokens": 0,
                "plan_truncated": False,
                "plan_text": "",
            }
            if cond == A_COND:
                msg = ensure_a_message(ex)
                rec = {
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
                    "correct": msg["correct"],
                    "message_text": msg["message_text"],
                    "message_bytes": msg["message_bytes"],
                    "sender_generation_time_s": msg["sender_generation_time_s"],
                    "sender_truncated": msg["truncated"],
                    "n_sender_new_tokens": msg["n_new_tokens"],
                }
            else:
                message = compact if cond.startswith("compact_") else sentences
                extra["message_text"] = message
                extra["message_bytes"] = len(message.encode("utf-8"))
                extra["message_format"] = "compact" if cond.startswith("compact_") else "sentence"
                extra["procedure"] = "reason" if cond.endswith("_reason") else "direct"
                if extra["procedure"] == "direct":
                    prompt = b_with_message_prompt(ex, message)
                    max_new = args.max_new_tokens
                else:
                    plan_prompt = b_reason_with_message_prompt(ex, message)
                    if "Answer: <letter>) <carrier>" in plan_prompt or "Stop after that line." in plan_prompt:
                        raise RuntimeError("Reasoning pass still asks for a one-line final answer.")
                    if message not in plan_prompt:
                        raise RuntimeError("Reasoning pass omitted the eligibility message.")
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
                    max_new = args.max_new_tokens
                if prompt_contains_a_private(prompt, ex):
                    raise RuntimeError(f"{cond} leaked A's raw operational sheet.")
                text, n_new, truncated, latency = runner.generate_text_message(prompt, max_new_tokens=max_new)
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
                f"status={rec.get('n_status_correct')} template={rec.get('copied_old_template')} "
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
    print("Do not drop individual failures. This diagnostic is not Study 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
