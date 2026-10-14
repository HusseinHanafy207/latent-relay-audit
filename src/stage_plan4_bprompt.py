"""Compare two oracle-to-B procedures on all of plan4_dev.

b_oracle_brief: previous few-sentence plan (first calibration).
b_oracle_lines: one short line per carrier, then a Choice line.

No A generation. No compressor. Do not tune on the six diagnostic families
alone. Preserve plan4_cal and plan4_diag.
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
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import LEVELS, assert_plan4_invariants, load_plan4_jsonl, oracle_intermediate_message
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    b_answer_after_brief_plan_prompt,
    b_answer_after_message_plan_prompt,
    b_reason_brief_prompt,
    b_reason_with_message_prompt,
    prompt_contains_a_private,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _subset
from src.stage_plan4_cal import _b_block, _b_row, _process_time_s
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

CONDS = ("b_oracle_brief", "b_oracle_lines")
PLAN_MAX_NEW = 128
PRESERVED = (
    "results_stage_plan4_cal",
    "results_stage_plan4_diag",
    "results_stage_plan4_cachecheck",
    "results_stage_study1",
)


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_bprompt" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_bprompt")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_core(out_root, extra_dest)


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "unit_note": "90 variants from 30 families. Oracle intermediates, not A's messages.",
        "receiver": {cond: _b_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired": {"lines_vs_brief": _paired(grouped_plain, "b_oracle_brief", "b_oracle_lines")},
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "lead": (
            "Oracle-to-B procedure comparison on all of plan4_dev. "
            "brief is the previous few-sentence plan. "
            "lines is one short eligibility/price/candidate line per carrier. "
            "Do not start compression or six-condition communication from this run."
        ),
        "deferred": ["full sender calibration repeat", "A-message-to-B", "mismatch control", "compression"],
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            cond: _b_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Oracle-B brief vs line plan on plan4_dev.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("B-prompt comparison uses plan4_dev only.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_bprompt" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_bprompt.")
        return 2
    examples = load_plan4_jsonl(path)
    for ex in examples:
        assert_plan4_invariants(ex)

    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-bprompt",
            "n_examples": len(examples),
            "conditions": list(CONDS),
            "plan_max_new": args.plan_max_new,
            "do_not_resume": list(PRESERVED),
            "deferred": ["full sender calibration", "A actual messages to B", "compression"],
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

    builders = {
        "b_oracle_brief": (b_reason_brief_prompt, b_answer_after_brief_plan_prompt),
        "b_oracle_lines": (b_reason_with_message_prompt, b_answer_after_message_plan_prompt),
    }

    for i, ex in enumerate(examples):
        remaining = [cond for cond in CONDS if (ex.example_id, cond) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        oracle = oracle_intermediate_message(ex)
        reset_peak_memory()
        for cond in remaining:
            reason_fn, answer_fn = builders[cond]
            plan_prompt = reason_fn(ex, oracle)
            if prompt_contains_a_private(plan_prompt, ex):
                raise RuntimeError("Oracle B leaked A's raw sheet.")
            if "Answer: <letter>) <carrier>" in plan_prompt:
                raise RuntimeError("Reasoning pass still asks for a one-line final answer.")
            plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
                plan_prompt, max_new_tokens=args.plan_max_new
            )
            prompt = answer_fn(ex, oracle, plan_text)
            text, n_new, truncated, latency = runner.generate_text_message(
                prompt, max_new_tokens=args.max_new_tokens
            )
            extra = {
                "encode_c0_s": 0.0,
                "rollout_c40_s": 0.0,
                "sender_generation_time_s": 0.0,
                "message_text": oracle,
                "message_bytes": len(oracle.encode("utf-8")),
                "plan_time_s": plan_time,
                "n_plan_tokens": n_plan,
                "plan_truncated": plan_trunc,
                "plan_text": plan_text,
                "plan_max_new": args.plan_max_new,
                "truncated": truncated,
                "n_new_tokens": n_new,
                "receiver_latency_s": latency,
                "max_new_tokens": args.max_new_tokens,
                "peak_gpu_gb": peak_gpu_memory_gb(),
            }
            extra["process_time_s"] = _process_time_s(extra)
            rec = _b_row(ex, prompt=prompt, text=text, extra=extra)
            rec["condition"] = cond
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                f"correct={rec['correct']} trap={rec['follows_trap']} "
                f"plan_trunc={rec.get('plan_truncated')} extract={rec.get('plan_extraction_status')}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
