"""Held-out 5×2 comparison. Resumes from evaluations.jsonl if present."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inventory import assert_no_leak, load_jsonl
from src.manifest import load_pinned, write_manifest
from src.paths import DATA_DIR, RESULTS_DIR
from src.precision import apply_precision_policy, reset_peak_memory
from src.model_revision import pin_model_revision, resolve_model_revision
from src.stage_b import _persist_checkpoint, _record, _summarize
from src.stats import (
    MAIN_ACCESSES,
    MAIN_RELAYS,
    SENSITIVITY_CONDS,
    by_example,
    condition_name,
    contrasts,
    paired_bootstrap,
    sensitivity_contrasts,
)
from src.two_agent import TwoAgentInventory

MAIN_SPECS = [
    ("full", "full", 32),
    ("headwise_32", "headwise", 32),
    ("hobf_32", "hobf_fast", 32),
    ("headwise_64", "headwise", 64),
    ("hobf_64", "hobf_fast", 64),
]


SENSITIVITY_SPECS = [
    ("full", "full", 32),
    ("headwise_32", "headwise", 32),
    ("hobf_32", "hobf_fast", 32),
]


def _jobs(ex):
    jobs = [
        {
            "condition": "direct_text",
            "relay": "direct_text",
            "sender": "none",
            "inventory": "true",
            "filler": False,
            "kv_budget": None,
            "access": "direct",
            "main": False,
            "sensitivity": False,
        }
    ]
    for label, compressor, budget in MAIN_SPECS:
        jobs.append(
            {
                "condition": condition_name(label, "sender_only"),
                "relay": compressor,
                "sender": "true",
                "inventory": None,
                "filler": True,
                "kv_budget": budget,
                "access": "sender_only",
                "main": True,
                "sensitivity": False,
            }
        )
        jobs.append(
            {
                "condition": condition_name(label, "shared"),
                "relay": compressor,
                "sender": "true",
                "inventory": "true",
                "filler": False,
                "kv_budget": budget,
                "access": "shared",
                "main": True,
                "sensitivity": False,
            }
        )
    for label, compressor, budget in SENSITIVITY_SPECS:
        jobs.append(
            {
                "condition": condition_name(label, "sender_only_no_filler"),
                "relay": compressor,
                "sender": "true",
                "inventory": None,
                "filler": False,
                "kv_budget": budget,
                "access": "sender_only_no_filler",
                "main": False,
                "sensitivity": True,
            }
        )
    jobs.append(
        {
            "condition": "mismatch_full_sender_only",
            "relay": "full",
            "sender": "alt",
            "inventory": None,
            "filler": True,
            "kv_budget": 32,
            "access": "mismatch",
            "main": False,
            "sensitivity": False,
        }
    )
    return jobs


def _load_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    if not jsonl_path.is_file():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Held-out 5×2 comparison with resume.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_steps", type=int, default=40)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--pca_rank", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "test":
        print("Stage C uses the held-out test split only. Use --split test.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    examples = load_jsonl(DATA_DIR / "test.jsonl")
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    for ex in examples:
        if ex.split != "test":
            print(f"Refusing non-test example {ex.example_id}.")
            return 2
        assert_no_leak(ex)

    out_root = RESULTS_DIR / "stage_c" / "run"
    kaggle_dest = Path("/kaggle/working/results_stage_c") if Path("/kaggle/working").exists() else None
    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = RESULTS_DIR / "stage_c" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "gates.json", "manifest.json", "bootstrap.json"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "C",
            "split": "test",
            "n_examples": len(examples),
            "main": "5 relays × 2 access = 800 receiver evals; inferential unit is the example",
            "sensitivity": "predeclared sender-only no-filler for full, headwise-32, hobf-32 = 240 evals",
            "n_main_plus_sensitivity": 1040,
            "extra_controls": ["direct_text", "mismatch_full_sender_only"],
            "filler": "main sender-only uses filler; no-filler at budget 32 is a predeclared sensitivity, not a fishing expedition",
            "h3": "not included; no reservation rule was frozen on development",
            "model_revision": pinned.get("model", {}).get("revision"),
            "resume_completed": len(done),
            "max_new_tokens": args.max_new_tokens,
        },
    )
    _persist_checkpoint(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=args.latent_steps,
        kv_budget=32,
        sink_size=args.sink_size,
        pca_rank=args.pca_rank,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha") and not pinned.get("model", {}).get("revision"):
        pin_model_revision(revision["sha"])
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            **json.loads((out_root / "manifest.json").read_text(encoding="utf-8")).get("extra", {}),
            "model_revision": revision.get("sha"),
            "model_revision_source": revision.get("source"),
            "resume_completed": len(done),
        },
    )
    _persist_checkpoint(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        jobs = _jobs(ex)
        remaining = [job for job in jobs if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        true_sender = None
        alt_sender = None
        if any(job["sender"] == "true" for job in remaining):
            true_sender = runner.run_sender(ex.true_inventory_text, ex, source="true")
        if any(job["sender"] == "alt" for job in remaining):
            alt_sender = runner.run_sender(ex.alt_inventory_text, ex, source="alt")

        senders = {"true": true_sender, "alt": alt_sender, "none": None}
        for job in remaining:
            sender = senders[job["sender"]]
            inventory = ex.true_inventory_text if job["inventory"] == "true" else None
            filler = ex.filler_text if job["filler"] else None
            result = runner.run_receiver(
                ex,
                condition=job["condition"],
                relay=job["relay"],
                sender=sender,
                inventory_text=inventory,
                filler_text=filler,
                kv_budget=job["kv_budget"],
            )
            rec = _record(
                ex,
                result,
                extra={
                    "access": job["access"],
                    "kv_budget": job["kv_budget"],
                    "main_comparison": job["main"],
                    "sensitivity": job["sensitivity"],
                    "true_full_relay_bytes": true_sender.full_relay_bytes if true_sender is not None else None,
                    "sender_source": sender.source if sender is not None else None,
                },
            )
            rows.append(rec)
            done.add((ex.example_id, job["condition"]))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {job['condition']}: kind={result.parsed.kind} truncated={result.truncated} "
                f"n_tok={result.n_new_tokens} relay_mb={result.relay_bytes / (1024 ** 2):.2f} "
                f"compression_s={result.compression_time_s:.3f} "
                f"receiver_s={result.receiver_latency_s:.2f}",
                flush=True,
            )
            summary = _summarize(rows)
            report = {
                "summary": summary,
                "contrasts": contrasts({k: v["accuracy"] for k, v in summary.items() if isinstance(v, dict) and "accuracy" in v}),
                "n_rows": len(rows),
                "n_examples_seen": len({row["example_id"] for row in rows}),
            }
            (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            _persist_checkpoint(out_root, kaggle_dest)

    summary = _summarize(rows)
    acc_table = {k: v["accuracy"] for k, v in summary.items() if isinstance(v, dict) and "accuracy" in v}
    grouped = by_example(rows)
    boot = paired_bootstrap(
        grouped,
        n_boot=args.n_boot,
        conditions=[condition_name(r, a) for r in MAIN_RELAYS for a in MAIN_ACCESSES] + list(SENSITIVITY_CONDS),
    )
    report = {
        "summary": summary,
        "contrasts": contrasts(acc_table),
        "sensitivity": sensitivity_contrasts(acc_table),
        "bootstrap": boot,
        "n_rows": len(rows),
        "n_examples_seen": len(grouped),
        "main_cells": 10,
        "sensitivity_cells": 3,
        "n_main_plus_sensitivity": 1040,
        "lead": "sender-only accuracy vs relay size for eviction and OBF; shared is interpretive; no-filler budget-32 is a predeclared sensitivity",
    }
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_root / "bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
    _persist_checkpoint(out_root, kaggle_dest)
    print(json.dumps(report["contrasts"], indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
