"""Development-set calibration. Does not touch the held-out test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from src.gates import evaluate_gates
from src.inventory import assert_no_leak, load_jsonl
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse import flags_from_kind
from src.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR
from src.precision import apply_precision_policy, reset_peak_memory
from src.two_agent import ReceiverResult, TwoAgentInventory


def _record(ex, result: ReceiverResult, extra: dict) -> dict:
    flags = flags_from_kind(result.parsed.kind)
    legacy_flags = flags_from_kind(result.parsed_legacy.kind)
    return {
        "example_id": ex.example_id,
        "split": ex.split,
        "condition": result.condition,
        "relay": result.relay,
        "gold_letter": ex.gold_letter,
        "gold_locker": ex.true_locker,
        "donor_letter": ex.donor_letter,
        "donor_locker": ex.alt_locker,
        "kind": result.parsed.kind,
        "parse_ok": result.parsed.parse_ok,
        "pred_letter": result.parsed.letter,
        "pred_locker": result.parsed.locker,
        "correct": flags["correct"],
        "follows_donor": flags["follows_donor"],
        "true_partial": flags["true_partial"],
        "donor_partial": flags["donor_partial"],
        "kind_legacy": result.parsed_legacy.kind,
        "parse_ok_legacy": result.parsed_legacy.parse_ok,
        "pred_letter_legacy": result.parsed_legacy.letter,
        "pred_locker_legacy": result.parsed_legacy.locker,
        "correct_legacy": legacy_flags["correct"],
        "follows_donor_legacy": legacy_flags["follows_donor"],
        "truncated": result.truncated,
        "n_new_tokens": result.n_new_tokens,
        "n_output_lines": result.n_output_lines,
        "max_new_tokens": result.max_new_tokens,
        "relay_bytes": result.relay_bytes,
        "compression_time_s": result.compression_time_s,
        "compression_core_s": result.compression_core_s,
        "receiver_latency_s": result.receiver_latency_s,
        "sender_latency_s": result.sender_latency_s,
        "peak_gpu_gb": result.peak_gpu_gb,
        "cache_finite": result.cache_finite,
        "obf_finite": result.obf_finite,
        "leaked_true_inventory": result.leaked_true_inventory,
        "raw_text": result.raw_text,
        **extra,
    }


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    out = {}
    for cond, items in by_cond.items():
        n = len(items)
        kinds = Counter(item["kind"] for item in items)
        n_true = sum(item["kind"] == "true" for item in items)
        n_donor = sum(item["kind"] == "donor" for item in items)
        n_true_p = sum(item["kind"] == "true_partial" for item in items)
        n_donor_p = sum(item["kind"] == "donor_partial" for item in items)
        out[cond] = {
            "n": n,
            "accuracy": n_true / n if n else None,
            "follow_donor": n_donor / n if n else None,
            "true_partial": n_true_p / n if n else None,
            "donor_partial": n_donor_p / n if n else None,
            "parse_fail": sum(not item["parse_ok"] for item in items) / n if n else None,
            "truncated": sum(item["truncated"] for item in items) / n if n else None,
            "mean_n_new_tokens": sum(item["n_new_tokens"] for item in items) / n if n else None,
            "mean_n_output_lines": sum(item.get("n_output_lines", 0) for item in items) / n if n else None,
            "kinds": dict(kinds),
            "mean_relay_mb": (sum(item["relay_bytes"] for item in items) / n / (1024 ** 2)) if n else None,
            "mean_compression_time_s": sum(item["compression_time_s"] for item in items) / n if n else None,
            "mean_compression_core_s": sum(item["compression_core_s"] for item in items) / n if n else None,
            "mean_receiver_latency_s": sum(item["receiver_latency_s"] for item in items) / n if n else None,
            "any_nonfinite_cache": any(not item["cache_finite"] for item in items),
            "any_nonfinite_obf": any(not item["obf_finite"] for item in items),
            "any_leak": any(item["leaked_true_inventory"] for item in items),
            "peak_gpu_gb_max": max((item["peak_gpu_gb"] or 0) for item in items),
        }
    filler = out.get("full_sender_only_filler", {})
    no_fill = out.get("full_sender_only_no_filler", {})
    head_f = out.get("headwise_sender_only_filler", {})
    head_n = out.get("headwise_sender_only_no_filler", {})
    if filler and no_fill and filler.get("accuracy") is not None and no_fill.get("accuracy") is not None:
        out["filler_sensitivity_full"] = {
            "acc_with_filler": filler["accuracy"],
            "acc_without_filler": no_fill["accuracy"],
            "delta": filler["accuracy"] - no_fill["accuracy"],
        }
    if filler and head_f and filler.get("accuracy") is not None and head_f.get("accuracy") is not None:
        out["compression_penalty_sender_only_filler"] = {
            "full": filler["accuracy"],
            "headwise": head_f["accuracy"],
            "penalty": filler["accuracy"] - head_f["accuracy"],
        }
    if no_fill and head_n and no_fill.get("accuracy") is not None and head_n.get("accuracy") is not None:
        out["compression_penalty_sender_only_no_filler"] = {
            "full": no_fill["accuracy"],
            "headwise": head_n["accuracy"],
            "penalty": no_fill["accuracy"] - head_n["accuracy"],
        }
    hobf_n = out.get("hobf_sender_only_no_filler", {})
    if no_fill and hobf_n and no_fill.get("accuracy") is not None and hobf_n.get("accuracy") is not None:
        out["compression_penalty_obf_no_filler"] = {
            "full": no_fill["accuracy"],
            "hobf": hobf_n["accuracy"],
            "penalty": no_fill["accuracy"] - hobf_n["accuracy"],
        }
        if head_n.get("accuracy") is not None:
            out["obf_recovery_no_filler"] = hobf_n["accuracy"] - head_n["accuracy"]
    hobf_f = out.get("hobf_sender_only_filler", {})
    if filler and hobf_f and head_f.get("accuracy") is not None and hobf_f.get("accuracy") is not None:
        out["obf_recovery_filler"] = hobf_f["accuracy"] - head_f["accuracy"]
    none_f = out.get("no_relay_filler", {})
    if none_f.get("accuracy") is not None:
        out["no_relay_baseline"] = {
            "accuracy": none_f["accuracy"],
            "note": "Empirical no-evidence baseline with filler. Do not substitute 0.25; report majority-class rate from gold letters.",
        }
    return out


def _persist_checkpoint(out_root: Path, extra_dest: Path | None = None) -> None:
    """Copy running files to a stable folder after every example."""
    names = ("evaluations.jsonl", "summary.json", "gates.json", "manifest.json", "bootstrap.json")
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _jobs(_ex=None) -> list[dict]:
    return [
        {"condition": "direct_text", "relay": "direct_text", "sender": "none", "inventory": "true", "filler": False},
        {"condition": "full_sender_only_filler", "relay": "full", "sender": "true", "inventory": None, "filler": True},
        {"condition": "full_sender_only_no_filler", "relay": "full", "sender": "true", "inventory": None, "filler": False},
        {"condition": "mismatch_full_sender_only", "relay": "full", "sender": "alt", "inventory": None, "filler": True},
        {"condition": "headwise_sender_only_filler", "relay": "headwise", "sender": "true", "inventory": None, "filler": True},
        {"condition": "headwise_sender_only_no_filler", "relay": "headwise", "sender": "true", "inventory": None, "filler": False},
        {"condition": "hobf_sender_only_filler", "relay": "hobf_fast", "sender": "true", "inventory": None, "filler": True},
        {"condition": "no_relay_filler", "relay": "none", "sender": "none", "inventory": None, "filler": True},
        {"condition": "hobf_sender_only_no_filler", "relay": "hobf_fast", "sender": "true", "inventory": None, "filler": False},
    ]


def _load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Development calibration. Refuses the held-out test split.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_steps", type=int, default=40)
    parser.add_argument("--kv_budget", type=int, default=32)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--pca_rank", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--only", default=None, help="comma-separated conditions to run")
    parser.add_argument(
        "--prior",
        default=str(REPO_ROOT / "results_stage_b" / "evaluations.jsonl"),
        help="existing evaluations.jsonl to merge and skip",
    )
    args = parser.parse_args()
    if args.split != "dev":
        print("Held-out test is frozen until extra development checks finish. Use --split dev.")
        return 2

    only = {item.strip() for item in args.only.split(",")} if args.only else None
    if only:
        known = {job["condition"] for job in _jobs()}
        unknown = only - known
        if unknown:
            print(f"Unknown --only conditions: {sorted(unknown)}")
            return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    examples = load_jsonl(DATA_DIR / "dev.jsonl")
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    for ex in examples:
        if ex.split != "dev":
            print(f"Refusing non-dev example {ex.example_id}. Do not load the held-out split.")
            return 2
        assert_no_leak(ex)
    gold_counts = Counter(ex.gold_letter for ex in examples)
    majority_n = max(gold_counts.values()) if gold_counts else 0
    majority_rate = majority_n / len(examples) if examples else None

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = RESULTS_DIR / "stage_b" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    extra_run = bool(only)
    kaggle_dest = None
    if Path("/kaggle/working").exists():
        kaggle_dest = Path("/kaggle/working/results_stage_b_extra" if extra_run else "/kaggle/working/results_stage_b")

    prior_rows = _load_rows(Path(args.prior))
    rows: list[dict] = list(prior_rows)
    if prior_rows:
        jsonl_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prior_rows),
            encoding="utf-8",
        )
        print(f"Loaded {len(prior_rows)} prior rows from {args.prior}")
    done = {(row["example_id"], row["condition"]) for row in rows}

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=args.latent_steps,
        kv_budget=args.kv_budget,
        sink_size=args.sink_size,
        pca_rank=args.pca_rank,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha") and not load_pinned().get("model", {}).get("revision"):
        pin_model_revision(revision["sha"])
        print(f"Pinned model revision {revision['sha']}")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "B-extra" if extra_run else "B",
            "split": "dev",
            "n_examples": len(examples),
            "held_out_untouched": True,
            "max_new_tokens": args.max_new_tokens,
            "latent_steps": args.latent_steps,
            "kv_budget": args.kv_budget,
            "sink_size": args.sink_size,
            "pca_rank": args.pca_rank,
            "only": sorted(only) if only else None,
            "model_revision": revision.get("sha"),
            "model_revision_source": revision.get("source"),
            "gold_letter_counts": dict(gold_counts),
            "majority_class_rate": majority_rate,
            "timing": "compression_time_s is compress() wall; compression_core_s is the operator; receiver_latency_s is B decode",
        },
    )
    _persist_checkpoint(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        jobs = [job for job in _jobs(ex) if only is None or job["condition"] in only]
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
            relay = "direct_text" if job["relay"] == "none" else job["relay"]
            result = runner.run_receiver(
                ex,
                condition=job["condition"],
                relay=relay,
                sender=sender,
                inventory_text=inventory,
                filler_text=filler,
            )
            rec = _record(
                ex,
                result,
                extra={
                    "true_full_relay_bytes": true_sender.full_relay_bytes if true_sender is not None else None,
                    "sender_source": sender.source if sender is not None else None,
                },
            )
            rows.append(rec)
            done.add((ex.example_id, job["condition"]))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _persist_checkpoint(out_root, kaggle_dest)
            print(
                f"  {job['condition']}: kind={result.parsed.kind} truncated={result.truncated} "
                f"n_tok={result.n_new_tokens} lines={result.n_output_lines} "
                f"relay_mb={result.relay_bytes / (1024 ** 2):.2f} "
                f"compression_s={result.compression_time_s:.3f} "
                f"compression_core_s={result.compression_core_s:.3f} "
                f"receiver_s={result.receiver_latency_s:.2f}",
                flush=True,
            )

        n_done = len({row["example_id"] for row in rows})
        summary = _summarize(rows)
        summary["dev_label_balance"] = {
            "gold_letter_counts": dict(gold_counts),
            "majority_class_rate": majority_rate,
        }
        gates = evaluate_gates(summary, n_examples=min(n_done, len(examples)))
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_root / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
        _persist_checkpoint(out_root, kaggle_dest)

    summary = _summarize(rows)
    summary["dev_label_balance"] = {
        "gold_letter_counts": dict(gold_counts),
        "majority_class_rate": majority_rate,
    }
    gates = evaluate_gates(summary, n_examples=len(examples))
    print(json.dumps(summary, indent=2))
    print(json.dumps(gates, indent=2))
    print(f"Wrote {jsonl_path}")
    extra_done = bool(
        summary.get("no_relay_filler", {}).get("n") == len(examples)
        and summary.get("hobf_sender_only_no_filler", {}).get("n") == len(examples)
    )
    if extra_done:
        print("Extra development conditions finished. Do not open data/test.jsonl until they are interpreted.")
        return 0
    if not gates["all_passed"]:
        print("Development gates not passed. Do not open data/test.jsonl.")
        return 1
    print("Original development gates passed; extra no-relay / OBF-no-filler cells still missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
