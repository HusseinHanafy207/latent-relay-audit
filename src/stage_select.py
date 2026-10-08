"""Diagnostic: does keeping the evidence line repair budget-32 relay?

Development set only. One sender cache per example, reused across methods.
Forced tokens occupy the existing kv_budget. This is not a proposed compressor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evidence_spans import (
    match_slot_count,
    retention_grid,
    sender_token_spans,
)
from src.inventory import assert_no_leak, load_jsonl
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.precision import apply_precision_policy, reset_peak_memory
from src.stage_b import _record
from src.two_agent import TwoAgentInventory

CONDS = (
    "full_sender_only_filler",
    "headwise_sender_only_filler",
    "random_sender_only_filler",
    "evidence_line_sender_only_filler",
    "unrelated_line_sender_only_filler",
)


def _example_seed(example_id: str) -> int:
    return 2026 + (zlib.adler32(example_id.encode("utf-8")) % 1_000_003)


def _jobs():
    return [
        {"condition": "full_sender_only_filler", "relay": "full", "selection_mode": None, "force": None},
        {"condition": "headwise_sender_only_filler", "relay": "headwise", "selection_mode": "attention", "force": None},
        {"condition": "random_sender_only_filler", "relay": "random", "selection_mode": "random", "force": None},
        {
            "condition": "evidence_line_sender_only_filler",
            "relay": "evidence_line",
            "selection_mode": "force",
            "force": "evidence",
        },
        {
            "condition": "unrelated_line_sender_only_filler",
            "relay": "unrelated_line",
            "selection_mode": "force",
            "force": "unrelated",
        },
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


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    names = ("evaluations.jsonl", "summary.json", "manifest.json", "retention.jsonl")
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _paired(grouped: dict, a: str, b: str) -> dict:
    n = fix = hurt = both_ok = both_bad = 0
    for cells in grouped.values():
        if a not in cells or b not in cells:
            continue
        n += 1
        ca = bool(cells[a]["correct"])
        cb = bool(cells[b]["correct"])
        if cb and not ca:
            fix += 1
        elif ca and not cb:
            hurt += 1
        elif ca and cb:
            both_ok += 1
        else:
            both_bad += 1
    acc_a = (both_ok + hurt) / n if n else None
    acc_b = (both_ok + fix) / n if n else None
    return {
        "n": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta_b_minus_a": (acc_b - acc_a) if n else None,
        "b_fixes": fix,
        "b_hurts": hurt,
        "both_ok": both_ok,
        "both_bad": both_bad,
    }


def _mean_frac(rows: list[dict], cond: str, key: str) -> float | None:
    vals = [row[key] for row in rows if row["condition"] == cond and row.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_cond[row["condition"]].append(row)
        grouped[row["example_id"]][row["condition"]] = row
    out: dict = {"accuracy": {}}
    for cond in CONDS:
        items = by_cond.get(cond, [])
        n = len(items)
        n_true = sum(bool(item["correct"]) for item in items)
        out["accuracy"][cond] = {
            "n": n,
            "correct": n_true,
            "accuracy": n_true / n if n else None,
            "truncated": sum(bool(item["truncated"]) for item in items) / n if n else None,
            "mean_relay_mb": (sum(item["relay_bytes"] for item in items) / n / (1024 ** 2)) if n else None,
            "any_leak": any(item.get("leaked_true_inventory") for item in items),
            "peak_gpu_gb_max": max((item.get("peak_gpu_gb") or 0) for item in items) if items else None,
        }
    grouped_plain = dict(grouped)
    out["paired"] = {
        "evidence_vs_headwise": _paired(grouped_plain, "headwise_sender_only_filler", "evidence_line_sender_only_filler"),
        "unrelated_vs_headwise": _paired(grouped_plain, "headwise_sender_only_filler", "unrelated_line_sender_only_filler"),
        "evidence_vs_unrelated": _paired(grouped_plain, "unrelated_line_sender_only_filler", "evidence_line_sender_only_filler"),
        "random_vs_headwise": _paired(grouped_plain, "headwise_sender_only_filler", "random_sender_only_filler"),
        "headwise_vs_full": _paired(grouped_plain, "full_sender_only_filler", "headwise_sender_only_filler"),
    }
    out["budget_slots"] = {
        "mean_evidence_line_tokens": _mean_frac(rows, "evidence_line_sender_only_filler", "n_evidence_line_tokens"),
        "mean_evidence_budget_tokens": _mean_frac(rows, "evidence_line_sender_only_filler", "n_evidence_budget_tokens"),
        "mean_unrelated_forced": _mean_frac(rows, "unrelated_line_sender_only_filler", "n_unrelated_forced"),
        "mean_k_eff": _mean_frac(rows, "headwise_sender_only_filler", "k_eff"),
    }
    out["ordinary_headwise_retention"] = {
        "frac_kept_package": _mean_frac(rows, "headwise_sender_only_filler", "frac_kept_package"),
        "frac_kept_locker": _mean_frac(rows, "headwise_sender_only_filler", "frac_kept_locker"),
        "frac_kept_both": _mean_frac(rows, "headwise_sender_only_filler", "frac_kept_both"),
        "frac_kept_line": _mean_frac(rows, "headwise_sender_only_filler", "frac_kept_line"),
        "note": "Fraction of (layer, head) cells that kept every token of the span. Finding the tokens on some other head does not count.",
    }
    out["lead"] = (
        "Diagnostic on development examples. If evidence-line retention helps and unrelated-line "
        "retention does not, there is recoverable headroom in selection. Do not say the whole "
        "problem is selection even if evidence-line hits 95%."
    )
    out["interpretation"] = {
        "evidence_helps_unrelated_does_not": "recoverable headroom in which record is kept",
        "both_help": "coherent spans or allocation may matter more than the specific evidence line",
        "neither_helps": "keeping the original line is not enough; inspect how B uses the cache before another selector",
        "random_matches_or_beats_headwise": "sender-attention ranking may be poorly suited to this task",
    }
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Evidence-preserving selection diagnostic on dev.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_steps", type=int, default=40)
    parser.add_argument("--kv_budget", type=int, default=32)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "dev":
        print("This diagnostic uses the development split only. Use --split dev.")
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
            print(f"Refusing non-dev example {ex.example_id}.")
            return 2
        assert_no_leak(ex)

    out_root = RESULTS_DIR / "stage_select" / "run"
    kaggle_dest = Path("/kaggle/working/results_stage_select") if Path("/kaggle/working").exists() else None
    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = RESULTS_DIR / "stage_select" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    retention_path = out_root / "retention.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json", "retention.jsonl"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "select-diagnostic",
            "split": "dev",
            "n_examples": len(examples),
            "kv_budget": args.kv_budget,
            "sink_size": args.sink_size,
            "same_sender_cache": True,
            "forced_uses_existing_budget": True,
            "n_receiver_evals": len(examples) * len(_jobs()),
            "conditions": list(CONDS),
            "note": "Diagnostic intervention, not a proposed compression method. OBF is not used.",
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=args.latent_steps,
        kv_budget=args.kv_budget,
        sink_size=args.sink_size,
        pca_rank=2,
        max_new_tokens=args.max_new_tokens,
    )
    tokenizer = runner.wrapper.tokenizer
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
    _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        jobs = _jobs()
        remaining = [job for job in jobs if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        sender = runner.run_sender(ex.true_inventory_text, ex, source="true")
        ids = sender.input_ids_cpu[0].tolist()
        evidence, unrelated = sender_token_spans(tokenizer, sender.rendered, ids, ex)
        sink_len = min(args.sink_size, int(sender.prompt_mask_cpu.sum().item()))
        evidence_budget = evidence.budget_tokens(sink_len)
        if len(evidence_budget) > args.kv_budget:
            raise RuntimeError(
                f"{ex.example_id}: evidence line needs {len(evidence_budget)} budget slots > kv_budget={args.kv_budget}"
            )
        unrelated_force = match_slot_count(unrelated, len(evidence_budget), sink_len)
        seed = _example_seed(ex.example_id)
        print(
            f"  evidence_line_tokens={evidence.n_line} budget={len(evidence_budget)} "
            f"package={evidence.n_package} locker={evidence.n_locker} "
            f"unrelated_forced={len(unrelated_force)}",
            flush=True,
        )

        for job in remaining:
            force_idx = None
            forced_spans = None
            if job["force"] == "evidence":
                force_idx = list(evidence.line)
                forced_spans = evidence
            elif job["force"] == "unrelated":
                force_idx = list(unrelated_force)
                forced_spans = unrelated
            result = runner.run_receiver(
                ex,
                condition=job["condition"],
                relay=job["relay"],
                sender=sender,
                inventory_text=None,
                filler_text=ex.filler_text,
                kv_budget=args.kv_budget,
                selection_mode=job["selection_mode"],
                force_prompt_idx=force_idx,
                rng_seed=seed,
            )
            meta = result.selection_meta or {}
            extra = {
                "access": "sender_only_filler",
                "kv_budget": args.kv_budget,
                "selection_mode": job["selection_mode"],
                "n_evidence_line_tokens": evidence.n_line,
                "n_evidence_budget_tokens": len(evidence_budget),
                "n_unrelated_forced": len(unrelated_force),
                "n_forced": meta.get("n_forced"),
                "n_forced_in_budget": meta.get("n_forced_in_budget"),
                "k_eff": meta.get("k_eff"),
                "sink_len": meta.get("sink_len", sink_len),
                "l_history": meta.get("l_history"),
                "l_latent": meta.get("l_latent"),
                "rng_seed": seed,
                "queried_package": ex.queried_package,
                "true_locker": ex.true_locker,
                "unrelated_record_index": unrelated.record_index,
                "true_full_relay_bytes": sender.full_relay_bytes,
                "sender_source": sender.source,
            }
            matrix_batch = meta.get("selected_prompt_positions_matrix")
            if matrix_batch:
                grid = retention_grid(matrix_batch[0], int(extra["sink_len"]), evidence)
                extra["frac_kept_package"] = grid["frac_kept_package"]
                extra["frac_kept_locker"] = grid["frac_kept_locker"]
                extra["frac_kept_both"] = grid["frac_kept_both"]
                extra["frac_kept_line"] = grid["frac_kept_line"]
                ret = {
                    "example_id": ex.example_id,
                    "condition": job["condition"],
                    "sink_len": extra["sink_len"],
                    "k_eff": extra["k_eff"],
                    "mode": job["selection_mode"],
                    "n_forced_in_budget": extra["n_forced_in_budget"],
                    "evidence": grid,
                }
                if forced_spans is not None:
                    ret["forced_span"] = retention_grid(matrix_batch[0], int(extra["sink_len"]), forced_spans)
                with retention_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ret, ensure_ascii=False) + "\n")
            rec = _record(ex, result, extra=extra)
            rows.append(rec)
            done.add((ex.example_id, job["condition"]))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {job['condition']}: kind={result.parsed.kind} "
                f"kept_both={extra.get('frac_kept_both')} "
                f"relay_mb={result.relay_bytes / (1024 ** 2):.2f}",
                flush=True,
            )
            report = _summarize(rows)
            report["n_rows"] = len(rows)
            report["n_examples_seen"] = len({row["example_id"] for row in rows})
            (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    report["n_rows"] = len(rows)
    report["n_examples_seen"] = len({row["example_id"] for row in rows})
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
