"""Practical selectors at budget 32: question-name match vs receiver-question probe.

Development set only. One sender cache per example. Does not use the gold
answer or relevant-record index for the practical methods. Oracle evidence
line is a ceiling on the same cache, not a proposed method. OBF is unused.
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

from src.evidence_spans import evidence_line_text, retention_grid, sender_token_spans
from src.inventory import assert_no_leak, load_jsonl
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.precision import apply_precision_policy, reset_peak_memory
from src.question_match import match_record_from_question
from src.stage_b import _record
from src.stage_select import _example_seed, _load_rows, _mean_frac, _paired, _persist
from src.two_agent import TwoAgentInventory

CONDS = (
    "full_sender_only_filler",
    "headwise_sender_only_filler",
    "question_match_sender_only_filler",
    "receiver_probe_sender_only_filler",
    "oracle_evidence_sender_only_filler",
)


def _jobs():
    return [
        {"condition": "full_sender_only_filler", "relay": "full", "selection_mode": None, "kind": "full"},
        {"condition": "headwise_sender_only_filler", "relay": "headwise", "selection_mode": "attention", "kind": "headwise"},
        {
            "condition": "question_match_sender_only_filler",
            "relay": "question_match",
            "selection_mode": "force",
            "kind": "question_match",
        },
        {
            "condition": "receiver_probe_sender_only_filler",
            "relay": "receiver_probe",
            "selection_mode": "attention",
            "kind": "probe",
        },
        {
            "condition": "oracle_evidence_sender_only_filler",
            "relay": "oracle_evidence",
            "selection_mode": "force",
            "kind": "oracle",
        },
    ]


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
            "mean_probe_time_s": _mean_frac(items, cond, "probe_time_s") if items else None,
            "mean_compression_time_s": (
                sum(item["compression_time_s"] for item in items) / n if n else None
            ),
            "peak_gpu_gb_max": max((item.get("peak_gpu_gb") or 0) for item in items) if items else None,
        }
    grouped_plain = dict(grouped)
    out["paired"] = {
        "question_match_vs_headwise": _paired(
            grouped_plain, "headwise_sender_only_filler", "question_match_sender_only_filler"
        ),
        "probe_vs_headwise": _paired(
            grouped_plain, "headwise_sender_only_filler", "receiver_probe_sender_only_filler"
        ),
        "question_match_vs_oracle": _paired(
            grouped_plain, "oracle_evidence_sender_only_filler", "question_match_sender_only_filler"
        ),
        "probe_vs_oracle": _paired(
            grouped_plain, "oracle_evidence_sender_only_filler", "receiver_probe_sender_only_filler"
        ),
        "probe_vs_question_match": _paired(
            grouped_plain, "question_match_sender_only_filler", "receiver_probe_sender_only_filler"
        ),
        "oracle_vs_full": _paired(
            grouped_plain, "full_sender_only_filler", "oracle_evidence_sender_only_filler"
        ),
    }
    out["retention"] = {
        cond: {
            "frac_kept_both": _mean_frac(rows, cond, "frac_kept_both"),
            "frac_kept_line": _mean_frac(rows, cond, "frac_kept_line"),
        }
        for cond in CONDS
        if cond != "full_sender_only_filler"
    }
    match_rows = by_cond.get("question_match_sender_only_filler", [])
    out["question_match"] = {
        "matched_equals_gold_rate": (
            sum(bool(item.get("matched_equals_gold")) for item in match_rows) / len(match_rows)
            if match_rows
            else None
        ),
        "note": "Exact-name matching is unusually easy on this inventory format. If it recovers oracle accuracy, a learned probe has to beat this baseline, not just headwise.",
    }
    out["lead"] = (
        "Can question-name matching or a receiver-question probe obtain the evidence-line "
        "recovery at budget 32? Probe time is extra compute, not extra communication, because "
        "the probe is simulated at the sender on the full cache. Sending that cache to B first "
        "would cancel the communication saving."
    )
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Practical selection: question match vs receiver probe.")
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
        print("This practical selector uses the development split only. Use --split dev.")
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

    out_root = RESULTS_DIR / "stage_practical" / "run"
    kaggle_dest = Path("/kaggle/working/results_stage_practical") if Path("/kaggle/working").exists() else None
    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = RESULTS_DIR / "stage_practical" / stamp
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
            "stage": "practical-select",
            "split": "dev",
            "n_examples": len(examples),
            "kv_budget": args.kv_budget,
            "sink_size": args.sink_size,
            "same_sender_cache": True,
            "n_receiver_evals": len(examples) * len(_jobs()),
            "conditions": list(CONDS),
            "probe": "one question+choices forward pass on A's full cache; discard new KV; select original prompt positions",
            "question_match": "parse package name from the question; keep that inventory line; no gold index",
            "oracle": "ceiling from experiment 1; uses the known relevant line; not a proposed method",
            "obf": "unused",
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
        gold_evidence, _unrelated = sender_token_spans(tokenizer, sender.rendered, ids, ex)
        sink_len = min(args.sink_size, int(sender.prompt_mask_cpu.sum().item()))
        matched = match_record_from_question(
            question=ex.question,
            sender_prompt_text=sender.prompt,
            rendered=sender.rendered,
            input_ids=ids,
            tokenizer=tokenizer,
        )
        matched_budget = matched.spans.budget_tokens(sink_len)
        if len(matched_budget) > args.kv_budget:
            raise RuntimeError(
                f"{ex.example_id}: matched line needs {len(matched_budget)} slots > kv_budget={args.kv_budget}"
            )
        matched_equals_gold = matched.line_text == evidence_line_text(ex)
        seed = _example_seed(ex.example_id)
        probe_attn = None
        probe_time = 0.0
        n_probe = 0
        if any(job["kind"] == "probe" for job in remaining):
            probe_attn, probe_time, n_probe = runner.probe_question_attentions(sender, ex)
            print(f"  probe_tokens={n_probe} probe_s={probe_time:.3f} match_eq_gold={matched_equals_gold}", flush=True)

        for job in remaining:
            force_idx = None
            attention_override = None
            this_probe_time = 0.0
            this_n_probe = 0
            if job["kind"] == "question_match":
                force_idx = list(matched.spans.line)
            elif job["kind"] == "oracle":
                force_idx = list(gold_evidence.line)
            elif job["kind"] == "probe":
                attention_override = probe_attn
                this_probe_time = probe_time
                this_n_probe = n_probe
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
                attention_override=attention_override,
                probe_time_s=this_probe_time,
                n_probe_tokens=this_n_probe,
            )
            meta = result.selection_meta or {}
            extra = {
                "access": "sender_only_filler",
                "kv_budget": args.kv_budget,
                "selection_mode": job["selection_mode"],
                "selector_kind": job["kind"],
                "matched_package": matched.package_name,
                "matched_line": matched.line_text,
                "matched_equals_gold": matched_equals_gold,
                "n_matched_line_tokens": matched.spans.n_line,
                "n_matched_budget_tokens": len(matched_budget),
                "n_forced": meta.get("n_forced"),
                "n_forced_in_budget": meta.get("n_forced_in_budget"),
                "k_eff": meta.get("k_eff"),
                "sink_len": meta.get("sink_len", sink_len),
                "l_history": meta.get("l_history"),
                "l_latent": meta.get("l_latent"),
                "probe_time_s": result.probe_time_s,
                "n_probe_tokens": result.n_probe_tokens,
                "true_full_relay_bytes": sender.full_relay_bytes,
                "sender_source": sender.source,
            }
            matrix_batch = meta.get("selected_prompt_positions_matrix")
            if matrix_batch:
                grid = retention_grid(matrix_batch[0], int(extra["sink_len"]), gold_evidence)
                extra["frac_kept_package"] = grid["frac_kept_package"]
                extra["frac_kept_locker"] = grid["frac_kept_locker"]
                extra["frac_kept_both"] = grid["frac_kept_both"]
                extra["frac_kept_line"] = grid["frac_kept_line"]
                with retention_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "example_id": ex.example_id,
                                "condition": job["condition"],
                                "kind": job["kind"],
                                "sink_len": extra["sink_len"],
                                "k_eff": extra["k_eff"],
                                "matched_equals_gold": matched_equals_gold,
                                "evidence": grid,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            rec = _record(ex, result, extra=extra)
            rows.append(rec)
            done.add((ex.example_id, job["condition"]))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {job['condition']}: kind={result.parsed.kind} "
                f"kept_both={extra.get('frac_kept_both')} "
                f"probe_s={result.probe_time_s:.3f} "
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
