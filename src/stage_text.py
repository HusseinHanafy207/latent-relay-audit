"""One-record text baseline on development examples.

Matcher finds the inventory sentence from B's question and sends that sentence
as text. No sender KV cache and no latent states. Compares with existing
development full / question-matched KV / question-probed KV results. Not a
new held-out confirmation.
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

from src.inventory import assert_no_leak, load_jsonl
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse import flags_from_kind, label_both
from src.paths import DATA_DIR, RESULTS_DIR
from src.precision import apply_precision_policy, reset_peak_memory
from src.question_match import matched_fact_sentence, package_name_from_question
from src.stage_b import _record
from src.stage_select import _load_rows, _mean_frac, _paired, _persist
from src.two_agent import TwoAgentInventory

CONDS = ("matched_sentence_text",)
KV_COMPARE = (
    "full_sender_only_filler",
    "question_match_sender_only_filler",
    "receiver_probe_sender_only_filler",
)


def _jobs():
    return [{"condition": "matched_sentence_text", "relay": "text", "kind": "text"}]


def _process_time_s(row: dict) -> float:
    return float(
        (row.get("match_time_s") or 0.0)
        + (row.get("sender_latency_s") or 0.0)
        + (row.get("probe_time_s") or 0.0)
        + (row.get("compression_time_s") or 0.0)
        + (row.get("receiver_latency_s") or 0.0)
    )


def _message_bytes(row: dict) -> int:
    if row.get("message_bytes") is not None:
        return int(row["message_bytes"])
    return int(row.get("relay_bytes") or 0)


def _shared_correct(row: dict) -> bool:
    if "raw_text" not in row:
        return bool(row.get("correct"))
    current, _legacy = label_both(
        row.get("raw_text") or "",
        gold_letter=row["gold_letter"],
        gold_locker=int(row["gold_locker"]),
        donor_letter=row["donor_letter"],
        donor_locker=int(row["donor_locker"]),
    )
    return flags_from_kind(current.kind)["correct"]


def _find_practical_jsonl() -> Path | None:
    candidates = [
        REPO_ROOT / "results_stage_practical" / "evaluations.jsonl",
        RESULTS_DIR / "stage_practical" / "run" / "evaluations.jsonl",
        RESULTS_DIR / "stage_practical" / "latest" / "evaluations.jsonl",
        Path("/kaggle/working/results_stage_practical/evaluations.jsonl"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _cond_block(items: list[dict], *, shared: bool) -> dict:
    n = len(items)
    if shared:
        n_true = sum(_shared_correct(item) for item in items)
    else:
        n_true = sum(bool(item.get("correct")) for item in items)
    times = [_process_time_s(item) for item in items]
    sizes = [_message_bytes(item) for item in items]
    return {
        "n": n,
        "correct": n_true,
        "accuracy": n_true / n if n else None,
        "truncated": sum(bool(item.get("truncated")) for item in items) / n if n else None,
        "mean_message_bytes": (sum(sizes) / n) if n else None,
        "mean_message_mb": (sum(sizes) / n / (1024 ** 2)) if n else None,
        "mean_process_time_s": (sum(times) / n) if n else None,
        "peak_gpu_gb_max": max((item.get("peak_gpu_gb") or 0) for item in items) if items else None,
    }


def _summarize(rows: list[dict], practical_rows: list[dict] | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_cond[row["condition"]].append(row)
        grouped[row["example_id"]][row["condition"]] = row
    out: dict = {"accuracy": {}}
    for cond in CONDS:
        items = by_cond.get(cond, [])
        out["accuracy"][cond] = _cond_block(items, shared=True)
        out["accuracy"][cond]["accuracy_legacy"] = _cond_block(items, shared=False)["accuracy"]
        out["accuracy"][cond]["mean_match_time_s"] = _mean_frac(items, cond, "match_time_s") if items else None
        out["accuracy"][cond]["mean_receiver_latency_s"] = (
            sum(item["receiver_latency_s"] for item in items) / len(items) if items else None
        )
    comparison = {}
    if practical_rows:
        for row in practical_rows:
            grouped[row["example_id"]][row["condition"]] = row
        practical_by: dict[str, list[dict]] = defaultdict(list)
        for row in practical_rows:
            practical_by[row["condition"]].append(row)
        for cond in KV_COMPARE:
            items = practical_by.get(cond, [])
            block = _cond_block(items, shared=True)
            original = _cond_block(items, shared=False)
            block["accuracy_original"] = original["accuracy"]
            block["correct_original"] = original["correct"]
            block["source"] = "results_stage_practical"
            comparison[cond] = block
        comparison["matched_sentence_text"] = out["accuracy"]["matched_sentence_text"]
        comparison["matched_sentence_text"]["source"] = "this_run"
        grouped_plain = dict(grouped)
        out["paired"] = {
            "text_vs_full": _paired(grouped_plain, "full_sender_only_filler", "matched_sentence_text"),
            "text_vs_match_kv": _paired(
                grouped_plain, "question_match_sender_only_filler", "matched_sentence_text"
            ),
            "text_vs_probe_kv": _paired(
                grouped_plain, "receiver_probe_sender_only_filler", "matched_sentence_text"
            ),
        }
        # Pairing uses stored `correct`. Practical jsonl is the original parser;
        # text rows use the shared extractor. Also report shared-parser pairing.
        shared_grouped: dict[str, dict[str, dict]] = defaultdict(dict)
        for eid, cells in grouped_plain.items():
            shared_grouped[eid] = {
                cond: {**cell, "correct": _shared_correct(cell)} for cond, cell in cells.items()
            }
        out["paired_shared"] = {
            "text_vs_full": _paired(dict(shared_grouped), "full_sender_only_filler", "matched_sentence_text"),
            "text_vs_match_kv": _paired(
                dict(shared_grouped), "question_match_sender_only_filler", "matched_sentence_text"
            ),
            "text_vs_probe_kv": _paired(
                dict(shared_grouped), "receiver_probe_sender_only_filler", "matched_sentence_text"
            ),
        }
    out["comparison"] = comparison
    out["lead"] = (
        "Development check, not held-out evidence. If the retrieved sentence matches "
        "KV accuracy, B does not need latent states for this one-record task. Message "
        "bytes are UTF-8 of the sentence vs relay_bytes of the cache. KV timings are "
        "from the existing practical run (same T4 class, not the same session)."
    )
    out["question"] = (
        "For this simple inventory lookup, is the matched sentence as text enough for B, "
        "or does B need the latent KV message?"
    )
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Matched-sentence text baseline on development examples.")
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
        print("This text baseline uses data/dev.jsonl only. Use --split dev. Do not load test.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path = DATA_DIR / "dev.jsonl"
    if not path.is_file():
        print("Missing data/dev.jsonl.")
        return 2
    examples = load_jsonl(path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    for ex in examples:
        if ex.split != "dev":
            print(f"Refusing non-dev example {ex.example_id}.")
            return 2
        assert_no_leak(ex)

    out_root = RESULTS_DIR / "stage_text" / "run"
    kaggle_dest = Path("/kaggle/working/results_stage_text") if Path("/kaggle/working").exists() else None
    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = RESULTS_DIR / "stage_text" / stamp
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
    practical_path = _find_practical_jsonl()
    practical_rows = _load_rows(practical_path) if practical_path is not None else []
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "text-baseline",
            "split": "dev",
            "n_examples": len(examples),
            "n_receiver_evals": len(examples) * len(_jobs()),
            "conditions": list(CONDS),
            "sender_kv": False,
            "latent_states": False,
            "compare_to": "existing development full / question_match KV / receiver_probe KV",
            "practical_jsonl": str(practical_path) if practical_path else None,
            "held_out": "not a new confirmation; data/test.jsonl is not loaded",
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
        asked = package_name_from_question(ex.question)
        t_match = time.perf_counter()
        sentence = matched_fact_sentence(ex.question, ex.true_inventory_text)
        match_time = time.perf_counter() - t_match
        gold_sentence = f"Package {ex.queried_package} is in locker {ex.true_locker}."
        matched_equals_gold = sentence == gold_sentence
        message_bytes = len(sentence.encode("utf-8"))
        print(
            f"  asked={asked} sentence={sentence!r} eq_gold={matched_equals_gold} bytes={message_bytes}",
            flush=True,
        )
        for job in remaining:
            result = runner.run_receiver(
                ex,
                condition=job["condition"],
                relay=job["relay"],
                sender=None,
                inventory_text=None,
                filler_text=None,
                retrieved_sentence=sentence,
            )
            extra = {
                "access": "matched_sentence_text",
                "selector_kind": job["kind"],
                "asked_package": asked,
                "matched_sentence": sentence,
                "matched_equals_gold": matched_equals_gold,
                "message_bytes": message_bytes,
                "match_time_s": match_time,
                "sender_kv": False,
                "latent_states": False,
            }
            rec = _record(ex, result, extra=extra)
            rows.append(rec)
            done.add((ex.example_id, job["condition"]))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {job['condition']}: kind={result.parsed.kind} "
                f"legacy={result.parsed_legacy.kind} "
                f"msg_bytes={message_bytes} recv_s={result.receiver_latency_s:.3f}",
                flush=True,
            )
            report = _summarize(rows, practical_rows)
            report["n_rows"] = len(rows)
            report["n_examples_seen"] = len({row["example_id"] for row in rows})
            (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows, practical_rows)
    report["n_rows"] = len(rows)
    report["n_examples_seen"] = len({row["example_id"] for row in rows})
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    if not practical_rows:
        print("No results_stage_practical/evaluations.jsonl found; comparison table is empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
