"""Two-record hop experiment: no filler in any receiver condition.

A encodes the inventory without the question. The question is available for
selection. B never sees the complete inventory. hop_dev is the 10-example
correction rerun; hop_test is the 40-example evaluation after that check.
Does not load the Stage C test split or rewrite one-record files.
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

from src.hop_inventory import assert_hop_no_leak, hop_evidence_absent_from_prompt, load_hop_jsonl
from src.hop_retrieve import oracle_message, retrieve_two_records, retrieved_message
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.precision import apply_precision_policy, reset_peak_memory
from src.prompts_inventory import probe_prompt, sender_helpful_message_prompt
from src.stage_b import _record
from src.stage_select import _load_rows, _paired, _persist
from src.two_agent import TwoAgentInventory

SENDER_TEXT_MAX_NEW = 128

DEV_CONDS = (
    "full_kv",
    "probe_kv_32",
    "probe_kv_64",
    "iterative_text",
    "sender_text",
    "oracle_text",
    "no_message",
)
EVAL_CONDS = (
    "full_kv",
    "probe_kv_32",
    "probe_kv_64",
    "iterative_text",
    "sender_text",
    "oracle_text",
)
CONDS = DEV_CONDS


def _jobs(*, include_no_message: bool = True):
    jobs = [
        {"condition": "full_kv", "relay": "full", "selection_mode": None, "kind": "full", "kv_budget": 32},
        {
            "condition": "probe_kv_32",
            "relay": "receiver_probe",
            "selection_mode": "attention",
            "kind": "probe",
            "kv_budget": 32,
        },
        {
            "condition": "probe_kv_64",
            "relay": "receiver_probe",
            "selection_mode": "attention",
            "kind": "probe",
            "kv_budget": 64,
        },
        {"condition": "iterative_text", "relay": "text", "selection_mode": None, "kind": "iterative_text", "kv_budget": None},
        {"condition": "sender_text", "relay": "text", "selection_mode": None, "kind": "sender_text", "kv_budget": None},
        {"condition": "oracle_text", "relay": "text", "selection_mode": None, "kind": "oracle_text", "kv_budget": None},
    ]
    if include_no_message:
        jobs.append(
            {"condition": "no_message", "relay": "none", "selection_mode": None, "kind": "no_message", "kv_budget": None}
        )
    return jobs


def _process_time_s(row: dict) -> float:
    return float(
        (row.get("prep_time_s") or 0.0)
        + (row.get("selection_time_s") or 0.0)
        + (row.get("compression_time_s") or 0.0)
        + (row.get("sender_generation_time_s") or 0.0)
        + (row.get("receiver_latency_s") or 0.0)
    )


def _message_bytes(row: dict) -> int:
    if row.get("message_bytes") is not None:
        return int(row["message_bytes"])
    return int(row.get("relay_bytes") or 0)


def _cond_block(items: list[dict]) -> dict:
    n = len(items)
    n_true = sum(bool(item.get("correct")) for item in items)
    n_parse = sum(bool(item.get("parse_ok")) for item in items)
    sizes = [_message_bytes(item) for item in items]
    found_vals = [item["found_both"] for item in items if "found_both" in item]
    return {
        "n": n,
        "correct": n_true,
        "accuracy": n_true / n if n else None,
        "parse_ok": n_parse / n if n else None,
        "truncated": sum(bool(item.get("truncated")) for item in items) / n if n else None,
        "sender_truncated": sum(bool(item.get("sender_truncated")) for item in items) / n if n else None,
        "mean_message_bytes": (sum(sizes) / n) if n else None,
        "mean_message_mb": (sum(sizes) / n / (1024 ** 2)) if n else None,
        "mean_prep_time_s": (sum(item.get("prep_time_s") or 0 for item in items) / n) if n else None,
        "mean_selection_time_s": (sum(item.get("selection_time_s") or 0 for item in items) / n) if n else None,
        "mean_compression_time_s": (sum(item.get("compression_time_s") or 0 for item in items) / n) if n else None,
        "mean_sender_generation_time_s": (
            sum(item.get("sender_generation_time_s") or 0 for item in items) / n if n else None
        ),
        "mean_receiver_latency_s": (sum(item.get("receiver_latency_s") or 0 for item in items) / n) if n else None,
        "mean_process_time_s": (sum(_process_time_s(item) for item in items) / n) if n else None,
        "found_both": (sum(bool(v) for v in found_vals) / len(found_vals)) if found_vals else None,
    }


def _differing_examples(grouped: dict, a: str, b: str) -> dict:
    a_only = []
    b_only = []
    for eid, cells in grouped.items():
        if a not in cells or b not in cells:
            continue
        ca = bool(cells[a]["correct"])
        cb = bool(cells[b]["correct"])
        if ca and not cb:
            a_only.append(eid)
        elif cb and not ca:
            b_only.append(eid)
    return {"a_correct_b_wrong": a_only, "b_correct_a_wrong": b_only}


def _checklist(by_cond: dict[str, list[dict]], *, split: str) -> dict:
    def acc(name: str) -> float | None:
        items = by_cond.get(name, [])
        if not items:
            return None
        return sum(bool(item.get("correct")) for item in items) / len(items)

    iterative = by_cond.get("iterative_text", [])
    found_both = (
        sum(bool(item.get("found_both")) for item in iterative) / len(iterative) if iterative else None
    )
    no_msg = acc("no_message")
    oracle = acc("oracle_text")
    full = acc("full_kv")
    notes = []
    if oracle is not None and oracle < 0.8:
        notes.append("Oracle text is weak: inspect the task or prompting before comparing compression.")
    if oracle is not None and oracle >= 0.8 and full is not None and full < 0.6:
        notes.append("Oracle text works but full relay does not: inspect the relay first.")
    if full is not None and oracle is not None and full < 0.6 and oracle < 0.8:
        notes.append("Both oracle text and full relay fail: inspect the task or prompting.")
    if found_both is not None and found_both < 1.0:
        notes.append("Iterative retriever missed at least one record.")
    if no_msg is not None and no_msg >= 0.5:
        notes.append("No-message accuracy is high; lockers may be guessable.")
    return {
        "oracle_text_accuracy": oracle,
        "full_relay_accuracy": full,
        "iterative_found_both": found_both,
        "no_message_accuracy": no_msg,
        "b_can_solve_from_two_sentences": oracle,
        "without_a_message_b_is_not_reliable": (no_msg is None) or (no_msg < 0.5),
        "notes": notes,
        "split": split,
        "not_held_out": split == "hop_dev",
        "filler": False,
    }


def _summarize(rows: list[dict], *, split: str) -> dict:
    conds = DEV_CONDS if split == "hop_dev" else EVAL_CONDS
    by_cond: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_cond[row["condition"]].append(row)
        grouped[row["example_id"]][row["condition"]] = row
    out: dict = {"accuracy": {}}
    for cond in conds:
        out["accuracy"][cond] = _cond_block(by_cond.get(cond, []))
    grouped_plain = dict(grouped)
    paired = {
        "probe32_vs_full": _paired(grouped_plain, "full_kv", "probe_kv_32"),
        "probe64_vs_full": _paired(grouped_plain, "full_kv", "probe_kv_64"),
        "iterative_vs_full": _paired(grouped_plain, "full_kv", "iterative_text"),
        "sender_text_vs_full": _paired(grouped_plain, "full_kv", "sender_text"),
        "oracle_vs_full": _paired(grouped_plain, "full_kv", "oracle_text"),
        "iterative_vs_oracle": _paired(grouped_plain, "oracle_text", "iterative_text"),
        "probe32_vs_iterative": _paired(grouped_plain, "iterative_text", "probe_kv_32"),
        "probe64_vs_iterative": _paired(grouped_plain, "iterative_text", "probe_kv_64"),
        "probe32_vs_sender_text": _paired(grouped_plain, "sender_text", "probe_kv_32"),
    }
    if split == "hop_dev":
        paired["no_message_vs_oracle"] = _paired(grouped_plain, "no_message", "oracle_text")
    out["paired"] = paired
    out["differing_examples"] = {
        "probe32_vs_full": _differing_examples(grouped_plain, "full_kv", "probe_kv_32"),
        "iterative_vs_full": _differing_examples(grouped_plain, "full_kv", "iterative_text"),
        "iterative_vs_oracle": _differing_examples(grouped_plain, "oracle_text", "iterative_text"),
        "probe32_vs_iterative": _differing_examples(grouped_plain, "iterative_text", "probe_kv_32"),
        "sender_text_vs_full": _differing_examples(grouped_plain, "full_kv", "sender_text"),
    }
    out["checklist"] = _checklist(by_cond, split=split)
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    out["peak_gpu_note"] = (
        "Peak GPU memory is reset once per example, before that example's conditions. "
        "Do not compare per-row peaks across methods."
    )
    if split == "hop_dev":
        out["lead"] = (
            "No-filler correction on the same 10 hop_dev examples. Old filled results are not resumed. "
            "If retrieval and oracle text stay healthy, run hop_test (40). Small n; do not declare a winner."
        )
    else:
        out["lead"] = (
            "Frozen hop_test evaluation, 40 examples, no filler, no no-message control. "
            "Report paired differences; do not declare a winner from one or two extra correct answers."
        )
    return out


def _paths_for_split(split: str) -> tuple[Path, Path, Path]:
    data = DATA_DIR / f"{split}.jsonl"
    if split == "hop_dev":
        out_root = RESULTS_DIR / "stage_hop_nofiller" / "run"
        kaggle = Path("/kaggle/working/results_stage_hop_nofiller")
    else:
        out_root = RESULTS_DIR / "stage_hop_test" / "run"
        kaggle = Path("/kaggle/working/results_stage_hop_test")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Two-record hop run with no filler.")
    parser.add_argument("--split", default="hop_dev")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_steps", type=int, default=40)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--sender_text_max_new", type=int, default=SENDER_TEXT_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split not in {"hop_dev", "hop_test"}:
        print("Use --split hop_dev or --split hop_test. Do not load the Stage C test split.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths_for_split(args.split)
    if not path.is_file():
        print(f"Missing {path}. Run: python -m src.hop_inventory")
        return 2
    examples = load_hop_jsonl(path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    for ex in examples:
        if ex.split != args.split:
            print(f"Refusing {ex.example_id} on --split {args.split}.")
            return 2
        assert_hop_no_leak(ex)

    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = out_root.parent / stamp
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
    include_no_message = args.split == "hop_dev"
    jobs_all = _jobs(include_no_message=include_no_message)
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": f"hop-{args.split}",
            "split": args.split,
            "n_examples": len(examples),
            "n_receiver_evals": len(examples) * len(jobs_all),
            "conditions": [job["condition"] for job in jobs_all],
            "filler": False,
            "sender_sees_question": False,
            "probe_budgets": [32, 64],
            "sender_text_max_new": args.sender_text_max_new,
            "text_methods_use_latent_rollout": False,
            "held_out_stage_c": "data/test.jsonl is not loaded",
            "old_filled_results": "results_stage_hop is not resumed",
            "obf": "unused",
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=args.latent_steps,
        kv_budget=32,
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
        jobs = _jobs(include_no_message=include_no_message)
        remaining = [job for job in jobs if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        needed = {job["kind"] for job in remaining}
        sender = None
        prep_time = 0.0
        if needed & {"full", "probe"}:
            sender = runner.run_sender(
                ex.true_inventory_text, ex, source="true", include_question=False
            )
            if ex.question in sender.prompt or "Choices:" in sender.prompt:
                raise RuntimeError("Sender prompt leaked the hop question.")
            prep_time = sender.sender_latency_s
        probe_attn = None
        probe_time = 0.0
        n_probe = 0
        if "probe" in needed:
            if sender is None:
                raise RuntimeError("Probe requires a sender cache.")
            probe_attn, probe_time, n_probe = runner.probe_text_attentions(sender, probe_prompt(ex))

        t_ret = time.perf_counter()
        retrieval = retrieve_two_records(ex.question, ex.true_inventory_text)
        retrieve_time = time.perf_counter() - t_ret
        iterative_payload = retrieved_message(retrieval)
        oracle_payload = oracle_message(ex)

        sender_msg = ""
        sender_gen_time = 0.0
        sender_n_new = 0
        sender_truncated = False
        if "sender_text" in needed:
            gen_prompt = sender_helpful_message_prompt(ex.true_inventory_text, ex)
            sender_msg, sender_n_new, sender_truncated, sender_gen_time = runner.generate_text_message(
                gen_prompt, max_new_tokens=args.sender_text_max_new
            )

        for job in remaining:
            kind = job["kind"]
            extra = {
                "access": "sender_only" if kind in {"full", "probe"} else kind,
                "selector_kind": kind,
                "sender_sees_question": False,
                "filler": False,
                "prep_time_s": prep_time if kind in {"full", "probe"} else 0.0,
                "selection_time_s": 0.0,
                "sender_generation_time_s": 0.0,
                "sender_truncated": False,
                "n_sender_new_tokens": 0,
                "kv_budget": job["kv_budget"],
            }
            result = None
            if kind in {"full", "probe"}:
                if sender is None:
                    raise RuntimeError("KV condition requires a sender cache.")
                result = runner.run_receiver(
                    ex,
                    condition=job["condition"],
                    relay=job["relay"],
                    sender=sender,
                    inventory_text=None,
                    filler_text=None,
                    kv_budget=job["kv_budget"],
                    selection_mode=job["selection_mode"],
                    attention_override=probe_attn if kind == "probe" else None,
                    probe_time_s=probe_time if kind == "probe" else 0.0,
                    n_probe_tokens=n_probe if kind == "probe" else 0,
                )
                extra["selection_time_s"] = probe_time if kind == "probe" else 0.0
                extra["message_bytes"] = result.relay_bytes
                extra["n_probe_tokens"] = result.n_probe_tokens
                extra["probe_time_s"] = result.probe_time_s
                if not hop_evidence_absent_from_prompt(result.prompt, ex):
                    raise RuntimeError(f"{job['condition']} leaked hop evidence into B's prompt.")
                if "Warehouse notes" in result.prompt or "filler" in result.prompt.lower():
                    raise RuntimeError(f"{job['condition']} still contains filler.")
            elif kind == "no_message":
                result = runner.run_receiver(
                    ex,
                    condition=job["condition"],
                    relay=job["relay"],
                    sender=None,
                    inventory_text=None,
                    filler_text=None,
                )
                extra["message_bytes"] = 0
                if not hop_evidence_absent_from_prompt(result.prompt, ex):
                    raise RuntimeError("no_message leaked hop evidence into B's prompt.")
                if "Warehouse notes" in result.prompt:
                    raise RuntimeError("no_message still contains filler.")
            else:
                if kind == "iterative_text":
                    payload = iterative_payload
                    source = "iterative"
                    extra["selection_time_s"] = retrieve_time
                    extra["matched_package"] = retrieval.package_name
                    extra["matched_shipment"] = retrieval.shipment_name
                    extra["found_both"] = retrieval.found_both
                    extra["found_package"] = retrieval.found_package
                    extra["found_shipment"] = retrieval.found_shipment
                elif kind == "oracle_text":
                    payload = oracle_payload
                    source = "oracle"
                else:
                    payload = sender_msg
                    source = "sender"
                    extra["sender_generation_time_s"] = sender_gen_time
                    extra["sender_truncated"] = sender_truncated
                    extra["n_sender_new_tokens"] = sender_n_new
                extra["message_text"] = payload
                extra["message_bytes"] = len(payload.encode("utf-8"))
                result = runner.run_receiver(
                    ex,
                    condition=job["condition"],
                    relay=job["relay"],
                    sender=None,
                    inventory_text=None,
                    filler_text=None,
                    retrieved_sentence=payload,
                    text_source=source,
                )
                if ex.true_inventory_text in result.prompt:
                    raise RuntimeError(f"{job['condition']} sent the complete inventory to B.")
            rec = _record(ex, result, extra=extra)
            rec["process_time_s"] = _process_time_s(rec)
            rows.append(rec)
            done.add((ex.example_id, job["condition"]))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {job['condition']}: kind={result.parsed.kind} "
                f"bytes={extra.get('message_bytes')} "
                f"prep={extra['prep_time_s']:.3f} sel={extra['selection_time_s']:.3f} "
                f"comp={result.compression_time_s:.3f} "
                f"gen={extra['sender_generation_time_s']:.3f} recv={result.receiver_latency_s:.3f}",
                flush=True,
            )
            report = _summarize(rows, split=args.split)
            report["n_rows"] = len(rows)
            report["n_examples_seen"] = len({row["example_id"] for row in rows})
            (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows, split=args.split)
    report["n_rows"] = len(rows)
    report["n_examples_seen"] = len({row["example_id"] for row in rows})
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    if args.split == "hop_dev":
        print("If this no-filler checklist is healthy, run: python -m src.stage_hop --split hop_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
