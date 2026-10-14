"""Study 0: does the delivery-planning task measure collaboration?

Text only. No new compressor. Development split only. Do not load
data/test.jsonl, hop_test, or a plan_test file. Do not drop individual
failures after looking at the rows.
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
from src.parse_plan import flags_from_plan_kind, label_plan_answer, parse_plan_answer
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan_task import (
    assert_plan_invariants,
    evaluate,
    load_plan_jsonl,
    mismatch_source_id,
    oracle_intermediate_message,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan import (
    a_alone_prompt,
    agent_a_message_prompt,
    b_alone_prompt,
    b_answer_after_plan_prompt,
    b_planning_prompt,
    b_with_message_prompt,
    pooled_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
)
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

CONDS = (
    "a_alone",
    "b_alone",
    "collab_text",
    "oracle_to_b",
    "pooled",
    "mismatch",
    "b_extra_compute",
)
A_MESSAGE_MAX_NEW = 256
B_ANSWER_MAX_NEW = 48
B_PLAN_MAX_NEW = 128


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_core(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        src = out_root / "a_messages.jsonl"
        if src.is_file():
            (dest / "a_messages.jsonl").write_bytes(src.read_bytes())


def _jobs():
    return [{"condition": name} for name in CONDS]


def _process_time_s(row: dict) -> float:
    return float(
        (row.get("sender_generation_time_s") or 0.0)
        + (row.get("plan_time_s") or 0.0)
        + (row.get("receiver_latency_s") or 0.0)
    )


def _by_example_condition(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    return grouped


def _cond_block(items: list[dict]) -> dict:
    n = len(items)
    n_true = sum(bool(item.get("correct")) for item in items)
    n_trap = sum(bool(item.get("follows_trap")) for item in items)
    n_parse = sum(bool(item.get("parse_ok")) for item in items)
    sizes = [int(item.get("message_bytes") or 0) for item in items]
    return {
        "n": n,
        "correct": n_true,
        "accuracy": n_true / n if n else None,
        "follows_trap": n_trap / n if n else None,
        "parse_ok": n_parse / n if n else None,
        "truncated": sum(bool(item.get("truncated")) for item in items) / n if n else None,
        "sender_truncated": sum(bool(item.get("sender_truncated")) for item in items) / n if n else None,
        "feasible_affordable": (
            sum(bool(item.get("feasible_affordable")) for item in items) / n if n else None
        ),
        "mean_message_bytes": (sum(sizes) / n) if n else None,
        "mean_sender_generation_time_s": (
            sum(item.get("sender_generation_time_s") or 0 for item in items) / n if n else None
        ),
        "mean_plan_time_s": (sum(item.get("plan_time_s") or 0 for item in items) / n if n else None),
        "mean_n_plan_tokens": (sum(item.get("n_plan_tokens") or 0 for item in items) / n if n else None),
        "mean_receiver_latency_s": (sum(item.get("receiver_latency_s") or 0 for item in items) / n if n else None),
        "mean_process_time_s": (sum(_process_time_s(item) for item in items) / n if n else None),
    }


def _subset(rows: list[dict], **kwargs) -> list[dict]:
    out = rows
    for key, value in kwargs.items():
        out = [row for row in out if row.get(key) == value]
    return out


def _flip_tracking(rows: list[dict], condition: str) -> dict:
    by_family: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("condition") != condition:
            continue
        by_family[row["family_id"]][row["variant"]] = row
    n = 0
    a_changed = a_unchanged = a_new_gold = a_both = 0
    b_changed = b_new_gold = b_both = 0
    for cells in by_family.values():
        if "base" not in cells or "a_flip" not in cells or "b_flip" not in cells:
            continue
        n += 1
        base_pred = cells["base"].get("pred_carrier")
        a_pred = cells["a_flip"].get("pred_carrier")
        b_pred = cells["b_flip"].get("pred_carrier")
        if a_pred and a_pred != base_pred:
            a_changed += 1
        else:
            a_unchanged += 1
        if cells["a_flip"].get("correct"):
            a_new_gold += 1
        if cells["base"].get("correct") and cells["a_flip"].get("correct"):
            a_both += 1
        if b_pred and b_pred != base_pred:
            b_changed += 1
        if cells["b_flip"].get("correct"):
            b_new_gold += 1
        if cells["base"].get("correct") and cells["b_flip"].get("correct"):
            b_both += 1
    return {
        "n_families": n,
        "a_flip_prediction_changed": a_changed / n if n else None,
        "a_flip_prediction_unchanged": a_unchanged / n if n else None,
        "a_flip_matches_new_gold": a_new_gold / n if n else None,
        "a_flip_both_base_and_changed_correct": a_both / n if n else None,
        "b_flip_prediction_changed": b_changed / n if n else None,
        "b_flip_matches_new_gold": b_new_gold / n if n else None,
        "b_flip_both_base_and_changed_correct": b_both / n if n else None,
    }


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped = _by_example_condition(rows)
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    out: dict = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "accuracy": {cond: _cond_block(items) for cond, items in by_cond.items()},
        "by_variant": {},
        "by_difficulty": {},
        "paired": {
            "collab_vs_a_alone": _paired(grouped_plain, "a_alone", "collab_text"),
            "collab_vs_b_alone": _paired(grouped_plain, "b_alone", "collab_text"),
            "oracle_vs_b_alone": _paired(grouped_plain, "b_alone", "oracle_to_b"),
            "oracle_vs_collab": _paired(grouped_plain, "collab_text", "oracle_to_b"),
            "mismatch_vs_collab": _paired(grouped_plain, "collab_text", "mismatch"),
            "extra_b_vs_collab": _paired(grouped_plain, "b_extra_compute", "collab_text"),
            "pooled_vs_collab": _paired(grouped_plain, "collab_text", "pooled"),
        },
        "flip_tracking": {
            cond: _flip_tracking(rows, cond) for cond in ("collab_text", "oracle_to_b", "pooled", "b_alone")
        },
        "timing_note": (
            "Reported timings cover local message preparation and receiver processing, "
            "including generation where applicable. They exclude network transfer."
        ),
        "lead": (
            "Study 0 corrected development rerun: explicit lowest-cost objective, "
            "shared collab/oracle receiver wrapper, two-pass extra B compute, "
            "compact A messages with 256-token allowance. First run in "
            "results_stage_plan0 is preserved and not resumed. "
            "These are 12 families with three versions each, not 36 independent scenarios. "
            "Primary score is cheapest eligible within budget. feasible_affordable is separate. "
            "Do not drop individual failures. No compressor. No held-out plan_test."
        ),
        "a_flip_note": (
            "On a_flip the gold carrier becomes B's cheapest-under-budget carrier. "
            "B-alone can look correct without seeing A's change. Read base collab "
            "(must beat the trap) and b_flip (only B's prices change) as the cleaner "
            "tests that both agents contribute."
        ),
    }
    for variant in ("base", "a_flip", "b_flip"):
        out["by_variant"][variant] = {
            cond: _cond_block(_subset(items, variant=variant)) for cond, items in by_cond.items()
        }
    for difficulty in ("easy", "combine"):
        out["by_difficulty"][difficulty] = {
            cond: _cond_block(_subset(items, difficulty=difficulty)) for cond, items in by_cond.items()
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _load_messages(path: Path) -> dict[str, dict]:
    out = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["example_id"]] = rec
    return out


def _paths():
    data = DATA_DIR / "plan_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan0_corrected" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan0_corrected")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _score(text: str, ex) -> dict:
    parsed = label_plan_answer(
        parse_plan_answer(text),
        gold_letter=ex.gold_letter,
        gold_carrier=ex.gold_carrier,
        trap_letter=ex.trap_letter,
        trap_carrier=ex.trap_carrier,
    )
    flags = flags_from_plan_kind(parsed.kind)
    ev = evaluate(
        {"boxes": ex.boxes, "refrigerated": ex.refrigerated},
        ex.a_facts,
        ex.b_facts,
        difficulty=ex.difficulty,
    )
    pred = parsed.carrier
    feasible = bool(pred) and pred in ev.affordable_eligible
    return {
        "kind": parsed.kind,
        "parse_ok": parsed.parse_ok,
        "pred_letter": parsed.letter,
        "pred_carrier": parsed.carrier,
        "correct": flags["correct"],
        "follows_trap": flags["follows_trap"],
        "true_partial": flags["true_partial"],
        "trap_partial": flags["trap_partial"],
        "feasible_affordable": feasible,
        "raw_text": text,
    }


def _record(ex, *, condition: str, prompt: str, text: str, extra: dict) -> dict:
    scored = _score(text, ex)
    return {
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "split": ex.split,
        "variant": ex.variant,
        "difficulty": ex.difficulty,
        "condition": condition,
        "gold_letter": ex.gold_letter,
        "gold_carrier": ex.gold_carrier,
        "trap_letter": ex.trap_letter,
        "trap_carrier": ex.trap_carrier,
        "prompt": prompt,
        **scored,
        **extra,
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Study 0 delivery-planning collaboration check.")
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
        print("Study 0 runs only on plan_dev. No plan_test file until development is interpreted.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan0_corrected" not in str(out_root):
        print("Refusing to write over the preserved first Study 0 run in results_stage_plan0.")
        return 2
    if not path.is_file():
        print(f"Missing {path}. Run: python -m src.plan_task --force")
        return 2
    all_examples = load_plan_jsonl(path)
    examples = all_examples[: args.max_examples] if args.max_examples is not None else all_examples
    for ex in all_examples:
        if ex.split != "plan_dev":
            print(f"Refusing {ex.example_id} on Study 0.")
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
            "stage": "plan0-plan_dev-corrected",
            "split": "plan_dev",
            "n_examples": len(examples),
            "n_families": len({ex.family_id for ex in examples}),
            "n_examples_in_file": len(all_examples),
            "n_receiver_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "channel": "text",
            "compressor": "none",
            "sender_text_max_new": args.sender_text_max_new,
            "plan_max_new": args.plan_max_new,
            "objective": "lowest-cost eligible carrier within budget",
            "held_out": "data/test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": "results_stage_plan0",
            "oracle": "same receiver wrapper as collab_text; only the message body changes",
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
    if revision.get("sha") and not pinned.get("model", {}).get("revision"):
        pin_model_revision(revision["sha"])
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
        oracle = oracle_intermediate_message(ex)
        if oracle in prompt:
            raise RuntimeError("A's ordinary prompt contained the oracle eligibility message.")
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
        }
        messages[ex.example_id] = rec
        with messages_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    needed_messages = set()
    for ex in examples:
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        kinds = {job["condition"] for job in remaining}
        if kinds & {"collab_text", "mismatch"}:
            needed_messages.add(ex.example_id)
        if "mismatch" in kinds:
            needed_messages.add(mismatch_source_id(ex, all_examples))

    print(f"Generating A messages for {len(needed_messages)} examples", flush=True)
    for eid in sorted(needed_messages):
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
        oracle = oracle_intermediate_message(ex)
        for job in remaining:
            cond = job["condition"]
            extra = {
                "sender_generation_time_s": 0.0,
                "sender_truncated": False,
                "n_sender_new_tokens": 0,
                "message_bytes": 0,
                "message_text": "",
                "mismatch_source_id": None,
                "max_new_tokens": args.max_new_tokens,
                "plan_time_s": 0.0,
                "n_plan_tokens": 0,
                "plan_truncated": False,
                "plan_text": "",
            }
            if cond == "a_alone":
                prompt = a_alone_prompt(ex)
                if prompt_contains_b_private(prompt, ex):
                    raise RuntimeError("a_alone leaked B's prices.")
                max_new = args.max_new_tokens
            elif cond == "b_alone":
                prompt = b_alone_prompt(ex)
                if prompt_contains_a_private(prompt, ex):
                    raise RuntimeError("b_alone leaked A's operations.")
                max_new = args.max_new_tokens
            elif cond == "b_extra_compute":
                plan_prompt = b_planning_prompt(ex)
                if prompt_contains_a_private(plan_prompt, ex):
                    raise RuntimeError("b_extra_compute leaked A's operations.")
                if "Answer: <letter>) <carrier>" in plan_prompt or "Stop after that line." in plan_prompt:
                    raise RuntimeError("Planning pass still asks for a one-line final answer.")
                plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
                    plan_prompt, max_new_tokens=args.plan_max_new
                )
                prompt = b_answer_after_plan_prompt(ex, plan_text)
                extra["plan_time_s"] = plan_time
                extra["n_plan_tokens"] = n_plan
                extra["plan_truncated"] = plan_trunc
                extra["plan_text"] = plan_text
                extra["plan_max_new"] = args.plan_max_new
                max_new = args.max_new_tokens
            elif cond == "pooled":
                prompt = pooled_prompt(ex)
                if not prompt_contains_a_private(prompt, ex) or not prompt_contains_b_private(prompt, ex):
                    raise RuntimeError("pooled is missing one of the private documents.")
                max_new = args.max_new_tokens
            elif cond == "oracle_to_b":
                prompt = b_with_message_prompt(ex, oracle)
                if prompt_contains_a_private(prompt, ex):
                    raise RuntimeError("oracle_to_b leaked A's raw operational sheet.")
                extra["message_text"] = oracle
                extra["message_bytes"] = len(oracle.encode("utf-8"))
                max_new = args.max_new_tokens
            elif cond == "collab_text":
                msg = ensure_a_message(ex)
                prompt = b_with_message_prompt(ex, msg["message_text"])
                if prompt_contains_a_private(prompt, ex):
                    raise RuntimeError("collab_text leaked A's raw operational sheet.")
                if oracle in prompt:
                    raise RuntimeError("collab_text inserted the evaluator oracle.")
                extra["sender_generation_time_s"] = msg["sender_generation_time_s"]
                extra["sender_truncated"] = msg["truncated"]
                extra["n_sender_new_tokens"] = msg["n_new_tokens"]
                extra["message_text"] = msg["message_text"]
                extra["message_bytes"] = msg["message_bytes"]
                max_new = args.max_new_tokens
            elif cond == "mismatch":
                src_id = mismatch_source_id(ex, all_examples)
                src_ex = by_id[src_id]
                msg = ensure_a_message(src_ex)
                prompt = b_with_message_prompt(ex, msg["message_text"])
                if prompt_contains_a_private(prompt, ex):
                    raise RuntimeError("mismatch leaked this example's operational sheet.")
                extra["mismatch_source_id"] = src_id
                extra["sender_generation_time_s"] = msg["sender_generation_time_s"]
                extra["sender_truncated"] = msg["truncated"]
                extra["n_sender_new_tokens"] = msg["n_new_tokens"]
                extra["message_text"] = msg["message_text"]
                extra["message_bytes"] = msg["message_bytes"]
                max_new = args.max_new_tokens
            else:
                raise RuntimeError(f"Unknown condition {cond}")

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
                f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                f"feasible={rec['feasible_affordable']} "
                f"bytes={extra['message_bytes']} gen={extra['sender_generation_time_s']:.3f} "
                f"plan={extra['plan_time_s']:.3f} n_plan={extra['n_plan_tokens']} "
                f"recv={latency:.3f}",
                flush=True,
            )
            report = _summarize(rows)
            (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Do not drop individual failures. If capability is poor, revise on this development split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
