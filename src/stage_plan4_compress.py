"""Fuller text plus two headwise budgets, same written A as full KV.

A writes calculations once. Then:
- text: four RESULT lines
- text_full: A's operational sheet plus that written working (no extra A generation)
- kv: full cache after the written generation
- kv_32 / kv_64: headwise eviction over the entire completed cache
  (sink 4 plus 32 or 64 selected positions per head; no reserved latent tail)

plan4_dev only. Frozen line-B. Do not resume plan4_comm or earlier folders.
Fresh examples wait until this procedure is settled.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse_plan4 import handoff_from_written, score_sender_message, written_working_metrics
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    b_answer_after_message_plan_prompt,
    b_reason_with_message_prompt,
    fuller_text_handoff,
    ordinary_sender_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_cal import _b_row, _process_time_s
from src.stage_plan4_comm import _a_extra, _comm_block, _run_kv_b, _sender_block
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

CONDS = ("text", "text_full", "kv", "kv_32", "kv_64")
A_WRITTEN_MAX_NEW = 768
B_PLAN_MAX_NEW = 128
SINK_SIZE = 4
BUDGETS = {"kv_32": 32, "kv_64": 64}
BUDGET_SCOPE = (
    "entire completed cache (prompt + written tokens); "
    "sink_size 4 initial positions plus kv_budget selected positions per head; "
    "no reserved latent tail (this run has no 40-step rollout)"
)
PRESERVED = (
    "results_stage_plan0",
    "results_stage_plan0_corrected",
    "results_stage_plan0_diag",
    "results_stage_plan0_e2e",
    "results_stage_study1",
    "results_stage_plan4_cal",
    "results_stage_plan4_diag",
    "results_stage_plan4_cachecheck",
    "results_stage_plan4_bprompt",
    "results_stage_plan4_chain",
    "results_stage_plan4_comm",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_compress" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_compress")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_core(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        src = out_root / "a_artifacts.jsonl"
        if src.is_file():
            (dest / "a_artifacts.jsonl").write_bytes(src.read_bytes())


def _family_delta(
    rows: list[dict],
    cond_a: str,
    cond_b: str,
    *,
    n_boot: int = 2000,
    seed: int = 2026,
    key: str = "correct",
) -> dict:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    fam_of: dict[str, str] = {}
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
        fam_of[row["example_id"]] = row["family_id"]
    by_fam: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for eid, cells in grouped.items():
        if cond_a in cells and cond_b in cells:
            by_fam[fam_of[eid]].append((bool(cells[cond_a].get(key)), bool(cells[cond_b].get(key))))
    fids = sorted(by_fam)
    if not fids:
        return {"n_families": 0}

    def delta(sample: list[str]) -> float:
        n = acc_a = acc_b = 0
        for fid in sample:
            for ca, cb in by_fam[fid]:
                n += 1
                acc_a += int(ca)
                acc_b += int(cb)
        return (acc_b / n - acc_a / n) if n else 0.0

    observed = delta(fids)
    rng = random.Random(seed)
    draws = sorted(delta([fids[rng.randrange(len(fids))] for _ in fids]) for _ in range(n_boot))
    lo = draws[int(round(0.025 * (len(draws) - 1)))]
    hi = draws[int(round(0.975 * (len(draws) - 1)))]
    return {
        "n_families": len(fids),
        "n_boot": n_boot,
        "key": key,
        "delta_b_minus_a": observed,
        "ci95": [lo, hi],
        "unit": "family (all variants kept together)",
    }


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    text_rows = by_cond.get("text", [])
    unique_a = [r for r in text_rows if r.get("variant") != "b_flip"]
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in text_rows if r.get("a_state_key")}),
        "unit_note": (
            "30 families with three versions each, not 90 independent scenarios. "
            "Same written A on every branch. text_full is source plus written working, "
            "not extra generation. kv_32/kv_64 apply headwise budgets to the entire "
            "completed cache; actual bytes are reported, not the nominal 32/64 labels. "
            "Comm reference: concise text 77/90, full KV 82/90, mismatch KV 28/90."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": {cond: _comm_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "kv_vs_text": _paired(grouped_plain, "text", "kv"),
            "kv_vs_text_full": _paired(grouped_plain, "text_full", "kv"),
            "text_full_vs_text": _paired(grouped_plain, "text", "text_full"),
            "kv_32_vs_kv": _paired(grouped_plain, "kv", "kv_32"),
            "kv_64_vs_kv": _paired(grouped_plain, "kv", "kv_64"),
        },
        "family_clustered_delta": {
            "kv_vs_text": _family_delta(rows, "text", "kv"),
            "kv_vs_text_full": _family_delta(rows, "text_full", "kv"),
            "text_full_vs_text": _family_delta(rows, "text", "text_full"),
            "kv_32_vs_kv": _family_delta(rows, "kv", "kv_32"),
            "kv_64_vs_kv": _family_delta(rows, "kv", "kv_64"),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "lead": (
            "Same written A. Concise RESULT text, fuller source-plus-working text, "
            "full KV, and headwise budgets 32 and 64 on the entire completed cache. "
            "Frozen line-B. No latent-40 sender path. Fresh split waits."
        ),
        "budget_note": BUDGET_SCOPE,
        "timing_note": (
            "process_time_s charges shared written generation, ranking/compression where used, "
            "and B's plan and answer. mean_b_only_s is B alone. mean_relay_mb is mebibytes "
            "(bytes/1024²): UTF-8 for text conditions, actual kept-cache bytes for KV. "
            "Network transfer is not measured. Some cache cloning sits outside the timers."
        ),
        "deferred": {
            "fresh_split": "After this communication procedure is settled.",
            "hobf": "Not in this bounded comparison; two headwise budgets only.",
        },
        "b_prompt": "frozen line-by-line; KV uses the same procedure over relayed memory",
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _comm_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _run_text_b(runner, ex, *, message: str, plan_max_new: int, answer_max_new: int, allow_a_source: bool):
    plan_prompt = b_reason_with_message_prompt(ex, message)
    if (not allow_a_source) and prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("Text B leaked A's raw sheet.")
    if "Answer: <letter>) <carrier>" in plan_prompt:
        raise RuntimeError("Text reasoning pass still asks for a one-line final answer.")
    plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
        plan_prompt, max_new_tokens=plan_max_new
    )
    prompt = b_answer_after_message_plan_prompt(ex, message, plan_text)
    text, n_new, truncated, latency = runner.generate_text_message(
        prompt, max_new_tokens=answer_max_new
    )
    extra = {
        "channel": "text",
        "message_text": message,
        "message_bytes": len(message.encode("utf-8")),
        "n_kv_positions": 0,
        "n_latent_positions": 0,
        "plan_time_s": plan_time,
        "n_plan_tokens": n_plan,
        "plan_truncated": plan_trunc,
        "plan_text": plan_text,
        "plan_max_new": plan_max_new,
        "truncated": truncated,
        "n_new_tokens": n_new,
        "receiver_latency_s": latency,
        "max_new_tokens": answer_max_new,
        "peak_gpu_gb": peak_gpu_memory_gb(),
        "compression_time_s": 0.0,
    }
    return _b_row(ex, prompt=prompt, text=text, extra=extra)


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="plan4_dev fuller text and two headwise budgets.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Compression comparison uses plan4_dev only. Do not load plan_dev or a plan_test file.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_compress" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_compress.")
        return 2
    if not path.is_file():
        print(f"Missing {path}.")
        return 2
    examples = load_plan4_jsonl(path)
    by_id = {ex.example_id: ex for ex in examples}
    for ex in examples:
        if ex.split != "plan4_dev":
            print(f"Refusing {ex.example_id}.")
            return 2
        assert_plan4_invariants(ex)

    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    artifacts_path = out_root / "a_artifacts.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json", "a_artifacts.jsonl"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    artifacts = _load_messages(artifacts_path)
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-compress",
            "split": "plan4_dev",
            "n_examples": len(examples),
            "n_families": 30,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "compressor": "headwise",
            "kv_budgets": [32, 64],
            "sink_size": SINK_SIZE,
            "budget_scope": BUDGET_SCOPE,
            "a_path": "ordinary written calculations on every branch; not latent-40",
            "b_prompt": "frozen line-by-line; KV uses relayed-memory wording",
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "deferred": ["fresh split after this procedure is settled"],
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=0,
        kv_budget=32,
        sink_size=SINK_SIZE,
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

    states: dict[str, dict] = {}

    def ensure_a(ex) -> dict:
        key = a_state_key(ex)
        if key in states:
            return states[key]
        src_ex = by_id[key]
        encode_prompt = a_local_task_prompt(src_ex)
        written = written_result_handoff()
        ordinary = ordinary_sender_prompt(encode_prompt, written)
        if prompt_contains_b_private(ordinary, src_ex):
            raise RuntimeError("A prompt leaked B's prices.")
        if prompt_contains_solver_labels(ordinary, src_ex):
            raise RuntimeError("A prompt contained solver q/eligibility lines.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text, n_new, truncated, gen_s, bundle = runner.generate_text_keep_cache(
            ordinary, max_new_tokens=args.written_max_new, source="written"
        )
        if int(bundle.latent_steps) != 0:
            raise RuntimeError("Written cache must have zero latent steps.")
        if n_kv_positions(bundle.cache_cpu) < int(bundle.prompt_mask_cpu.sum().item()):
            raise RuntimeError("Written cache is shorter than the prompt.")
        rec = {
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "variant": src_ex.variant,
            "difficulty": src_ex.difficulty,
            "a_state_key": key,
            "encode_prompt": encode_prompt,
            "written_handoff": written,
            "ordinary_prompt": ordinary,
            "written": text,
            "written_n_new": n_new,
            "written_truncated": truncated,
            "written_s": gen_s,
            "written_bytes": len(text.encode("utf-8")),
            "handoff": handoff_from_written(text),
            "fuller_text": fuller_text_handoff(src_ex, text),
            "n_kv_positions": n_kv_positions(bundle.cache_cpu),
            "n_prompt_positions": int(bundle.prompt_mask_cpu.sum().item()),
            "full_relay_bytes": int(bundle.full_relay_bytes),
            "latent_steps": 0,
            "written_score": score_sender_message(
                text, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_working": written_working_metrics(text),
        }
        artifacts[key] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        states[key] = {"bundle": bundle, "meta": rec, "attentions": None, "attention_s": 0.0}
        return states[key]

    def ensure_rank(pack: dict) -> None:
        if pack["attentions"] is not None:
            return
        attn, attn_s = runner.last_query_attentions(pack["bundle"])
        pack["attentions"] = attn
        pack["attention_s"] = attn_s

    needed = set()
    for ex in examples:
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if remaining:
            needed.add(a_state_key(ex))
    print(f"Encoding written A caches for {len(needed)} unique operational sheets", flush=True)
    for eid in sorted(needed):
        src = by_id.get(eid) or next(ex for ex in examples if a_state_key(ex) == eid)
        ensure_a(src)
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        pack = ensure_a(ex)
        extra_a = _a_extra(pack)
        for cond in remaining:
            if cond == "text":
                rec = _run_text_b(
                    runner,
                    ex,
                    message=pack["meta"]["handoff"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    allow_a_source=False,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["handoff"] = "results_lines"
            elif cond == "text_full":
                rec = _run_text_b(
                    runner,
                    ex,
                    message=pack["meta"]["fuller_text"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    allow_a_source=True,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["handoff"] = "source_and_written"
            elif cond == "kv":
                rec = _run_kv_b(
                    runner,
                    ex,
                    pack["bundle"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["compression_time_s"] = 0.0
            elif cond in BUDGETS:
                ensure_rank(pack)
                compressed, wall, core, info = runner.compress_sender(
                    pack["bundle"],
                    kv_budget=BUDGETS[cond],
                    attentions=pack["attentions"],
                    sink_size=SINK_SIZE,
                )
                rec = _run_kv_b(
                    runner,
                    ex,
                    compressed,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["handoff"] = cond
                rec["compression_time_s"] = float(wall) + float(pack["attention_s"] or 0.0)
                rec["compression_core_s"] = core
                rec["attention_time_s"] = pack["attention_s"]
                rec.update(info)
                rec["budget_scope"] = BUDGET_SCOPE
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            rec["process_time_s"] = _process_time_s(rec)
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                f"correct={rec['correct']} bytes={rec.get('message_bytes')} "
                f"kv={rec.get('n_kv_positions')} kept={rec.get('n_kept')} "
                f"plan_trunc={rec.get('plan_truncated')}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Interpret fuller text vs full KV vs the two budgets before a fresh split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
