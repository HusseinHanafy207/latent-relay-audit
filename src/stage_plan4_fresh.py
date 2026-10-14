"""Frozen four-way comparison on fresh families, plus collaboration controls.

Unchanged procedures:
- text: compact RESULT lines
- text_full: A's source sheet plus already-written calculations
- kv: full completed cache
- probe_256: question-conditioned whole-cache headwise, sink 4 + 256

Controls: no_message and mismatch (compact text from another family).
Smaller-budget and OBF runs stay development diagnostics.

plan4_fresh only. Do not resume plan4_dev folders. Do not retune prompts.
"""

from __future__ import annotations

import argparse
import json
import os
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
    mismatch_source_id,
)
from src.precision import apply_precision_policy, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    b_need_probe_prompt,
    fuller_text_handoff,
    ordinary_sender_prompt,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_chain import _run_b
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import BUDGET_SCOPE, SINK_SIZE, _family_delta, _run_text_b
from src.stage_plan4_probe import (
    A_WRITTEN_MAX_NEW,
    B_PLAN_MAX_NEW,
    _persist,
    _process_time_s,
    _probe_block,
)
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

CONDS = ("text", "text_full", "kv", "probe_256", "no_message", "mismatch")
CORE_CONDS = ("text", "text_full", "kv", "probe_256")
KV_BUDGET = 256
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
    "results_stage_plan4_compress",
    "results_stage_plan4_probe",
    "results_stage_plan4_budget",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_fresh.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_fresh" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_fresh")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


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
            "30 fresh families with three versions each, not 90 independent scenarios. "
            "Not plan4_dev. Same written A on every branch of an example. "
            "Development reference: compact text 77/90, fuller text 81/90, "
            "full KV 82/90, probe-256 41/90."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": {cond: _probe_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "kv_vs_text": _paired(grouped_plain, "text", "kv"),
            "kv_vs_text_full": _paired(grouped_plain, "text_full", "kv"),
            "text_full_vs_text": _paired(grouped_plain, "text", "text_full"),
            "probe_256_vs_kv": _paired(grouped_plain, "kv", "probe_256"),
            "probe_256_vs_text": _paired(grouped_plain, "text", "probe_256"),
            "probe_256_vs_text_full": _paired(grouped_plain, "text_full", "probe_256"),
            "text_vs_no_message": _paired(grouped_plain, "no_message", "text"),
            "text_vs_mismatch": _paired(grouped_plain, "mismatch", "text"),
        },
        "family_clustered_delta": {
            "kv_vs_text": _family_delta(rows, "text", "kv"),
            "kv_vs_text_full": _family_delta(rows, "text_full", "kv"),
            "text_full_vs_text": _family_delta(rows, "text", "text_full"),
            "probe_256_vs_kv": _family_delta(rows, "kv", "probe_256"),
            "text_vs_no_message": _family_delta(rows, "no_message", "text"),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "lead": (
            "Frozen comparison on fresh families: compact RESULT text, fuller "
            "source-plus-working text, full KV, and question-conditioned KV at "
            "budget 256. no_message and mismatch stay as collaboration controls. "
            "No budget 32/64 or OBF on this split."
        ),
        "budget_note": BUDGET_SCOPE,
        "probe_note": (
            "Probe request asks for capacity, eligibility, and shipment conditions. "
            "It does not include prices or budget. Ranking uses completed-cache keys."
        ),
        "obf_note": "Unused. Smaller-budget and OBF results remain development diagnostics.",
        "timing_note": (
            "mean_process_time_s averages recorded process_time_s "
            "(written A, probe/compression where used, and B). "
            "Network is not measured."
        ),
        "b_prompt": "frozen line-by-line; KV uses the same procedure over relayed memory",
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _probe_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Frozen four-way comparison on plan4_fresh.")
    parser.add_argument("--split", default="plan4_fresh")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_fresh":
        print("Fresh comparison uses plan4_fresh only. Do not load plan4_dev, plan_dev, or a plan_test file.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_fresh" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_fresh.")
        return 2
    if not path.is_file():
        print(f"Missing {path}. Run: python -m src.plan4_task --fresh")
        return 2
    examples = load_plan4_jsonl(path)
    by_id = {ex.example_id: ex for ex in examples}
    for ex in examples:
        if ex.split != "plan4_fresh":
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
            "stage": "plan4-fresh",
            "split": "plan4_fresh",
            "n_examples": len(examples),
            "n_families": 30,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "core": list(CORE_CONDS),
            "controls": ["no_message", "mismatch"],
            "compressor": "headwise",
            "kv_budgets": [KV_BUDGET],
            "sink_size": SINK_SIZE,
            "budget_scope": BUDGET_SCOPE,
            "obf": False,
            "a_path": "ordinary written calculations on every branch; not latent-40",
            "b_prompt": "frozen line-by-line; KV uses relayed-memory wording",
            "probe": "capacity/eligibility/conditions request; no prices or budget",
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan4_dev, plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "diagnostics_not_repeated": ["probe_32", "probe_64", "obf_32", "obf_64", "dummy_token"],
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=0,
        kv_budget=KV_BUDGET,
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
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
        if n_full < n_prompt:
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
            "n_kv_positions": n_full,
            "n_prompt_positions": n_prompt,
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
        states[key] = {"bundle": bundle, "meta": rec, "attentions": None, "probe_s": 0.0, "n_probe": 0}
        return states[key]

    def ensure_probe(pack: dict, ex) -> None:
        if pack["attentions"] is not None:
            return
        probe_text = b_need_probe_prompt(ex)
        if prompt_contains_b_private(probe_text, ex):
            raise RuntimeError("Probe request leaked B's prices.")
        attn, probe_s, n_probe = runner.probe_completed_cache_attentions(pack["bundle"], probe_text)
        cache_len = n_kv_positions(pack["bundle"].cache_cpu)
        if int(attn[0][0].shape[-1]) != cache_len:
            raise RuntimeError("Probe scores are not the completed-cache length.")
        n_prompt = int(pack["bundle"].prompt_mask_cpu.sum().item())
        if cache_len <= n_prompt:
            raise RuntimeError("Completed cache has no written tokens to score.")
        pack["attentions"] = attn
        pack["probe_s"] = probe_s
        pack["n_probe"] = n_probe

    needed = set()
    for ex in examples:
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if remaining:
            needed.add(a_state_key(ex))
        if "mismatch" in remaining:
            needed.add(mismatch_source_id(ex, examples))
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
                rec["compression_time_s"] = 0.0
                rec["probe_time_s"] = 0.0
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
                rec["compression_time_s"] = 0.0
                rec["probe_time_s"] = 0.0
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
                rec["probe_time_s"] = 0.0
            elif cond == "probe_256":
                ensure_probe(pack, ex)
                compressed, wall, core, info = runner.compress_sender(
                    pack["bundle"],
                    kv_budget=KV_BUDGET,
                    attentions=pack["attentions"],
                    sink_size=SINK_SIZE,
                    relay="headwise",
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
                rec["probe_time_s"] = float(pack["probe_s"] or 0.0)
                rec["n_probe_tokens"] = int(pack["n_probe"] or 0)
                rec["compression_core_s"] = core
                rec["selection_time_s"] = float(wall)
                rec.update(info)
                rec["budget_scope"] = BUDGET_SCOPE
                rec["compression_time_s"] = float(wall) + float(pack["probe_s"] or 0.0)
            elif cond == "no_message":
                rec = _run_b(
                    runner,
                    ex,
                    message=None,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["compression_time_s"] = 0.0
                rec["probe_time_s"] = 0.0
            elif cond == "mismatch":
                donor_id = mismatch_source_id(ex, examples)
                donor = by_id[donor_id]
                donor_pack = ensure_a(donor)
                rec = _run_b(
                    runner,
                    ex,
                    message=donor_pack["meta"]["handoff"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["mismatch_source_id"] = donor_id
                rec["compression_time_s"] = 0.0
                rec["probe_time_s"] = 0.0
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
                f"kv={rec.get('n_kv_positions')} kept={rec.get('n_kept')}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Interpret the frozen four-way comparison on these fresh families.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
