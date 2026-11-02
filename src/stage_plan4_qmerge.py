"""Merged receiver and fair handoff timing for the frozen query adapter (twelve carriers).

An implementation and timing follow-up. No training: the frozen epoch-1
adapter, the selected states, the support and the prompts stay as they were.

The unmerged adapter runs two extra fp32 matmuls in every adapted layer
(measured: ~7.8% more decode time per token than unadapted compact KV). LoRA
allows folding the update into the projection weights. Here the merged weights
are kept beside the originals and swapped in only for B, so A, support
construction and every baseline still run on the exact original model.

Phases:
- agree: dev sheets (12 base + 12 a_flip). Merged vs unmerged: forced logits on
         the unmerged outputs, eligibility labels, final choices, formatting.
         Registered tolerances; the benchmark needs this to pass.
- integrate: dev sheets, merged receiver. The frozen R (selected and re-rotated
         on the CPU when the states were built) vs the R that A's GPU
         preparation actually produces: tensor differences, forced logits,
         labels, formatting and final choices, through B's standard path and
         through the direct path the benchmark times. The benchmark runs only
         if this agrees within the same tolerances.
- bench: eval sheets reused as timing inputs (not another confirmation), for a
         declared deployment: same_gpu (direct reuse) or device_transfer
         (device-to-host and host-to-device copies; network not measured).
         Six receivers, warm-up, synchronized timing, three repetitions per
         sheet in a random condition order. Sender preparation, cache movement,
         receiver preparation, prefill and decode are reported separately, plus
         the complete handoff. Every condition prefills the prompt tokens its
         plan and answer passes share once; text also gets its public header
         prefilled once, as KV gets its public support.
- textadapt: diagnostic only. The frozen adapter applied to the text receiver.
         A negative result would not show what a text-trained receiver could do.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.stage_plan4_qadapt as sq
from src.cache_utils import kv_blob_size_bytes, n_kv_positions
from src.manifest import load_pinned
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_qadapt_task import ensure_split, file_sha256
from src.plan4_task import a_state_key
from src.plan4_timing_task import load_timing12_jsonl
from src.precision import peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import prompt_contains_a_private
from src.qadapt import argmax_agreement, kl_sum, logit_delta
from src.qmerge import (
    attached_as_built,
    cache_layers,
    cat_layers,
    clone_layers,
    compact_sender,
    extend,
    lcp,
    merge_stats,
    merged_q_weights,
    now,
    q_weight_digest,
    receiver_weights,
    to_device,
    two_pass,
)
from src.span_select import decode_positions
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_kvmix12 import _append, _twelve_choice, _twelve_labels
from src.stage_plan4_query12 import _cluster_delta, _copy_first
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_timing import B_PLAN_MAX_NEW_12, _prompts_for, _row_n, _run_kv_b_timed, _run_text_b_timed
from src.stage_select import _load_rows

REPS = 3
BENCH_SEED = 2026
PLACEHOLDER_PLAN = "Choice: P at $1"
AGREE_MIN_ARGMAX = 0.99
AGREE_MAX_LOSS = {"labels": 1, "all_twelve": 1, "final_choice": 1}
EVAL_MAX_LOSS = {"labels": 2, "all_twelve": 2, "final_choice": 2}
CONDS = ("merged_kv", "unmerged_kv", "unadapted_kv", "fk_fv", "text_result", "full_kv")
SPEC = {
    "merged_kv": {"handoff": "compact", "mode": "merged", "eval_ref": "adapted_correct_r"},
    "unmerged_kv": {"handoff": "compact", "mode": "unmerged", "eval_ref": "adapted_correct_r"},
    "unadapted_kv": {"handoff": "compact", "mode": "unadapted", "eval_ref": "unadapted_correct_r"},
    "fk_fv": {"handoff": "reconstruct", "mode": "unadapted", "eval_ref": "fk_fv"},
    "text_result": {"handoff": "text", "mode": "unadapted", "eval_ref": "text_result"},
    "full_kv": {"handoff": "full", "mode": "unadapted", "eval_ref": "full_kv"},
}
COMPARE_TO = ("text_result", "fk_fv", "full_kv", "unmerged_kv", "unadapted_kv")
COMPONENTS = (
    "sender_prep_s",
    "movement_s",
    "receiver_prep_s",
    "encode_s",
    "r_prefill_s",
    "shared_prefill_s",
    "plan_prefill_s",
    "answer_prefill_s",
    "plan_decode_s",
    "answer_decode_s",
    "crop_s",
)
PREFILL_PARTS = ("r_prefill_s", "shared_prefill_s", "plan_prefill_s", "answer_prefill_s")
DEPLOYMENTS = {
    "same_gpu": (
        "Direct reuse on one GPU: A and B share the device. A's result states are selected and re-rotated on the GPU "
        "and placed straight into B's cache; nothing leaves the device. For full KV, B takes over A's cache; the copy "
        "the benchmark needs to repeat a handoff is timed separately and excluded."
    ),
    "device_transfer": (
        "Transfer between devices, approximated on one T4: after A's preparation, what B receives is copied "
        "device-to-host, then host-to-device before B's preparation (pageable memory, synchronous copies). "
        "The network between hosts is not measured; bytes are reported."
    ),
}
INTEGRATE_RUNS = ("frozen_std", "actual_std", "actual_direct")
QADAPT_INPUTS = {
    "frozen_candidate.json": "qadapt_frozen_candidate.json",
    "dev_evaluations.jsonl": "qadapt_dev_evaluations.jsonl",
    "eval_evaluations.jsonl": "qadapt_eval_evaluations.jsonl",
    "eval_construction.json": "qadapt_eval_construction.json",
    "train_summary.json": "qadapt_train_summary.json",
    "a_artifacts_eval.jsonl": "a_artifacts_eval.jsonl",
    "prep_a_artifacts_dev.jsonl": "prep_a_artifacts_dev.jsonl",
    "prep_timing12_a_artifacts_12.jsonl": "prep_timing12_a_artifacts_12.jsonl",
    "prep_check_summary.json": "prep_check_summary.json",
}
PERSIST_NAMES = (
    "manifest.json",
    "agree_evaluations.jsonl",
    "agree_logits.jsonl",
    "agree_summary.json",
    "bench_checks.jsonl",
    "bench_evaluations.jsonl",
    "bench_summary.json",
    "integrate_evaluations.jsonl",
    "integrate_r.jsonl",
    "integrate_summary.json",
    *(f"bench_{d}_{s}" for d in ("same_gpu", "device_transfer") for s in ("checks.jsonl", "evaluations.jsonl", "summary.json")),
    "textadapt_evaluations.jsonl",
    "textadapt_summary.json",
)
PRESERVED = ("results_stage_plan4_qadapt",) + sq.PRESERVED
SCOPE = (
    "Implementation and timing comparison. Dev sheets were already used for checkpoint selection; "
    "eval sheets were already used for the candidate's evaluation and are reused here as timing inputs. "
    "Neither is a new independent confirmation."
)
TIMING_NOTE = (
    "Single T4. Compact KV is built from the R that A's GPU preparation actually produces (checked against the "
    "frozen R on dev first). Movement depends on the declared deployment (see 'deployment'). Public support (KV) "
    "and the public text header are prefilled once and kept on B's GPU; their one-time cost is reported "
    "separately. Every condition prefills the prompt tokens shared by its plan and answer passes once."
)


def _paths():
    out_root = RESULTS_DIR / "stage_plan4_qmerge" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_qmerge")
    return out_root, (kaggle if Path("/kaggle/working").exists() else None)


def _persist(out_root: Path, extra: Path | None) -> None:
    for dest in [out_root.parent / "latest"] + ([extra] if extra is not None else []):
        dest.mkdir(parents=True, exist_ok=True)
        for name in PERSIST_NAMES + tuple(QADAPT_INPUTS.values()):
            src, out = out_root / name, dest / name
            if src.is_file() and src.resolve() != out.resolve():
                if not (out.is_file() and out.stat().st_size == src.stat().st_size and out.stat().st_mtime >= src.stat().st_mtime):
                    out.write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _copy_inputs(out_root: Path) -> dict:
    dirs = [REPO_ROOT / "results_stage_plan4_qadapt", Path("/kaggle/working/results_stage_plan4_qadapt")]
    prep = [REPO_ROOT / "results_stage_plan4_qadapt_prep", Path("/kaggle/working/results_stage_plan4_qadapt_prep")]
    paths = {}
    for name, local in QADAPT_INPUTS.items():
        sources = [d / name for d in dirs]
        if name.startswith("prep_"):
            sources += [d / name[len("prep_") :] for d in prep]
        paths[name] = _copy_first(out_root, local, sources, what=f"qadapt {name}")
    frozen = json.loads(paths["frozen_candidate.json"].read_text(encoding="utf-8"))
    ckpt = _copy_first(out_root, f"qadapt_{frozen['checkpoint']}", [d / frozen["checkpoint"] for d in dirs], what="frozen checkpoint")
    if file_sha256(ckpt) != frozen["sha256"]:
        raise RuntimeError("The frozen checkpoint's hash changed. Refusing to continue.")
    paths["checkpoint"] = ckpt
    paths["frozen"] = frozen
    return paths


def _qadapt_inputs(paths: dict) -> dict:
    """The names the qadapt stage helpers expect."""
    return {
        "a_artifacts_dev.jsonl": paths["prep_a_artifacts_dev.jsonl"],
        "timing12_a_artifacts_12.jsonl": paths["prep_timing12_a_artifacts_12.jsonl"],
    }


def _load_candidate(runner, paths: dict):
    adapter = sq._adapter(runner)
    adapter.load(torch.load(paths["checkpoint"], map_location="cpu"))
    originals = [m.base.weight for m in adapter.modules]
    digest = q_weight_digest(adapter)
    t0 = now()
    merged = merged_q_weights(adapter)
    merge_s = now() - t0
    return adapter, originals, digest, merged, merge_s


def _weights_intact(adapter, originals, digest) -> dict:
    out = {"wrappers_and_weights_restored": attached_as_built(adapter, originals), "original_digest_equal": q_weight_digest(adapter) == digest}
    out["ok"] = all(out.values())
    return out


# ---------------------------------------------------------------- agree (dev)


def _loss(left: int, right: int) -> int:
    return max(0, int(left) - int(right))


def behaviour_gates(ref_rows: list[dict], test_rows: list[dict], logit_rows: list[dict] | None, intact: dict) -> dict:
    """test against ref on the same sheets; logit_rows=None skips the forced-logit gate."""
    if not test_rows or sorted(r["example_id"] for r in test_rows) != sorted(r["example_id"] for r in ref_rows):
        return {"gates": {}, "label_predictions_changed": None, "passed": False}
    u, m = _twelve_labels(ref_rows), _twelve_labels(test_rows)
    uc, mc = _twelve_choice(ref_rows), _twelve_choice(test_rows)

    def malformed(b):
        return (b["n_cases"] - b["parse_ok"]) + b["plan_truncated"] + b["answer_truncated"] + b["plan_lines_missing"]

    gates = {}
    if logit_rows is not None:
        toks = sum(r["n_tokens"] for r in logit_rows)
        agree = sum(r["argmax_agree_tokens"] for r in logit_rows) / toks if toks else 0.0
        gates["forced_argmax_agreement"] = {"value": agree, "min": AGREE_MIN_ARGMAX, "pass": agree >= AGREE_MIN_ARGMAX}
    gates.update(
        {
            "labels": {"ref": u["labels_correct"], "test": m["labels_correct"], "pass": _loss(u["labels_correct"], m["labels_correct"]) <= AGREE_MAX_LOSS["labels"]},
            "all_twelve": {
                "ref": u["cases_all_twelve_correct"],
                "test": m["cases_all_twelve_correct"],
                "pass": _loss(u["cases_all_twelve_correct"], m["cases_all_twelve_correct"]) <= AGREE_MAX_LOSS["all_twelve"],
            },
            "final_choice": {
                "ref": uc["final_choice_correct"],
                "test": mc["final_choice_correct"],
                "pass": _loss(uc["final_choice_correct"], mc["final_choice_correct"]) <= AGREE_MAX_LOSS["final_choice"],
            },
            "formatting": {"ref": malformed(u), "test": malformed(m), "pass": malformed(m) <= malformed(u)},
            "original_weights_intact": {**intact, "pass": bool(intact.get("ok"))},
        }
    )
    changed = sum(len(d["carriers_changed"]) for d in _differences(ref_rows, test_rows))
    return {"gates": gates, "label_predictions_changed": changed, "passed": all(g["pass"] for g in gates.values())}


def agree_gates(rows: list[dict], logit_rows: list[dict], intact: dict) -> dict:
    return behaviour_gates(
        [r for r in rows if r["mode"] == "unmerged"], [r for r in rows if r["mode"] == "merged"], logit_rows, intact
    )


def _differences(ref_rows: list[dict], test_rows: list[dict]) -> list[dict]:
    ref = {r["example_id"]: r for r in ref_rows}
    out = []
    for m in sorted(test_rows, key=lambda r: r["example_id"]):
        eid = m["example_id"]
        u = ref.get(eid)
        if u is None:
            continue
        if u.get("plan_text") != m.get("plan_text") or u.get("raw_text") != m.get("raw_text"):
            out.append(
                {
                    "example_id": eid,
                    "labels": [u.get("plan_n_status_correct"), m.get("plan_n_status_correct")],
                    "choice": [u.get("correct"), m.get("correct")],
                    "carriers_changed": sorted(
                        c for c in (u.get("plan_pred_status") or {}) if (u["plan_pred_status"] or {}).get(c) != (m.get("plan_pred_status") or {}).get(c)
                    ),
                    "plan_text_equal": u.get("plan_text") == m.get("plan_text"),
                    "answer_text_equal": u.get("raw_text") == m.get("raw_text"),
                }
            )
    return out


def _run_agree(args, out_root, kaggle_dest, splits, runner, paths, protocol) -> int:
    adapter, originals, digest, merged, merge_s = _load_candidate(runner, paths)
    stats = merge_stats(adapter, merged)
    dev, states = sq._dev_setup(runner, splits, _qadapt_inputs(paths))
    saved = {
        r["example_id"]: r
        for r in _load_rows(paths["dev_evaluations.jsonl"])
        if r["condition"] == "adapted_correct_r"
    }
    path, lpath = out_root / "agree_evaluations.jsonl", out_root / "agree_logits.jsonl"
    rows, lrows = _load_rows(path), _load_rows(lpath)
    done = {(r["example_id"], r["mode"]) for r in rows}
    ldone = {r["example_id"] for r in lrows}
    todo = dev if args.limit is None else dev[: max(0, args.limit)]
    for ex in todo:
        bundle = states[ex.example_id]["kvmix"][sq.STUDENT]
        for mode in ("unmerged", "merged"):
            if (ex.example_id, mode) in done:
                continue
            with receiver_weights(adapter, mode, merged):
                rec = _run_kv_b_timed(
                    runner, ex, runner.clone_sender(bundle), plan_max_new=args.plan_max_new_12, answer_max_new=args.max_new_tokens
                )
            ref = saved.get(ex.example_id) or {}
            rec.update(
                {
                    "mode": mode,
                    "equals_saved_plan": rec.get("plan_text") == ref.get("plan_text"),
                    "equals_saved_answer": rec.get("raw_text") == ref.get("raw_text"),
                    "saved_labels": ref.get("plan_n_status_correct"),
                }
            )
            rows.append(rec)
            done.add((ex.example_id, mode))
            _append(path, rec)
            print(f"  {ex.example_id} {mode}: {sq._brief(rec)} same-as-saved={rec['equals_saved_plan']}", flush=True)
        if ex.example_id not in ldone:
            u = next(r for r in rows if r["example_id"] == ex.example_id and r["mode"] == "unmerged")
            plan_prompt = _prompts_for(ex)["kv_plan"]
            seqs = {
                "plan": sq._sequence(runner, plan_prompt, u["plan_text"] or "", truncated=bool(u.get("plan_truncated"))),
                "answer": sq._sequence(runner, u["prompt"], u["raw_text"] or "", truncated=bool(u.get("truncated"))),
            }
            for part, seq in seqs.items():
                if not seq["response_ids"]:
                    continue
                with torch.no_grad():
                    with receiver_weights(adapter, "unmerged"):
                        lu = sq._logits(runner, bundle, seq).float()
                    with receiver_weights(adapter, "merged", merged):
                        lm = sq._logits(runner, bundle, seq).float()
                d = logit_delta(lu, lm)
                n = len(seq["response_ids"])
                lrow = {
                    "example_id": ex.example_id,
                    "part": part,
                    "n_tokens": n,
                    "max_abs": d["max_abs"],
                    "mean_abs": d["mean_abs"],
                    "argmax_agree": d["argmax_agree"],
                    "argmax_agree_tokens": int(round(d["argmax_agree"] * n)),
                    "kl_mean": float(kl_sum(lu, lm)) / n,
                    "unmerged_follows_own_output": argmax_agreement(lu, seq["response_ids"]),
                    "merged_follows_unmerged_output": argmax_agreement(lm, seq["response_ids"]),
                }
                lrows.append(lrow)
                _append(lpath, lrow)
                del lu, lm
            ldone.add(ex.example_id)
        _persist(out_root, kaggle_dest)
    intact = _weights_intact(adapter, originals, digest)
    summary = {
        "scope": SCOPE,
        "candidate": {k: paths["frozen"][k] for k in ("selected_epoch", "sha256")},
        "merge": {"merge_s": merge_s, "stats": stats["worst"], "n_layers": stats["n_layers"]},
        "labels": {
            v: {mode: _twelve_labels([r for r in rows if r["mode"] == mode and r.get("variant") == v]) for mode in ("unmerged", "merged")}
            for v in ("base", "a_flip")
        },
        "final_choice": {
            v: {mode: _twelve_choice([r for r in rows if r["mode"] == mode and r.get("variant") == v]) for mode in ("unmerged", "merged")}
            for v in ("base", "a_flip")
        },
        "logits": {
            part: {
                "n_tokens": sum(r["n_tokens"] for r in lrows if r["part"] == part),
                "max_abs": max((r["max_abs"] for r in lrows if r["part"] == part), default=None),
                "mean_abs": statistics.fmean([r["mean_abs"] for r in lrows if r["part"] == part]) if any(r["part"] == part for r in lrows) else None,
                "kl_mean": statistics.fmean([r["kl_mean"] for r in lrows if r["part"] == part]) if any(r["part"] == part for r in lrows) else None,
            }
            for part in ("plan", "answer")
        },
        "differences": _differences([r for r in rows if r["mode"] == "unmerged"], [r for r in rows if r["mode"] == "merged"]),
        "unmerged_rerun_matches_saved": {
            "plan_text": sum(bool(r["equals_saved_plan"]) for r in rows if r["mode"] == "unmerged"),
            "answer_text": sum(bool(r["equals_saved_answer"]) for r in rows if r["mode"] == "unmerged"),
            "n": sum(r["mode"] == "unmerged" for r in rows),
        },
        "complete": len(rows) == 2 * len(dev),
    }
    summary.update(agree_gates(rows, lrows, intact))
    summary["passed"] = summary["passed"] and summary["complete"]
    _write_json(out_root / "agree_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("merge", "gates", "passed", "unmerged_rerun_matches_saved")}, indent=1))
    return 0


# ---------------------------------------------------------------- integrate (dev)


def _forced_pair(runner, adapter, merged, ref_bundle, test_bundle, seq: dict, part: str, eid: str) -> dict:
    with torch.no_grad(), receiver_weights(adapter, "merged", merged):
        lf = sq._logits(runner, ref_bundle, seq).float()
        la = sq._logits(runner, test_bundle, seq).float()
    d = logit_delta(lf, la)
    n = len(seq["response_ids"])
    return {
        "example_id": eid,
        "part": part,
        "n_tokens": n,
        "equal": d["equal"],
        "max_abs": d["max_abs"],
        "mean_abs": d["mean_abs"],
        "argmax_agree": d["argmax_agree"],
        "argmax_agree_tokens": int(round(d["argmax_agree"] * n)),
        "kl_mean": float(kl_sum(lf, la)) / n,
    }


def integrate_summary(rows: list[dict], r_rows: list[dict], agree_merged: dict, intact: dict, n_dev: int) -> dict:
    by_run = {run: [r for r in rows if r["run"] == run] for run in INTEGRATE_RUNS}
    checks = [r["r_check"] for r in r_rows]
    logit_rows = [lg for r in r_rows for lg in r["logits"]]
    r_cmp = {
        "n_sheets": len(checks),
        "shapes_equal_all": bool(checks) and all(c["shapes_equal"] for c in checks),
        "values_equal_all": bool(checks) and all(c.get("values_equal") for c in checks),
        "sheets_keys_bit_equal": sum(bool(c.get("keys_bit_equal")) for c in checks),
        "keys_max_abs": max((c.get("keys_max_abs") or 0.0 for c in checks), default=None),
        "keys_mean_abs": (sum(c.get("keys_mean_abs") or 0.0 for c in checks) / len(checks)) if checks else None,
        "min_keys_bit_equal_fraction": min((c.get("min_keys_bit_equal_fraction", 0.0) for c in checks), default=None),
    }
    actual = behaviour_gates(by_run["frozen_std"], by_run["actual_std"], logit_rows, intact)
    direct = behaviour_gates(by_run["frozen_std"], by_run["actual_direct"], None, intact)
    actual["gates"]["r_values_and_shapes"] = {"pass": r_cmp["shapes_equal_all"] and r_cmp["values_equal_all"]}
    actual["passed"] = actual["passed"] and actual["gates"]["r_values_and_shapes"]["pass"]
    frozen = by_run["frozen_std"]
    complete = all(len(by_run[run]) == n_dev for run in INTEGRATE_RUNS) and len(r_rows) == n_dev
    if not actual["passed"]:
        cause = "the GPU-prepared R changes B's behaviour beyond the tolerance"
    elif not direct["passed"]:
        cause = "the GPU-prepared R agrees, but the direct path (R kept on the GPU, shared-prefix prefill) changes B's behaviour"
    else:
        cause = None
    return {
        "scope": "Dev sheets, merged receiver. Same checkpoint, support, prompts and selection; only where R is prepared and how it reaches B changes.",
        "runs": {
            "frozen_std": "frozen R (CPU selection and re-rotation at state build), B's standard two-call path",
            "actual_std": "R from A's GPU selection and re-rotation, same standard path (isolates the R difference)",
            "actual_direct": "R from the GPU placed straight into B's cache on the GPU, shared-prefix two-pass generation (the benchmark's same_gpu path)",
        },
        "r_comparison": r_cmp,
        "logits_frozen_vs_actual_r": {
            part: {
                "n_tokens": sum(r["n_tokens"] for r in logit_rows if r["part"] == part),
                "all_equal": all(r["equal"] for r in logit_rows if r["part"] == part),
                "max_abs": max((r["max_abs"] for r in logit_rows if r["part"] == part), default=None),
                "kl_mean_max": max((r["kl_mean"] for r in logit_rows if r["part"] == part), default=None),
            }
            for part in ("plan", "answer")
        },
        "labels": {v: {run: _twelve_labels([r for r in by_run[run] if r.get("variant") == v]) for run in INTEGRATE_RUNS} for v in ("base", "a_flip")},
        "final_choice": {v: {run: _twelve_choice([r for r in by_run[run] if r.get("variant") == v]) for run in INTEGRATE_RUNS} for v in ("base", "a_flip")},
        "frozen_std_reproduces_agree_merged": {
            "plan_text": sum(r.get("plan_text") == (agree_merged.get(r["example_id"]) or {}).get("plan_text") for r in frozen),
            "answer_text": sum(r.get("raw_text") == (agree_merged.get(r["example_id"]) or {}).get("raw_text") for r in frozen),
            "n": len(frozen),
        },
        "actual_r_vs_frozen_r": actual,
        "direct_path_vs_frozen_r": direct,
        "differences": {
            "actual_std": _differences(frozen, by_run["actual_std"]),
            "actual_direct": _differences(frozen, by_run["actual_direct"]),
        },
        "complete": complete,
        "passed": complete and actual["passed"] and direct["passed"],
        "cause": cause,
        "next": (
            "The benchmark feeds B the GPU-prepared R directly."
            if complete and actual["passed"] and direct["passed"]
            else "Document the difference before interpreting deployment timing; the benchmark refuses to run."
        ),
    }


def _run_integrate(args, out_root, kaggle_dest, splits, runner, paths, protocol) -> int:
    agree = _read_json(out_root / "agree_summary.json")
    if agree is None or not agree.get("passed"):
        print("The dev agreement check (merged vs unmerged) must pass first.")
        return 2
    adapter, originals, digest, merged, _merge_s = _load_candidate(runner, paths)
    dev = list(splits["dev"])
    artifacts = _load_messages(paths["prep_a_artifacts_dev.jsonl"])
    if any(a_state_key(ex) not in artifacts for ex in dev):
        raise RuntimeError("Dev A sheets are missing. Do not regenerate them here.")
    frozen_a = _load_messages(paths["prep_timing12_a_artifacts_12.jsonl"])
    timing12 = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    setup = sq._prep_setup(runner, dev, artifacts, frozen_a, timing12)
    agree_merged = {r["example_id"]: r for r in _load_rows(out_root / "agree_evaluations.jsonl") if r["mode"] == "merged"}
    path, rpath = out_root / "integrate_evaluations.jsonl", out_root / "integrate_r.jsonl"
    rows, r_rows = _load_rows(path), _load_rows(rpath)
    done = {(r["example_id"], r["run"]) for r in rows}
    r_done = {r["example_id"] for r in r_rows}
    todo = [ex for ex in dev if ex.example_id not in r_done or any((ex.example_id, run) not in done for run in INTEGRATE_RUNS)]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]
    print(f"Integrate: {len(r_rows)} sheets done, {len(todo)} to run", flush=True)
    for ex in todo:
        state, built, mix, _s = sq._prep_state(runner, ex, artifacts, setup)
        if not (mix["ok"] and built["support_held"]):
            raise RuntimeError(f"Construction failed for {ex.example_id}.")
        ctx = _kv_ctx(runner, ex, state)
        frozen_bundle = state["kvmix"][sq.STUDENT]
        actual_bundle = _actual_bundle(frozen_bundle, ctx)
        del state
        if ctx["tk_ids"][ctx["rs"] :] != ctx["r_ids"]:
            raise RuntimeError(f"R token IDs drifted for {ex.example_id}.")
        bundles = {"frozen_std": frozen_bundle, "actual_std": actual_bundle}
        for run in INTEGRATE_RUNS:
            if (ex.example_id, run) in done:
                continue
            if run == "actual_direct":
                rec = _handoff(runner, ctx, "merged_kv", adapter=adapter, merged=merged, header_gpu=None, args=args, deployment="same_gpu")
            else:
                with receiver_weights(adapter, "merged", merged):
                    rec = _run_kv_b_timed(
                        runner, ex, runner.clone_sender(bundles[run]), plan_max_new=args.plan_max_new_12, answer_max_new=args.max_new_tokens
                    )
            rec.update({"run": run, "a_state_key": a_state_key(ex), "candidate_sha256": paths["frozen"]["sha256"]})
            rows.append(rec)
            done.add((ex.example_id, run))
            _append(path, rec)
            print(f"  {ex.example_id} {run}: {sq._brief(rec)}", flush=True)
        if ex.example_id not in r_done:
            ref = next(r for r in rows if r["example_id"] == ex.example_id and r["run"] == "frozen_std")
            seqs = {
                "plan": sq._sequence(runner, _prompts_for(ex)["kv_plan"], ref["plan_text"] or "", truncated=bool(ref.get("plan_truncated"))),
                "answer": sq._sequence(runner, ref["prompt"], ref["raw_text"] or "", truncated=bool(ref.get("truncated"))),
            }
            logits = [
                _forced_pair(runner, adapter, merged, frozen_bundle, actual_bundle, seq, part, ex.example_id)
                for part, seq in seqs.items()
                if seq["response_ids"]
            ]
            r_row = {"example_id": ex.example_id, "variant": ex.variant, "r_check": ctx["r_check"], "n_result": len(ctx["r_ids"]), "logits": logits}
            r_rows.append(r_row)
            r_done.add(ex.example_id)
            _append(rpath, r_row)
            c = ctx["r_check"]
            print(
                f"    R keys max|d|={c.get('keys_max_abs')} bit-equal={c.get('keys_bit_equal')} values equal={c.get('values_equal')} "
                f"| logits max|d| " + " ".join(f"{lg['part']}={lg['max_abs']:.3g}" for lg in logits),
                flush=True,
            )
        _persist(out_root, kaggle_dest)
        del ctx, frozen_bundle, actual_bundle, bundles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    intact = _weights_intact(adapter, originals, digest)
    summary = integrate_summary(rows, r_rows, agree_merged, intact, len(dev))
    _write_json(out_root / "integrate_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("r_comparison", "logits_frozen_vs_actual_r", "passed", "cause", "next")}, indent=1))
    return 0


# ---------------------------------------------------------------- bench (eval, reused)


def bench_order(example_id: str, rep: int) -> list[str]:
    order = list(CONDS)
    random.Random(f"{BENCH_SEED}:{example_id}:{rep}").shuffle(order)
    return order


def _ids(runner, prompt: str) -> list[int]:
    return [int(t) for t in runner._encode(prompt)[1].view(-1).tolist()]


def _text_header(runner, examples, messages: dict) -> list[int]:
    """Public tokens before the message: identical on every sheet, ending before any message token."""
    text_plan = lambda ex, msg: _prompts_for(ex)["text_plan"](ex, msg)  # noqa: E731
    n = None
    ref = None
    for ex in examples:
        msg = messages[ex.example_id]
        ids = _ids(runner, text_plan(ex, msg))
        other = _ids(runner, text_plan(ex, "#"))
        k = lcp(ids, other)
        rendered = runner._encode(text_plan(ex, msg))[0]
        at = rendered.find(msg.strip())
        if at < 0:
            raise RuntimeError(f"Message not found in the rendered text prompt for {ex.example_id}.")
        before = rendered[:at]
        head = runner.wrapper.tokenizer.decode(ids[:k], skip_special_tokens=False)
        if not before.startswith(head):
            raise RuntimeError("Text header runs into the message.")
        ref = ids if ref is None else ref
        n = k if n is None else min(n, k, lcp(ids, ref))
    return [int(t) for t in ref[:n]]


def _dynamic_cache(layers):
    from transformers import DynamicCache

    return DynamicCache.from_legacy_cache(tuple(layers))


def _empty_cache():
    from transformers import DynamicCache

    return DynamicCache()


def r_compare(frozen, actual) -> dict:
    """Frozen R (selected and re-rotated on the CPU when the states were built) vs R from A's GPU preparation."""
    per = []
    for (fk, fv), (ak, av) in zip(frozen, actual):
        ak, av = ak.detach().cpu(), av.detach().cpu()
        if ak.shape != fk.shape or av.shape != fv.shape:
            per.append({"shapes_equal": False})
            continue
        d = (ak.float() - fk.float()).abs()
        per.append(
            {
                "shapes_equal": True,
                "keys_max_abs": float(d.max()),
                "keys_mean_abs": float(d.mean()),
                "keys_bit_equal_fraction": float((ak == fk).float().mean()),
                "values_equal": bool(torch.equal(av, fv)),
            }
        )
    ok_shape = len(per) == len(frozen) > 0 and all(p["shapes_equal"] for p in per)
    out = {"n_layers": len(per), "shapes_equal": ok_shape}
    if ok_shape:
        out.update(
            {
                "keys_max_abs": max(p["keys_max_abs"] for p in per),
                "keys_mean_abs": sum(p["keys_mean_abs"] for p in per) / len(per),
                "min_keys_bit_equal_fraction": min(p["keys_bit_equal_fraction"] for p in per),
                "keys_bit_equal": all(p["keys_bit_equal_fraction"] == 1.0 for p in per),
                "values_equal": all(p["values_equal"] for p in per),
            }
        )
    return out


def _kv_ctx(runner, ex, state) -> dict:
    """Everything the KV handoffs need for one sheet, including the R that A's GPU preparation produces."""
    dev = runner.device
    rs = int(state["slots_extra"]["result_start"])
    tk, a_bundle = state["kvmix"][sq.STUDENT], state["bundle"]
    piece = next(p for p in state["composed_extra"]["pieces"] if p["kind"] == "result")
    src_pos = [int(p) for p in piece["source_pos"]]
    r_ids = [int(t) for t in piece["ids"]]
    n_c = n_kv_positions(tk.cache_cpu)
    new_pos = list(range(rs, n_c))
    support = [(k[..., :rs, :].contiguous(), v[..., :rs, :].contiguous()) for k, v in tk.cache_cpu["layers"]]
    frozen = [(k[..., rs:, :].contiguous(), v[..., rs:, :].contiguous()) for k, v in tk.cache_cpu["layers"]]
    a_gpu = to_device(a_bundle.cache_cpu["layers"], dev)
    actual = compact_sender(a_gpu, src_pos, new_pos, runner.wrapper.model)
    prompts = _prompts_for(ex)
    kv_ids = _ids(runner, prompts["kv_plan"])
    return {
        "ex": ex,
        "rs": rs,
        "n_compact": n_c,
        "src_pos": src_pos,
        "new_pos": new_pos,
        "r_ids": r_ids,
        "tk_ids": [int(t) for t in tk.full_ids_cpu.view(-1).tolist()],
        "a_ids": [int(t) for t in a_bundle.full_ids_cpu.view(-1).tolist()],
        "support_cpu": support,
        "support_gpu": to_device(support, dev),
        "frozen_r_cpu": frozen,
        "actual_r_gpu": actual,
        "r_check": r_compare(frozen, actual),
        "a_gpu": a_gpu,
        "n_full": n_kv_positions(a_bundle.cache_cpu),
        "full_cursor": int(a_bundle.pos_cursor_cpu.view(-1)[0].item()),
        "kv_plan_ids": kv_ids,
        "kv_shared": lcp(kv_ids, _ids(runner, prompts["kv_answer"](ex, PLACEHOLDER_PLAN))),
        "bytes": {
            "compact_r": kv_blob_size_bytes({"layers": frozen}),
            "full_kv": kv_blob_size_bytes(a_bundle.cache_cpu),
            "r_token_ids": 4 * len(r_ids),
        },
    }


def _actual_bundle(tk, ctx):
    """tk_tv with its R replaced by the GPU-produced R (copied to the host exactly) for B's standard path."""
    layers = [
        (torch.cat([sk, ak.cpu()], dim=-2), torch.cat([sv, av.cpu()], dim=-2))
        for (sk, sv), (ak, av) in zip(ctx["support_cpu"], ctx["actual_r_gpu"])
    ]
    return dataclasses.replace(tk, cache_cpu={**tk.cache_cpu, "layers": layers}, source="qmerge_actual_r")


def _sheet_ctx(runner, ex, state, header_ids, decode, saved_message: str) -> tuple[dict, dict]:
    dev = runner.device
    ctx = _kv_ctx(runner, ex, state)
    tk, fk, a_bundle = state["kvmix"][sq.STUDENT], state["kvmix"][sq.TEACHER], state["bundle"]
    rs, r_ids, src_pos, a_ids = ctx["rs"], ctx["r_ids"], ctx["src_pos"], ctx["a_ids"]
    message = (state.get("instr") or {}).get("result_text") or ""
    prompts = _prompts_for(ex)
    text_ids = _ids(runner, prompts["text_plan"](ex, message))
    text_shared = lcp(text_ids, _ids(runner, prompts["text_answer"](ex, message, PLACEHOLDER_PLAN)))
    ctx.update(
        {
            "decode": decode,
            "message": message,
            "text_plan_ids": text_ids,
            "text_shared": text_shared,
            "header_len": len(header_ids),
        }
    )
    ctx["bytes"]["text"] = len(message.encode("utf-8"))
    rebuilt = _dynamic_cache(clone_layers(ctx["support_gpu"]))
    extend(runner.wrapper.model, rebuilt, r_ids, n_past=rs, cursor=rs, device=dev)
    fk_layers = [(k.cpu(), v.cpu()) for k, v in cache_layers(rebuilt)]
    fk_k = max(float((a.float() - b.float()).abs().max()) for (a, _), (b, _) in zip(fk_layers, fk.cache_cpu["layers"]))
    fk_v = max(float((a.float() - b.float()).abs().max()) for (_, a), (_, b) in zip(fk_layers, fk.cache_cpu["layers"]))
    del rebuilt, fk_layers
    tk_ids = ctx["tk_ids"]
    check = {
        "example_id": ex.example_id,
        "variant": ex.variant,
        "result_start": rs,
        "n_result": len(r_ids),
        "n_full": ctx["n_full"],
        "r_ids_match_compact": tk_ids[rs:] == r_ids and len(ctx["new_pos"]) == len(r_ids),
        "r_ids_match_a": [a_ids[p] for p in src_pos] == r_ids,
        "fk_ids_equal_tk": [int(t) for t in fk.full_ids_cpu.view(-1).tolist()] == tk_ids,
        "cursor_is_length": int(tk.pos_cursor_cpu.view(-1)[0].item()) == ctx["n_compact"],
        "full_cursor_is_length": ctx["full_cursor"] == ctx["n_full"],
        "masks_all_ones": all(bool((b.past_mask_cpu > 0).all()) for b in (tk, a_bundle)),
        "message_from_r_tokens": decode_positions(a_ids, src_pos, decode) == message and bool(message.strip()),
        "message_equals_saved_eval": message == saved_message,
        "text_header_matches": text_ids[: len(header_ids)] == list(header_ids),
        "kv_shared": ctx["kv_shared"],
        "text_shared": text_shared,
        "shared_after_header": text_shared > len(header_ids),
        "actual_r_vs_frozen": ctx["r_check"],
        "rebuilt_fk_keys_max_abs": fk_k,
        "rebuilt_fk_values_max_abs": fk_v,
        "bytes": ctx["bytes"],
    }
    check["ok"] = bool(
        check["r_ids_match_compact"]
        and check["r_ids_match_a"]
        and check["fk_ids_equal_tk"]
        and check["cursor_is_length"]
        and check["masks_all_ones"]
        and check["message_from_r_tokens"]
        and check["message_equals_saved_eval"]
        and check["text_header_matches"]
        and check["shared_after_header"]
        and ctx["r_check"]["shapes_equal"]
        and ctx["r_check"].get("values_equal")
        and 0 < ctx["kv_shared"] < len(ctx["kv_plan_ids"])
    )
    return ctx, check


def _move(layers, deployment: str, device, t: dict):
    """same_gpu: B uses the tensors where they are. device_transfer: device-to-host, then host-to-device."""
    if deployment == "same_gpu":
        return layers
    a = now()
    host = [(k.to("cpu"), v.to("cpu")) for k, v in layers]
    t["movement_out_s"] = now() - a
    a = now()
    out = to_device(host, device)
    t["movement_in_s"] = now() - a
    t["movement_s"] = t["movement_out_s"] + t["movement_in_s"]
    return out


def _handoff(runner, ctx, cond: str, *, adapter, merged, header_gpu, args, deployment: str) -> dict:
    if deployment not in DEPLOYMENTS:
        raise ValueError(f"Unknown deployment {deployment}")
    spec, ex, dev = SPEC[cond], ctx["ex"], runner.device
    model = runner.wrapper.model
    prompts = _prompts_for(ex)
    t: dict = {"r_prefill_s": 0.0, "movement_s": 0.0, "movement_out_s": 0.0, "movement_in_s": 0.0, "untimed_bench_copy_s": 0.0}
    kind = spec["handoff"]
    owned = None
    if kind == "full" and deployment == "same_gpu":
        a = now()
        owned = clone_layers(ctx["a_gpu"])
        t["untimed_bench_copy_s"] = now() - a
    reset_peak_memory()
    with receiver_weights(adapter, spec["mode"], merged):
        t_start = now()
        message = None
        if kind == "compact":
            a = now()
            sent = compact_sender(ctx["a_gpu"], ctx["src_pos"], ctx["new_pos"], model)
            t["sender_prep_s"] = now() - a
            r = _move(sent, deployment, dev, t)
            a = now()
            cache = _dynamic_cache(cat_layers(ctx["support_gpu"], r))
            t["receiver_prep_s"] = now() - a
            del sent, r
            n_past = cursor = ctx["n_compact"]
            bytes_moved = ctx["bytes"]["compact_r"]
        elif kind == "reconstruct":
            a = now()
            ids = list(ctx["r_ids"])
            t["sender_prep_s"] = now() - a
            a = now()
            x = torch.tensor(ids, dtype=torch.long).to(dev)
            t["movement_s"] = t["movement_in_s"] = now() - a
            a = now()
            cache = _dynamic_cache(clone_layers(ctx["support_gpu"]))
            t["receiver_prep_s"] = now() - a
            a = now()
            extend(model, cache, x, n_past=ctx["rs"], cursor=ctx["rs"], device=dev)
            t["r_prefill_s"] = now() - a
            n_past = cursor = ctx["n_compact"]
            bytes_moved = ctx["bytes"]["r_token_ids"]
        elif kind == "text":
            a = now()
            message = decode_positions(ctx["a_ids"], ctx["src_pos"], ctx["decode"])
            t["sender_prep_s"] = now() - a
            a = now()
            cache = _dynamic_cache(clone_layers(header_gpu))
            t["receiver_prep_s"] = now() - a
            n_past = cursor = ctx["header_len"]
            bytes_moved = len(message.encode("utf-8"))
        else:
            t["sender_prep_s"] = 0.0
            layers = owned if deployment == "same_gpu" else _move(ctx["a_gpu"], deployment, dev, t)
            a = now()
            cache = _dynamic_cache(layers)
            t["receiver_prep_s"] = now() - a
            n_past, cursor = ctx["n_full"], ctx["full_cursor"]
            bytes_moved = ctx["bytes"]["full_kv"]

        a = time.perf_counter()
        if kind == "text":
            plan_prompt = prompts["text_plan"](ex, message)
            answer_prompt_of = lambda plan: prompts["text_answer"](ex, message, plan)  # noqa: E731
            start, shared, expect = ctx["header_len"], ctx["text_shared"], ctx["text_plan_ids"]
        else:
            plan_prompt = prompts["kv_plan"]
            answer_prompt_of = lambda plan: prompts["kv_answer"](ex, plan)  # noqa: E731
            start, shared, expect = 0, ctx["kv_shared"], ctx["kv_plan_ids"]
        if prompt_contains_a_private(plan_prompt, ex):
            raise RuntimeError("B's plan prompt leaked A's raw sheet.")
        plan_ids = _ids(runner, plan_prompt)
        t["encode_s"] = time.perf_counter() - a
        if plan_ids != expect:
            raise RuntimeError("Plan prompt tokens changed between setup and handoff.")

        def answer_of(plan_text: str):
            prompt = answer_prompt_of(plan_text)
            if prompt_contains_a_private(prompt, ex):
                raise RuntimeError("B's answer prompt leaked A's raw sheet.")
            return _ids(runner, prompt), prompt

        out = two_pass(
            runner.wrapper,
            cache=cache,
            n_past=n_past,
            cursor=cursor,
            plan_ids=plan_ids,
            start=start,
            shared=shared,
            answer_of=answer_of,
            plan_max=args.plan_max_new_12,
            answer_max=args.max_new_tokens,
            device=dev,
        )
        total = now() - t_start
    plan, ans = out["plan"], out["answer"]
    t.update(
        {
            "encode_s": t["encode_s"] + out["answer_encode_s"],
            "shared_prefill_s": out["shared_prefill_s"],
            "plan_prefill_s": plan["prefill_s"],
            "plan_decode_s": plan["decode_s"],
            "answer_prefill_s": ans["prefill_s"],
            "answer_decode_s": ans["decode_s"],
            "crop_s": out["crop_s"],
        }
    )
    extra = {
        "plan_text": plan["text"],
        "plan_truncated": plan["truncated"],
        "truncated": ans["truncated"],
        "n_plan_tokens": plan["n_new"],
        "n_new_tokens": ans["n_new"],
        "n_shared_tokens": out["n_shared_tokens"],
        "n_plan_suffix_tokens": plan["n_prefill_tokens"],
        "n_answer_suffix_tokens": ans["n_prefill_tokens"],
        "n_kv_received": n_past,
        "bytes_moved": bytes_moved,
        "peak_gpu_gb": peak_gpu_memory_gb(),
        **t,
        "component_sum_s": sum(t[c] for c in COMPONENTS),
        "handoff_total_s": total,
    }
    rec = _row_n(ex, prompt=out["answer_prompt"], text=ans["text"], extra=extra)
    rec.update(
        {
            "condition": cond,
            "mode": spec["mode"],
            "handoff": spec["handoff"],
            "deployment": deployment,
            "r_source": "gpu_preparation" if kind == "compact" else None,
        }
    )
    del cache, out
    return rec


def _median(xs):
    return statistics.median(xs) if xs else None


def per_sheet_medians(rows: list[dict], field: str) -> dict:
    """(condition, example_id) -> median over repetitions."""
    acc: dict = {}
    for r in rows:
        acc.setdefault((r["condition"], r["example_id"]), []).append(float(r.get(field) or 0.0))
    return {k: _median(v) for k, v in acc.items()}


def _family_means(med: dict, cond: str, fam_of: dict) -> dict:
    acc: dict = {}
    for (c, eid), v in med.items():
        if c == cond:
            acc.setdefault(fam_of[eid], []).append(v)
    return {f: sum(v) / len(v) for f, v in acc.items()}


def paired_time(rows: list[dict], left: str, right: str, field: str = "handoff_total_s") -> dict:
    """left minus right, per sheet median over repetitions, bootstrap over families; plus each repetition alone."""
    fam_of = {r["example_id"]: r["family_id"] for r in rows}
    med = per_sheet_medians(rows, field)
    out = _cluster_delta(_family_means(med, right, fam_of), _family_means(med, left, fam_of))
    out.pop("per_case", None)
    by_rep = {}
    for rep in sorted({int(r["rep"]) for r in rows}):
        sub = [r for r in rows if int(r["rep"]) == rep]
        a = {r["example_id"]: float(r.get(field) or 0.0) for r in sub if r["condition"] == left}
        b = {r["example_id"]: float(r.get(field) or 0.0) for r in sub if r["condition"] == right}
        common = sorted(set(a) & set(b))
        by_rep[rep] = sum(a[e] - b[e] for e in common) / len(common) if common else None
    out["mean_delta_by_rep"] = by_rep
    reps = [v for v in by_rep.values() if v is not None]
    ci = out.get("ci95_per_case") or [0.0, 0.0]
    out["repeatable_saving"] = bool(reps) and all(v < 0 for v in reps) and ci[1] < 0
    out["repeatable_cost"] = bool(reps) and all(v > 0 for v in reps) and ci[0] > 0
    return out


def _component_table(rows: list[dict], cond: str) -> dict:
    fields = COMPONENTS + ("movement_out_s", "movement_in_s", "untimed_bench_copy_s", "component_sum_s", "handoff_total_s", "n_plan_tokens", "n_new_tokens", "n_kv_received", "bytes_moved", "peak_gpu_gb")
    out = {}
    for f in fields:
        med = per_sheet_medians([r for r in rows if r["condition"] == cond], f)
        out[f] = statistics.fmean(med.values()) if med else None
    toks = (out["n_plan_tokens"] or 0) + (out["n_new_tokens"] or 0)
    out["decode_s_per_token"] = ((out["plan_decode_s"] or 0) + (out["answer_decode_s"] or 0)) / toks if toks else None
    out["prefill_total_s"] = sum(out[k] or 0 for k in PREFILL_PARTS)
    out["decode_total_s"] = (out["plan_decode_s"] or 0) + (out["answer_decode_s"] or 0)
    out["preparation_total_s"] = sum(out[k] or 0 for k in ("sender_prep_s", "movement_s", "receiver_prep_s"))
    by_rep = {}
    for rep in sorted({int(r["rep"]) for r in rows}):
        sub = [float(r["handoff_total_s"]) for r in rows if r["condition"] == cond and int(r["rep"]) == rep]
        by_rep[rep] = statistics.fmean(sub) if sub else None
    out["handoff_total_by_rep"] = by_rep
    return out


def _rep_consistency(rows: list[dict], cond: str) -> dict:
    texts: dict = {}
    for r in rows:
        if r["condition"] == cond:
            texts.setdefault(r["example_id"], set()).add((r.get("plan_text"), r.get("raw_text")))
    return {"n_sheets": len(texts), "identical_across_reps": sum(len(v) == 1 for v in texts.values())}


def reliability_check(rows: list[dict]) -> dict:
    first = [r for r in rows if int(r["rep"]) == 1]
    m = _twelve_labels([r for r in first if r["condition"] == "merged_kv"])
    u = _twelve_labels([r for r in first if r["condition"] == "unmerged_kv"])
    mc = _twelve_choice([r for r in first if r["condition"] == "merged_kv"])
    uc = _twelve_choice([r for r in first if r["condition"] == "unmerged_kv"])
    losses = {
        "labels": _loss(u.get("labels_correct", 0), m.get("labels_correct", 0)),
        "all_twelve": _loss(u.get("cases_all_twelve_correct", 0), m.get("cases_all_twelve_correct", 0)),
        "final_choice": _loss(uc.get("final_choice_correct", 0), mc.get("final_choice_correct", 0)),
    }
    return {
        "merged_minus_unmerged_losses": losses,
        "max_loss": EVAL_MAX_LOSS,
        "pass": m.get("n_cases", 0) > 0 and m.get("n_cases") == u.get("n_cases") and all(losses[k] <= EVAL_MAX_LOSS[k] for k in losses),
    }


def decide(reliability_ok: bool, paired: dict, one_time_s: float | None) -> dict:
    out = {"reliability_preserved": bool(reliability_ok)}
    for other in ("text_result", "fk_fv", "full_kv"):
        p = paired.get(other) or {}
        saving = -float(p.get("mean_delta_per_case") or 0.0)
        out[f"vs_{other}"] = {
            "repeatable_saving": bool(p.get("repeatable_saving")),
            "repeatable_cost": bool(p.get("repeatable_cost")),
            "mean_saving_s_per_handoff": saving,
            "break_even_handoffs": (one_time_s / saving) if (one_time_s and saving > 0 and p.get("repeatable_saving")) else None,
        }
    out["useful_practical_result"] = bool(reliability_ok and out["vs_text_result"]["repeatable_saving"])
    if not reliability_ok:
        out["statement"] = "Merging did not preserve reliability within the registered tolerance; no speed result is claimed."
    elif out["useful_practical_result"]:
        out["statement"] = "The merged receiver keeps reliability and is repeatably faster than text per complete handoff."
    elif out["vs_text_result"]["repeatable_cost"]:
        out["statement"] = "The merged receiver keeps reliability but is repeatably slower than text per complete handoff."
    else:
        out["statement"] = "The merged receiver keeps reliability; no repeatable difference from text in complete handoff time."
    return out


def bench_summary(rows, checks, eval_rows, *, merge_info=None, one_time=None, intact=None) -> dict:
    _add_component_totals(rows)
    labels = {
        v: {c: _twelve_labels([r for r in rows if r["condition"] == c and r.get("variant") == v and int(r["rep"]) == 1]) for c in CONDS}
        for v in ("base", "a_flip")
    }
    choice = {
        v: {c: _twelve_choice([r for r in rows if r["condition"] == c and r.get("variant") == v and int(r["rep"]) == 1]) for c in CONDS}
        for v in ("base", "a_flip")
    }
    saved = {(r["example_id"], r["condition"]): r for r in eval_rows}
    vs_saved = {}
    for c in CONDS:
        sub = [r for r in rows if r["condition"] == c and int(r["rep"]) == 1]
        refs = [(r, saved.get((r["example_id"], SPEC[c]["eval_ref"]))) for r in sub]
        refs = [(r, s) for r, s in refs if s is not None]
        vs_saved[c] = {
            "eval_ref": SPEC[c]["eval_ref"],
            "n": len(refs),
            "plan_text_identical": sum(r.get("plan_text") == s.get("plan_text") for r, s in refs),
            "labels_bench": sum(int(r.get("plan_n_status_correct") or 0) for r, _ in refs),
            "labels_saved": sum(int(s.get("plan_n_status_correct") or 0) for _, s in refs),
        }
    paired = {other: paired_time(rows, "merged_kv", other) for other in COMPARE_TO}
    components = {
        other: {
            f: {k: v for k, v in paired_time(rows, "merged_kv", other, f).items() if k in ("mean_delta_per_case", "ci95_per_case")}
            for f in ("prefill_total", "decode_total", "preparation_total")
        }
        for other in ("text_result", "fk_fv", "full_kv")
    }
    rel = reliability_check(rows)
    one_time_s = None
    if one_time:
        one_time_s = sum(float(v or 0) for k, v in one_time.items() if k.endswith("_s"))
    out = {
        "scope": SCOPE,
        "timing_note": TIMING_NOTE,
        "n_rows": len(rows),
        "reps": REPS,
        "checks_ok": bool(checks) and all(c["ok"] for c in checks),
        "checks_worst": {
            "actual_r_keys_max_abs_vs_frozen": max((c["actual_r_vs_frozen"].get("keys_max_abs") or 0.0 for c in checks), default=None),
            "actual_r_keys_bit_equal_sheets": sum(bool(c["actual_r_vs_frozen"].get("keys_bit_equal")) for c in checks),
            "actual_r_values_equal_all": all(c["actual_r_vs_frozen"].get("values_equal") for c in checks) if checks else None,
            "rebuilt_fk_keys_max_abs": max((c["rebuilt_fk_keys_max_abs"] for c in checks), default=None),
            "rebuilt_fk_values_max_abs": max((c["rebuilt_fk_values_max_abs"] for c in checks), default=None),
        },
        "labels": labels,
        "final_choice": choice,
        "consistency_across_reps": {c: _rep_consistency(rows, c) for c in CONDS},
        "vs_saved_eval_rows": vs_saved,
        "timing": {c: _component_table(rows, c) for c in CONDS},
        "paired_handoff_time": paired,
        "paired_components": components,
        "reliability_check": rel,
        "merge": merge_info,
        "one_time": one_time,
        "original_weights": intact,
    }
    out["decision"] = decide(rel["pass"], paired, one_time_s)
    return out


def _add_component_totals(rows: list[dict]) -> None:
    for r in rows:
        r["prefill_total"] = sum(float(r.get(k) or 0) for k in PREFILL_PARTS)
        r["decode_total"] = float(r.get("plan_decode_s") or 0) + float(r.get("answer_decode_s") or 0)
        r["preparation_total"] = sum(float(r.get(k) or 0) for k in ("sender_prep_s", "movement_s", "receiver_prep_s"))


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _run_bench(args, out_root, kaggle_dest, splits, runner, paths, protocol) -> int:
    agree = _read_json(out_root / "agree_summary.json")
    if agree is None or not agree.get("passed"):
        print("The dev agreement check must pass before benchmarking the merged receiver.")
        return 2
    integration = _read_json(out_root / "integrate_summary.json")
    if integration is None or not integration.get("complete"):
        print("Run --phase integrate first: the benchmark feeds B the R that A's GPU preparation produces.")
        return 2
    if not integration.get("passed"):
        print("The GPU-prepared R does not behave like the frozen R within tolerance. Document the difference "
              "(integrate_summary.json) before interpreting any deployment timing; the benchmark will not run.")
        return 2
    deployment = args.deployment
    if deployment is None:
        print(f"Declare the deployment being measured: --deployment {' | '.join(DEPLOYMENTS)}.")
        return 2
    adapter, originals, digest, merged, merge_s = _load_candidate(runner, paths)
    examples, artifacts, frozen_a, timing12 = sq._eval_inputs(out_root, splits, _qadapt_inputs(paths))
    setup = sq._prep_setup(runner, examples, artifacts, frozen_a, timing12)
    generics = setup[4]
    decode = _decode_fn(runner)
    eval_rows = _load_rows(paths["eval_evaluations.jsonl"])
    messages = {r["example_id"]: r["message_text"] for r in eval_rows if r["condition"] == "text_result"}
    header_ids = _text_header(runner, examples, messages)
    t0 = now()
    header_cache = _empty_cache()
    extend(runner.wrapper.model, header_cache, header_ids, n_past=0, cursor=0, device=runner.device)
    header_prefill_s = now() - t0
    header_gpu = clone_layers(cache_layers(header_cache))
    del header_cache

    path, cpath = out_root / f"bench_{deployment}_evaluations.jsonl", out_root / f"bench_{deployment}_checks.jsonl"
    rows, checks = _load_rows(path), _load_rows(cpath)
    done = {(r["example_id"], int(r["rep"]), r["condition"]) for r in rows}
    checked = {c["example_id"] for c in checks}
    todo = [ex for ex in examples if any((ex.example_id, rep, c) not in done for rep in range(1, REPS + 1) for c in CONDS)]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]
    print(f"Bench [{deployment}]: {len(rows)} handoffs done, {len(todo)} sheets to run; text header {len(header_ids)} tokens", flush=True)
    print(f"  {DEPLOYMENTS[deployment]}", flush=True)
    warmed = False
    for i, ex in enumerate(todo):
        state, built, mix, _s = sq._prep_state(runner, ex, artifacts, setup)
        if not (mix["ok"] and built["support_held"]):
            raise RuntimeError(f"Construction failed for {ex.example_id}.")
        ctx, check = _sheet_ctx(runner, ex, state, header_ids, decode, messages[ex.example_id])
        del state
        if not check["ok"]:
            raise RuntimeError(f"Benchmark preparation check failed for {ex.example_id}: {check}")
        if ex.example_id not in checked:
            checks.append(check)
            checked.add(ex.example_id)
            _append(cpath, check)
        if not warmed:
            for cond in CONDS:
                _handoff(runner, ctx, cond, adapter=adapter, merged=merged, header_gpu=header_gpu, args=args, deployment=deployment)
            warmed = True
            print("  warm-up done (one handoff per condition, discarded)", flush=True)
        for rep in range(1, REPS + 1):
            for k, cond in enumerate(bench_order(ex.example_id, rep)):
                if (ex.example_id, rep, cond) in done:
                    continue
                rec = _handoff(runner, ctx, cond, adapter=adapter, merged=merged, header_gpu=header_gpu, args=args, deployment=deployment)
                rec.update({"rep": rep, "order_index": k, "candidate_sha256": paths["frozen"]["sha256"]})
                rows.append(rec)
                done.add((ex.example_id, rep, cond))
                _append(path, rec)
            print(
                f"  [{i + 1}/{len(todo)}] {ex.example_id} rep {rep}: "
                + " ".join(
                    f"{c}={next(r['handoff_total_s'] for r in rows if r['example_id'] == ex.example_id and int(r['rep']) == rep and r['condition'] == c):.2f}s/"
                    f"{next(r['plan_n_status_correct'] for r in rows if r['example_id'] == ex.example_id and int(r['rep']) == rep and r['condition'] == c)}"
                    for c in CONDS
                ),
                flush=True,
            )
        _persist(out_root, kaggle_dest)
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    intact = _weights_intact(adapter, originals, digest)
    if not intact["ok"]:
        raise RuntimeError(f"Original weights were not restored: {intact}")
    train = json.loads(paths["train_summary.json"].read_text(encoding="utf-8"))
    one_time = {
        "teacher_target_generation_s": train.get("teacher_target_generation_s"),
        "training_state_build_s": train.get("state_build_s"),
        "training_s": sum(float(b.get("train_s") or 0) for b in (train.get("by_epoch") or {}).values()),
        "merge_s": merge_s,
        "note": "Adapter costs only. Public support and text header prefills are listed separately and are paid once per template either way.",
    }
    public = {
        "text_header_tokens": len(header_ids),
        "text_header_prefill_s": header_prefill_s,
        "kv_support_template_prefill_s": {name: float(g.get("prefill_s") or 0.0) for name, g in generics.items()},
    }
    stats = merge_stats(adapter, merged)
    summary = bench_summary(
        rows,
        checks,
        eval_rows,
        merge_info={"merge_s": merge_s, "stats": stats["worst"], "merged_weight_bytes": sum(w.numel() * w.element_size() for w in merged)},
        one_time=one_time,
        intact=intact,
    )
    summary["public_prefixes"] = public
    summary["deployment"] = {"name": deployment, "description": DEPLOYMENTS[deployment]}
    summary["integration"] = {k: integration.get(k) for k in ("passed", "r_comparison", "cause")}
    summary["complete"] = len(rows) == len(examples) * REPS * len(CONDS)
    _write_json(out_root / f"bench_{deployment}_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"decision": summary["decision"], "complete": summary["complete"]}, indent=1))
    return 0


# ---------------------------------------------------------------- text + frozen adapter (diagnostic)


def _run_textadapt(args, out_root, kaggle_dest, splits, runner, paths, protocol) -> int:
    adapter, originals, digest, _merged, _s = _load_candidate(runner, paths)
    examples = list(splits["eval"])
    eval_rows = _load_rows(paths["eval_evaluations.jsonl"])
    messages = {r["example_id"]: r["message_text"] for r in eval_rows if r["condition"] == "text_result"}
    path = out_root / "textadapt_evaluations.jsonl"
    rows = _load_rows(path)
    done = {r["example_id"] for r in rows}
    todo = [ex for ex in examples if ex.example_id not in done]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]
    for ex in todo:
        with receiver_weights(adapter, "unmerged"):
            rec = _run_text_b_timed(
                runner, ex, message=messages[ex.example_id], plan_max_new=args.plan_max_new_12, answer_max_new=args.max_new_tokens
            )
        rec.update({"condition": "text_adapted", "a_state_key": a_state_key(ex)})
        rows.append(rec)
        _append(path, rec)
        print(f"  {ex.example_id} text_adapted: {sq._brief(rec)}", flush=True)
    _persist(out_root, kaggle_dest)
    intact = _weights_intact(adapter, originals, digest)
    by = lambda cond, v: [r for r in eval_rows if r["condition"] == cond and r.get("variant") == v]  # noqa: E731
    summary = {
        "scope": "Diagnostic only, on reused eval sheets. The adapter was trained for the KV receiver; "
        "a negative result here does not show what a separately text-trained receiver could achieve.",
        "labels": {
            v: {
                "text_adapted": _twelve_labels([r for r in rows if r.get("variant") == v]),
                "text_result_saved": _twelve_labels(by("text_result", v)),
                "adapted_kv_saved": _twelve_labels(by("adapted_correct_r", v)),
            }
            for v in ("base", "a_flip")
        },
        "final_choice": {
            v: {
                "text_adapted": _twelve_choice([r for r in rows if r.get("variant") == v]),
                "text_result_saved": _twelve_choice(by("text_result", v)),
            }
            for v in ("base", "a_flip")
        },
        "original_weights": intact,
        "complete": len(rows) == len(examples),
    }
    _write_json(out_root / "textadapt_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary["labels"], indent=1))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Merged receiver and fair handoff timing.")
    parser.add_argument("--phase", choices=("agree", "integrate", "bench", "textadapt"), default="agree")
    parser.add_argument("--deployment", choices=tuple(DEPLOYMENTS), default=None, help="bench only; required")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.max_new_tokens != 48 or args.plan_max_new_12 != 384:
        print("Decoding allowances are frozen at 384 plan / 48 answer tokens.")
        return 2
    protocol = pinned.get("study_plan4_qmerge") or {}
    if not protocol.get("registered_before_running"):
        print("Register study_plan4_qmerge in configs/pinned.json before running.")
        return 2
    mism = sq.registered_hash_mismatches(pinned.get("study_plan4_qadapt") or {})
    if mism:
        print(f"Split files differ from the registered hashes: {mism}.")
        return 2
    out_root, kaggle_dest = _paths()
    if "stage_plan4_qmerge" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_qmerge.")
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    for folder in [d for d in (kaggle_dest, REPO_ROOT / "results_stage_plan4_qmerge") if d is not None]:
        for name in PERSIST_NAMES:
            src, dest = folder / name, out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())
    paths = _copy_inputs(out_root)
    if not json.loads(paths["prep_check_summary.json"].read_text(encoding="utf-8")).get("all_passed"):
        print("The prep construction check did not pass.")
        return 2
    splits = {name: ensure_split(name) for name in (("dev",) if args.phase in ("agree", "integrate") else ("eval",))}
    policy, runner = sq._prepare_runner(
        args,
        out_root,
        extra={
            "stage": "plan4-qmerge",
            "phase": args.phase,
            "conditions": list(CONDS),
            "do_not_overwrite": list(PRESERVED),
            "candidate": {k: paths["frozen"][k] for k in ("selected_epoch", "sha256")},
            "scope": SCOPE,
            "timing_note": TIMING_NOTE,
            "protocol": protocol,
        },
    )
    if sq.current_attention(runner.wrapper.model) != sq.GEN_ATTN:
        print(f"Generation must use {sq.GEN_ATTN} attention.")
        return 2
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase}")
    run = (args, out_root, kaggle_dest, splits, runner, paths, protocol)
    return {"agree": _run_agree, "integrate": _run_integrate, "bench": _run_bench, "textadapt": _run_textadapt}[args.phase](*run)


if __name__ == "__main__":
    raise SystemExit(main())
