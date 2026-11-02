"""Relevant supporting context vs equal-size random additions on plan4_dev.

Start from cons_instr_sink: result lines, first four positions, consecutive
RoPE, and the frozen framing sentence. Add the first eligibility-instruction
paragraph and the written calculation block, or the same number of randomly
chosen remaining positions. Reconstruct A's cache from recorded token IDs.
Do not regenerate A's answers.

plan4_hold stays out. Do not resume the interface factorial.
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
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    SINK_INITIAL,
    a_state_key,
    assert_plan4_invariants,
    load_plan4_jsonl,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    ELIGIBILITY_RULE,
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta, _run_text_b
from src.stage_plan4_hold import _plan_fields
from src.stage_plan4_initctrl import _checked_bool
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_select import _load_rows, _paired
from src.support_select import RANDOM_SEEDS, extract_support_pack
from src.two_agent import TwoAgentInventory

LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
RAND_CONDS = tuple(f"compact_rand_{seed}" for seed in RANDOM_SEEDS)
CONDS = (
    "kv",
    "text_result",
    "compact",
    "compact_support",
    "compact_rand_2101",
    "compact_rand_2102",
    "compact_rand_2103",
    "text_support",
)
COMPACT_KV = ("compact", "compact_support") + RAND_CONDS
REFS = ("kv", "text_result", "text_support")
SPEC = {
    "kv": {"kind": "full", "layout": "original", "framing": "baseline"},
    "text_result": {"kind": "text", "text": "result"},
    "compact": {"kind": "kv", "layout": "consecutive", "framing": "instruct", "positions": "baseline"},
    "compact_support": {
        "kind": "kv",
        "layout": "consecutive",
        "framing": "instruct",
        "positions": "support",
    },
    "compact_rand_2101": {
        "kind": "kv",
        "layout": "consecutive",
        "framing": "instruct",
        "positions": "random",
        "seed": 2101,
    },
    "compact_rand_2102": {
        "kind": "kv",
        "layout": "consecutive",
        "framing": "instruct",
        "positions": "random",
        "seed": 2102,
    },
    "compact_rand_2103": {
        "kind": "kv",
        "layout": "consecutive",
        "framing": "instruct",
        "positions": "random",
        "seed": 2103,
    },
    "text_support": {"kind": "text", "text": "support"},
}
PAIRED = (
    ("support_vs_compact", "compact", "compact_support"),
    ("support_vs_rand_2101", "compact_rand_2101", "compact_support"),
    ("support_vs_rand_2102", "compact_rand_2102", "compact_support"),
    ("support_vs_rand_2103", "compact_rand_2103", "compact_support"),
    ("support_vs_kv", "kv", "compact_support"),
    ("support_vs_text_result", "text_result", "compact_support"),
    ("support_vs_text_support", "text_support", "compact_support"),
    ("text_support_vs_text_result", "text_result", "text_support"),
    ("kv_vs_text_result", "text_result", "kv"),
    ("compact_vs_kv", "kv", "compact"),
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
    "results_stage_plan4_compress",
    "results_stage_plan4_probe",
    "results_stage_plan4_budget",
    "results_stage_plan4_fresh",
    "results_stage_plan4_span",
    "results_stage_plan4_kvdiag",
    "results_stage_plan4_initctrl",
    "results_stage_plan4_stable",
    "results_stage_plan4_hold",
    "results_stage_plan4_iface",
)
INTERPRETATION_GUIDE = (
    "Relevant support beats both the baseline and random additions: the specified "
    "supporting context helps beyond merely increasing message size. That does not "
    "prove the added entries restore the original computation. "
    "Relevant and random additions improve similarly: more retained cache, structure, "
    "or overlap with useful content may explain the change. "
    "Relevant support helps but remains well below full KV and text: the selected-state "
    "handoff still has unresolved limitations. "
    "Supported KV remains poor while supported text works: the passages are sufficient "
    "through text, but reusing their contextual states remains unreliable. "
    "Both supported KV and supported text deteriorate: inspect conflicting instructions, "
    "sender errors, formatting, and truncation before attributing the result to KV reuse. "
    "Supported KV approaches the references: apply the existing development advancement "
    "rule. Qualification selects a confirmation candidate; it does not itself establish recovery."
)


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        src = out_root / "construction.json"
        if src.is_file():
            (dest / "construction.json").write_bytes(src.read_bytes())


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_support" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_support")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def candidate_qualifies(block: dict, kv_block: dict, text_block: dict) -> bool:
    n = int(block.get("n") or 0)
    if n == 0 or int(kv_block.get("n") or 0) == 0 or int(text_block.get("n") or 0) == 0:
        return False
    if int(block.get("parse_ok") or 0) != n:
        return False
    if int(block.get("truncated") or 0) != 0:
        return False
    cand = 100.0 * int(block.get("plan_all_status_correct") or 0) / n
    kv = 100.0 * int(kv_block.get("plan_all_status_correct") or 0) / int(kv_block["n"])
    text = 100.0 * int(text_block.get("plan_all_status_correct") or 0) / int(text_block["n"])
    return cand + 1e-9 >= kv - LABEL_MARGIN and cand + 1e-9 >= text - LABEL_MARGIN


def choose_candidate(receiver: dict) -> dict:
    kv_block = receiver.get("kv") or {}
    text_block = receiver.get("text_result") or {}
    block = receiver.get("compact_support") or {}
    ok = candidate_qualifies(block, kv_block, text_block)
    return {
        "margin_points": LABEL_MARGIN,
        "candidate": "compact_support",
        "qualifies": ok,
        "plan_all_status_correct": block.get("plan_all_status_correct"),
        "parse_ok": block.get("parse_ok"),
        "truncated": block.get("truncated"),
        "selected": "compact_support" if ok else None,
        "note": (
            "A qualifying candidate needs parseable untruncated outputs in every "
            "case and all-label accuracy within five points of full KV and result text. "
            "If those references score 90/90, that requires at least 86/90. "
            "This selects a confirmation candidate; it does not itself establish recovery."
        ),
    }


def _process_time_s(row: dict) -> float:
    return float(
        (row.get("sender_generation_time_s") or 0.0)
        + (row.get("rebuild_time_s") or 0.0)
        + (row.get("selection_time_s") or 0.0)
        + (row.get("plan_time_s") or 0.0)
        + (row.get("receiver_latency_s") or 0.0)
    )


def _cond_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    bytes_ = [int(item.get("message_bytes") or 0) for item in items]
    mean_bytes = sum(bytes_) / n
    recorded = [float(item["process_time_s"]) for item in items if item.get("process_time_s") is not None]
    return {
        "n": n,
        "parse_ok": sum(bool(item.get("parse_ok")) for item in items),
        "think_collapse": sum(bool(item.get("think_collapse")) for item in items),
        "truncated": sum(bool(item.get("truncated") or item.get("plan_truncated")) for item in items),
        "plan_all_status_correct": sum(bool(item.get("plan_all_status_correct")) for item in items),
        "mean_n_status_correct": sum(int(item.get("plan_n_status_correct") or 0) for item in items) / n,
        "plan_ineligible_marked_eligible": sum(
            int(item.get("plan_ineligible_marked_eligible") or 0) for item in items
        ),
        "plan_eligible_marked_ineligible": sum(
            int(item.get("plan_eligible_marked_ineligible") or 0) for item in items
        ),
        "correct": sum(bool(item.get("correct")) for item in items),
        "follows_trap": sum(bool(item.get("follows_trap")) for item in items),
        "plan_answer_agrees": _checked_bool(items, "plan_answer_agrees"),
        "correct_with_wrong_eligibility": sum(
            bool(item.get("correct_with_wrong_eligibility")) for item in items
        ),
        "mean_n_kept": sum(int(item.get("n_kept") or item.get("n_kv_positions") or 0) for item in items) / n,
        "mean_n_result_kept": sum(int(item.get("n_result_kept") or 0) for item in items) / n,
        "mean_n_sink_kept": sum(int(item.get("n_sink_kept") or 0) for item in items) / n,
        "mean_n_extra": sum(int(item.get("n_extra") or 0) for item in items) / n,
        "mean_n_added": sum(int(item.get("n_added") or 0) for item in items) / n,
        "mean_n_full": sum(int(item.get("n_full") or 0) for item in items) / n,
        "mean_overlap_support_extra_frac": (
            sum(float(item.get("overlap_support_extra_frac") or 0.0) for item in items) / n
        ),
        "mean_selection_time_s": sum(float(item.get("selection_time_s") or 0.0) for item in items) / n,
        "mean_rebuild_time_s": sum(float(item.get("rebuild_time_s") or 0.0) for item in items) / n,
        "span_extract_ok": sum(bool(item.get("span_extract_ok")) for item in items),
        "mean_message_bytes": mean_bytes,
        "mean_relay_mb": mean_bytes / (1024 * 1024),
        "mean_process_time_s": (sum(recorded) / len(recorded)) if recorded else None,
        "ids_source": sorted({str(item.get("ids_source") or "") for item in items if item.get("ids_source")}),
        "cache_source": sorted({str(item.get("cache_source") or "") for item in items if item.get("cache_source")}),
        "sampling_mode": sorted({str(item.get("sampling_mode") or "") for item in items if item.get("sampling_mode")}),
    }


def _family_delta_vs_mean(
    rows: list[dict],
    cond_a: str,
    cond_bs: Sequence[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    key: str = "plan_all_status_correct",
) -> dict:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    fam_of: dict[str, str] = {}
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
        fam_of[row["example_id"]] = row["family_id"]
    by_fam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for eid, cells in grouped.items():
        if cond_a not in cells or any(name not in cells for name in cond_bs):
            continue
        a_val = float(bool(cells[cond_a].get(key)))
        b_val = sum(float(bool(cells[name].get(key))) for name in cond_bs) / len(cond_bs)
        by_fam[fam_of[eid]].append((a_val, b_val))
    fids = sorted(by_fam)
    if not fids:
        return {"n_families": 0}

    def delta(sample: list[str]) -> float:
        n = acc_a = acc_b = 0.0
        for fid in sample:
            for ca, cb in by_fam[fid]:
                n += 1.0
                acc_a += ca
                acc_b += cb
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
        "cond_a": cond_a,
        "cond_bs": list(cond_bs),
        "delta_mean_b_minus_a": observed,
        "ci95": [lo, hi],
        "unit": "family (variants and random seeds kept together)",
    }


def _random_mean_block(rows: list[dict]) -> dict:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    items = []
    for eid, cells in grouped.items():
        if any(name not in cells for name in RAND_CONDS):
            continue
        seeds = [cells[name] for name in RAND_CONDS]
        proto = dict(seeds[0])
        proto["condition"] = "compact_rand_mean"
        proto["example_id"] = eid
        proto["plan_all_status_correct"] = (
            sum(bool(item.get("plan_all_status_correct")) for item in seeds) / len(seeds)
        )
        proto["correct"] = sum(bool(item.get("correct")) for item in seeds) / len(seeds)
        proto["parse_ok"] = sum(bool(item.get("parse_ok")) for item in seeds) / len(seeds)
        proto["truncated"] = sum(bool(item.get("truncated") or item.get("plan_truncated")) for item in seeds) / len(
            seeds
        )
        proto["correct_with_wrong_eligibility"] = (
            sum(bool(item.get("correct_with_wrong_eligibility")) for item in seeds) / len(seeds)
        )
        proto["plan_ineligible_marked_eligible"] = (
            sum(int(item.get("plan_ineligible_marked_eligible") or 0) for item in seeds) / len(seeds)
        )
        proto["plan_eligible_marked_ineligible"] = (
            sum(int(item.get("plan_eligible_marked_ineligible") or 0) for item in seeds) / len(seeds)
        )
        proto["n_kept"] = sum(int(item.get("n_kept") or 0) for item in seeds) / len(seeds)
        proto["n_extra"] = sum(int(item.get("n_extra") or 0) for item in seeds) / len(seeds)
        proto["message_bytes"] = sum(int(item.get("message_bytes") or 0) for item in seeds) / len(seeds)
        proto["overlap_support_extra_frac"] = (
            sum(float(item.get("overlap_support_extra_frac") or 0.0) for item in seeds) / len(seeds)
        )
        proto["process_time_s"] = sum(float(item.get("process_time_s") or 0.0) for item in seeds) / len(seeds)
        items.append(proto)
    if not items:
        return {"n": 0}
    n = len(items)
    bytes_ = [float(item.get("message_bytes") or 0) for item in items]
    mean_bytes = sum(bytes_) / n
    return {
        "n": n,
        "parse_ok": sum(float(item.get("parse_ok") or 0) for item in items),
        "truncated": sum(float(item.get("truncated") or 0) for item in items),
        "plan_all_status_correct": sum(float(item.get("plan_all_status_correct") or 0) for item in items),
        "correct": sum(float(item.get("correct") or 0) for item in items),
        "correct_with_wrong_eligibility": sum(
            float(item.get("correct_with_wrong_eligibility") or 0) for item in items
        ),
        "plan_ineligible_marked_eligible": sum(
            float(item.get("plan_ineligible_marked_eligible") or 0) for item in items
        ),
        "plan_eligible_marked_ineligible": sum(
            float(item.get("plan_eligible_marked_ineligible") or 0) for item in items
        ),
        "mean_n_kept": sum(float(item.get("n_kept") or 0) for item in items) / n,
        "mean_n_extra": sum(float(item.get("n_extra") or 0) for item in items) / n,
        "mean_overlap_support_extra_frac": (
            sum(float(item.get("overlap_support_extra_frac") or 0) for item in items) / n
        ),
        "mean_message_bytes": mean_bytes,
        "mean_relay_mb": mean_bytes / (1024 * 1024),
        "mean_process_time_s": sum(float(item.get("process_time_s") or 0) for item in items) / n,
        "note": "Per-example mean of the three random seeds. Seeds are not independent extra cases.",
    }


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _artifact_candidates(out_root: Path, explicit: str | None = None) -> list[Path]:
    paths = []
    if explicit:
        paths.append(Path(explicit))
    paths.extend(
        [
            out_root / "a_artifacts.jsonl",
            REPO_ROOT / "results_stage_plan4_iface" / "a_artifacts.jsonl",
            RESULTS_DIR / "stage_plan4_iface" / "run" / "a_artifacts.jsonl",
            Path("/kaggle/working/results_stage_plan4_iface/a_artifacts.jsonl"),
        ]
    )
    if Path("/kaggle/input").is_dir():
        for root, dirs, files in os.walk("/kaggle/input"):
            dirs[:] = [d for d in dirs if d not in {".git", ".pytest_cache"}]
            if "a_artifacts.jsonl" in files and "plan4_iface" in root.replace("\\", "/"):
                paths.append(Path(root) / "a_artifacts.jsonl")
    return paths


def _copy_frozen_artifacts(out_root: Path, *, explicit: str | None = None) -> Path:
    dest = out_root / "a_artifacts.jsonl"
    if dest.is_file():
        return dest
    for src in _artifact_candidates(out_root, explicit):
        if src.is_file() and src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
            print(f"Copied frozen A artifacts from {src}")
            return dest
    raise FileNotFoundError(
        "Need results_stage_plan4_iface/a_artifacts.jsonl with recorded token IDs. "
        "Do not regenerate A's answers."
    )


def _iface_reference() -> dict | None:
    for path in (
        REPO_ROOT / "results_stage_plan4_iface" / "summary.json",
        RESULTS_DIR / "stage_plan4_iface" / "run" / "summary.json",
        Path("/kaggle/working/results_stage_plan4_iface/summary.json"),
    ):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = data.get("receiver") or {}
        out = {"source": str(path), "conditions": {}}
        for name in ("kv", "text_result", "cons_instr_sink"):
            block = rec.get(name) or {}
            if block:
                out["conditions"][name] = {
                    "n": block.get("n"),
                    "correct": block.get("correct"),
                    "plan_all_status_correct": block.get("plan_all_status_correct"),
                    "parse_ok": block.get("parse_ok"),
                    "truncated": block.get("truncated"),
                }
        return out
    return None


def _make_decode(tokenizer):
    def decode(ids):
        return tokenizer.decode(list(ids), skip_special_tokens=False)

    return decode


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _positions_for(cond: str, spans: dict) -> tuple[list[int], dict]:
    spec = SPEC[cond]
    extra = {
        "n_result_kept": len(spans.get("result_pos") or []),
        "n_sink_kept": int(spans.get("n_sink_extra") or 0),
        "n_extra": 0,
        "n_added": 0,
        "overlap_instruction": 0,
        "overlap_calc": 0,
        "overlap_support_extra": 0,
        "overlap_support_extra_frac": 0.0,
        "n_from_prompt": 0,
        "n_from_written": 0,
        "sampling_mode": None,
        "random_runs": None,
        "support_extra_runs": spans.get("support_extra_runs"),
    }
    kind = spec.get("positions")
    if kind == "baseline":
        return list(spans["baseline_pos"]), extra
    if kind == "support":
        extra["n_extra"] = int(spans.get("n_extra") or 0)
        extra["n_added"] = int(spans.get("n_extra") or 0)
        extra["n_from_prompt"] = sum(p < int(spans["n_prompt"]) for p in spans.get("support_extra_pos") or [])
        extra["n_from_written"] = sum(p >= int(spans["n_prompt"]) for p in spans.get("support_extra_pos") or [])
        return list(spans["support_pos"]), extra
    if kind == "random":
        rec = (spans.get("random") or {})[str(spec["seed"])]
        extra["n_added"] = int(rec.get("n") or 0)
        extra["n_extra"] = int(rec.get("n") or 0)
        extra["overlap_instruction"] = int(rec.get("overlap_instruction") or 0)
        extra["overlap_calc"] = int(rec.get("overlap_calc") or 0)
        extra["overlap_support_extra"] = int(rec.get("overlap_support_extra") or 0)
        extra["overlap_support_extra_frac"] = float(rec.get("overlap_support_extra_frac") or 0.0)
        extra["n_from_prompt"] = int(rec.get("n_from_prompt") or 0)
        extra["n_from_written"] = int(rec.get("n_from_written") or 0)
        extra["sampling_mode"] = rec.get("sampling_mode")
        extra["random_runs"] = rec.get("runs")
        return list(rec["union_positions"]), extra
    raise KeyError(cond)


def _summarize(rows: list[dict], construction: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("kv", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    receiver["compact_rand_mean"] = _random_mean_block(rows)
    family = {name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED) for name, a, b in PAIRED}
    family_elig = {
        name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED, key="plan_all_status_correct")
        for name, a, b in PAIRED
    }
    family_elig["support_vs_rand_mean"] = _family_delta_vs_mean(
        rows, "compact_support", RAND_CONDS, n_boot=N_BOOT, seed=BOOT_SEED
    )
    family["support_vs_rand_mean"] = _family_delta_vs_mean(
        rows, "compact_support", RAND_CONDS, n_boot=N_BOOT, seed=BOOT_SEED, key="correct"
    )
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "n_evals_expected": 90 * len(CONDS),
        "unit_note": (
            "plan4_dev only: 30 families with three versions each. "
            "plan4_hold is unused. Same reconstructed A on every branch. "
            "The three random seeds do not turn 90 cases into 270 independent examples."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
        "by_difficulty": {},
        "paired_receiver": {name: _paired(grouped_plain, a, b) for name, a, b in PAIRED},
        "family_clustered_delta": family,
        "family_clustered_eligibility": family_elig,
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "decision": choose_candidate(receiver),
        "framing_change": FRAMING_INSTRUCTION,
        "random_seeds": list(RANDOM_SEEDS),
        "construction": None
        if construction is None
        else {
            "n": construction.get("n"),
            "n_ok": construction.get("n_ok"),
            "all_ok": construction.get("all_ok"),
            "mean_n_extra": construction.get("mean_n_extra"),
            "mean_overlap_frac": construction.get("mean_overlap_frac"),
            "n_a_errors_preserved": construction.get("n_a_errors_preserved"),
            "sampling_modes": construction.get("sampling_modes"),
        },
        "reconstruction_check": {
            "cache_source": "prefill_recorded_ids",
            "iface_reference": _iface_reference(),
            "this_run": {
                name: {
                    "correct": (receiver.get(name) or {}).get("correct"),
                    "plan_all_status_correct": (receiver.get(name) or {}).get("plan_all_status_correct"),
                    "parse_ok": (receiver.get(name) or {}).get("parse_ok"),
                    "truncated": (receiver.get(name) or {}).get("truncated"),
                }
                for name in ("kv", "text_result", "compact")
            },
            "note": (
                "Full KV, result text, and the compact baseline were rerun on caches "
                "rebuilt from recorded token IDs. Compare them with the interface factorial "
                "before attributing a change to supporting context."
            ),
        },
        "lead": (
            "Primary measure: all four eligibility labels. Also report final-choice "
            "accuracy, parseability, truncation, eligibility confusions, and "
            "correct choices with wrong labels. Malformed outputs stay in the denominator."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "timing_note": (
            "process_time_s includes recorded A generation time, cache reconstruction, "
            "gather/re-rotation, and B. Network is not measured."
        ),
        "after_this_run": (
            "These are development results. Do not load plan4_hold. "
            "A professor cost comparison at four and twelve carriers waits until a candidate qualifies."
        ),
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _cond_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _inspect_pack(pack: dict) -> dict:
    instr = pack.get("instruction") or {}
    calc = pack.get("calc") or {}
    return {
        "a_state_key": pack.get("a_state_key"),
        "ok": pack.get("ok"),
        "failures": pack.get("failures"),
        "a_error_recorded": pack.get("a_error_recorded"),
        "n_full": pack.get("n_full"),
        "n_prompt": pack.get("n_prompt"),
        "n_extra": pack.get("n_extra"),
        "n_baseline": len(pack.get("baseline_pos") or []),
        "n_support": len(pack.get("support_pos") or []),
        "instruction_text": (instr.get("text") or "")[:500],
        "instruction_contains_rule": ELIGIBILITY_RULE in (instr.get("text") or ""),
        "calc_head": (calc.get("text") or "")[:400],
        "calc_tail": (calc.get("text") or "")[-400:],
        "result_text": pack.get("result_text"),
        "result_carriers": pack.get("result_carriers"),
        "first_four_once": pack.get("first_four_once"),
        "equal_random_counts": pack.get("equal_random_counts"),
        "random_overlap_frac": {
            seed: (pack.get("random") or {}).get(seed, {}).get("overlap_support_extra_frac")
            for seed in (str(s) for s in RANDOM_SEEDS)
        },
        "sampling_mode": {
            seed: (pack.get("random") or {}).get(seed, {}).get("sampling_mode")
            for seed in (str(s) for s in RANDOM_SEEDS)
        },
    }


def _construction_summary(packs: list[dict]) -> dict:
    overlaps = []
    extras = []
    modes = []
    for pack in packs:
        extras.append(int(pack.get("n_extra") or 0))
        for seed in RANDOM_SEEDS:
            rec = (pack.get("random") or {}).get(str(seed)) or {}
            overlaps.append(float(rec.get("overlap_support_extra_frac") or 0.0))
            modes.append(rec.get("sampling_mode"))
    n = len(packs)
    n_ok = sum(bool(p.get("ok")) for p in packs)
    return {
        "n": n,
        "n_ok": n_ok,
        "all_ok": bool(n) and n_ok == n,
        "n_a_errors_preserved": sum(bool(p.get("a_error_recorded")) for p in packs),
        "mean_n_extra": (sum(extras) / n) if n else None,
        "min_n_extra": min(extras) if extras else None,
        "max_n_extra": max(extras) if extras else None,
        "mean_overlap_frac": (sum(overlaps) / len(overlaps)) if overlaps else None,
        "n_random_majority_overlap": sum(v >= 0.5 for v in overlaps),
        "sampling_modes": sorted({m for m in modes if m}),
        "instruction_occurrence": "first",
        "random_seeds": list(RANDOM_SEEDS),
        "used_gold_or_b_private": any(p.get("used_gold_or_b_private") for p in packs),
        "inspect": [_inspect_pack(p) for p in packs[:3]],
        "failures": [
            {"a_state_key": p.get("a_state_key"), "failures": p.get("failures")}
            for p in packs
            if not p.get("ok")
        ],
        "frozen_rule": (
            "First eligibility-instruction paragraph in the prompt, written tokens "
            "before the RESULTS marker, result lines after that marker, first four "
            "positions, consecutive RoPE, instruct framing. Random samples M extra "
            "positions with seeds 2101/2102/2103 keyed by a_state_key."
        ),
    }


def _run_construct(args, out_root, kaggle_dest, examples, artifacts, decode) -> int:
    by_id = {ex.example_id: ex for ex in examples}
    unique = []
    seen = set()
    for ex in examples:
        key = a_state_key(ex)
        if key not in seen:
            unique.append(ex)
            seen.add(key)
    packs = []
    slim = {}
    print(f"Constructing frozen support spans on {len(unique)} unique A sheets", flush=True)
    for i, ex in enumerate(unique):
        key = a_state_key(ex)
        rec = artifacts.get(key)
        if rec is None:
            raise RuntimeError(f"Missing frozen A artifact for {key}.")
        src = by_id.get(key) or ex
        ordinary = rec.get("ordinary_prompt") or ""
        if prompt_contains_b_private(ordinary, src):
            raise RuntimeError("Recorded A prompt leaked B's prices or budget.")
        if prompt_contains_solver_labels(ordinary, src):
            raise RuntimeError("Recorded A prompt leaked solver labels.")
        full_ids = rec.get("full_ids") or []
        n_full = int(rec.get("n_kv_positions") or len(full_ids))
        n_prompt = int(rec.get("n_prompt_positions") or 0)
        score = rec.get("written_score") or {}
        pack = extract_support_pack(
            full_ids,
            n_prompt=n_prompt,
            n_full=n_full,
            decode=decode,
            a_state_key=key,
            a_error_recorded=bool(rec.get("a_error_recorded")) or not bool(score.get("all_four_correct")),
        )
        packs.append(pack)
        slim[key] = pack
        print(
            f"[{i + 1}/{len(unique)}] {key} ok={pack['ok']} extra={pack['n_extra']} "
            f"a_error={pack['a_error_recorded']} failures={pack['failures']}",
            flush=True,
        )
    summary = _construction_summary(packs)
    payload = {
        **summary,
        "by_a_state": slim,
    }
    _write_json(out_root / "construction.json", payload)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("n", "n_ok", "all_ok", "mean_n_extra", "mean_overlap_frac")}, indent=2))
    print("Inspected examples:", json.dumps(summary.get("inspect"), indent=2)[:8000])
    return 0 if summary["all_ok"] else 1


def _prepare_runner(args, out_root: Path, extra: dict) -> tuple:
    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-support",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "framing_change": FRAMING_INSTRUCTION,
            "sink_initial": SINK_INITIAL,
            "random_seeds": list(RANDOM_SEEDS),
            "cache_source": "prefill_recorded_ids",
            "baseline": "cons_instr_sink",
            **extra,
        },
    )
    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=0,
        kv_budget=32,
        sink_size=4,
        pca_rank=2,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha") and not extra.get("skip_pin"):
        pin_model_revision(revision["sha"])
    return policy, runner, revision


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, construction) -> int:
    if not construction.get("all_ok") and not args.force_compare:
        print("Construction did not pass. Fix span extraction before the comparison, or pass --force-compare.")
        return 2
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    by_id = {ex.example_id: ex for ex in examples}
    states: dict = {}
    needed = set()
    for ex in examples:
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if remaining:
            needed.add(a_state_key(ex))
    print(f"Rebuilding {len(needed)} unique A caches from recorded token IDs", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        ordinary = rec.get("ordinary_prompt") or ""
        if prompt_contains_b_private(ordinary, src):
            raise RuntimeError("Recorded A prompt leaked B's prices or budget.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bundle, rebuild_s = runner.rebuild_cache_from_ids(
            rec["full_ids"],
            prompt=ordinary,
            n_prompt=int(rec["n_prompt_positions"]),
            pos_cursor=int(rec["pos_cursor"]),
            source="rebuild",
            token_align=rec.get("token_align"),
            sender_latency_s=float(rec.get("written_s") or 0.0),
        )
        if int(bundle.latent_steps) != 0:
            raise RuntimeError("Rebuilt cache must have zero latent steps.")
        if n_kv_positions(bundle.cache_cpu) != len(rec["full_ids"]):
            raise RuntimeError("Rebuilt cache length does not match recorded token IDs.")
        states[key] = {
            "bundle": bundle,
            "meta": rec,
            "rebuild_time_s": rebuild_s,
            "spans": construction["by_a_state"][key],
        }
        print(f"  rebuilt {key} n_kv={n_kv_positions(bundle.cache_cpu)} s={rebuild_s:.2f}", flush=True)
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        key = a_state_key(ex)
        pack = states[key]
        extra_a = _a_extra(pack)
        extra_a["rebuild_time_s"] = float(pack["rebuild_time_s"])
        extra_a["cache_source"] = "prefill_recorded_ids"
        extra_a["a_error_recorded"] = bool(pack["spans"].get("a_error_recorded"))
        spans = pack["spans"]
        token_align = dict(pack["meta"].get("token_align") or {})
        inspect = pack["meta"].get("token_inspect")

        for cond in remaining:
            spec = SPEC[cond]
            if spec["kind"] == "text":
                message = spans["result_text"] if spec["text"] == "result" else spans["supported_text"]
                rec = _run_text_b(
                    runner,
                    ex,
                    message=message,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    allow_a_source=True,
                )
                rec.update(extra_a)
                rec["n_kept"] = 0
                rec["n_kv_positions"] = 0
                rec["n_result_kept"] = len(spans.get("result_pos") or [])
                rec["n_sink_kept"] = 0
                rec["n_extra"] = int(spans.get("n_extra") or 0) if spec["text"] == "support" else 0
                rec["n_added"] = rec["n_extra"]
                rec["positions"] = list(spans.get("result_pos") or [])
                rec["assigned_positions"] = list(spans.get("result_pos") or [])
                rec["selected_text"] = message
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
                rec["layout"] = None
                rec["framing"] = "baseline"
                rec["framing_instruction"] = None
                rec["message_bytes"] = len(message.encode("utf-8"))
                rec["overlap_support_extra_frac"] = 0.0
                rec["sampling_mode"] = None
            elif spec["kind"] == "full":
                rec = _run_kv_b(
                    runner,
                    ex,
                    runner.clone_sender(pack["bundle"]),
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["positions"] = list(range(int(spans["n_full"])))
                rec["assigned_positions"] = list(range(int(spans["n_full"])))
                rec["n_kept"] = int(rec.get("n_kv_positions") or 0)
                rec["n_result_kept"] = 0
                rec["n_sink_kept"] = 0
                rec["n_extra"] = 0
                rec["n_added"] = 0
                rec["selected_text"] = pack["meta"].get("written")
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
                rec["layout"] = "original"
                rec["framing"] = "baseline"
                rec["framing_instruction"] = None
                rec["overlap_support_extra_frac"] = 0.0
                rec["sampling_mode"] = None
            else:
                positions, pos_extra = _positions_for(cond, spans)
                cloned = runner.clone_sender(pack["bundle"])
                compressed, wall, info = runner.keep_shared_positions(
                    cloned, positions, layout=spec["layout"]
                )
                rec = _run_kv_b(
                    runner,
                    ex,
                    compressed,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    plan_prompt=b_reason_kv_instruct_prompt(ex),
                    make_answer_prompt=b_answer_kv_instruct_prompt,
                )
                rec.update(extra_a)
                rec["selection_time_s"] = float(wall)
                rec["compression_time_s"] = float(wall)
                rec.update(info)
                rec.update(pos_extra)
                rec["n_kept"] = int(info["n_kept"])
                rec["positions"] = list(info.get("positions") or positions)
                rec["assigned_positions"] = list(info.get("assigned_positions") or [])
                rec["selected_text"] = spans.get("result_text")
                rec["layout"] = spec["layout"]
                rec["framing"] = spec["framing"]
                rec["framing_instruction"] = FRAMING_INSTRUCTION
                extra_framing = len(FRAMING_INSTRUCTION.encode("utf-8"))
                rec["framing_bytes"] = extra_framing
                rec["message_bytes"] = int(compressed.full_relay_bytes) + extra_framing
                rec["cache_finite"] = bool(info.get("cache_finite"))

            rec["condition"] = cond
            rec["span_extract_ok"] = bool(spans.get("ok"))
            rec["result_fallback"] = bool(spans.get("result_fallback"))
            rec["n_full"] = int(spans.get("n_full") or 0)
            rec["ids_source"] = pack["meta"].get("ids_source")
            rec["cache_source"] = "prefill_recorded_ids"
            rec["loop_eq_retokenize"] = token_align.get("loop_eq_retokenize")
            rec["loop_len_eq_n_kv"] = token_align.get("loop_len_eq_n_kv")
            rec["eos_counted_not_cached"] = token_align.get("eos_counted_not_cached")
            rec["token_inspect"] = inspect
            rec["shared_across_heads"] = True
            rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            rec["process_time_s"] = _process_time_s(rec)
            _plan_fields(rec, ex)
            if spec["kind"] == "text" and prompt_contains_a_private(str(rec.get("prompt") or ""), ex):
                rec["text_contains_a_sheet"] = True
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec.get('pred_carrier')} correct={rec.get('correct')} "
                f"elig={rec.get('plan_all_status_correct')} parse={rec.get('parse_ok')} "
                f"trunc={rec.get('truncated')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    summary = _summarize(rows, construction)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary.get("decision"), indent=2))
    print(json.dumps(summary.get("reconstruction_check"), indent=2))
    print(json.dumps({k: summary["receiver"].get(k) for k in list(CONDS) + ["compact_rand_mean"]}, indent=2)[:12000])
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="plan4_dev relevant supporting context vs equal-size random additions."
    )
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--phase", choices=("construct", "compare", "all", "rescore"), default="construct")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Supporting-context comparison uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_support" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_support.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in ("evaluations.jsonl", "summary.json", "construction.json", "a_artifacts.jsonl", "manifest.json"):
            src = kaggle_dest / name
            dest = out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    examples = load_plan4_jsonl(data)
    for ex in examples:
        assert_plan4_invariants(ex)
    if len(examples) != 90:
        print(f"Expected 90 plan4_dev examples, got {len(examples)}.")
        return 2

    artifacts_path = _copy_frozen_artifacts(out_root, explicit=args.artifacts)
    artifacts = _load_messages(artifacts_path)
    if len(artifacts) != 60:
        print(f"Expected 60 unique frozen A sheets, got {len(artifacts)}.")
        return 2
    for rec in artifacts.values():
        if not rec.get("full_ids"):
            print("Frozen artifacts are missing token IDs. Do not regenerate A.")
            return 2

    if args.phase == "rescore":
        construction = {}
        cpath = out_root / "construction.json"
        if cpath.is_file():
            construction = json.loads(cpath.read_text(encoding="utf-8"))
        rows = _load_rows(out_root / "evaluations.jsonl")
        summary = _summarize(rows, construction or None)
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps(summary.get("decision"), indent=2))
        return 0

    if args.phase in {"construct", "all"}:
        tokenizer = _load_tokenizer(args.model_name)
        decode = _make_decode(tokenizer)
        rc = _run_construct(args, out_root, kaggle_dest, examples, artifacts, decode)
        if rc != 0 and args.phase == "all" and not args.force_compare:
            print("Construction failed; stopping before the eight-condition comparison.")
            return rc
        if args.phase == "construct":
            return rc

    construction_path = out_root / "construction.json"
    if not construction_path.is_file():
        print("Run --phase construct first.")
        return 2
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    policy, runner, revision = _prepare_runner(
        args,
        out_root,
        extra={"phase": args.phase, "conditions": list(CONDS), "n_evals": 90 * len(CONDS)},
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, construction)


if __name__ == "__main__":
    raise SystemExit(main())
