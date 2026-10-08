"""Development-set pass/fail checks. The 80-example split is not used here."""

from __future__ import annotations

from typing import Any

from src.paths import UPSTREAM_DIR

CHANCE = 0.25
DIRECT_MIN_ACC = 0.70
RELAY_MIN_ACC = 0.45
MAX_TRUNCATED = 0.15
MAX_PEAK_GPU_GB = 15.0
HOBF_FP32_MARKERS = (
    "X = v[b, h].index_select(dim=0, index=kept_abs).to(torch.float32)",
    "Y = v[b, h].index_select(dim=0, index=disc_abs).to(torch.float32)",
    "torch.linalg.qr",
    "patch.to(dtype=new_v.dtype)",
)


def inspect_hobf_fp32() -> dict[str, Any]:
    path = UPSTREAM_DIR / "compression_methods" / "HOBFFast.py"
    if not path.is_file():
        return {"ok": False, "path": str(path), "error": "HOBFFast.py missing; fetch upstream first"}
    text = path.read_text(encoding="utf-8")
    found = {marker: marker in text for marker in HOBF_FP32_MARKERS}
    return {"ok": all(found.values()), "path": str(path), "markers_found": found}


def _cond(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return summary.get(name) or {}


def evaluate_gates(summary: dict[str, Any], *, n_examples: int, n_dev: int = 20) -> dict[str, Any]:
    direct = _cond(summary, "direct_text")
    full_fill = _cond(summary, "full_sender_only_filler")
    full_none = _cond(summary, "full_sender_only_no_filler")
    mismatch = _cond(summary, "mismatch_full_sender_only")
    head_fill = _cond(summary, "headwise_sender_only_filler")
    head_none = _cond(summary, "headwise_sender_only_no_filler")
    hobf = _cond(summary, "hobf_sender_only_filler")
    complete = n_examples >= n_dev
    fp32 = inspect_hobf_fp32()

    def truncated_ok(block: dict[str, Any]) -> bool:
        rate = block.get("truncated")
        return rate is not None and rate <= MAX_TRUNCATED

    answers_complete = all(
        truncated_ok(block)
        for block in (direct, full_fill, mismatch)
        if block.get("n")
    ) and bool(direct.get("n"))

    gate1 = {
        "name": "direct_text",
        "question": "Can B answer the inventory question from text?",
        "accuracy": direct.get("accuracy"),
        "truncated": direct.get("truncated"),
        "passed": bool(
            complete
            and direct.get("accuracy") is not None
            and direct["accuracy"] >= DIRECT_MIN_ACC
            and truncated_ok(direct)
        ),
        "need": f"accuracy >= {DIRECT_MIN_ACC} and truncated <= {MAX_TRUNCATED} on {n_dev} examples",
    }
    gate2 = {
        "name": "full_relay_sender_only",
        "question": "Can B obtain the answer through A's uncompressed cache?",
        "accuracy": full_fill.get("accuracy"),
        "truncated": full_fill.get("truncated"),
        "chance": CHANCE,
        "passed": bool(
            complete
            and full_fill.get("accuracy") is not None
            and full_fill["accuracy"] >= RELAY_MIN_ACC
            and truncated_ok(full_fill)
        ),
        "need": f"sender-only full-relay accuracy >= {RELAY_MIN_ACC} (above chance {CHANCE}) and truncated <= {MAX_TRUNCATED}. The no-relay filler cell is the empirical baseline, not 0.25.",
        "note": "This is the finding that matters before comparing compressors.",
    }
    follow_mismatch = mismatch.get("follow_donor")
    follow_true = full_fill.get("follow_donor")
    acc_mismatch = mismatch.get("accuracy")
    acc_true = full_fill.get("accuracy")
    mismatch_moves = (
        follow_mismatch is not None
        and follow_true is not None
        and acc_mismatch is not None
        and acc_true is not None
        and follow_mismatch > follow_true
        and acc_mismatch < acc_true
    )
    gate3 = {
        "name": "matched_mismatch",
        "question": "Does changing the transmitted inventory move B toward the alternative assignment?",
        "true_cache_accuracy": acc_true,
        "true_cache_follow_donor": follow_true,
        "alt_cache_accuracy": acc_mismatch,
        "alt_cache_follow_donor": follow_mismatch,
        "passed": bool(complete and gate2["passed"] and mismatch_moves and truncated_ok(mismatch)),
        "need": "with full relay working: alt-cache follow_donor > true-cache follow_donor and alt-cache accuracy < true-cache accuracy",
        "blocked_by_full_relay": not gate2["passed"],
    }
    filler_delta = None
    if full_fill.get("accuracy") is not None and full_none.get("accuracy") is not None:
        filler_delta = full_fill["accuracy"] - full_none["accuracy"]
    penalty_fill = None
    penalty_none = None
    if full_fill.get("accuracy") is not None and head_fill.get("accuracy") is not None:
        penalty_fill = full_fill["accuracy"] - head_fill["accuracy"]
    if full_none.get("accuracy") is not None and head_none.get("accuracy") is not None:
        penalty_none = full_none["accuracy"] - head_none["accuracy"]
    gate4 = {
        "name": "filler_sensitivity",
        "question": "Does adding filler change performance or the compression penalty?",
        "acc_full_filler": full_fill.get("accuracy"),
        "acc_full_no_filler": full_none.get("accuracy"),
        "delta_full": filler_delta,
        "compression_penalty_filler": penalty_fill,
        "compression_penalty_no_filler": penalty_none,
        "passed": bool(
            complete
            and full_fill.get("n")
            and full_none.get("n")
            and head_fill.get("n")
            and head_none.get("n")
        ),
        "need": "all four full/headwise × filler/no-filler cells finished. Completed does not mean filler is harmless; report the penalty delta.",
        "note": "A larger eviction penalty with filler than without means filler may amplify compression loss.",
    }
    peaks = [
        block.get("peak_gpu_gb_max")
        for block in (direct, full_fill, head_fill, hobf)
        if block.get("peak_gpu_gb_max") is not None
    ]
    peak_max = max(peaks) if peaks else None
    nonfinite = any(
        block.get("any_nonfinite_cache") or block.get("any_nonfinite_obf")
        for block in (full_fill, head_fill, hobf)
        if block
    )
    gate5 = {
        "name": "compression_health",
        "question": "Are OBF outputs finite, its sensitive math FP32, and peak memory acceptable?",
        "hobf_fp32_source": fp32,
        "any_nonfinite": nonfinite,
        "peak_gpu_gb_max": peak_max,
        "mean_hobf_compression_time_s": hobf.get("mean_compression_time_s"),
        "mean_hobf_compression_core_s": hobf.get("mean_compression_core_s"),
        "mean_hobf_receiver_latency_s": hobf.get("mean_receiver_latency_s"),
        "timing_note": "compression_time_s is the compress() wall; compression_core_s is the operator; receiver_latency_s is B's decode. Do not call the decode gap 'OBF time'.",
        "passed": bool(
            complete
            and hobf.get("n")
            and fp32.get("ok")
            and not nonfinite
            and (peak_max is None or peak_max < MAX_PEAK_GPU_GB)
        ),
        "need": f"finite caches, HOBFFast QR/SVD/injection in FP32, peak GPU < {MAX_PEAK_GPU_GB} GB",
    }
    hobf_none = _cond(summary, "hobf_sender_only_no_filler")
    no_relay = _cond(summary, "no_relay_filler")
    extra_complete = bool(no_relay.get("n") and hobf_none.get("n") and no_relay["n"] >= n_dev and hobf_none["n"] >= n_dev)
    extra = {
        "name": "extra_controls",
        "question": "No-relay+filler baseline and OBF without filler are both present?",
        "no_relay_filler_accuracy": no_relay.get("accuracy"),
        "hobf_no_filler_accuracy": hobf_none.get("accuracy"),
        "majority_class_rate": (summary.get("dev_label_balance") or {}).get("majority_class_rate"),
        "passed": extra_complete,
        "need": "20 no_relay_filler evals and 20 hobf_sender_only_no_filler evals. Compare no-relay to majority-class rate, not 0.25.",
    }
    gates = [gate1, gate2, gate3, gate4, gate5]
    return {
        "n_examples_seen": n_examples,
        "n_dev": n_dev,
        "complete_run": complete,
        "short_answers_complete": answers_complete,
        "short_answers_note": "Truncation below threshold is not the same as every output finishing; inspect any truncated raw_text.",
        "all_passed": complete and answers_complete and all(g["passed"] for g in gates),
        "extra_controls_complete": extra_complete,
        "do_not_open_test": True,
        "next_finding": "whether full relay transfers private evidence, not whether OBF already beats eviction",
        "gates": {g["name"]: g for g in gates} | {"extra_controls": extra},
    }
