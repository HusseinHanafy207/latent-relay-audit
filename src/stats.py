"""Example-level contrasts and paired bootstrap. No GPU."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

MAIN_RELAYS = ("full", "headwise_32", "hobf_32", "headwise_64", "hobf_64")
MAIN_ACCESSES = ("sender_only", "shared")
NO_FILLER_RELAYS = ("full", "headwise_32", "hobf_32")
EVICTION_AT = {32: "headwise_32", 64: "headwise_64"}
OBF_AT = {32: "hobf_32", 64: "hobf_64"}


def condition_name(relay: str, access: str) -> str:
    return f"{relay}_{access}"


SENSITIVITY_CONDS = tuple(condition_name(r, "sender_only_no_filler") for r in NO_FILLER_RELAYS)


def by_example(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        out[row["example_id"]][row["condition"]] = row
    return dict(out)


def accuracy_by_condition(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(int(bool(row["correct"])))
    return {cond: (sum(vals) / len(vals) if vals else float("nan")) for cond, vals in grouped.items()}


def _acc(table: dict[str, float], relay: str, access: str) -> float | None:
    name = condition_name(relay, access)
    if name not in table:
        return None
    return table[name]


def contrasts(table: dict[str, float]) -> dict[str, Any]:
    """D, I, and R from condition accuracies. Missing cells are omitted."""
    out: dict[str, Any] = {"accuracy": {}}
    for access in MAIN_ACCESSES:
        for relay in MAIN_RELAYS:
            val = _acc(table, relay, access)
            if val is not None:
                out["accuracy"][condition_name(relay, access)] = val

    penalties: dict[str, float] = {}
    for access in MAIN_ACCESSES:
        full = _acc(table, "full", access)
        if full is None:
            continue
        for relay in MAIN_RELAYS:
            if relay == "full":
                continue
            other = _acc(table, relay, access)
            if other is None:
                continue
            penalties[condition_name(relay, access)] = full - other
    out["compression_penalty_D"] = penalties

    interaction: dict[str, float] = {}
    for relay in MAIN_RELAYS:
        if relay == "full":
            continue
        d_so = penalties.get(condition_name(relay, "sender_only"))
        d_sh = penalties.get(condition_name(relay, "shared"))
        if d_so is None or d_sh is None:
            continue
        interaction[relay] = d_so - d_sh
    out["interaction_I"] = interaction

    recovery: dict[str, float] = {}
    for budget, evict in EVICTION_AT.items():
        obf = OBF_AT[budget]
        for access in MAIN_ACCESSES:
            a_obf = _acc(table, obf, access)
            a_ev = _acc(table, evict, access)
            if a_obf is None or a_ev is None:
                continue
            recovery[f"{access}_{budget}"] = a_obf - a_ev
    out["obf_recovery_R"] = recovery
    out["lead"] = {
        "note": "Lead with sender-only accuracy, D, and R. Shared A and I interpret those curves. If shared is at ceiling, do not hang the claim on I.",
        "sender_only_accuracy": {
            relay: _acc(table, relay, "sender_only") for relay in MAIN_RELAYS if _acc(table, relay, "sender_only") is not None
        },
    }
    return out


def sensitivity_contrasts(table: dict[str, float]) -> dict[str, Any]:
    """Predeclared no-filler sender-only check at budget 32. Not part of the 800."""
    acc = {}
    for relay in NO_FILLER_RELAYS:
        val = _acc(table, relay, "sender_only_no_filler")
        if val is not None:
            acc[condition_name(relay, "sender_only_no_filler")] = val
        filled = _acc(table, relay, "sender_only")
        if filled is not None:
            acc[condition_name(relay, "sender_only")] = filled
    full_nf = _acc(table, "full", "sender_only_no_filler")
    head_nf = _acc(table, "headwise_32", "sender_only_no_filler")
    hobf_nf = _acc(table, "hobf_32", "sender_only_no_filler")
    penalties = {}
    recovery = None
    if full_nf is not None and head_nf is not None:
        penalties["headwise_32_sender_only_no_filler"] = full_nf - head_nf
    if full_nf is not None and hobf_nf is not None:
        penalties["hobf_32_sender_only_no_filler"] = full_nf - hobf_nf
    if hobf_nf is not None and head_nf is not None:
        recovery = hobf_nf - head_nf
    filler_delta = {}
    for relay in NO_FILLER_RELAYS:
        with_f = _acc(table, relay, "sender_only")
        without = _acc(table, relay, "sender_only_no_filler")
        if with_f is not None and without is not None:
            filler_delta[relay] = with_f - without
    return {
        "accuracy": acc,
        "compression_penalty_D_no_filler": penalties,
        "obf_recovery_R_no_filler_32": recovery,
        "filler_minus_no_filler": filler_delta,
        "note": "Predeclared sensitivity. Lead with the filler sender-only 5×2; use this table to say whether filler changed the budget-32 story.",
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def paired_bootstrap(
    grouped: dict[str, dict[str, dict[str, Any]]],
    *,
    n_boot: int = 10000,
    seed: int = 2026,
    conditions: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Resample examples, keeping every condition for an example together."""
    ids = sorted(grouped)
    if not ids:
        return {"n_examples": 0, "n_boot": n_boot, "intervals": {}}
    conds = list(conditions) if conditions is not None else sorted({c for ex in grouped.values() for c in ex})
    rng = random.Random(seed)
    acc_draws: dict[str, list[float]] = {c: [] for c in conds}
    for _ in range(n_boot):
        sample = [grouped[ids[rng.randrange(len(ids))]] for _ in ids]
        for cond in conds:
            vals = [int(bool(ex[cond]["correct"])) for ex in sample if cond in ex]
            acc_draws[cond].append(sum(vals) / len(vals) if vals else float("nan"))

    intervals = {}
    for cond, draws in acc_draws.items():
        finite = sorted(x for x in draws if x == x)
        intervals[cond] = {
            "mean": sum(finite) / len(finite) if finite else None,
            "ci95": [_percentile(finite, 0.025), _percentile(finite, 0.975)] if finite else None,
        }

    contrast_draws: dict[str, list[float]] = defaultdict(list)
    for i in range(n_boot):
        table = {cond: acc_draws[cond][i] for cond in conds if acc_draws[cond]}
        c = contrasts(table)
        for key, val in c["compression_penalty_D"].items():
            contrast_draws[f"D_{key}"].append(val)
        for key, val in c["interaction_I"].items():
            contrast_draws[f"I_{key}"].append(val)
        for key, val in c["obf_recovery_R"].items():
            contrast_draws[f"R_{key}"].append(val)

    contrast_intervals = {}
    for name, draws in contrast_draws.items():
        finite = sorted(x for x in draws if x == x)
        contrast_intervals[name] = {
            "mean": sum(finite) / len(finite) if finite else None,
            "ci95": [_percentile(finite, 0.025), _percentile(finite, 0.975)] if finite else None,
        }
    return {
        "n_examples": len(ids),
        "n_boot": n_boot,
        "seed": seed,
        "unit": "example",
        "accuracy": intervals,
        "contrasts": contrast_intervals,
    }
