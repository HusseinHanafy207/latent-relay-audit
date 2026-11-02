"""Instruction-only repair of the compact handoff on plan4_dev.

Question: can the designated general eligibility paragraph make B use A's
selected result states, without adding example-specific calculations?

R is the current fixed-prefix transferred baseline: reconstructed five-token
prefix plus A's result-line keys and values. Extra KV is placed after that
prefix and before R, with one frozen newline between the extra block and the
result lines. Equal-size random controls match extra length, piece order, R's
packed slots, and B's starting position.

A is not regenerated. plan4_hold stays out. Do not resume kvresult. Do not
train an adapter. Do not run the calculation-block comparison from this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.boundary_layout import compact_slot_ranges, compose_compact_with_extra
from src.cache_utils import cache_slice_delta, n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
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
    insert_after_framing,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_boundary import (
    PAYLOAD_NOTE,
    _added_token_ids,
    _annotate_plan,
    _carrier_block,
    candidate_qualifies,
)
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta
from src.stage_plan4_kvresult import PREFIX_SOURCE, _rebuild_pair
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _copy_frozen_artifacts
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _family_delta_vs_mean, _process_time_s
from src.stage_select import _load_rows, _paired
from src.support_select import INSTRUCTION_RANDOM_SEEDS, extract_instruction_pack
from src.two_agent import TwoAgentInventory

LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
EXTRA_SEP = True
RAND_CONDS = tuple(f"rand_{seed}" for seed in INSTRUCTION_RANDOM_SEEDS)
CONDS = ("r", "instr_kv", "r_instr_text") + RAND_CONDS + ("reconstruct",)
EQUAL_SIZE = ("instr_kv",) + RAND_CONDS
SPEC = {
    "r": {"kind": "r", "extra": None, "instruction_text": False, "cache_source": "mix_result_a_k_a_v"},
    "instr_kv": {
        "kind": "extra",
        "extra": "instruction",
        "instruction_text": False,
        "cache_source": "transfer_instr_kv",
    },
    "r_instr_text": {
        "kind": "r",
        "extra": None,
        "instruction_text": True,
        "cache_source": "mix_result_a_k_a_v",
    },
    "rand_1": {"kind": "extra", "extra": "random", "seed": 1, "instruction_text": False, "cache_source": "transfer_rand_1"},
    "rand_2": {"kind": "extra", "extra": "random", "seed": 2, "instruction_text": False, "cache_source": "transfer_rand_2"},
    "rand_3": {"kind": "extra", "extra": "random", "seed": 3, "instruction_text": False, "cache_source": "transfer_rand_3"},
    "reconstruct": {
        "kind": "reconstruct",
        "extra": None,
        "instruction_text": False,
        "cache_source": "reconstruct_prefill",
    },
}
PAIRED = (
    ("instr_kv_vs_r", "r", "instr_kv"),
    ("r_instr_text_vs_r", "r", "r_instr_text"),
    ("instr_kv_vs_r_instr_text", "r_instr_text", "instr_kv"),
    ("instr_kv_vs_rand_1", "rand_1", "instr_kv"),
    ("instr_kv_vs_rand_2", "rand_2", "instr_kv"),
    ("instr_kv_vs_rand_3", "rand_3", "instr_kv"),
    ("reconstruct_vs_r", "r", "reconstruct"),
    ("reconstruct_vs_instr_kv", "instr_kv", "reconstruct"),
    ("reconstruct_vs_r_instr_text", "r_instr_text", "reconstruct"),
)
PRESERVED = (
    "results_stage_plan4_kvresult",
    "results_stage_plan4_kvswap",
    "results_stage_plan4_boundary",
    "results_stage_plan4_boundary1",
    "results_stage_plan4_recon",
    "results_stage_plan4_iface",
    "results_stage_plan4_support",
    "results_stage_plan4_hold",
    "results_stage_plan4_stable",
    "results_stage_plan4_fresh",
)
FROZEN_PLACEMENT = (
    "Equal-size extra conditions pack: reconstructed five-token prefix, then the "
    "extra block, then one newline, then R. R-alone, instruction-text, and "
    "reconstructed R omit the extra block and that newline. Instruction text "
    "keeps the R-alone cache and inserts the paragraph after the framing sentence."
)
INTERPRETATION_GUIDE = (
    "Instruction KV helps more than random additions: the designated general "
    "guidance is useful beyond adding the same amount of cache. "
    "Instruction text also recovers performance: clearer receiver instructions "
    "may provide the simpler repair. A learned interface would need a stronger "
    "justification. "
    "Instruction KV helps, but instruction text does not: the cached instruction "
    "representations merit further investigation; this would not by itself prove "
    "a unique mechanism. "
    "Random additions perform similarly: recovery is not specific to the "
    "instruction block. "
    "Neither instruction version helps enough: proceed to the calculation-block "
    "comparison. "
    "These are exploratory development comparisons. Adding context changes "
    "message length and attention, so improvement over R alone does not establish "
    "that instruction content matters. Equal-size random comparisons address that. "
    "The text-instruction condition is a practical alternative; it does not hold "
    "every computation identical."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_instr" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_instr")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


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


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _frozen_refs() -> dict:
    r_alone = {
        "n": 90,
        "plan_all_status_correct": 32,
        "correct": 51,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_kvresult a_k_a_v",
    }
    reconstruct = {
        "n": 90,
        "plan_all_status_correct": 89,
        "correct": 74,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_kvresult reconstruct",
    }
    for folder in (
        REPO_ROOT / "results_stage_plan4_kvresult1" / "summary (1).json",
        REPO_ROOT / "results_stage_plan4_kvresult1" / "summary.json",
        REPO_ROOT / "results_stage_plan4_kvresult" / "summary.json",
        Path("/kaggle/working/results_stage_plan4_kvresult/summary.json"),
    ):
        if not folder.is_file():
            continue
        data = json.loads(folder.read_text(encoding="utf-8"))
        rec = data.get("receiver") or {}
        a_block = rec.get("a_k_a_v") or {}
        r_block = rec.get("reconstruct") or {}
        if int(a_block.get("n") or 0) == 90:
            r_alone = {
                "n": 90,
                "plan_all_status_correct": a_block.get("plan_all_status_correct"),
                "correct": a_block.get("correct"),
                "parse_ok": a_block.get("parse_ok"),
                "truncated": a_block.get("truncated"),
                "source": str(folder).replace("\\", "/"),
            }
        if int(r_block.get("n") or 0) == 90:
            reconstruct = {
                "n": 90,
                "plan_all_status_correct": r_block.get("plan_all_status_correct"),
                "correct": r_block.get("correct"),
                "parse_ok": r_block.get("parse_ok"),
                "truncated": r_block.get("truncated"),
                "source": str(folder).replace("\\", "/"),
            }
        break
    return {"r": r_alone, "reconstruct": reconstruct}


def _ids_at(full_ids: Sequence[int], positions: Sequence[int]) -> list[int]:
    return [int(full_ids[int(p)]) for p in positions]


def _extra_pos(spans: dict, spec: dict) -> list[int]:
    kind = spec.get("extra")
    if kind == "instruction":
        return list(spans["instruction_pos"])
    if kind == "random":
        rec = (spans.get("instruction_random") or {})[str(spec["seed"])]
        return list(rec["positions"])
    raise KeyError(spec)


def _overlap_extra(spans: dict, extra_pos: Sequence[int]) -> dict:
    extra = {int(p) for p in extra_pos}
    instr = set(spans.get("instruction_pos") or [])
    calc = set(spans.get("calc_pos") or [])
    n = len(extra)
    return {
        "overlap_instruction": len(extra & instr),
        "overlap_calc": len(extra & calc),
        "overlap_instruction_frac": (len(extra & instr) / n) if n else 0.0,
        "overlap_calc_frac": (len(extra & calc) / n) if n else 0.0,
        "n_from_prompt": sum(p < int(spans["n_prompt"]) for p in extra_pos),
        "n_from_written": sum(p >= int(spans["n_prompt"]) for p in extra_pos),
    }


def _compose_extra(pair: dict, extra_pos: Sequence[int]) -> dict:
    extra_pos = [int(p) for p in extra_pos]
    a_ids = [int(t) for t in pair["bundle"].full_ids_cpu.view(-1).tolist()]
    extra_ids = _ids_at(a_ids, extra_pos)
    composed = compose_compact_with_extra(
        pair["pack"]["initial_ids"],
        extra_ids,
        pair["pack"]["result_ids"],
        sep_ids=_piece_sep_ids(pair["composed"]),
        extra_pos=extra_pos,
        result_pos=pair["pack"]["split"]["result_pos"],
        extra_sep=EXTRA_SEP,
    )
    slots = compact_slot_ranges(composed)
    if slots["n_prefix"] != 5:
        raise RuntimeError(f"Expected a five-token reconstructed prefix, got {slots['n_prefix']}.")
    if slots["result_start"] != slots["n_prefix"] + slots["n_extra"] + slots["n_extra_sep"]:
        raise RuntimeError("Result packed start drifted from the frozen extra layout.")
    if slots["n_extra"] != len(extra_pos):
        raise RuntimeError("Packed extra length does not match the selected extra positions.")
    return composed, slots


def _piece_sep_ids(composed: dict) -> list[int]:
    for piece in composed.get("pieces") or []:
        if piece.get("kind") == "separator":
            return [int(t) for t in piece["ids"]]
    raise RuntimeError("Sep layout is missing the newline piece.")


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
        proto["condition"] = "rand_mean"
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
        proto["n_kept"] = sum(int(item.get("n_kept") or 0) for item in seeds) / len(seeds)
        proto["n_extra"] = sum(int(item.get("n_extra") or 0) for item in seeds) / len(seeds)
        proto["message_bytes"] = sum(int(item.get("message_bytes") or 0) for item in seeds) / len(seeds)
        proto["process_time_s"] = sum(float(item.get("process_time_s") or 0.0) for item in seeds) / len(seeds)
        proto["overlap_instruction"] = sum(int(item.get("overlap_instruction") or 0) for item in seeds) / len(seeds)
        proto["overlap_calc"] = sum(int(item.get("overlap_calc") or 0) for item in seeds) / len(seeds)
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
        "mean_n_kept": sum(float(item.get("n_kept") or 0) for item in items) / n,
        "mean_n_extra": sum(float(item.get("n_extra") or 0) for item in items) / n,
        "mean_overlap_instruction": sum(float(item.get("overlap_instruction") or 0) for item in items) / n,
        "mean_overlap_calc": sum(float(item.get("overlap_calc") or 0) for item in items) / n,
        "mean_message_bytes": mean_bytes,
        "mean_relay_mb": mean_bytes / (1024 * 1024),
        "mean_process_time_s": sum(float(item.get("process_time_s") or 0) for item in items) / n,
        "note": "Per-example mean of the three random seeds. Seeds are not independent extra cases.",
    }


def _inspect_pack(pack: dict) -> dict:
    instr = pack.get("instruction") or {}
    return {
        "a_state_key": pack.get("a_state_key"),
        "ok": pack.get("ok"),
        "failures": pack.get("failures"),
        "n_instruction": pack.get("n_instruction"),
        "instruction_text": (instr.get("text") or "")[:500],
        "instruction_contains_rule": pack.get("instruction_contains_rule"),
        "instruction_before_facts": pack.get("instruction_before_facts"),
        "n_result": len(pack.get("result_pos") or []),
        "n_retained": len(pack.get("retained_pos") or []),
        "result_text": pack.get("result_text"),
        "random_overlap_instruction": {
            seed: (pack.get("instruction_random") or {}).get(seed, {}).get("overlap_instruction")
            for seed in (str(s) for s in INSTRUCTION_RANDOM_SEEDS)
        },
        "random_overlap_calc": {
            seed: (pack.get("instruction_random") or {}).get(seed, {}).get("overlap_calc")
            for seed in (str(s) for s in INSTRUCTION_RANDOM_SEEDS)
        },
        "sampling_mode": {
            seed: (pack.get("instruction_random") or {}).get(seed, {}).get("sampling_mode")
            for seed in (str(s) for s in INSTRUCTION_RANDOM_SEEDS)
        },
    }


def _construction_summary(packs: list[dict]) -> dict:
    n = len(packs)
    n_ok = sum(bool(p.get("ok")) for p in packs)
    n_instr = [int(p.get("n_instruction") or 0) for p in packs]
    overlaps_i = []
    overlaps_c = []
    modes = []
    for pack in packs:
        for seed in INSTRUCTION_RANDOM_SEEDS:
            rec = (pack.get("instruction_random") or {}).get(str(seed)) or {}
            overlaps_i.append(int(rec.get("overlap_instruction") or 0))
            overlaps_c.append(int(rec.get("overlap_calc") or 0))
            modes.append(rec.get("sampling_mode"))
    return {
        "n": n,
        "n_ok": n_ok,
        "all_ok": bool(n) and n_ok == n,
        "n_a_errors_preserved": sum(bool(p.get("a_error_recorded")) for p in packs),
        "mean_n_instruction": (sum(n_instr) / n) if n else None,
        "min_n_instruction": min(n_instr) if n_instr else None,
        "max_n_instruction": max(n_instr) if n_instr else None,
        "instruction_n_tokens_constant": bool(n_instr) and min(n_instr) == max(n_instr),
        "mean_random_overlap_instruction": (sum(overlaps_i) / len(overlaps_i)) if overlaps_i else None,
        "mean_random_overlap_calc": (sum(overlaps_c) / len(overlaps_c)) if overlaps_c else None,
        "sampling_modes": sorted({m for m in modes if m}),
        "instruction_occurrence": "first",
        "random_seeds": list(INSTRUCTION_RANDOM_SEEDS),
        "extra_separator": EXTRA_SEP,
        "prefix_source": PREFIX_SOURCE,
        "layout": LAYOUT,
        "used_gold_or_b_private": any(p.get("used_gold_or_b_private") for p in packs),
        "inspect": [_inspect_pack(p) for p in packs[:3]],
        "failures": [
            {"a_state_key": p.get("a_state_key"), "failures": p.get("failures")}
            for p in packs
            if not p.get("ok")
        ],
        "frozen_rule": FROZEN_PLACEMENT,
    }


def _summarize(rows: list[dict], construction: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("r", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    receiver["rand_mean"] = _random_mean_block(rows)
    refs = _frozen_refs()
    family = {name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED) for name, a, b in PAIRED}
    family_elig = {
        name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED, key="plan_all_status_correct")
        for name, a, b in PAIRED
    }
    family_elig["instr_kv_vs_rand_mean"] = _family_delta_vs_mean(
        rows, "instr_kv", RAND_CONDS, n_boot=N_BOOT, seed=BOOT_SEED
    )
    family["instr_kv_vs_rand_mean"] = _family_delta_vs_mean(
        rows, "instr_kv", RAND_CONDS, n_boot=N_BOOT, seed=BOOT_SEED, key="correct"
    )
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "n_evals_expected": 90 * len(CONDS),
        "layout": LAYOUT,
        "prefix_source": PREFIX_SOURCE,
        "extra_separator": EXTRA_SEP,
        "baseline": "fixed-prefix transferred R (reconstructed prefix + A's result KV)",
        "unit_note": (
            "plan4_dev only: 30 families with three versions each. plan4_hold is unused. "
            "The three random seeds do not turn 90 cases into 270 independent examples. "
            "Equal-size conditions keep R's packed positions and B's starting position identical."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
        "carrier_label_errors": {cond: _carrier_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {name: _paired(grouped_plain, a, b) for name, a, b in PAIRED},
        "family_clustered_delta": family,
        "family_clustered_eligibility": family_elig,
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "frozen_r": refs["r"],
        "frozen_reconstruct": refs["reconstruct"],
        "qualifies_vs_reconstruct": {
            cond: candidate_qualifies(receiver.get(cond) or {}, refs["reconstruct"]) for cond in CONDS
        },
        "spec": SPEC,
        "construction": None
        if construction is None
        else {
            "n": construction.get("n"),
            "n_ok": construction.get("n_ok"),
            "all_ok": construction.get("all_ok"),
            "mean_n_instruction": construction.get("mean_n_instruction"),
            "instruction_n_tokens_constant": construction.get("instruction_n_tokens_constant"),
            "mean_random_overlap_instruction": construction.get("mean_random_overlap_instruction"),
            "mean_random_overlap_calc": construction.get("mean_random_overlap_calc"),
        },
        "framing_change": FRAMING_INSTRUCTION,
        "frozen_placement": FROZEN_PLACEMENT,
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels. Also report final-choice "
            "accuracy, correct choices with wrong labels, parsing, truncation, "
            "message size, and preparation cost."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "Interpret this instruction-only comparison before any calculation-block "
            "run. Do not load plan4_hold. Do not train an adapter yet."
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


def _prepare_runner(args, out_root: Path, extra: dict) -> tuple:
    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-instr",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "layout": LAYOUT,
            "prefix_source": PREFIX_SOURCE,
            "extra_separator": EXTRA_SEP,
            "conditions": list(CONDS),
            "n_evals": 90 * len(CONDS),
            "framing_change": FRAMING_INSTRUCTION,
            "frozen_placement": FROZEN_PLACEMENT,
            "payload_note": PAYLOAD_NOTE,
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


def _prompts_for(ex, spans: dict, spec: dict):
    plan = b_reason_kv_instruct_prompt(ex)
    if not spec.get("instruction_text"):
        return plan, b_answer_kv_instruct_prompt
    paragraph = ((spans.get("instruction") or {}).get("text") or "")
    plan = insert_after_framing(plan, paragraph)

    def make_answer(example, plan_text, _paragraph=paragraph):
        return insert_after_framing(b_answer_kv_instruct_prompt(example, plan_text), _paragraph)

    return plan, make_answer


def _run_one(runner, ex, extra_a, spans, composed, slots, cond, args, *, bundle, wall, extra_meta):
    spec = SPEC[cond]
    plan_prompt, make_answer = _prompts_for(ex, spans, spec)
    rec = _run_kv_b(
        runner,
        ex,
        bundle,
        plan_max_new=args.plan_max_new,
        answer_max_new=args.max_new_tokens,
        plan_prompt=plan_prompt,
        make_answer_prompt=make_answer,
    )
    rec.update(extra_a)
    rec.update(extra_meta)
    rec["condition"] = cond
    rec["instruction_text"] = bool(spec.get("instruction_text"))
    rec["prefix_source"] = PREFIX_SOURCE
    rec["layout"] = composed.get("layout") if composed else LAYOUT
    rec["layout_separator"] = True
    rec["layout_turn_close"] = False
    rec["extra_separator"] = EXTRA_SEP if spec.get("extra") else False
    rec["n_prefix"] = int(slots["n_prefix"])
    rec["n_extra"] = int(slots.get("n_extra") or 0) if spec.get("extra") else 0
    rec["n_extra_sep"] = int(slots.get("n_extra_sep") or 0) if spec.get("extra") else 0
    rec["n_result_slots"] = int(slots["n_result"])
    rec["result_start"] = int(slots.get("result_start") or slots["result"][0])
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = n_kv_positions(bundle.cache_cpu)
    rec["n_result_kept"] = len(spans["result_pos"])
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["n_added"] = int(composed.get("n_added") or 0) if composed else 1
    rec["n_full"] = int(spans["n_full"])
    rec["selected_text"] = spans["result_text"]
    rec["selected_token_ids"] = list(composed["ids"]) if composed else None
    rec["instruction_paragraph"] = (spans.get("instruction") or {}).get("text")
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
    rec["message_bytes"] = int(bundle.full_relay_bytes) + rec["framing_bytes"]
    if spec.get("instruction_text"):
        rec["message_bytes"] += len(((spans.get("instruction") or {}).get("text") or "").encode("utf-8"))
    rec["cache_source"] = spec["cache_source"]
    rec["shared_across_heads"] = True
    rec["n_kv_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["plan_prompt"] = plan_prompt
    rec["payload_note"] = PAYLOAD_NOTE
    rec["process_time_s"] = _process_time_s(rec)
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    _annotate_plan(rec, ex)
    return rec


def _run_construct(out_root, kaggle_dest, examples, artifacts, decode) -> int:
    unique = []
    seen = set()
    for ex in examples:
        key = a_state_key(ex)
        if key not in seen:
            unique.append(ex)
            seen.add(key)
    packs = []
    slim = {}
    print(f"Constructing instruction spans on {len(unique)} unique A sheets", flush=True)
    for i, ex in enumerate(unique):
        key = a_state_key(ex)
        rec = artifacts[key]
        pack = extract_instruction_pack(
            rec["full_ids"],
            n_prompt=int(rec["n_prompt_positions"]),
            n_full=int(rec.get("n_kv_positions") or len(rec["full_ids"])),
            decode=decode,
            a_state_key=key,
            a_error_recorded=bool(rec.get("a_error_recorded"))
            or not bool((rec.get("written_score") or {}).get("all_four_correct")),
        )
        packs.append(pack)
        slim[key] = pack
        print(
            f"[{i + 1}/{len(unique)}] {key} ok={pack['ok']} n_instr={pack['n_instruction']} "
            f"before_facts={pack['instruction_before_facts']} failures={pack['failures']}",
            flush=True,
        )
    summary = _construction_summary(packs)
    payload = {**summary, "by_a_state": slim}
    _write_json(out_root / "construction.json", payload)
    _persist(out_root, kaggle_dest)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "n",
                    "n_ok",
                    "all_ok",
                    "mean_n_instruction",
                    "instruction_n_tokens_constant",
                    "mean_random_overlap_instruction",
                    "mean_random_overlap_calc",
                )
            },
            indent=2,
        )
    )
    print("Inspected examples:", json.dumps(summary.get("inspect"), indent=2)[:8000])
    print(FROZEN_PLACEMENT)
    return 0 if summary["all_ok"] else 1


def _build_extra_bundle(runner, pair, extra_pos: Sequence[int], *, source: str):
    composed, slots = _compose_extra(pair, extra_pos)
    prefix = runner.prefix_bundle(pair["reconstruct"], slots["n_prefix"])
    bundle, wall, info = runner.assemble_compact_cache(
        pair["bundle"], composed, prefix=prefix, source=source
    )
    prefix_held = cache_slice_delta(bundle.cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
    if not prefix_held:
        raise RuntimeError("Assembled extra cache did not keep the reconstructed prefix.")
    if int(bundle.pos_cursor_cpu.view(-1)[0].item()) != int(composed["n"]):
        raise RuntimeError("B starting position does not match packed extra length.")
    info = dict(info)
    info["prefix_held_from_reconstruct"] = True
    info["result_start"] = slots["result_start"]
    return bundle, wall, composed, slots, info


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, construction) -> int:
    if not construction.get("all_ok") and not args.force_compare:
        print("Construction did not pass. Fix the instruction span before compare, or pass --force-compare.")
        return 2
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in CONDS)}
    print(f"Instruction comparison layout={LAYOUT} rebuilding {len(needed)} unique A caches", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pair = _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids)
        spans = construction["by_a_state"][key]
        if list(spans.get("result_pos") or []) != list(pair["spans"]["result_pos"]):
            raise RuntimeError(f"Instruction-pack result positions drifted from compact R for {key}.")
        pair["instr"] = spans
        r_alone = runner.mix_sender_kv_at(
            pair["transfer"],
            pair["transfer"],
            base=pair["reconstruct"],
            positions=pair["slots"]["result"],
            source="mix_result_a_k_a_v",
        )
        prefix_held = cache_slice_delta(r_alone.cache_cpu, pair["reconstruct"].cache_cpu, pair["slots"]["prefix"])[
            "equal"
        ]
        if not prefix_held:
            raise RuntimeError(f"R-alone did not keep the reconstructed prefix for {key}.")
        mixed = {"r": r_alone, "r_instr_text": r_alone, "reconstruct": pair["reconstruct"]}
        composed = {"r": pair["composed"], "r_instr_text": pair["composed"], "reconstruct": pair["composed"]}
        slots = {"r": pair["slots"], "r_instr_text": pair["slots"], "reconstruct": pair["slots"]}
        times = {
            "r": float(pair["times"]["a_k_a_v"]),
            "r_instr_text": float(pair["times"]["a_k_a_v"]),
            "reconstruct": float(pair["times"]["reconstruct"]),
        }
        info = {}
        for cond in EQUAL_SIZE:
            extra_pos = _extra_pos(spans, SPEC[cond])
            bundle, wall, extra_composed, extra_slots, extra_info = _build_extra_bundle(
                runner, pair, extra_pos, source=SPEC[cond]["cache_source"]
            )
            mixed[cond] = bundle
            composed[cond] = extra_composed
            slots[cond] = extra_slots
            times[cond] = wall + float(pair["times"]["a_k_a_v"])
            info[cond] = extra_info
        starts = {cond: int(slots[cond]["result_start"]) for cond in EQUAL_SIZE}
        cursors = {cond: int(mixed[cond].pos_cursor_cpu.view(-1)[0].item()) for cond in EQUAL_SIZE}
        extras = {cond: int(slots[cond]["n_extra"]) for cond in EQUAL_SIZE}
        if len(set(starts.values())) != 1 or len(set(cursors.values())) != 1 or len(set(extras.values())) != 1:
            raise RuntimeError(f"Equal-size extra conditions drifted for {key}: start={starts} cursor={cursors}")
        pair["mixed"] = mixed
        pair["composed_by"] = composed
        pair["slots_by"] = slots
        pair["times_by"] = times
        pair["info_by"] = info
        states[key] = pair
        print(
            f"  rebuilt {key} n_instr={spans['n_instruction']} result_start={starts['instr_kv']} "
            f"b_pos={cursors['instr_kv']}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [c for c in CONDS if (ex.example_id, c) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        state = states[a_state_key(ex)]
        extra_a = _a_extra(state)
        extra_a["rebuild_time_s"] = float(state["rebuild_time_s"])
        extra_a["ids_source"] = state["meta"].get("ids_source")
        extra_a["reused_a_from"] = (
            state["meta"]["a_state_key"] if state["meta"]["a_state_key"] != ex.example_id else None
        )
        spans = state["instr"]
        for cond in remaining:
            spec = SPEC[cond]
            extra_meta = {
                "span_extract_ok": bool(spans.get("ok")),
                "n_instruction": int(spans.get("n_instruction") or 0),
                "instruction_before_facts": bool(spans.get("instruction_before_facts")),
                "instruction_contains_rule": ELIGIBILITY_RULE
                in ((spans.get("instruction") or {}).get("text") or ""),
                "sampling_mode": None,
                "overlap_instruction": 0,
                "overlap_calc": 0,
                "overlap_instruction_frac": 0.0,
                "overlap_calc_frac": 0.0,
            }
            if spec.get("extra"):
                extra_pos = _extra_pos(spans, spec)
                extra_meta.update(_overlap_extra(spans, extra_pos))
                extra_meta["extra_pos"] = list(extra_pos)
                if spec.get("extra") == "random":
                    rec_rand = (spans.get("instruction_random") or {})[str(spec["seed"])]
                    extra_meta["sampling_mode"] = rec_rand.get("sampling_mode")
                    extra_meta["random_runs"] = rec_rand.get("runs")
            rec_out = _run_one(
                runner,
                ex,
                extra_a,
                spans,
                state["composed_by"][cond],
                state["slots_by"][cond],
                cond,
                args,
                bundle=runner.clone_sender(state["mixed"][cond]),
                wall=state["times_by"].get(cond, 0.0),
                extra_meta=extra_meta,
            )
            rows.append(rec_out)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec_out.get('pred_carrier')} correct={rec_out.get('correct')} "
                f"elig={rec_out.get('plan_all_status_correct')} parse={rec_out.get('parse_ok')} "
                f"n_kept={rec_out.get('n_kept')} result_start={rec_out.get('result_start')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    summary = _summarize(rows, construction)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in list(CONDS) + ["rand_mean"]}, indent=2)[:14000])
    print(json.dumps(summary.get("family_clustered_eligibility"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="Instruction-only compact-handoff comparison on plan4_dev."
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
        print("Instruction comparison uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_instr" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_instr.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "construction.json",
            "a_artifacts.jsonl",
            "manifest.json",
        ):
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
        summary = _summarize(_load_rows(out_root / "evaluations.jsonl"), construction or None)
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:8000])
        return 0

    if args.phase in {"construct", "all"}:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

        def decode(ids):
            return tokenizer.decode(list(ids), skip_special_tokens=False)

        rc = _run_construct(out_root, kaggle_dest, examples, artifacts, decode)
        if rc != 0 and args.phase == "all" and not args.force_compare:
            print("Construction failed; stopping before the seven-condition comparison.")
            return rc
        if args.phase == "construct":
            return rc

    construction_path = out_root / "construction.json"
    if not construction_path.is_file():
        print("Run --phase construct first.")
        return 2
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    policy, runner, revision = _prepare_runner(args, out_root, extra={"phase": args.phase})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} layout={LAYOUT} n={len(examples)}")
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, construction)


if __name__ == "__main__":
    raise SystemExit(main())
