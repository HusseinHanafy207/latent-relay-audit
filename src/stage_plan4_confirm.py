"""Held-out confirmation of the frozen generic-support candidate.

plan4_hold only. Do not overwrite results_stage_plan4_hold. Do not regenerate A.
Construction, B prompts, scoring, and the generic+R selection rule stay fixed.

These 90 example IDs already appeared in results_stage_plan4_hold (13 Oct).
The sender-artifact file is identical. This evaluates the frozen candidate on
previously evaluated holdout families, disjoint from development. It is not a
fresh independent confirmation on untouched examples.

Conditions, same 90 hold examples:
- generic_r: independently prepared generic instruction plus this example's R
- result_kv: reconstructed prefix plus transferred result states only
- full_kv: the rebuilt full sender cache
- text_result: the corresponding RESULT lines as text
- generic_mismatch: generic support plus the paired A-flip R

Do not search more support variants. Do not train an adapter.
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

from src.cache_utils import cache_slice_delta, n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    a_state_key,
    assert_plan4_invariants,
    load_plan4_jsonl,
    overlap_with_prior,
    prior_plan4_examples,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    ELIGIBILITY_RULE,
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_boundary import (
    PAYLOAD_NOTE,
    _added_token_ids,
    _annotate_plan,
    _carrier_block,
)
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta, _run_text_b
from src.stage_plan4_generic import _collect_templates, _prefill_templates
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvresult import PREFIX_SOURCE
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_rdep import (
    _annotate_rdep,
    _bundle_ids,
    _check_row,
    _family_members,
    _rebuild_family,
    _selected_meta,
    _shift_block,
    coerce_sheet_ok,
    construction_ok_from_row,
    partner_a_key,
    sheet_example,
)
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("generic_r", "result_kv", "full_kv", "text_result", "generic_mismatch")
SPEC = {
    "generic_r": {"kind": "kv", "r_source": "this", "cache_source": "generic_instr_same_R", "framing": "instruct"},
    "result_kv": {"kind": "kv", "r_source": "this", "cache_source": "mix_result_a_k_a_v", "framing": "instruct"},
    "full_kv": {"kind": "kv", "r_source": "full", "cache_source": "rebuilt_full", "framing": "instruct"},
    "text_result": {"kind": "text", "r_source": "text", "cache_source": "text", "framing": "baseline"},
    "generic_mismatch": {
        "kind": "kv",
        "r_source": "paired_a_flip",
        "cache_source": "generic_instr_mismatch_R",
        "framing": "instruct",
    },
}
PAIRED = (
    ("generic_r_vs_result_kv", "result_kv", "generic_r"),
    ("generic_r_vs_full_kv", "full_kv", "generic_r"),
    ("generic_r_vs_text_result", "text_result", "generic_r"),
    ("generic_r_vs_generic_mismatch", "generic_mismatch", "generic_r"),
    ("result_kv_vs_text_result", "text_result", "result_kv"),
    ("full_kv_vs_text_result", "text_result", "full_kv"),
)
PRESERVED = (
    "results_stage_plan4_hold",
    "results_stage_plan4_rdep",
    "results_stage_plan4_nbound",
    "results_stage_plan4_neutral",
    "results_stage_plan4_generic",
    "results_stage_plan4_ifresh",
    "results_stage_plan4_instr",
    "results_stage_plan4_kvresult",
    "results_stage_plan4_kvswap",
    "results_stage_plan4_boundary",
    "results_stage_plan4_boundary1",
    "results_stage_plan4_recon",
    "results_stage_plan4_iface",
    "results_stage_plan4_support",
    "results_stage_plan4_stable",
    "results_stage_plan4_fresh",
)
FROZEN_PLACEMENT = (
    "generic_r keeps the independently prepared generic task-context instruction "
    "on the reconstructed five-token prefix plus one newline, then this example's "
    "transferred R. result_kv keeps that reconstructed prefix and A's result "
    "states only. full_kv is the rebuilt completed sender cache. text_result is "
    "the same RESULT lines as ordinary text. generic_mismatch keeps the generic "
    "support and this example's B prompt, substituting the paired A-flip R."
)
INTERPRETATION_GUIDE = (
    "This evaluates the frozen candidate on previously evaluated plan4_hold "
    "families, disjoint from development. All 90 example IDs already appeared "
    "in results_stage_plan4_hold; the sender artifacts are identical. Being "
    "disjoint from development does not make an already examined set untouched "
    "or a fresh independent confirmation. Primary measure: all four eligibility "
    "labels. Compare generic+R with result-only KV, full KV, and result text, "
    "and keep the A-flip R substitution as the message-dependence control. "
    "Do not retune prompts, spans, or selection. Do not overwrite "
    "results_stage_plan4_hold. A later independent confirmation would need "
    "genuinely new families with the method and criteria fixed beforehand."
)


def _paths():
    data = DATA_DIR / "plan4_hold.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_confirm" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_confirm")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("check_summary.json", "construction.json"):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _rewrite_check_export(out_root: Path, extra_dest: Path | None = None, *, persist: bool = True) -> int:
    check_path = out_root / "check_summary.json"
    if not check_path.is_file():
        print(f"Missing {check_path}. Cannot rewrite construction ok without a check file.")
        return 2
    check = json.loads(check_path.read_text(encoding="utf-8"))
    rows = check.get("rows") or []
    if not rows:
        print("check_summary.json has no per-sheet rows.")
        return 2
    changed = []
    for row in rows:
        new_ok = construction_ok_from_row(row)
        if bool(row.get("ok")) != new_ok:
            changed.append(
                {
                    "a_state_key": row.get("a_state_key"),
                    "old_ok": row.get("ok"),
                    "new_ok": new_ok,
                    "span_extract_ok": row.get("span_extract_ok"),
                    "span_failures": row.get("span_failures"),
                }
            )
        row["ok"] = bool(new_ok)
    n_ok = sum(coerce_sheet_ok(row.get("ok")) for row in rows)
    all_passed = bool(rows) and all(coerce_sheet_ok(row.get("ok")) for row in rows)
    check["rows"] = rows
    check["n"] = len(rows)
    check["all_passed"] = all_passed
    construction = check.get("construction") or {}
    construction["n_ok"] = n_ok
    construction["all_passed"] = all_passed
    construction["n_unique_a_sheets"] = len(rows)
    check["construction"] = construction
    _write_json(check_path, check)
    cons_path = out_root / "construction.json"
    if cons_path.is_file():
        cons = json.loads(cons_path.read_text(encoding="utf-8"))
        cons["n_ok"] = n_ok
        cons["all_passed"] = all_passed
        cons["all_ok"] = all_passed
        _write_json(cons_path, cons)
    if persist:
        _persist(out_root, extra_dest)
    print(
        json.dumps(
            {
                "n_unique_a_sheets": len(rows),
                "n_ok": n_ok,
                "all_passed": all_passed,
                "n_changed": len(changed),
                "changed": changed,
            },
            indent=2,
        )
    )
    return 0 if all_passed else 1


def _copy_hold_artifacts(out_root: Path, *, explicit: str | None = None) -> Path:
    dest = out_root / "a_artifacts.jsonl"
    if dest.is_file():
        return dest
    paths = []
    if explicit:
        paths.append(Path(explicit))
    paths.extend(
        [
            REPO_ROOT / "results_stage_plan4_hold" / "a_artifacts.jsonl",
            RESULTS_DIR / "stage_plan4_hold" / "run" / "a_artifacts.jsonl",
            Path("/kaggle/working/results_stage_plan4_hold/a_artifacts.jsonl"),
        ]
    )
    for src in paths:
        if src.is_file() and src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
            print(f"Copied frozen hold A artifacts from {src}")
            return dest
    raise FileNotFoundError(
        "Need results_stage_plan4_hold/a_artifacts.jsonl with recorded token IDs. "
        "Do not regenerate A's answers."
    )


def _development_refs() -> dict:
    return {
        "generic_r": {
            "n": 90,
            "plan_all_status_correct": 87,
            "correct": 69,
            "source": "results_stage_plan4_rdep generic_r / results_stage_plan4_nbound instr_generic",
        },
        "result_kv": {
            "n": 90,
            "plan_all_status_correct": 32,
            "correct": 51,
            "source": "results_stage_plan4_kvresult a_k_a_v / results_stage_plan4_instr r",
        },
        "full_kv": {
            "n": 90,
            "plan_all_status_correct": 90,
            "correct": 82,
            "source": "results_stage_plan4_iface kv",
        },
        "text_result": {
            "n": 90,
            "plan_all_status_correct": 90,
            "correct": 79,
            "source": "results_stage_plan4_recon text_result: 90/90 labels, 79/90 final choices",
        },
        "generic_mismatch": {
            "n": 90,
            "plan_all_status_correct": 1,
            "n_all_match_src": 86,
            "n_match_src_better": 88,
            "source": "results_stage_plan4_rdep generic_mismatch",
        },
    }


def _attach_baselines(runner, state: dict) -> None:
    slots = state["slots"]
    r_alone = runner.mix_sender_kv_at(
        state["transfer"],
        state["transfer"],
        base=state["reconstruct"],
        positions=slots["result"],
        source="mix_result_a_k_a_v",
    )
    prefix_held = cache_slice_delta(
        r_alone.cache_cpu, state["reconstruct"].cache_cpu, slots["prefix"]
    )["equal"]
    if not prefix_held:
        raise RuntimeError("Result-only KV did not keep the reconstructed prefix.")
    state["mixed"]["result_kv"] = r_alone
    state["mixed"]["full_kv"] = state["bundle"]
    state["result_kv_prefix_held"] = True
    state["times_by"]["result_kv"] = float((state.get("times") or {}).get("a_k_a_v") or 0.0)
    state["times_by"]["full_kv"] = float(state.get("rebuild_time_s") or 0.0)
    state["times_by"]["text_result"] = 0.0


def _confirm_check_row(state: dict) -> dict:
    row = _check_row(state)
    row["ok"] = coerce_sheet_ok(row.get("ok")) and bool(state.get("result_kv_prefix_held"))
    row["result_kv_prefix_held"] = bool(state.get("result_kv_prefix_held"))
    row["n_result_compact"] = int(state["slots"]["n_result"])
    return row


def _rebuild_confirm_family(
    runner,
    family_id,
    members,
    artifacts,
    by_id,
    decode,
    sep_ids,
    close_ids,
    spans_by_key,
    generics,
    *,
    require_ok: bool = True,
):
    states = _rebuild_family(
        runner,
        family_id,
        members,
        artifacts,
        by_id,
        decode,
        sep_ids,
        close_ids,
        spans_by_key,
        generics,
        require_ok=False,
    )
    for key in (f"{family_id}-base", f"{family_id}-a_flip"):
        _attach_baselines(runner, states[key])
        states[key]["check_row"] = _confirm_check_row(states[key])
        if require_ok and not states[key]["check_row"]["ok"]:
            raise RuntimeError(f"Hold confirmation construction failed for {key}.")
    return states


def _summarize_check(rows: list[dict], templates: dict) -> dict:
    n = len(rows)
    return {
        "n_unique_a_sheets": n,
        "n_ok": sum(coerce_sheet_ok(r.get("ok")) for r in rows),
        "all_passed": bool(n) and all(coerce_sheet_ok(r.get("ok")) for r in rows),
        "n_templates": len(templates),
        "templates": {
            name: {
                "n": rec["n"],
                "n_prefix": rec["n_prefix"],
                "instruction_start": rec["instruction_start"],
                "instruction_end": rec["instruction_end"],
                "difficulties": rec["difficulties"],
            }
            for name, rec in templates.items()
        },
        "n_ids_equal": sum(bool(r.get("ids_equal")) for r in rows),
        "n_result_equal": sum(bool(r.get("result_equal")) for r in rows),
        "n_prefix_equal": sum(bool(r.get("prefix_equal")) for r in rows),
        "n_mismatch_ok": sum(bool(r.get("mismatch_ok")) for r in rows),
        "n_result_kv_prefix_held": sum(bool(r.get("result_kv_prefix_held")) for r in rows),
        "n_result_len_differ": sum(bool(r.get("result_len_differ")) for r in rows),
        "mean_n_instruction": (sum(int(r.get("n_instruction") or 0) for r in rows) / n) if n else None,
        "note": (
            "Hold A sheets are reused, not rewritten. Generic prefixes are independent "
            "prefills of the recorded hold tokens through the eligibility instruction. "
            "Per-sheet ok must be a Boolean."
        ),
    }


def _summarize(rows: list[dict], *, check: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("generic_r", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "n_evals_expected": 90 * len(CONDS),
        "layout": LAYOUT,
        "prefix_source": PREFIX_SOURCE,
        "extra_separator": EXTRA_SEP,
        "split": "plan4_hold",
        "baseline": "frozen generic task-context instruction plus this example's transferred R",
        "unit_note": (
            "plan4_hold only. 30 families with three versions each, disjoint from "
            "plan4_dev and plan4_fresh, but previously evaluated in "
            "results_stage_plan4_hold on 13 Oct with identical sender artifacts. "
            "Same written A on every branch. Construction, instruct B prompts, "
            "scoring, and generic+R selection stay frozen."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
        "carrier_label_errors": {cond: _carrier_block(by_cond.get(cond, [])) for cond in CONDS},
        "label_shift": {cond: _shift_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "by_variant": {},
        "paired_receiver": {name: _paired(grouped_plain, a, b) for name, a, b in PAIRED},
        "family_clustered_delta": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED) for name, a, b in PAIRED
        },
        "family_clustered_eligibility": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED, key="plan_all_status_correct")
            for name, a, b in PAIRED
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "development_reference": _development_refs(),
        "spec": SPEC,
        "check": None
        if check is None
        else {
            "all_passed": check.get("all_passed"),
            "n": check.get("n"),
            **(check.get("construction") or {}),
        },
        "framing_change": FRAMING_INSTRUCTION,
        "frozen_placement": FROZEN_PLACEMENT,
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels. Evaluate the frozen "
            "generic+R candidate on previously evaluated plan4_hold families "
            "against result-only KV, full KV, result text, and paired A-flip R. "
            "Not a fresh independent confirmation."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "Keep the candidate frozen. Generic support substantially improved "
            "selected-state reuse; complete recovery to full KV and result text "
            "is unestablished (paired intervals include values beyond a five-point "
            "gap). Next is the 4- and 12-carrier timing comparison, reporting that "
            "accuracy gap alongside costs. Do not train an adapter. Do not add "
            "the calculation block. Do not search more support variants."
        ),
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _cond_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            },
            "label_shift": {
                cond: _shift_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            },
        }
    for variant in ("base", "a_flip", "b_flip"):
        out["by_variant"][variant] = {
            "receiver": {
                cond: _cond_block(_subset(by_cond.get(cond, []), variant=variant)) for cond in CONDS
            },
            "label_shift": {
                cond: _shift_block(_subset(by_cond.get(cond, []), variant=variant)) for cond in CONDS
            },
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
            "stage": "plan4-confirm",
            "split": "plan4_hold",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_dev.jsonl", "data/plan4_fresh.jsonl"],
            "do_not_overwrite": ["results_stage_plan4_hold"],
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


def _run_one(runner, ex, extra_a, state, cond, args, *, src_ex):
    spans = state["instr"]
    spec = SPEC[cond]
    extra_meta = {
        "span_extract_ok": bool(spans.get("ok")),
        "n_instruction": int(spans.get("n_instruction") or 0),
        "instruction_before_facts": bool(spans.get("instruction_before_facts")),
        "instruction_contains_rule": ELIGIBILITY_RULE
        in ((spans.get("instruction") or {}).get("text") or ""),
        "template_id": spans.get("template_id"),
        "partner_key": state.get("partner_key"),
        "r_source": spec["r_source"],
        "cache_source": spec["cache_source"],
    }
    if cond == "text_result":
        rec = _run_text_b(
            runner,
            ex,
            message=spans.get("result_text") or "",
            plan_max_new=args.plan_max_new,
            answer_max_new=args.max_new_tokens,
            allow_a_source=True,
        )
        rec.update(extra_a)
        rec.update(extra_meta)
        rec["condition"] = cond
        rec["n_kept"] = 0
        rec["n_kv_positions"] = 0
        rec["n_result_kept"] = len(spans.get("result_pos") or [])
        rec["n_result_slots"] = 0
        rec["selected_text"] = spans.get("result_text") or ""
        rec["selected_token_ids"] = list(state["spans"].get("selected_ids") or [])
        rec["selection_time_s"] = 0.0
        rec["compression_time_s"] = 0.0
        rec["layout"] = None
        rec["framing"] = "baseline"
        rec["framing_instruction"] = None
        rec["prefix_source"] = None
        rec["extra_separator"] = EXTRA_SEP
        rec["message_bytes"] = len((spans.get("result_text") or "").encode("utf-8"))
        rec["plan_prompt"] = None
        rec["payload_note"] = PAYLOAD_NOTE
        rec["process_time_s"] = _process_time_s(rec)
        rec["peak_gpu_gb"] = peak_gpu_memory_gb()
        rec["result_len_differ"] = False
        _annotate_plan(rec, ex)
        _annotate_rdep(rec, ex, src_ex)
        return rec

    if cond in ("generic_r", "generic_mismatch"):
        selected = _selected_meta(state, cond, state.get("_other"))
        extra_meta.update(selected)
        composed = state["composed_extra"]
        slots = state["slots_extra"]
        bundle = runner.clone_sender(state["mixed"][cond])
        wall = float(state["times_by"].get(cond) or 0.0)
    elif cond == "result_kv":
        composed = state["composed"]
        slots = state["slots"]
        bundle = runner.clone_sender(state["mixed"][cond])
        wall = float(state["times_by"].get(cond) or 0.0)
        extra_meta.update(
            {
                "n_result_slots": int(slots["n_result"]),
                "n_result_src": int(slots["n_result"]),
                "result_len_differ": False,
                "selected_text": spans.get("result_text") or "",
                "selected_token_ids": list(composed["ids"]),
                "result_held_from_transferred": True,
            }
        )
    else:
        composed = {"layout": None, "n_added": 0, "ids": _bundle_ids(state["bundle"])}
        slots = {
            "n_prefix": 0,
            "n_extra": 0,
            "n_extra_sep": 0,
            "n_result": len(spans.get("result_pos") or []),
            "result_start": 0,
        }
        bundle = runner.clone_sender(state["mixed"][cond])
        wall = float(state["times_by"].get(cond) or 0.0)
        extra_meta.update(
            {
                "n_result_slots": int(slots["n_result"]),
                "n_result_src": int(slots["n_result"]),
                "result_len_differ": False,
                "selected_text": spans.get("result_text") or "",
                "selected_token_ids": _bundle_ids(bundle),
                "result_held_from_transferred": True,
            }
        )

    rec = _run_kv_b(
        runner,
        ex,
        bundle,
        plan_max_new=args.plan_max_new,
        answer_max_new=args.max_new_tokens,
        plan_prompt=b_reason_kv_instruct_prompt(ex),
        make_answer_prompt=b_answer_kv_instruct_prompt,
    )
    rec.update(extra_a)
    rec.update(extra_meta)
    rec["condition"] = cond
    rec["prefix_source"] = PREFIX_SOURCE if cond != "full_kv" else "full"
    rec["layout"] = composed.get("layout")
    rec["layout_separator"] = cond != "full_kv"
    rec["extra_separator"] = EXTRA_SEP
    rec["n_prefix"] = int(slots.get("n_prefix") or 0)
    rec["n_extra"] = int(slots.get("n_extra") or 0)
    rec["n_extra_sep"] = int(slots.get("n_extra_sep") or 0)
    rec["n_result_this"] = int(state["slots_extra"]["n_result"])
    rec["n_result_slots"] = int(extra_meta.get("n_result_slots") or 0)
    rec["result_start"] = int(slots.get("result_start") or 0)
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = n_kv_positions(bundle.cache_cpu)
    rec["n_result_kept"] = int(extra_meta.get("n_result_slots") or 0)
    rec["n_full"] = int(spans["n_full"])
    rec["template_id"] = spans.get("template_id")
    rec["selected_text"] = extra_meta.get("selected_text")
    rec["selected_token_ids"] = extra_meta.get("selected_token_ids") or _bundle_ids(bundle)
    rec["framing"] = spec["framing"]
    rec["framing_instruction"] = FRAMING_INSTRUCTION if spec["framing"] == "instruct" else None
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8")) if spec["framing"] == "instruct" else 0
    rec["message_bytes"] = int(bundle.full_relay_bytes) + int(rec["framing_bytes"])
    rec["shared_across_heads"] = True
    rec["n_kv_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["plan_prompt"] = b_reason_kv_instruct_prompt(ex)
    rec["payload_note"] = PAYLOAD_NOTE
    rec["process_time_s"] = _process_time_s(rec)
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    rec["result_len_differ"] = bool(extra_meta.get("result_len_differ"))
    _annotate_plan(rec, ex)
    _annotate_rdep(rec, ex, src_ex)
    return rec


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    templates, spans_by_key = _collect_templates(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    families = _family_members(examples)
    rows = []
    print(
        f"Hold confirmation check on {len(artifacts)} unique A sheets / {len(families)} families, layout={LAYOUT}",
        flush=True,
    )
    for i, family_id in enumerate(sorted(families)):
        members = families[family_id]
        states = _rebuild_confirm_family(
            runner,
            family_id,
            members,
            artifacts,
            by_id,
            decode,
            sep_ids,
            close_ids,
            spans_by_key,
            generics,
            require_ok=False,
        )
        for key in (f"{family_id}-base", f"{family_id}-a_flip"):
            row = states[key]["check_row"]
            rows.append(row)
            print(
                f"[{i + 1}/{len(families)}] {key} ok={row['ok']} partner={row['partner_key']} "
                f"n_R={row['n_result']} compact_R={row.get('n_result_compact')} "
                f"mismatch={row['mismatch_ok']} result_kv={row.get('result_kv_prefix_held')} "
                f"span_ok={row.get('span_extract_ok')}",
                flush=True,
            )
            if not row["ok"] or row.get("span_extract_ok") is False:
                print(f"    span_failures={row.get('span_failures')}", flush=True)
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _persist(out_root, kaggle_dest)

    construction = _summarize_check(rows, templates)
    slim_spans = {}
    for key, spans in spans_by_key.items():
        slim_spans[key] = {
            "ok": spans.get("ok"),
            "n_instruction": spans.get("n_instruction"),
            "instruction_pos": spans.get("instruction_pos"),
            "result_pos": spans.get("result_pos"),
            "result_text": spans.get("result_text"),
            "n_full": spans.get("n_full"),
            "causal_prefix": spans.get("causal_prefix"),
            "template_id": spans.get("template_id"),
            "a_state_key": spans.get("a_state_key"),
        }
    summary = {
        "n": construction["n_unique_a_sheets"],
        "all_passed": construction["all_passed"],
        "layout": LAYOUT,
        "construction": construction,
        "rows": rows,
        "lead": (
            "If all_passed, run --phase compare: generic+R, result-only KV, full KV, "
            "result text, and paired A-flip R on plan4_hold."
        ),
        "frozen_placement": FROZEN_PLACEMENT,
    }
    _write_json(
        out_root / "construction.json",
        {
            **construction,
            "by_a_state": slim_spans,
            "templates": {name: {k: v for k, v in rec.items() if k != "bundle"} for name, rec in generics.items()},
            "all_ok": construction["all_passed"],
        },
    )
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(
        json.dumps(
            {
                k: construction[k]
                for k in (
                    "n_unique_a_sheets",
                    "all_passed",
                    "n_templates",
                    "n_result_equal",
                    "n_mismatch_ok",
                    "n_result_kv_prefix_held",
                )
            },
            indent=2,
        )
    )
    return 0 if construction["all_passed"] else 1


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    check_path = out_root / "check_summary.json"
    check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.is_file() else None
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first. Pass --force-compare only to skip that gate.")
        return 2
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    templates, spans_by_key = _collect_templates(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    families = _family_members(examples)
    print(f"Hold confirmation compare layout={LAYOUT} n_families={len(families)} n_evals={90 * len(CONDS)}", flush=True)

    for i, family_id in enumerate(sorted(families)):
        members = families[family_id]
        remaining_any = any((ex.example_id, c) not in done for ex in members for c in CONDS)
        if not remaining_any:
            print(f"[{i + 1}/{len(families)}] {family_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(families)}] {family_id} rebuilding paired A sheets", flush=True)
        states = _rebuild_confirm_family(
            runner,
            family_id,
            members,
            artifacts,
            by_id,
            decode,
            sep_ids,
            close_ids,
            spans_by_key,
            generics,
            require_ok=True,
        )
        for ex in members:
            remaining = [c for c in CONDS if (ex.example_id, c) not in done]
            if not remaining:
                print(f"  {ex.example_id} already complete", flush=True)
                continue
            print(f"  {ex.example_id} remaining={len(remaining)}", flush=True)
            reset_peak_memory()
            key = a_state_key(ex)
            state = states[key]
            state["_other"] = states[partner_a_key(key)]
            extra_a = _a_extra(state)
            extra_a["rebuild_time_s"] = float(state["rebuild_time_s"])
            extra_a["ids_source"] = state["meta"].get("ids_source")
            extra_a["reused_a_from"] = (
                state["meta"]["a_state_key"] if state["meta"]["a_state_key"] != ex.example_id else None
            )
            src_by_cond = {
                "generic_r": sheet_example(members, key),
                "result_kv": sheet_example(members, key),
                "full_kv": sheet_example(members, key),
                "text_result": sheet_example(members, key),
                "generic_mismatch": sheet_example(members, partner_a_key(key)),
            }
            for cond in remaining:
                rec_out = _run_one(
                    runner,
                    ex,
                    extra_a,
                    state,
                    cond,
                    args,
                    src_ex=src_by_cond[cond],
                )
                rows.append(rec_out)
                done.add((ex.example_id, cond))
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                print(
                    f"    {cond}: pred={rec_out.get('pred_carrier')} correct={rec_out.get('correct')} "
                    f"elig={rec_out.get('plan_all_status_correct')} "
                    f"match_this={rec_out.get('n_status_match_this')} "
                    f"match_src={rec_out.get('n_status_match_src')} parse={rec_out.get('parse_ok')}",
                    flush=True,
                )
                _persist(out_root, kaggle_dest)
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = _summarize(rows, check=check)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:12000])
    print(json.dumps(summary.get("label_shift"), indent=2)[:8000])
    print(json.dumps(summary.get("family_clustered_eligibility"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="Confirm the frozen generic+R candidate on plan4_hold."
    )
    parser.add_argument("--split", default="plan4_hold")
    parser.add_argument(
        "--phase",
        choices=("check", "compare", "all", "rescore", "rewrite_check"),
        default="check",
    )
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args()
    if args.split != "plan4_hold":
        print("Confirmation uses plan4_hold only. Do not load plan4_dev or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_confirm" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_confirm.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "check_summary.json",
            "construction.json",
            "a_artifacts.jsonl",
            "manifest.json",
        ):
            src = kaggle_dest / name
            dest = out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    if args.phase == "rewrite_check":
        shared = REPO_ROOT / "results_stage_plan4_confirm"
        seen = set()
        rc = 2
        for path, do_persist in (
            (out_root, True),
            (out_root.parent / "latest", False),
            (shared, False),
            (kaggle_dest, False),
        ):
            if path is None or not (path / "check_summary.json").is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            print(f"Rewriting construction ok in {path}", flush=True)
            rc = _rewrite_check_export(path, kaggle_dest if do_persist else None, persist=do_persist)
        if not seen:
            print("No check_summary.json found to rewrite.")
            return 2
        return rc

    if args.phase == "rescore":
        shared = REPO_ROOT / "results_stage_plan4_confirm"
        seen = set()
        rc = 2
        for path, do_persist in (
            (out_root, True),
            (out_root.parent / "latest", False),
            (shared, False),
            (kaggle_dest, False),
        ):
            if path is None or not (path / "evaluations.jsonl").is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            check = None
            cpath = path / "check_summary.json"
            if cpath.is_file():
                check = json.loads(cpath.read_text(encoding="utf-8"))
            summary = _summarize(_load_rows(path / "evaluations.jsonl"), check=check)
            _write_json(path / "summary.json", summary)
            if do_persist:
                _persist(path, kaggle_dest)
            print(f"Rewrote reporting fields in {path}", flush=True)
            print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:8000])
            refs = summary.get("development_reference") or {}
            print(json.dumps(refs.get("text_result"), indent=2))
            rc = 0
        if not seen:
            print("No evaluations.jsonl found to rescore.")
            return 2
        return rc

    examples = load_plan4_jsonl(data)
    for ex in examples:
        if ex.split != "plan4_hold":
            print(f"Refusing {ex.example_id}.")
            return 2
        assert_plan4_invariants(ex)
    if len(examples) != 90:
        print(f"Expected 90 plan4_hold examples, got {len(examples)}.")
        return 2
    overlap = overlap_with_prior(examples, prior_plan4_examples())
    if any(overlap[k] for k in ("n_sheet_overlap", "n_example_id_overlap", "n_family_id_overlap")):
        print(f"plan4_hold overlaps a previously used split: {overlap}")
        return 2

    artifacts_path = _copy_hold_artifacts(out_root, explicit=args.artifacts)
    artifacts = _load_messages(artifacts_path)
    if len(artifacts) != 60:
        print(f"Expected 60 unique frozen hold A sheets, got {len(artifacts)}.")
        return 2
    if any(not str(key).startswith("plan4_hold") for key in artifacts):
        print("Refusing non-hold A artifacts. Do not copy plan4_dev iface sheets.")
        return 2
    for rec in artifacts.values():
        if not rec.get("full_ids"):
            print("Frozen hold artifacts are missing token IDs. Do not regenerate A.")
            return 2

    policy, runner, revision = _prepare_runner(args, out_root, extra={"phase": args.phase})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} layout={LAYOUT} n={len(examples)}")
    if args.phase == "check":
        return _run_check(args, out_root, kaggle_dest, examples, runner, artifacts)
    if args.phase == "compare":
        return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)
    rc = _run_check(args, out_root, kaggle_dest, examples, runner, artifacts)
    if rc != 0 and not args.force_compare:
        print("Check failed; stopping.")
        return rc
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
