"""Reuse independently prepared generic instruction states on plan4_dev.

Question: can instruction states built from shared task text, without shipment
facts, replace per-example transferred instruction states while keeping A's
selected result states?

Three conditions, same R tensors, packed slots, B prompt, and decoding:
- instr_kv: transferred instruction states from each sender's full cache
- instr_fresh: the same instruction tokens processed on the short reconstructed prefix
- instr_generic: one independent prefill per shared causal prefix, reused

Do not add calculations. Do not train an adapter. Do not run the neutral-context
or R-dependence controls from this file. plan4_hold stays out.
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
from src.generic_select import (
    N_EXPECTED_TEMPLATES,
    causal_prefix_from_ids,
    group_generic_templates,
    template_id_for,
)
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
from src.stage_plan4_ifresh import _build_fresh_from_transferred, _extra_sep_ids
from src.stage_plan4_instr import EXTRA_SEP, _build_extra_bundle
from src.stage_plan4_kvresult import PREFIX_SOURCE, _rebuild_pair
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _copy_frozen_artifacts
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.support_select import extract_instruction_pack
from src.two_agent import TwoAgentInventory

LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("instr_kv", "instr_fresh", "instr_generic")
SPEC = {
    "instr_kv": {"instruction": "transfer", "cache_source": "transfer_instr_kv"},
    "instr_fresh": {"instruction": "fresh", "cache_source": "fresh_instr_same_R"},
    "instr_generic": {"instruction": "generic", "cache_source": "generic_instr_same_R"},
}
PAIRED = (
    ("instr_generic_vs_instr_kv", "instr_kv", "instr_generic"),
    ("instr_generic_vs_instr_fresh", "instr_fresh", "instr_generic"),
    ("instr_fresh_vs_instr_kv", "instr_kv", "instr_fresh"),
)
PRESERVED = (
    "results_stage_plan4_ifresh",
    "results_stage_plan4_instr",
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
    "All three keep the reconstructed five-token prefix, one newline, and the same "
    "transferred R. instr_kv gathers instruction states from each sender's full cache. "
    "instr_fresh forwards the instruction tokens on that short prefix. instr_generic "
    "prefills the original tokens only through the eligibility instruction, once per "
    "shared template, then relocates those states onto the same packed instruction slots."
)
INTERPRETATION_GUIDE = (
    "If rebuilt generic context reproduces approximately 87/90, the support can be "
    "prepared independently and reused; A still contributes the example-specific results. "
    "If it does not reproduce the original states or behavior, first investigate "
    "construction and numerical differences. Given identical causal prefixes, that "
    "would not immediately support a new semantic explanation. "
    "This is a construction and reuse check. Reproducing a causal prefix is expected. "
    "Do not treat it as a novel mechanism. Do not run the neutral-context comparison "
    "or the R-dependence controls until this construction is interpreted."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_generic" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_generic")
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


def _frozen_refs() -> dict:
    instr_kv = {
        "n": 90,
        "plan_all_status_correct": 87,
        "correct": 70,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_ifresh instr_kv",
    }
    instr_fresh = {
        "n": 90,
        "plan_all_status_correct": 71,
        "correct": 64,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_ifresh instr_fresh",
    }
    reconstruct = {
        "n": 90,
        "plan_all_status_correct": 89,
        "correct": 74,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_instr reconstruct",
    }
    for folder, kv_name, fresh_name, recon_name in (
        (REPO_ROOT / "results_stage_plan4_ifresh" / "summary.json", "instr_kv", "instr_fresh", None),
        (Path("/kaggle/working/results_stage_plan4_ifresh/summary.json"), "instr_kv", "instr_fresh", None),
        (REPO_ROOT / "results_stage_plan4_instr" / "summary.json", "instr_kv", None, "reconstruct"),
    ):
        if not folder.is_file():
            continue
        data = json.loads(folder.read_text(encoding="utf-8"))
        rec = data.get("receiver") or {}
        if kv_name:
            block = rec.get(kv_name) or {}
            if int(block.get("n") or 0) == 90:
                instr_kv = {
                    "n": 90,
                    "plan_all_status_correct": block.get("plan_all_status_correct"),
                    "correct": block.get("correct"),
                    "parse_ok": block.get("parse_ok"),
                    "truncated": block.get("truncated"),
                    "source": str(folder).replace("\\", "/"),
                }
        if fresh_name:
            block = rec.get(fresh_name) or {}
            if int(block.get("n") or 0) == 90:
                instr_fresh = {
                    "n": 90,
                    "plan_all_status_correct": block.get("plan_all_status_correct"),
                    "correct": block.get("correct"),
                    "parse_ok": block.get("parse_ok"),
                    "truncated": block.get("truncated"),
                    "source": str(folder).replace("\\", "/"),
                }
        if recon_name:
            block = rec.get(recon_name) or {}
            if int(block.get("n") or 0) == 90:
                reconstruct = {
                    "n": 90,
                    "plan_all_status_correct": block.get("plan_all_status_correct"),
                    "correct": block.get("correct"),
                    "parse_ok": block.get("parse_ok"),
                    "truncated": block.get("truncated"),
                    "source": str(folder).replace("\\", "/"),
                }
    return {"instr_kv": instr_kv, "instr_fresh": instr_fresh, "reconstruct": reconstruct}


def _extract_spans(rec, decode, key: str) -> dict:
    return extract_instruction_pack(
        rec["full_ids"],
        n_prompt=int(rec["n_prompt_positions"]),
        n_full=int(rec.get("n_kv_positions") or len(rec["full_ids"])),
        decode=decode,
        a_state_key=key,
        a_error_recorded=bool(rec.get("a_error_recorded"))
        or not bool((rec.get("written_score") or {}).get("all_four_correct")),
    )


def _build_generic_from_transferred(runner, pair, transferred, composed, slots, generic, extra_pos):
    prefix = runner.prefix_bundle(pair["reconstruct"], slots["n_prefix"])
    extra_ids = [int(t) for t in composed["extra_ids"]]
    if extra_ids != [int(t) for t in generic["instruction_ids"]]:
        raise RuntimeError("Generic instruction token IDs do not match this example's instruction tokens.")
    mid = runner.concat_gathered_block(
        prefix,
        generic["bundle"],
        src_pos=extra_pos,
        token_ids=extra_ids,
        source="generic_instr",
    )
    head, wall = runner.extend_cache_with_tokens(mid, _extra_sep_ids(composed), source="generic_instr")
    if n_kv_positions(head.cache_cpu) != int(slots["result_start"]):
        raise RuntimeError(
            f"Generic instruction+newline length {n_kv_positions(head.cache_cpu)} "
            f"!= result start {slots['result_start']}."
        )
    bundle = runner.append_packed_slots(
        head, transferred, tail_positions=slots["result"], source="generic_instr_same_R"
    )
    got = [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]
    if got != list(composed["ids"]):
        raise RuntimeError("Generic instruction cache token IDs drifted from the transferred layout.")
    prefix_held = cache_slice_delta(bundle.cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
    r_held = cache_slice_delta(bundle.cache_cpu, transferred.cache_cpu, slots["result"])["equal"]
    extra_vs_kv = cache_slice_delta(bundle.cache_cpu, transferred.cache_cpu, slots["extra"])
    if not prefix_held:
        raise RuntimeError("Generic instruction cache did not keep the reconstructed prefix.")
    if not r_held:
        raise RuntimeError("Generic instruction cache did not keep the transferred R tensors.")
    if int(bundle.pos_cursor_cpu.view(-1)[0].item()) != int(composed["n"]):
        raise RuntimeError("B starting position drifted after splicing R onto generic instruction.")
    info = {
        "prefix_held_from_reconstruct": True,
        "result_held_from_transferred": True,
        "extra_equal": extra_vs_kv["equal"],
        "extra_keys_max_abs": extra_vs_kv["keys_max_abs"],
        "extra_values_max_abs": extra_vs_kv["values_max_abs"],
        "template_id": generic["template_id"],
        "result_start": slots["result_start"],
        "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
        "n_kept": n_kv_positions(bundle.cache_cpu),
        "extend_time_s": float(wall),
        "generic_prefill_s": float(generic.get("prefill_s") or 0.0),
    }
    return bundle, wall, info


def _summarize_check(rows: list[dict], templates: dict) -> dict:
    n = len(rows)
    return {
        "n_unique_a_sheets": n,
        "n_ok": sum(bool(r.get("ok")) for r in rows),
        "all_passed": bool(n) and all(bool(r.get("ok")) for r in rows),
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
        "n_original_instr_equal": sum(bool(r.get("original_instr_equal")) for r in rows),
        "n_packed_extra_equal": sum(bool(r.get("extra_equal")) for r in rows),
        "max_original_keys_max_abs": max((float(r.get("original_keys_max_abs") or 0) for r in rows), default=None),
        "max_original_values_max_abs": max((float(r.get("original_values_max_abs") or 0) for r in rows), default=None),
        "max_extra_keys_max_abs": max((float(r.get("extra_keys_max_abs") or 0) for r in rows), default=None),
        "max_extra_values_max_abs": max((float(r.get("extra_values_max_abs") or 0) for r in rows), default=None),
        "mean_n_instruction": (sum(int(r.get("n_instruction") or 0) for r in rows) / n) if n else None,
        "note": (
            "Generic prefixes are independent prefills of the recorded tokens through "
            "the eligibility instruction. Exact tensor equality with A's full-cache "
            "instruction states is not required; record the numerical gap. Token IDs, "
            "reconstructed prefix, transferred R, and B's starting position must match."
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
    kv_rows = by_cond.get("instr_kv", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    refs = _frozen_refs()
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "n_evals_expected": 90 * len(CONDS),
        "layout": LAYOUT,
        "prefix_source": PREFIX_SOURCE,
        "extra_separator": EXTRA_SEP,
        "baseline": "transferred instruction KV plus the same transferred R",
        "unit_note": (
            "plan4_dev only. Same reconstructed prefix, instruction token IDs, newline, "
            "transferred R tensors and packed slots, B prompt, and decoding. "
            "instr_generic reuses one independent prefill per shared causal prefix."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": receiver,
        "carrier_label_errors": {cond: _carrier_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {name: _paired(grouped_plain, a, b) for name, a, b in PAIRED},
        "family_clustered_delta": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED) for name, a, b in PAIRED
        },
        "family_clustered_eligibility": {
            name: _family_delta(rows, a, b, n_boot=N_BOOT, seed=BOOT_SEED, key="plan_all_status_correct")
            for name, a, b in PAIRED
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "frozen_instr_kv": refs["instr_kv"],
        "frozen_instr_fresh": refs["instr_fresh"],
        "frozen_reconstruct": refs["reconstruct"],
        "qualifies_vs_reconstruct": {
            cond: candidate_qualifies(receiver.get(cond) or {}, refs["reconstruct"]) for cond in CONDS
        },
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
            "Primary measure: all four eligibility labels. Compare transferred "
            "instruction states, short-prefix fresh instruction, and independently "
            "prepared reusable generic instruction, holding R fixed."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "Interpret this reuse check before matched neutral contexts or R-dependence "
            "controls. Do not load plan4_hold. Do not train an adapter. Do not add the "
            "calculation block."
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
            "stage": "plan4-generic",
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


def _collect_templates(examples, artifacts, decode) -> tuple[dict, dict]:
    records = []
    spans_by_key = {}
    by_id = {ex.example_id: ex for ex in examples}
    seen = set()
    unique = []
    for ex in examples:
        key = a_state_key(ex)
        if key not in seen:
            unique.append(ex)
            seen.add(key)
    for ex in unique:
        key = a_state_key(ex)
        rec = artifacts[key]
        spans = _extract_spans(rec, decode, key)
        src = by_id.get(key) or ex
        prefix = causal_prefix_from_ids(
            rec["full_ids"], spans["instruction"], decode=decode
        )
        prefix["a_state_key"] = key
        prefix["difficulty"] = src.difficulty
        records.append(prefix)
        spans_by_key[key] = spans
        spans_by_key[key]["causal_prefix"] = {
            k: prefix[k]
            for k in (
                "n",
                "instruction_start",
                "instruction_end",
                "instruction_pos",
                "instruction_ids",
                "contains_rule",
                "contains_shipment",
                "private_facts",
            )
        }
    templates = group_generic_templates(records)
    if len(templates) != N_EXPECTED_TEMPLATES:
        raise RuntimeError(f"Expected {N_EXPECTED_TEMPLATES} generic templates, found {len(templates)}.")
    prefix_by_key = {r["a_state_key"]: r for r in records}
    for key, spans in spans_by_key.items():
        spans["template_id"] = template_id_for(prefix_by_key[key], templates)
    return templates, spans_by_key


def _prefill_templates(runner, templates: dict) -> dict:
    out = {}
    print(f"Prefilling {len(templates)} generic instruction prefixes", flush=True)
    for name, rec in templates.items():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bundle, wall = runner.prefill_token_sequence(rec["ids"], source=f"generic_{name}")
        if n_kv_positions(bundle.cache_cpu) != rec["n_prefix"]:
            raise RuntimeError(f"Generic prefill length drifted for template {name}.")
        out[name] = {
            **rec,
            "bundle": bundle,
            "prefill_s": float(wall),
        }
        print(f"  template {name} n={rec['n']} n_prefix={rec['n_prefix']} s={wall:.2f}", flush=True)
    return out


def _pair_with_three(runner, rec, src, decode, sep_ids, close_ids, spans, generics):
    pair = _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids)
    if list(spans.get("result_pos") or []) != list(pair["spans"]["result_pos"]):
        raise RuntimeError("Instruction-pack result positions drifted from compact R.")
    extra_pos = list(spans["instruction_pos"])
    transferred, trans_s, composed, slots, trans_info = _build_extra_bundle(
        runner, pair, extra_pos, source="transfer_instr_kv"
    )
    fresh, fresh_s, fresh_info = _build_fresh_from_transferred(
        runner, pair, transferred, composed, slots
    )
    template_id = spans["template_id"]
    generic_rec = generics[template_id]
    generic, gen_s, gen_info = _build_generic_from_transferred(
        runner, pair, transferred, composed, slots, generic_rec, extra_pos
    )
    original_delta = cache_slice_delta(
        generic_rec["bundle"].cache_cpu, pair["bundle"].cache_cpu, extra_pos
    )
    trans_ids = [int(t) for t in transferred.full_ids_cpu.view(-1).tolist()]
    fresh_ids = [int(t) for t in fresh.full_ids_cpu.view(-1).tolist()]
    gen_ids = [int(t) for t in generic.full_ids_cpu.view(-1).tolist()]
    result_eq = cache_slice_delta(generic.cache_cpu, transferred.cache_cpu, slots["result"])["equal"]
    prefix_eq = cache_slice_delta(generic.cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
    cursor_eq = int(generic.pos_cursor_cpu.view(-1)[0].item()) == int(
        transferred.pos_cursor_cpu.view(-1)[0].item()
    )
    ok = (
        bool(spans.get("ok"))
        and trans_ids == fresh_ids == gen_ids == list(composed["ids"])
        and result_eq
        and prefix_eq
        and cursor_eq
        and int(slots["n_prefix"]) == 5
        and EXTRA_SEP
        and not bool(spans.get("causal_prefix", {}).get("private_facts"))
    )
    pair.update(
        {
            "instr": spans,
            "composed_extra": composed,
            "slots_extra": slots,
            "mixed": {"instr_kv": transferred, "instr_fresh": fresh, "instr_generic": generic},
            "times_by": {
                "instr_kv": trans_s,
                "instr_fresh": trans_s + fresh_s,
                "instr_generic": trans_s + gen_s,
            },
            "info_by": {"instr_kv": trans_info, "instr_fresh": fresh_info, "instr_generic": gen_info},
            "check_row": {
                "a_state_key": spans.get("a_state_key"),
                "ok": ok,
                "template_id": template_id,
                "ids_equal": trans_ids == fresh_ids == gen_ids,
                "result_equal": result_eq,
                "prefix_equal": prefix_eq,
                "cursor_equal": cursor_eq,
                "original_instr_equal": original_delta["equal"],
                "original_keys_max_abs": original_delta["keys_max_abs"],
                "original_values_max_abs": original_delta["values_max_abs"],
                "extra_equal": gen_info.get("extra_equal"),
                "extra_keys_max_abs": gen_info.get("extra_keys_max_abs"),
                "extra_values_max_abs": gen_info.get("extra_values_max_abs"),
                "n_instruction": int(spans.get("n_instruction") or 0),
                "n_causal_prefix": int(spans.get("causal_prefix", {}).get("n") or generic_rec["n_prefix"]),
                "result_start": int(slots["result_start"]),
                "pos_cursor": int(generic.pos_cursor_cpu.view(-1)[0].item()),
                "instruction_before_facts": bool(spans.get("instruction_before_facts")),
                "instruction_contains_rule": ELIGIBILITY_RULE
                in ((spans.get("instruction") or {}).get("text") or ""),
                "private_facts": bool(spans.get("causal_prefix", {}).get("private_facts")),
            },
        }
    )
    return pair


def _run_one(runner, ex, extra_a, spans, composed, slots, cond, args, *, bundle, wall, extra_meta):
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
    rec["instruction_source"] = SPEC[cond]["instruction"]
    rec["prefix_source"] = PREFIX_SOURCE
    rec["layout"] = composed.get("layout")
    rec["layout_separator"] = True
    rec["layout_turn_close"] = False
    rec["extra_separator"] = EXTRA_SEP
    rec["n_prefix"] = int(slots["n_prefix"])
    rec["n_extra"] = int(slots["n_extra"])
    rec["n_extra_sep"] = int(slots["n_extra_sep"])
    rec["n_result_slots"] = int(slots["n_result"])
    rec["result_start"] = int(slots["result_start"])
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = n_kv_positions(bundle.cache_cpu)
    rec["n_result_kept"] = len(spans["result_pos"])
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["n_added"] = int(composed.get("n_added") or 0)
    rec["n_full"] = int(spans["n_full"])
    rec["n_causal_prefix"] = int(spans.get("causal_prefix", {}).get("n") or 0)
    rec["template_id"] = spans.get("template_id")
    rec["selected_text"] = spans["result_text"]
    rec["selected_token_ids"] = list(composed["ids"])
    rec["instruction_paragraph"] = (spans.get("instruction") or {}).get("text")
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
    rec["message_bytes"] = int(bundle.full_relay_bytes) + rec["framing_bytes"]
    rec["cache_source"] = SPEC[cond]["cache_source"]
    rec["shared_across_heads"] = True
    rec["n_kv_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["plan_prompt"] = b_reason_kv_instruct_prompt(ex)
    rec["payload_note"] = PAYLOAD_NOTE
    rec["process_time_s"] = _process_time_s(rec)
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    _annotate_plan(rec, ex)
    return rec


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    templates, spans_by_key = _collect_templates(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    unique = []
    seen = set()
    for ex in examples:
        key = a_state_key(ex)
        if key not in seen:
            unique.append(ex)
            seen.add(key)
    rows = []
    print(f"Generic-instruction check on {len(unique)} unique A sheets, layout={LAYOUT}", flush=True)
    for i, ex in enumerate(unique):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pair = _pair_with_three(
            runner, rec, src, decode, sep_ids, close_ids, spans_by_key[key], generics
        )
        row = pair["check_row"]
        rows.append(row)
        print(
            f"[{i + 1}/{len(unique)}] {key} ok={row['ok']} template={row['template_id']} "
            f"orig_eq={row['original_instr_equal']} k_max={row['original_keys_max_abs']:.4g} "
            f"R_eq={row['result_equal']}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    construction = _summarize_check(rows, templates)
    slim_spans = {}
    for key, spans in spans_by_key.items():
        slim_spans[key] = {
            "ok": spans.get("ok"),
            "failures": spans.get("failures"),
            "n_instruction": spans.get("n_instruction"),
            "instruction": spans.get("instruction"),
            "instruction_pos": spans.get("instruction_pos"),
            "result_pos": spans.get("result_pos"),
            "result_text": spans.get("result_text"),
            "n_full": spans.get("n_full"),
            "n_prompt": spans.get("n_prompt"),
            "n_sink_extra": spans.get("n_sink_extra"),
            "instruction_before_facts": spans.get("instruction_before_facts"),
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
            "If all_passed, run --phase compare: transferred instruction, short-prefix "
            "fresh instruction, and reused generic instruction, with R held fixed."
        ),
        "frozen_placement": FROZEN_PLACEMENT,
    }
    template_meta = {
        name: {k: v for k, v in rec.items() if k != "bundle"}
        for name, rec in generics.items()
    }
    _write_json(
        out_root / "construction.json",
        {
            **construction,
            "by_a_state": slim_spans,
            "templates": template_meta,
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
                    "n_original_instr_equal",
                    "n_result_equal",
                    "max_original_keys_max_abs",
                )
            },
            indent=2,
        )
    )
    print(json.dumps(construction.get("templates"), indent=2))
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
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in CONDS)}
    print(f"Generic-instruction compare layout={LAYOUT} rebuilding {len(needed)} unique A caches", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pair = _pair_with_three(
            runner, rec, src, decode, sep_ids, close_ids, spans_by_key[key], generics
        )
        if not pair["check_row"]["ok"]:
            raise RuntimeError(f"Generic-instruction construction failed for {key}.")
        states[key] = pair
        print(
            f"  rebuilt {key} template={spans_by_key[key]['template_id']} "
            f"result_start={pair['slots_extra']['result_start']}",
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
        gen_info = state["info_by"].get("instr_generic") or {}
        for cond in remaining:
            extra_meta = {
                "span_extract_ok": bool(spans.get("ok")),
                "n_instruction": int(spans.get("n_instruction") or 0),
                "instruction_before_facts": bool(spans.get("instruction_before_facts")),
                "instruction_contains_rule": ELIGIBILITY_RULE
                in ((spans.get("instruction") or {}).get("text") or ""),
                "result_held_from_transferred": True,
                "template_id": spans.get("template_id"),
                "extra_equal": gen_info.get("extra_equal"),
                "extra_keys_max_abs": gen_info.get("extra_keys_max_abs"),
                "extra_values_max_abs": gen_info.get("extra_values_max_abs"),
            }
            rec_out = _run_one(
                runner,
                ex,
                extra_a,
                spans,
                state["composed_extra"],
                state["slots_extra"],
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
                f"elig={rec_out.get('plan_all_status_correct')} parse={rec_out.get('parse_ok')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    summary = _summarize(rows, check=check)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:12000])
    print(json.dumps(summary.get("family_clustered_eligibility"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="Reuse independently prepared generic instruction states, holding R fixed."
    )
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--phase", choices=("check", "compare", "all", "rescore"), default="check")
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
        print("Generic-instruction reuse uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_generic" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_generic.")
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
        check = None
        cpath = out_root / "check_summary.json"
        if cpath.is_file():
            check = json.loads(cpath.read_text(encoding="utf-8"))
        summary = _summarize(_load_rows(out_root / "evaluations.jsonl"), check=check)
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:8000])
        return 0

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

