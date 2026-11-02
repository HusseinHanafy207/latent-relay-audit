"""Matched-context encoding of the eligibility instruction, R held fixed.

Question: does meaningful general task context create more useful instruction
states than equally long ordinary text with no shipment facts or eligibility
guidance?

Three conditions, same instruction token IDs, packed slots, transferred R,
separator, B prompt, and decoding:
- instr_generic: original shared task prefix (A's role and capacity rules)
- instr_neutral_1: first predeclared matched-length neutral prefix
- instr_neutral_2: second predeclared matched-length neutral prefix

Only the instruction states are sent, not the preceding context. Do not add
calculations. Do not train an adapter. Do not run R-dependence from this file.
plan4_hold stays out.
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
from src.generic_select import N_EXPECTED_TEMPLATES
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.neutral_select import (
    CONTEXT_NEUTRAL_1,
    CONTEXT_NEUTRAL_2,
    CONTEXT_TASK,
    NEUTRAL_PASSAGES,
    splice_neutral_prefix,
)
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
from src.stage_plan4_generic import (
    _build_generic_from_transferred,
    _collect_templates,
    _prefill_templates,
)
from src.stage_plan4_instr import EXTRA_SEP, _build_extra_bundle
from src.stage_plan4_kvresult import PREFIX_SOURCE, _rebuild_pair
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _copy_frozen_artifacts
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("instr_generic", "instr_neutral_1", "instr_neutral_2")
SPEC = {
    "instr_generic": {
        "instruction": CONTEXT_TASK,
        "cache_source": "generic_instr_same_R",
        "passage_id": None,
    },
    "instr_neutral_1": {
        "instruction": CONTEXT_NEUTRAL_1,
        "cache_source": "neutral_1_instr_same_R",
        "passage_id": CONTEXT_NEUTRAL_1,
    },
    "instr_neutral_2": {
        "instruction": CONTEXT_NEUTRAL_2,
        "cache_source": "neutral_2_instr_same_R",
        "passage_id": CONTEXT_NEUTRAL_2,
    },
}
PAIRED = (
    ("instr_generic_vs_instr_neutral_1", "instr_neutral_1", "instr_generic"),
    ("instr_generic_vs_instr_neutral_2", "instr_neutral_2", "instr_generic"),
    ("instr_neutral_1_vs_instr_neutral_2", "instr_neutral_2", "instr_neutral_1"),
)
PRESERVED = (
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
    "results_stage_plan4_hold",
    "results_stage_plan4_stable",
    "results_stage_plan4_fresh",
)
FROZEN_PLACEMENT = (
    "All three keep the reconstructed five-token prefix, the same 64-token "
    "eligibility paragraph at its original encoding positions, one newline, and "
    "the same transferred R. Only the instruction states are relocated onto the "
    "packed extra slots. instr_generic prefills A's original general task text. "
    "instr_neutral_1 and instr_neutral_2 replace that preceding body with two "
    "predeclared ordinary passages of matching token length, keeping the chat "
    "header. Do not treat pad or mask tokens as a contextual prefix."
)
INTERPRETATION_GUIDE = (
    "If the meaningful task context consistently beats both neutrals, general "
    "task content helps construct useful instruction states. If the neutrals "
    "perform similarly to the task context, the benefit may come from more "
    "general properties of the encoding context. If the two neutrals disagree "
    "substantially, the method is context-sensitive and robustness is next. "
    "None of these outcomes alone proves an attention mechanism. Do not add "
    "calculations or train an adapter until this comparison is interpreted. "
    "Do not run R-dependence or load plan4_hold from this file."
)
PASSAGE_NOTE = (
    "Both neutral passages were written before any receiver scores from this "
    "stage. They contain no shipment facts, answers, or eligibility guidance."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_neutral" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_neutral")
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


def _encode_plain(tokenizer):
    def encode(text: str):
        return tokenizer.encode(text, add_special_tokens=False)

    return encode


def _token_id(tokenizer, name: str):
    value = getattr(tokenizer, name, None)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _frozen_refs() -> dict:
    generic = {
        "n": 90,
        "plan_all_status_correct": 87,
        "correct": 69,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_generic instr_generic",
    }
    reconstruct = {
        "n": 90,
        "plan_all_status_correct": 89,
        "correct": 74,
        "parse_ok": 90,
        "truncated": 0,
        "source": "results_stage_plan4_instr reconstruct",
    }
    for folder, key, name in (
        (REPO_ROOT / "results_stage_plan4_generic" / "summary.json", "generic", "instr_generic"),
        (Path("/kaggle/working/results_stage_plan4_generic/summary.json"), "generic", "instr_generic"),
        (REPO_ROOT / "results_stage_plan4_instr" / "summary.json", "reconstruct", "reconstruct"),
        (Path("/kaggle/working/results_stage_plan4_instr/summary.json"), "reconstruct", "reconstruct"),
    ):
        if not folder.is_file():
            continue
        data = json.loads(folder.read_text(encoding="utf-8"))
        block = (data.get("receiver") or {}).get(name) or {}
        if int(block.get("n") or 0) != 90:
            continue
        rec = {
            "n": 90,
            "plan_all_status_correct": block.get("plan_all_status_correct"),
            "correct": block.get("correct"),
            "parse_ok": block.get("parse_ok"),
            "truncated": block.get("truncated"),
            "source": str(folder).replace("\\", "/"),
        }
        if key == "generic":
            generic = rec
        else:
            reconstruct = rec
    return {"instr_generic": generic, "reconstruct": reconstruct}


def _splice_one(template: dict, passage_id: str, encode, decode, tokenizer) -> dict:
    return splice_neutral_prefix(
        template["ids"],
        instruction_start=template["instruction_start"],
        instruction_end=template["instruction_end"],
        passage=NEUTRAL_PASSAGES[passage_id],
        encode=encode,
        decode=decode,
        pad_token_id=_token_id(tokenizer, "pad_token_id"),
        mask_token_id=_token_id(tokenizer, "mask_token_id"),
        passage_id=passage_id,
    )


def _prefill_contexts(runner, templates: dict, tokenizer, decode) -> dict:
    encode = _encode_plain(tokenizer)
    task = _prefill_templates(runner, templates)
    out = {}
    print(f"Prefilling {len(templates)} templates x 2 neutral contexts", flush=True)
    for name, rec in templates.items():
        splices = {}
        for passage_id in (CONTEXT_NEUTRAL_1, CONTEXT_NEUTRAL_2):
            spliced = _splice_one(rec, passage_id, encode, decode, tokenizer)
            if spliced["instruction_ids"] != list(rec["instruction_ids"]):
                raise RuntimeError(f"Neutral {passage_id} moved instruction tokens on {name}.")
            if spliced["ids"][: spliced["header_len"]] != list(rec["ids"][: spliced["header_len"]]):
                raise RuntimeError(f"Neutral {passage_id} dropped the chat header on {name}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bundle, wall = runner.prefill_token_sequence(
                spliced["ids"], source=f"{passage_id}_{name}"
            )
            if n_kv_positions(bundle.cache_cpu) != rec["n_prefix"]:
                raise RuntimeError(f"Neutral prefill length drifted for {passage_id} {name}.")
            splices[passage_id] = {**spliced, "bundle": bundle, "prefill_s": float(wall)}
            print(
                f"  {name} {passage_id} n_body={spliced['n_body']} "
                f"header={spliced['header_len']} s={wall:.2f}",
                flush=True,
            )
        n1 = splices[CONTEXT_NEUTRAL_1]["ids"]
        n2 = splices[CONTEXT_NEUTRAL_2]["ids"]
        if n1 == n2:
            raise RuntimeError(f"The two neutral prefixes are identical for template {name}.")
        if n1 == list(rec["ids"]) or n2 == list(rec["ids"]):
            raise RuntimeError(f"A neutral prefix copied the task prefix for template {name}.")
        out[name] = {
            "instr_generic": {
                **rec,
                "bundle": task[name]["bundle"],
                "prefill_s": task[name]["prefill_s"],
                "context_kind": CONTEXT_TASK,
                "passage_id": None,
                "header_len": splices[CONTEXT_NEUTRAL_1]["header_len"],
                "ids": list(rec["ids"]),
            },
            "instr_neutral_1": {
                **rec,
                **splices[CONTEXT_NEUTRAL_1],
                "context_kind": CONTEXT_NEUTRAL_1,
                "template_id": name,
            },
            "instr_neutral_2": {
                **rec,
                **splices[CONTEXT_NEUTRAL_2],
                "context_kind": CONTEXT_NEUTRAL_2,
                "template_id": name,
            },
        }
    return out


def _summarize_check(rows: list[dict], templates: dict, contexts: dict) -> dict:
    n = len(rows)
    slim_ctx = {}
    for name in templates:
        slim_ctx[name] = {}
        for cond in CONDS:
            rec = contexts[name][cond]
            slim_ctx[name][cond] = {
                k: rec[k]
                for k in rec
                if k != "bundle"
            }
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
        "contexts": slim_ctx,
        "n_ids_equal": sum(bool(r.get("ids_equal")) for r in rows),
        "n_result_equal": sum(bool(r.get("result_equal")) for r in rows),
        "n_prefix_equal": sum(bool(r.get("prefix_equal")) for r in rows),
        "n_instruction_held": sum(bool(r.get("instruction_held")) for r in rows),
        "n_header_held": sum(bool(r.get("header_held")) for r in rows),
        "n_used_pad": sum(bool(r.get("used_pad")) for r in rows),
        "max_neutral1_keys_max_abs": max(
            (float(r.get("neutral1_keys_max_abs") or 0) for r in rows), default=None
        ),
        "max_neutral2_keys_max_abs": max(
            (float(r.get("neutral2_keys_max_abs") or 0) for r in rows), default=None
        ),
        "mean_n_instruction": (sum(int(r.get("n_instruction") or 0) for r in rows) / n) if n else None,
        "passages": {k: {"n_chars": len(v), "text": v} for k, v in NEUTRAL_PASSAGES.items()},
        "passage_note": PASSAGE_NOTE,
        "note": (
            "Instruction token IDs, reconstructed prefix, transferred R, and B's "
            "starting position must match. Packed extra states are expected to "
            "differ across contexts; record the numerical gap. Pad/mask tokens are "
            "not a valid contextual prefix."
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
    kv_rows = by_cond.get("instr_generic", [])
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
        "baseline": "independently prepared generic task-context instruction plus the same transferred R",
        "unit_note": (
            "plan4_dev only. Same reconstructed prefix, eligibility-instruction token "
            "IDs, newline, transferred R tensors and packed slots, B prompt, and "
            "decoding. Neutral prefixes match token length and keep the chat header."
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
        "frozen_instr_generic": refs["instr_generic"],
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
        "passage_note": PASSAGE_NOTE,
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels. Compare task-context "
            "instruction states with two matched-length neutral encoding contexts, "
            "holding R fixed."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "Interpret this matched-context comparison before R-dependence, hold "
            "confirmation, timing, calculations, or adapter training. Do not load "
            "plan4_hold."
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
            "stage": "plan4-neutral",
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
            "passage_note": PASSAGE_NOTE,
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


def _pair_with_contexts(runner, rec, src, decode, sep_ids, close_ids, spans, contexts):
    pair = _rebuild_pair(runner, rec, src, decode, sep_ids, close_ids)
    if list(spans.get("result_pos") or []) != list(pair["spans"]["result_pos"]):
        raise RuntimeError("Instruction-pack result positions drifted from compact R.")
    extra_pos = list(spans["instruction_pos"])
    transferred, trans_s, composed, slots, trans_info = _build_extra_bundle(
        runner, pair, extra_pos, source="transfer_instr_kv"
    )
    template_id = spans["template_id"]
    mixed = {}
    times_by = {}
    info_by = {}
    packed_ids = {}
    for cond in CONDS:
        ctx = contexts[template_id][cond]
        bundle, wall, info = _build_generic_from_transferred(
            runner, pair, transferred, composed, slots, ctx, extra_pos
        )
        mixed[cond] = bundle
        times_by[cond] = trans_s + wall
        info["context_kind"] = ctx.get("context_kind")
        info["passage_id"] = ctx.get("passage_id")
        info["header_len"] = ctx.get("header_len")
        info_by[cond] = info
        packed_ids[cond] = [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]
    generic_extra = cache_slice_delta(
        mixed["instr_generic"].cache_cpu, transferred.cache_cpu, slots["extra"]
    )
    n1_vs_task = cache_slice_delta(
        mixed["instr_neutral_1"].cache_cpu, mixed["instr_generic"].cache_cpu, slots["extra"]
    )
    n2_vs_task = cache_slice_delta(
        mixed["instr_neutral_2"].cache_cpu, mixed["instr_generic"].cache_cpu, slots["extra"]
    )
    ids_equal = packed_ids["instr_generic"] == packed_ids["instr_neutral_1"] == packed_ids["instr_neutral_2"] == list(
        composed["ids"]
    )
    result_eq = all(
        cache_slice_delta(mixed[cond].cache_cpu, transferred.cache_cpu, slots["result"])["equal"]
        for cond in CONDS
    )
    prefix_eq = all(
        cache_slice_delta(mixed[cond].cache_cpu, pair["reconstruct"].cache_cpu, slots["prefix"])["equal"]
        for cond in CONDS
    )
    cursors = {cond: int(mixed[cond].pos_cursor_cpu.view(-1)[0].item()) for cond in CONDS}
    cursor_eq = len(set(cursors.values())) == 1 and cursors["instr_generic"] == int(composed["n"])
    n1_ctx = contexts[template_id]["instr_neutral_1"]
    n2_ctx = contexts[template_id]["instr_neutral_2"]
    header_held = bool(n1_ctx.get("header_equal")) and bool(n2_ctx.get("header_equal"))
    instruction_held = bool(n1_ctx.get("instruction_equal")) and bool(n2_ctx.get("instruction_equal"))
    used_pad = bool(n1_ctx.get("used_pad")) or bool(n2_ctx.get("used_pad"))
    ok = (
        bool(spans.get("ok"))
        and ids_equal
        and result_eq
        and prefix_eq
        and cursor_eq
        and header_held
        and instruction_held
        and not used_pad
        and int(slots["n_prefix"]) == 5
        and EXTRA_SEP
        and not bool(spans.get("causal_prefix", {}).get("private_facts"))
        and n1_ctx["ids"] != n2_ctx["ids"]
        and n1_ctx["ids"] != list(contexts[template_id]["instr_generic"]["ids"])
    )
    pair.update(
        {
            "instr": spans,
            "composed_extra": composed,
            "slots_extra": slots,
            "mixed": mixed,
            "times_by": times_by,
            "info_by": info_by,
            "scaffold_info": trans_info,
            "check_row": {
                "a_state_key": spans.get("a_state_key"),
                "ok": ok,
                "template_id": template_id,
                "ids_equal": ids_equal,
                "result_equal": result_eq,
                "prefix_equal": prefix_eq,
                "cursor_equal": cursor_eq,
                "header_held": header_held,
                "instruction_held": instruction_held,
                "used_pad": used_pad,
                "header_len": n1_ctx.get("header_len"),
                "n_body": n1_ctx.get("n_body"),
                "extra_equal_generic_vs_transfer": generic_extra["equal"],
                "neutral1_equal": n1_vs_task["equal"],
                "neutral1_keys_max_abs": n1_vs_task["keys_max_abs"],
                "neutral1_values_max_abs": n1_vs_task["values_max_abs"],
                "neutral2_equal": n2_vs_task["equal"],
                "neutral2_keys_max_abs": n2_vs_task["keys_max_abs"],
                "neutral2_values_max_abs": n2_vs_task["values_max_abs"],
                "n_instruction": int(spans.get("n_instruction") or 0),
                "n_causal_prefix": int(spans.get("causal_prefix", {}).get("n") or contexts[template_id]["instr_generic"]["n_prefix"]),
                "result_start": int(slots["result_start"]),
                "pos_cursor": cursors["instr_generic"],
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
    rec["passage_note"] = PASSAGE_NOTE
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
    if len(templates) != N_EXPECTED_TEMPLATES:
        raise RuntimeError(f"Expected {N_EXPECTED_TEMPLATES} templates, found {len(templates)}.")
    contexts = _prefill_contexts(runner, templates, tokenizer, decode)
    unique = []
    seen = set()
    for ex in examples:
        key = a_state_key(ex)
        if key not in seen:
            unique.append(ex)
            seen.add(key)
    rows = []
    print(f"Neutral-context check on {len(unique)} unique A sheets, layout={LAYOUT}", flush=True)
    for i, ex in enumerate(unique):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pair = _pair_with_contexts(
            runner, rec, src, decode, sep_ids, close_ids, spans_by_key[key], contexts
        )
        row = pair["check_row"]
        rows.append(row)
        print(
            f"[{i + 1}/{len(unique)}] {key} ok={row['ok']} template={row['template_id']} "
            f"n1_eq={row['neutral1_equal']} n2_eq={row['neutral2_equal']} "
            f"k1={row['neutral1_keys_max_abs']:.4g} R_eq={row['result_equal']}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    construction = _summarize_check(rows, templates, contexts)
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
            "If all_passed, run --phase compare: task-context instruction versus two "
            "matched-length neutral contexts, with R held fixed."
        ),
        "frozen_placement": FROZEN_PLACEMENT,
        "passage_note": PASSAGE_NOTE,
    }
    _write_json(
        out_root / "construction.json",
        {
            **construction,
            "by_a_state": slim_spans,
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
                    "n_instruction_held",
                    "n_result_equal",
                    "n_used_pad",
                    "max_neutral1_keys_max_abs",
                    "max_neutral2_keys_max_abs",
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
    contexts = _prefill_contexts(runner, templates, tokenizer, decode)
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in CONDS)}
    print(f"Neutral-context compare layout={LAYOUT} rebuilding {len(needed)} unique A caches", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pair = _pair_with_contexts(
            runner, rec, src, decode, sep_ids, close_ids, spans_by_key[key], contexts
        )
        if not pair["check_row"]["ok"]:
            raise RuntimeError(f"Neutral-context construction failed for {key}.")
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
        for cond in remaining:
            info = state["info_by"].get(cond) or {}
            extra_meta = {
                "span_extract_ok": bool(spans.get("ok")),
                "n_instruction": int(spans.get("n_instruction") or 0),
                "instruction_before_facts": bool(spans.get("instruction_before_facts")),
                "instruction_contains_rule": ELIGIBILITY_RULE
                in ((spans.get("instruction") or {}).get("text") or ""),
                "result_held_from_transferred": True,
                "template_id": spans.get("template_id"),
                "context_kind": SPEC[cond]["instruction"],
                "passage_id": SPEC[cond]["passage_id"],
                "header_len": info.get("header_len"),
                "extra_equal": info.get("extra_equal"),
                "extra_keys_max_abs": info.get("extra_keys_max_abs"),
                "extra_values_max_abs": info.get("extra_values_max_abs"),
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
        description="Compare task-context instruction states with matched-length neutral contexts."
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
        print("Neutral-context comparison uses plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_neutral" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_neutral.")
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
