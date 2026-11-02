"""Matched reconstruction of the twelve-carrier compact message.

Use the existing nine timing-12 examples and frozen A outputs. Do not
regenerate A. Do not retune the candidate. Do not train an adapter.

Before the twelve-carrier comparison, run the generalized N-carrier assembly
on the original four-carrier timing bases with their original prompts. That
checks selected IDs, slot order, RoPE positions, masks, and continuation
logits against the established four-carrier path.

Twelve-carrier conditions, same B prompt and decoding:
- full_kv: rebuilt completed sender cache
- text_result: RESULT lines as text
- generic_r: generic support plus transferred result states
- generic_fresh_r: the identical support cache, then freshly process the same
  result token IDs at the same packed positions
- reconstruct: freshly process the entire compact sequence (prefix, instruction,
  separators, result lines)

If generic_fresh_r recovers, changing the result states is enough.
If only reconstruct recovers, the support cache or its combination with R
also matters. If both fail while text works, inspect framing/presentation
before cache adaptation.
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

from src.cache_utils import cache_slice_delta, cache_to_device, clone_cache_blob, n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    a_state_key,
    assert_plan4_invariants,
    load_plan4_jsonl,
)
from src.plan4_timing_task import (
    N_CARRIERS_12,
    N_TIMING_FAMILIES,
    assert_timing12_invariants,
    ensure_timing12_jsonl,
    select_timing4_examples,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prefix_logits import logit_delta_stats
from src.prompts_plan4 import FRAMING_INSTRUCTION, b_reason_kv_instruct_prompt, prompt_contains_a_private
from src.span_select import caches_equal
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_boundary import PAYLOAD_NOTE, _added_token_ids
from src.stage_plan4_comm import _a_extra
from src.stage_plan4_confirm import _copy_hold_artifacts
from src.stage_plan4_generic import _collect_templates, _prefill_templates
from src.stage_plan4_instr import EXTRA_SEP
from src.stage_plan4_kvresult import PREFIX_SOURCE
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_probe import _persist as _persist_probe
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_plan4_timing import (
    B_PLAN_MAX_NEW_12,
    _collect_templates_n,
    _pair_state_4,
    _pair_state_n,
    _prompts_for,
    _row_n,
)
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("full_kv", "text_result", "generic_r", "generic_fresh_r", "reconstruct")
SPEC = {
    "full_kv": {"kind": "kv", "r_source": "full", "cache_source": "rebuilt_full"},
    "text_result": {"kind": "text", "r_source": "text", "cache_source": "text"},
    "generic_r": {"kind": "kv", "r_source": "transfer", "cache_source": "generic_instr_same_R"},
    "generic_fresh_r": {"kind": "kv", "r_source": "fresh_after_support", "cache_source": "generic_instr_fresh_R"},
    "reconstruct": {"kind": "kv", "r_source": "fresh_compact", "cache_source": "reconstruct_compact"},
}
PRESERVED = (
    "results_stage_plan4_timing",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
    "results_stage_plan4_rdep",
    "results_stage_plan4_generic",
    "results_stage_plan4_ifresh",
    "results_stage_plan4_recon",
)
FROZEN_PLACEMENT = (
    "generic_r keeps independently prepared generic support and appends A's "
    "transferred result states. generic_fresh_r keeps that same support cache, "
    "result token IDs, packed result positions, B prompt, and decoding, and "
    "only replaces the result states by forwarding those tokens after the "
    "support. reconstruct prefills the entire compact token sequence."
)
INTERPRETATION_GUIDE = (
    "If generic_fresh_r recovers labels, changing the result states is enough "
    "and reuse of A's contextualized R is implicated. If only reconstruct "
    "recovers, the support cache or its combination with the result states "
    "also matters. If both fail while ordinary result text works, examine the "
    "compact message's framing and presentation before pursuing cache adaptation. "
    "Do not treat twelve-carrier timing accuracy as a confirmation claim. "
    "Directional label errors on the timing summary were a scorer omission; "
    "generic+R was 44 ineligible-marked-eligible and 1 eligible-marked-ineligible."
)
CONT_LEN = 64


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    hold = DATA_DIR / "plan4_hold.jsonl"
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_recon12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_recon12")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return hold, timing12, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("parity_summary.json", "check_summary.json", "a_artifacts_12.jsonl"):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _copy_timing12_artifacts(out_root: Path, *, explicit: str | None = None) -> Path:
    dest = out_root / "a_artifacts_12.jsonl"
    if dest.is_file():
        return dest
    paths = []
    if explicit:
        paths.append(Path(explicit))
    paths.extend(
        [
            REPO_ROOT / "results_stage_plan4_timing" / "a_artifacts_12.jsonl",
            RESULTS_DIR / "stage_plan4_timing" / "run" / "a_artifacts_12.jsonl",
            Path("/kaggle/working/results_stage_plan4_timing/a_artifacts_12.jsonl"),
        ]
    )
    for src in paths:
        if src.is_file() and src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
            print(f"Copied frozen 12-carrier A artifacts from {src}")
            return dest
    raise FileNotFoundError(
        "Need results_stage_plan4_timing/a_artifacts_12.jsonl. Do not regenerate A's answers."
    )


def _slots_plain(slots: dict) -> dict:
    return {
        "prefix": list(slots.get("prefix") or []),
        "extra": list(slots.get("extra") or []),
        "extra_sep": list(slots.get("extra_sep") or []),
        "result": list(slots.get("result") or []),
        "n_prefix": int(slots.get("n_prefix") or 0),
        "n_extra": int(slots.get("n_extra") or 0),
        "n_extra_sep": int(slots.get("n_extra_sep") or 0),
        "n_result": int(slots.get("n_result") or 0),
        "result_start": int(slots.get("result_start") or 0),
    }


def _mask_plain(bundle) -> dict:
    mask = [int(x) for x in bundle.past_mask_cpu.detach().cpu().view(-1).tolist()]
    return {
        "n": len(mask),
        "all_ones": bool(mask) and all(v == 1 for v in mask),
        "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
        "n_kv": n_kv_positions(bundle.cache_cpu),
    }


def _continuation_from_bundle(runner, bundle, cont_ids: torch.Tensor) -> torch.Tensor:
    past = cache_to_device(clone_cache_blob(bundle.cache_cpu), runner.device)
    past_mask = bundle.past_mask_cpu.to(runner.device)
    if past_mask.size(1) != n_kv_positions(bundle.cache_cpu):
        past_mask = torch.ones((1, n_kv_positions(bundle.cache_cpu)), dtype=torch.long, device=runner.device)
    cursor = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    logits = runner._continuation_logits(
        past=past, past_mask=past_mask, pos_cursor=cursor, cont_ids=cont_ids
    )
    del past, past_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def _fresh_r_after_support(runner, state: dict):
    head = state["head"]
    composed = state["composed_extra"]
    slots = state["slots_extra"]
    result_ids = [int(composed["ids"][p]) for p in slots["result"]]
    if result_ids != [int(t) for t in composed["ids"][int(slots["result_start"]) :]]:
        raise RuntimeError("Packed result IDs are not a suffix of the compact sequence.")
    bundle, wall = runner.extend_cache_with_tokens(head, result_ids, source="generic_instr_fresh_R")
    got = [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]
    if got != list(composed["ids"]):
        raise RuntimeError("Fresh-R cache token IDs drifted from the compact sequence.")
    support = list(range(int(slots["result_start"])))
    support_held = cache_slice_delta(bundle.cache_cpu, state["mixed"]["generic_r"].cache_cpu, support)["equal"]
    if not support_held:
        raise RuntimeError("Fresh-R construction overwrote the generic support tensors.")
    r_vs_transfer = cache_slice_delta(bundle.cache_cpu, state["transferred"].cache_cpu, slots["result"])
    cursor = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    if cursor != int(composed["n"]):
        raise RuntimeError(f"Fresh-R cursor {cursor} != compact length {composed['n']}.")
    info = {
        "support_held": True,
        "ids_equal": True,
        "result_equal_transferred": bool(r_vs_transfer["equal"]),
        "result_keys_max_abs": r_vs_transfer["keys_max_abs"],
        "result_values_max_abs": r_vs_transfer["values_max_abs"],
        "result_start": int(slots["result_start"]),
        "pos_cursor": cursor,
        "n_kept": n_kv_positions(bundle.cache_cpu),
        "extend_time_s": float(wall),
    }
    return bundle, wall, info


def _fresh_compact(runner, state: dict):
    composed = state["composed_extra"]
    bundle, wall = runner.prefill_token_sequence(
        composed["ids"], pos_cursor=composed["n"], source="reconstruct_compact"
    )
    got = [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]
    if got != list(composed["ids"]):
        raise RuntimeError("Reconstructed compact token IDs drifted.")
    cursor = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    if cursor != int(composed["n"]):
        raise RuntimeError(f"Reconstruct cursor {cursor} != compact length {composed['n']}.")
    slots = state["slots_extra"]
    vs_generic = cache_slice_delta(bundle.cache_cpu, state["mixed"]["generic_r"].cache_cpu, slots["result"])
    support = list(range(int(slots["result_start"])))
    vs_support = cache_slice_delta(bundle.cache_cpu, state["mixed"]["generic_r"].cache_cpu, support)
    info = {
        "ids_equal": True,
        "pos_cursor": cursor,
        "n_kept": n_kv_positions(bundle.cache_cpu),
        "result_equal_generic_r": bool(vs_generic["equal"]),
        "support_equal_generic_r": bool(vs_support["equal"]),
        "prefill_s": float(wall),
    }
    return bundle, wall, info


def _attach_recon12(runner, state: dict) -> None:
    fresh_r, fresh_s, fresh_info = _fresh_r_after_support(runner, state)
    recon, recon_s, recon_info = _fresh_compact(runner, state)
    state["mixed"]["generic_fresh_r"] = fresh_r
    state["mixed"]["reconstruct"] = recon
    state["info_by"] = dict(state.get("info_by") or {})
    state["info_by"]["generic_fresh_r"] = fresh_info
    state["info_by"]["reconstruct"] = recon_info
    state["times_by"]["generic_fresh_r"] = float(state.get("generic_construct_s") or 0.0) + float(fresh_s)
    state["times_by"]["reconstruct"] = float(recon_s)
    state["check_row"] = {
        "a_state_key": (state.get("instr") or {}).get("a_state_key"),
        "ok": bool(fresh_info.get("support_held")) and bool(fresh_info.get("ids_equal")) and bool(recon_info.get("ids_equal")),
        "n_result": int(state["slots_extra"]["n_result"]),
        "result_start": int(state["slots_extra"]["result_start"]),
        "generic_r_n": n_kv_positions(state["mixed"]["generic_r"].cache_cpu),
        "fresh_r_n": n_kv_positions(fresh_r.cache_cpu),
        "reconstruct_n": n_kv_positions(recon.cache_cpu),
        "fresh_r_result_equal_transferred": fresh_info.get("result_equal_transferred"),
        "reconstruct_support_equal": recon_info.get("support_equal_generic_r"),
        "template_id": (state.get("instr") or {}).get("template_id"),
    }


def _carrier_block_n(items: list[dict]) -> dict:
    carriers = []
    for item in items:
        for name in (item.get("gold_status") or {}):
            if name not in carriers:
                carriers.append(name)
    out = {}
    for carrier in carriers:
        n_mismatch = 0
        elig_as_inelig = 0
        inelig_as_elig = 0
        for item in items:
            gold = (item.get("gold_status") or {}).get(carrier)
            pred = (item.get("plan_pred_status") or {}).get(carrier)
            if not gold or not pred:
                continue
            if gold != pred:
                n_mismatch += 1
            if gold == "eligible" and pred == "ineligible":
                elig_as_inelig += 1
            if gold == "ineligible" and pred == "eligible":
                inelig_as_elig += 1
        out[carrier] = {
            "n_mismatch": n_mismatch,
            "eligible_marked_ineligible": elig_as_inelig,
            "ineligible_marked_eligible": inelig_as_elig,
        }
    return out


def _summarize(rows: list[dict], *, check: dict | None = None, parity: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    return {
        "n_rows": len(rows),
        "n_examples": len({row["example_id"] for row in rows}),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_evals_expected": N_TIMING_FAMILIES * len(CONDS),
        "layout": LAYOUT,
        "prefix_source": PREFIX_SOURCE,
        "extra_separator": EXTRA_SEP,
        "split": "plan4_timing12",
        "receiver": receiver,
        "carrier_label_errors": {cond: _carrier_block_n(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {
            difficulty: {
                cond: _cond_block([r for r in by_cond.get(cond, []) if r.get("difficulty") == difficulty])
                for cond in CONDS
            }
            for difficulty in LEVELS
        },
        "spec": SPEC,
        "check": None if check is None else {"all_passed": check.get("all_passed"), "n": check.get("n")},
        "parity": None if parity is None else {"all_passed": parity.get("all_passed"), "n": parity.get("n")},
        "frozen_placement": FROZEN_PLACEMENT,
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Matched reconstruction on nine frozen twelve-carrier sheets: full KV, "
            "result text, generic+transferred R, fresh R after the same support, "
            "and a fresh compact prefill."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "Interpret which of the three reconstruction possibilities holds "
            "before trying a new method. Do not train an adapter."
        ),
    }


def _prepare_runner(args, out_root: Path, extra: dict):
    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-recon12",
            "do_not_resume": list(PRESERVED),
            "do_not_overwrite": [
                "results_stage_plan4_timing",
                "results_stage_plan4_hold",
                "results_stage_plan4_confirm",
            ],
            "layout": LAYOUT,
            "conditions": list(CONDS),
            "frozen_placement": FROZEN_PLACEMENT,
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


def _run_kv_n(runner, ex, bundle, *, plan_max_new, answer_max_new):
    prompts = _prompts_for(ex)
    plan_prompt = prompts["kv_plan"]
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("KV B leaked A's raw sheet.")
    plan_text, n_plan, plan_trunc, plan_time = runner.generate_from_cache(
        bundle, plan_prompt, max_new_tokens=plan_max_new
    )
    prompt = prompts["kv_answer"](ex, plan_text)
    if prompt_contains_a_private(prompt, ex):
        raise RuntimeError("KV answer pass leaked A's raw sheet.")
    text, n_new, truncated, latency = runner.generate_from_cache(
        bundle, prompt, max_new_tokens=answer_max_new
    )
    extra = {
        "channel": "kv",
        "message_text": "",
        "message_bytes": int(bundle.full_relay_bytes),
        "n_kv_positions": n_kv_positions(bundle.cache_cpu),
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
    }
    return _row_n(ex, prompt=prompt, text=text, extra=extra)


def _run_text_n(runner, ex, *, message: str, plan_max_new, answer_max_new):
    prompts = _prompts_for(ex)
    plan_prompt = prompts["text_plan"](ex, message)
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("Text B leaked A's raw sheet.")
    plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
        plan_prompt, max_new_tokens=plan_max_new
    )
    prompt = prompts["text_answer"](ex, message, plan_text)
    text, n_new, truncated, latency = runner.generate_text_message(
        prompt, max_new_tokens=answer_max_new
    )
    extra = {
        "channel": "text",
        "message_text": message,
        "message_bytes": len(message.encode("utf-8")),
        "n_kv_positions": 0,
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
        "construct_s": 0.0,
    }
    return _row_n(ex, prompt=prompt, text=text, extra=extra)


def _run_one(runner, ex, extra_a, state, cond, args):
    spans = state["instr"]
    spec = SPEC[cond]
    if cond == "text_result":
        rec = _run_text_n(
            runner,
            ex,
            message=spans.get("result_text") or "",
            plan_max_new=args.plan_max_new_12,
            answer_max_new=args.max_new_tokens,
        )
        rec.update(extra_a)
        rec["condition"] = cond
        rec["cache_source"] = spec["cache_source"]
        rec["r_source"] = spec["r_source"]
        rec["n_carriers"] = N_CARRIERS_12
        rec["n_kept"] = 0
        rec["process_time_s"] = _process_time_s(rec)
        rec["payload_note"] = PAYLOAD_NOTE
        rec["framing_instruction"] = None
        return rec
    bundle = runner.clone_sender(state["mixed"][cond])
    rec = _run_kv_n(
        runner,
        ex,
        bundle,
        plan_max_new=args.plan_max_new_12,
        answer_max_new=args.max_new_tokens,
    )
    rec.update(extra_a)
    rec["condition"] = cond
    rec["cache_source"] = spec["cache_source"]
    rec["r_source"] = spec["r_source"]
    rec["n_carriers"] = N_CARRIERS_12
    rec["n_kept"] = n_kv_positions(bundle.cache_cpu)
    rec["n_result_kept"] = int(state["slots_extra"]["n_result"])
    rec["result_start"] = int(state["slots_extra"]["result_start"])
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["selection_time_s"] = float(state["times_by"].get(cond) or 0.0)
    rec["construct_s"] = float(state["times_by"].get(cond) or 0.0)
    rec["message_bytes"] = int(bundle.full_relay_bytes)
    rec["layout"] = LAYOUT
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["payload_note"] = PAYLOAD_NOTE
    rec["fresh_r_info"] = state.get("info_by", {}).get(cond)
    rec["process_time_s"] = _process_time_s(rec)
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    return rec


def _run_parity(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    templates, spans_est = _collect_templates(examples, artifacts, decode)
    templates_n, spans_n = _collect_templates_n(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    rows = []
    print(f"Assembly parity on {len(examples)} four-carrier bases", flush=True)
    for i, ex in enumerate(examples):
        key = a_state_key(ex)
        rec = artifacts[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        established = _pair_state_4(
            runner, rec, ex, decode, sep_ids, close_ids, spans_est[key], generics
        )
        gen_spans = spans_n[key]
        generic_rec = generics.get(gen_spans.get("template_id")) or generics.get(spans_est[key].get("template_id"))
        if generic_rec is None:
            raise RuntimeError(f"No generic template for {key}.")
        generics_n = {gen_spans["template_id"]: {**generic_rec, "template_id": gen_spans["template_id"]}}
        try:
            generalized = _pair_state_n(
                runner, rec, ex, decode, sep_ids, close_ids, gen_spans, generics_n
            )
        except RuntimeError as exc:
            row = {
                "a_state_key": key,
                "ok": False,
                "error": str(exc),
                "instruction_pos_equal": list(spans_est[key].get("instruction_pos") or [])
                == list(gen_spans.get("instruction_pos") or []),
                "result_pos_equal": list(spans_est[key].get("result_pos") or [])
                == list(gen_spans.get("result_pos") or []),
            }
            rows.append(row)
            print(f"  [{i + 1}/{len(examples)}] {key} ok=False error={exc}", flush=True)
            del established
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        est_ids = list(established["composed_extra"]["ids"])
        gen_ids = list(generalized["composed_extra"]["ids"])
        est_slots = _slots_plain(established["slots_extra"])
        gen_slots = _slots_plain(generalized["slots_extra"])
        est_mask = _mask_plain(established["mixed"]["generic_r"])
        gen_mask = _mask_plain(generalized["mixed"]["generic_r"])
        cache_eq = caches_equal(established["mixed"]["generic_r"].cache_cpu, generalized["mixed"]["generic_r"].cache_cpu)
        _rendered, input_ids, _attn = runner._encode(b_reason_kv_instruct_prompt(ex))
        cont = input_ids[:, : min(CONT_LEN, int(input_ids.size(1)))]
        logits_est = _continuation_from_bundle(runner, established["mixed"]["generic_r"], cont)
        logits_gen = _continuation_from_bundle(runner, generalized["mixed"]["generic_r"], cont)
        logits = logit_delta_stats(logits_est, logits_gen)
        ok = (
            est_ids == gen_ids
            and est_slots == gen_slots
            and est_mask == gen_mask
            and cache_eq
            and bool(logits.get("finite"))
            and float(logits.get("max_abs") or 1.0) <= 2.0
            and float(logits.get("argmax_agree") or 0.0) == 1.0
        )
        row = {
            "a_state_key": key,
            "ok": bool(ok),
            "ids_equal": est_ids == gen_ids,
            "slots_equal": est_slots == gen_slots,
            "mask_equal": est_mask == gen_mask,
            "caches_equal": bool(cache_eq),
            "pos_cursor_established": est_mask["pos_cursor"],
            "pos_cursor_generalized": gen_mask["pos_cursor"],
            "result_start_established": est_slots["result_start"],
            "result_start_generalized": gen_slots["result_start"],
            "instruction_pos_equal": list(spans_est[key].get("instruction_pos") or [])
            == list(gen_spans.get("instruction_pos") or []),
            "result_pos_equal": list(spans_est[key].get("result_pos") or [])
            == list(gen_spans.get("result_pos") or []),
            "logits": logits,
            "established_slots": est_slots,
            "generalized_slots": gen_slots,
        }
        rows.append(row)
        print(
            f"  [{i + 1}/{len(examples)}] {key} ok={ok} ids={row['ids_equal']} "
            f"slots={row['slots_equal']} cache={cache_eq} logit_max={logits.get('max_abs')}",
            flush=True,
        )
        del established, generalized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _persist(out_root, kaggle_dest)
    all_passed = bool(rows) and all(r["ok"] for r in rows)
    summary = {
        "n": len(rows),
        "all_passed": all_passed,
        "rows": rows,
        "lead": (
            "Generalized twelve-carrier assembly on original four-carrier examples "
            "and prompts. Task difficulty is unchanged."
        ),
    }
    _write_json(out_root / "parity_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"n": len(rows), "all_passed": all_passed}, indent=2))
    return 0 if all_passed else 1


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    rows = []
    print(f"Recon12 check n={len(examples)}", flush=True)
    for i, ex in enumerate(examples):
        key = a_state_key(ex)
        rec = artifacts[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state = _pair_state_n(runner, rec, ex, decode, sep_ids, close_ids, spans_by_key[key], generics)
        _attach_recon12(runner, state)
        row = state["check_row"]
        rows.append(row)
        print(
            f"  [{i + 1}/{len(examples)}] {key} ok={row['ok']} n_R={row['n_result']} "
            f"fresh_R_eq_transfer={row['fresh_r_result_equal_transferred']}",
            flush=True,
        )
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _persist(out_root, kaggle_dest)
    all_passed = bool(rows) and all(r["ok"] for r in rows)
    summary = {"n": len(rows), "all_passed": all_passed, "rows": rows, "frozen_placement": FROZEN_PLACEMENT}
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"n": len(rows), "all_passed": all_passed}, indent=2))
    return 0 if all_passed else 1


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    check = None
    cpath = out_root / "check_summary.json"
    if cpath.is_file():
        check = json.loads(cpath.read_text(encoding="utf-8"))
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first.")
        return 2
    parity = None
    ppath = out_root / "parity_summary.json"
    if ppath.is_file():
        parity = json.loads(ppath.read_text(encoding="utf-8"))
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    print(f"Recon12 compare n={len(examples)} n_evals={len(examples) * len(CONDS)}", flush=True)
    for i, ex in enumerate(examples):
        remaining = [c for c in CONDS if (ex.example_id, c) not in done]
        if not remaining:
            print(f"  [{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        key = a_state_key(ex)
        rec_a = artifacts[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state = _pair_state_n(runner, rec_a, ex, decode, sep_ids, close_ids, spans_by_key[key], generics)
        _attach_recon12(runner, state)
        extra_a = _a_extra(state)
        extra_a["rebuild_time_s"] = float(state["rebuild_time_s"])
        extra_a["n_carriers"] = N_CARRIERS_12
        reset_peak_memory()
        for cond in remaining:
            rec_out = _run_one(runner, ex, extra_a, state, cond, args)
            rows.append(rec_out)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(
                f"    {ex.example_id} {cond}: elig={rec_out.get('plan_all_status_correct')} "
                f"I->E={rec_out.get('plan_ineligible_marked_eligible')} "
                f"E->I={rec_out.get('plan_eligible_marked_ineligible')} "
                f"correct={rec_out.get('correct')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _summarize(rows, check=check, parity=parity)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:12000])
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Matched reconstruction of twelve-carrier compact KV.")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument(
        "--phase",
        choices=("parity", "check", "compare", "all", "rescore"),
        default="parity",
    )
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--artifacts12", default=None)
    args = parser.parse_args()
    if args.split not in {"plan4_timing12", "plan4_timing", "plan4_hold"}:
        print("Recon12 uses frozen timing-12 sheets and four-carrier hold bases. Do not load plan4_dev.")
        return 2

    hold_path, timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_recon12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_recon12.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "check_summary.json",
            "parity_summary.json",
            "a_artifacts_12.jsonl",
            "a_artifacts.jsonl",
            "manifest.json",
        ):
            src = kaggle_dest / name
            dest = out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    hold_all = load_plan4_jsonl(hold_path)
    for ex in hold_all:
        assert_plan4_invariants(ex)
    hold_ex = select_timing4_examples(hold_all)
    ex12 = ensure_timing12_jsonl(timing12_path)
    for ex in ex12:
        assert_timing12_invariants(ex)
    if len(hold_ex) != N_TIMING_FAMILIES or len(ex12) != N_TIMING_FAMILIES:
        print("Expected nine four-carrier bases and nine twelve-carrier sheets.")
        return 2

    if args.phase == "rescore":
        check = json.loads((out_root / "check_summary.json").read_text(encoding="utf-8")) if (out_root / "check_summary.json").is_file() else None
        parity = json.loads((out_root / "parity_summary.json").read_text(encoding="utf-8")) if (out_root / "parity_summary.json").is_file() else None
        shared = REPO_ROOT / "results_stage_plan4_recon12"
        eval_path = out_root / "evaluations.jsonl"
        if not eval_path.is_file() and (shared / "evaluations.jsonl").is_file():
            eval_path = shared / "evaluations.jsonl"
            out_root = shared
        summary = _summarize(_load_rows(eval_path), check=check, parity=parity)
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:8000])
        return 0

    artifacts4 = _load_messages(_copy_hold_artifacts(out_root, explicit=args.artifacts))
    artifacts12 = _load_messages(_copy_timing12_artifacts(out_root, explicit=args.artifacts12))
    if any(a_state_key(ex) not in artifacts4 for ex in hold_ex):
        print("Frozen hold artifacts are missing the four-carrier timing bases.")
        return 2
    if any(a_state_key(ex) not in artifacts12 for ex in ex12):
        print("Frozen 12-carrier artifacts are missing. Do not regenerate A.")
        return 2

    policy, runner, revision = _prepare_runner(args, out_root, extra={"phase": args.phase})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n4={len(hold_ex)} n12={len(ex12)}")

    if args.phase == "parity":
        return _run_parity(args, out_root, kaggle_dest, hold_ex, runner, artifacts4)
    if args.phase == "check":
        return _run_check(args, out_root, kaggle_dest, ex12, runner, artifacts12)
    if args.phase == "compare":
        return _run_compare(args, out_root, kaggle_dest, ex12, runner, artifacts12)
    rc = _run_parity(args, out_root, kaggle_dest, hold_ex, runner, artifacts4)
    if rc != 0 and not args.force_compare:
        print("Four-carrier assembly parity failed; stopping.")
        return rc
    rc = _run_check(args, out_root, kaggle_dest, ex12, runner, artifacts12)
    if rc != 0 and not args.force_compare:
        print("Twelve-carrier reconstruction check failed; stopping.")
        return rc
    return _run_compare(args, out_root, kaggle_dest, ex12, runner, artifacts12)


if __name__ == "__main__":
    raise SystemExit(main())
