"""Message-boundary layouts on frozen cons_instr_sink compact IDs.

First compare four reconstructed prefixes on plan4_dev:
current, newline separator, turn close, and both.
Result wording, B instructions, decoding, and scoring stay unchanged.
Inspect saves the complete rendered token sequence, including special tokens.

Transfer of the same layout is a later step: added tokens are forwarded,
gathered entries are RoPE-corrected, and masks/cursors stay matched.

A is not regenerated. plan4_hold stays out. Do not resume recon, iface, or support.
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

from src.boundary_layout import (
    LAYOUTS,
    compose_compact_ids,
    newline_ids,
    render_handoff_view,
    split_sink_positions,
    turn_close_ids,
)
from src.cache_utils import n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    CARRIERS,
    LEVELS,
    PLAN4_KVDIAG_FAMILIES,
    SINK_INITIAL,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    load_plan4_jsonl,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    FRAMING_INSTRUCTION,
    b_answer_kv_instruct_prompt,
    b_reason_kv_instruct_prompt,
)
from src.span_select import caches_equal
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta
from src.stage_plan4_hold import _plan_fields
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist as _persist_probe
from src.stage_plan4_recon import _compact_spans, _copy_frozen_artifacts, _rebuild_a
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block, _process_time_s
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

LABEL_MARGIN = 5.0
N_BOOT = 2000
BOOT_SEED = 2026
CONDS = LAYOUTS
PAIRED = (
    ("sep_vs_current", "current", "sep"),
    ("close_vs_current", "current", "close"),
    ("both_vs_current", "current", "both"),
    ("both_vs_sep", "sep", "both"),
    ("both_vs_close", "close", "both"),
)
PRESERVED = (
    "results_stage_plan4_recon",
    "results_stage_plan4_iface",
    "results_stage_plan4_support",
    "results_stage_plan4_hold",
    "results_stage_plan4_stable",
    "results_stage_plan4_fresh",
)
INTERPRETATION_GUIDE = (
    "Both reconstruction and transfer recover: presentation explains much of the "
    "failure; report that plainly. "
    "Reconstruction recovers but transfer remains poor: that is a stronger matched "
    "reference for investigating contextual-state reuse and attention. "
    "Reconstruction remains substantially below ordinary text: its message "
    "presentation still needs investigation. It is not yet a sufficiently reliable "
    "reference for training an adapter."
)
PAYLOAD_NOTE = (
    "KV byte counts are constructed-cache size at B. If reconstruction happens at B, "
    "the sender could transmit token IDs instead. Do not treat those bytes as the "
    "required transmitted payload. Timings still combine shared sender work and replay."
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_boundary" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_boundary")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_probe(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in (
            "inspect.json",
            "inspect.jsonl",
            "check_summary.json",
            "transfer_evaluations.jsonl",
            "transfer_summary.json",
        ):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _frozen_text_ref() -> dict:
    path = REPO_ROOT / "results_stage_plan4_recon" / "summary.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        block = (data.get("receiver") or {}).get("text_result") or {}
        if int(block.get("n") or 0) == 90:
            return {
                "n": 90,
                "plan_all_status_correct": block.get("plan_all_status_correct"),
                "correct": block.get("correct"),
                "parse_ok": block.get("parse_ok"),
                "truncated": block.get("truncated"),
                "source": "results_stage_plan4_recon/summary.json",
            }
    return {
        "n": 90,
        "plan_all_status_correct": 90,
        "correct": 79,
        "parse_ok": 90,
        "truncated": 0,
        "source": "recorded recon text_result",
    }


def candidate_qualifies(block: dict, text_block: dict) -> bool:
    n = int(block.get("n") or 0)
    if n == 0 or int(text_block.get("n") or 0) == 0:
        return False
    if int(block.get("parse_ok") or 0) != n:
        return False
    if int(block.get("truncated") or 0) != 0:
        return False
    cand = 100.0 * int(block.get("plan_all_status_correct") or 0) / n
    text = 100.0 * int(text_block.get("plan_all_status_correct") or 0) / int(text_block["n"])
    return cand + 1e-9 >= text - LABEL_MARGIN


def _carrier_block(items: list[dict]) -> dict:
    out = {}
    for carrier in CARRIERS:
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


def _summarize(rows: list[dict], *, inspect: dict | None = None, text_ref: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("current", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS}
    text_ref = text_ref or _frozen_text_ref()
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "n_evals_expected": 90 * len(CONDS),
        "baseline": "reconstructed cons_instr_sink current layout",
        "cache_source": "reconstruct_prefill",
        "unit_note": (
            "plan4_dev only. Frozen cons_instr_sink result IDs. Four reconstructed "
            "prefixes: current, newline separator, turn close, both. B instructions "
            "and result wording are unchanged. Ordinary result text is the frozen "
            "recon reference, not rerun."
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
        "frozen_text_result": text_ref,
        "qualifies_vs_text": {
            cond: candidate_qualifies(receiver.get(cond) or {}, text_ref) for cond in CONDS
        },
        "framing_change": FRAMING_INSTRUCTION,
        "inspect": None
        if inspect is None
        else {"all_passed": inspect.get("all_passed"), "n": inspect.get("n")},
        "payload_note": PAYLOAD_NOTE,
        "lead": (
            "Primary measure: all four eligibility labels on reconstructed compact KV. "
            "Separator vs turn closure vs both, against the current YouP layout."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "after_this_run": (
            "These are development reconstructed-layout results. Choose a layout, then "
            "run --phase transfer --layout <name>. Do not load plan4_hold."
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
            "stage": "plan4-boundary",
            "split": "plan4_dev",
            "do_not_resume": list(PRESERVED),
            "do_not_load": ["data/plan4_hold.jsonl", "data/plan4_fresh.jsonl"],
            "baseline": "cons_instr_sink",
            "layouts": list(LAYOUTS),
            "framing_change": FRAMING_INSTRUCTION,
            "sink_initial": SINK_INITIAL,
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


def _added_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    return newline_ids(tokenizer), turn_close_ids(tokenizer)


def _layouts_for_spans(spans: dict, *, sep_ids: list[int], close_ids: list[int]) -> dict:
    split = split_sink_positions(spans["sink_pos"])
    initial_ids = [int(spans["selected_ids"][i]) for i in range(len(split["initial_pos"]))]
    result_ids = [int(spans["selected_ids"][i]) for i in range(len(split["initial_pos"]), len(spans["selected_ids"]))]
    if initial_ids + result_ids != list(spans["selected_ids"]):
        raise RuntimeError("Initial/result split does not match frozen sink token IDs.")
    composed = {
        name: compose_compact_ids(
            initial_ids, result_ids, sep_ids=sep_ids, close_ids=close_ids, layout=name
        )
        for name in LAYOUTS
    }
    return {
        "split": split,
        "initial_ids": initial_ids,
        "result_ids": result_ids,
        "composed": composed,
    }


def _build_recon(runner, pack: dict, layout: str):
    composed = pack["composed"][layout]
    bundle, wall = runner.prefill_token_sequence(
        composed["ids"], pos_cursor=composed["n"], source="reconstruct"
    )
    return bundle, wall, composed


def _build_transfer(runner, a_bundle, pack: dict, layout: str):
    composed = pack["composed"][layout]
    split = pack["split"]
    bundle, wall, info = runner.transfer_boundary_cache(
        a_bundle,
        initial_pos=split["initial_pos"],
        result_pos=split["result_pos"],
        composed=composed,
        source="transfer",
    )
    return bundle, wall, info, composed


def _annotate_plan(rec: dict, ex) -> None:
    _plan_fields(rec, ex)
    rec["gold_status"] = gold_eligibility_statuses(ex)


def _run_recon_one(runner, ex, extra_a, spans, pack, layout, args, *, bundle, wall, composed):
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
    rec["condition"] = layout
    rec["selection_time_s"] = float(wall)
    rec["compression_time_s"] = float(wall)
    rec["n_kept"] = int(composed["n"])
    rec["n_result_kept"] = len(spans["result_pos"])
    rec["n_sink_kept"] = int(spans.get("n_sink_extra") or 0)
    rec["n_added"] = int(composed["n_added"])
    rec["n_full"] = int(spans["n_full"])
    rec["selected_text"] = spans["result_text"]
    rec["selected_token_ids"] = list(composed["ids"])
    rec["result_token_ids"] = list(pack["result_ids"])
    rec["layout"] = layout
    rec["layout_separator"] = bool(composed["separator"])
    rec["layout_turn_close"] = bool(composed["turn_close"])
    rec["framing"] = "instruct"
    rec["framing_instruction"] = FRAMING_INSTRUCTION
    rec["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
    rec["message_bytes"] = int(bundle.full_relay_bytes) + rec["framing_bytes"]
    rec["cache_source"] = "reconstruct_prefill"
    rec["result_fallback"] = bool(spans.get("result_fallback"))
    rec["shared_across_heads"] = True
    rec["pos_cursor"] = int(bundle.pos_cursor_cpu.view(-1)[0].item())
    rec["n_kv_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["n_prompt_positions"] = n_kv_positions(bundle.cache_cpu)
    rec["plan_prompt"] = b_reason_kv_instruct_prompt(ex)
    rec["payload_note"] = PAYLOAD_NOTE
    rec["process_time_s"] = _process_time_s(rec)
    rec["peak_gpu_gb"] = peak_gpu_memory_gb()
    _annotate_plan(rec, ex)
    return rec


def _wanted_examples(examples):
    wanted = []
    seen = set()
    for ex in examples:
        if ex.family_id in PLAN4_KVDIAG_FAMILIES and a_state_key(ex) not in seen:
            wanted.append(ex)
            seen.add(a_state_key(ex))
    return wanted


def _run_inspect(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    wanted = _wanted_examples(examples)
    inspect_rows = []
    construction = []
    print(f"Boundary inspect on {len(wanted)} unique A sheets", flush=True)
    print(f"sep_ids={sep_ids} close_ids={close_ids}", flush=True)
    for i, ex in enumerate(wanted):
        key = a_state_key(ex)
        rec = artifacts[key]
        src = by_id.get(key) or ex
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        a_bundle, rebuild_s = _rebuild_a(runner, rec, src)
        spans = _compact_spans(
            rec["full_ids"],
            n_prompt=int(rec["n_prompt_positions"]),
            n_full=int(rec["n_kv_positions"]),
            decode=decode,
        )
        pack = _layouts_for_spans(spans, sep_ids=sep_ids, close_ids=close_ids)
        current_ids = pack["composed"]["current"]["ids"]
        if current_ids != list(spans["selected_ids"]):
            raise RuntimeError("Current layout must keep the frozen compact token IDs.")
        b_prompt = b_reason_kv_instruct_prompt(ex)
        _rendered, b_ids, _mask = runner._encode(b_prompt)
        b_turn = [int(t) for t in b_ids.detach().cpu().long().view(-1).tolist()]
        baseline, _w, base_info = runner.keep_shared_positions(
            a_bundle, spans["sink_pos"], layout="consecutive"
        )
        layout_ok = {}
        views = {}
        for layout in LAYOUTS:
            recon, recon_s, composed = _build_recon(runner, pack, layout)
            transfer, trans_s, info, _composed = _build_transfer(runner, a_bundle, pack, layout)
            recon_ids = [int(t) for t in recon.full_ids_cpu.view(-1).tolist()]
            trans_ids = [int(t) for t in transfer.full_ids_cpu.view(-1).tolist()]
            view = render_handoff_view(tokenizer, recon_ids, b_turn)
            views[layout] = view
            same_cursor = int(recon.pos_cursor_cpu.view(-1)[0].item()) == int(
                transfer.pos_cursor_cpu.view(-1)[0].item()
            )
            same_len = n_kv_positions(recon.cache_cpu) == n_kv_positions(transfer.cache_cpu)
            ok = (
                recon_ids == composed["ids"]
                and trans_ids == composed["ids"]
                and same_cursor
                and same_len
                and int(recon.pos_cursor_cpu.view(-1)[0].item()) == composed["n"]
                and int(recon.past_mask_cpu.size(1)) == composed["n"]
                and int(transfer.past_mask_cpu.size(1)) == composed["n"]
            )
            if layout == "current":
                ok = ok and caches_equal(baseline.cache_cpu, transfer.cache_cpu)
            flags = view["junction"]
            if layout in {"current", "close"}:
                ok = ok and flags["has_youp"] and not flags["has_you_newline_p"]
            if layout in {"sep", "both"}:
                ok = ok and flags["has_you_newline_p"] and not flags["has_youp"]
            if layout in {"close", "both"}:
                ok = ok and flags["has_im_end"] and flags["im_end_before_b_turn"]
            if layout in {"current", "sep"}:
                ok = ok and not flags["has_im_end"]
            layout_ok[layout] = {
                "ok": ok,
                "n": composed["n"],
                "n_added": composed["n_added"],
                "recon_s": recon_s,
                "transfer_s": trans_s,
                "ids_match": recon_ids == trans_ids == composed["ids"],
                "caches_equal_pair": caches_equal(recon.cache_cpu, transfer.cache_cpu),
                "current_transfer_eq_baseline": (
                    caches_equal(baseline.cache_cpu, transfer.cache_cpu) if layout == "current" else None
                ),
                "junction": flags,
                "prefix_head": view["prefix_head"],
                "prefix_tail": view["prefix_tail"],
            }
            inspect_rows.append(
                {
                    "a_state_key": key,
                    "example_id": ex.example_id,
                    "layout": layout,
                    "ok": ok,
                    "n_prefix": view["n_prefix"],
                    "n_b_turn": view["n_b_turn"],
                    "n_full": view["n_full"],
                    "n_added": composed["n_added"],
                    "result_text": spans["result_text"],
                    "prefix_text": view["prefix_text"],
                    "full_text": view["full_text"],
                    "prefix_ids": view["prefix_ids"],
                    "full_ids": view["full_ids"],
                    "junction": flags,
                    "rebuild_time_s": rebuild_s,
                    "baseline_n_kept": int(base_info["n_kept"]),
                }
            )
        construction.append(
            {
                "a_state_key": key,
                "n_current": pack["composed"]["current"]["n"],
                "n_result": pack["composed"]["current"]["n_result"],
                "result_ids_shared": all(
                    pack["composed"][name]["result_ids"] == pack["result_ids"] for name in LAYOUTS
                ),
                "layouts": layout_ok,
            }
        )
        print(
            f"[{i + 1}/{len(wanted)}] {key} current_youp={layout_ok['current']['junction']['has_youp']} "
            f"sep_nl={layout_ok['sep']['junction']['has_you_newline_p']} "
            f"close_end={layout_ok['close']['junction']['has_im_end']}",
            flush=True,
        )
        _persist(out_root, kaggle_dest)

    all_ok = bool(construction) and all(
        item["result_ids_shared"] and all(rec["ok"] for rec in item["layouts"].values()) for item in construction
    )
    summary = {
        "n": len(construction),
        "families": list(PLAN4_KVDIAG_FAMILIES),
        "all_passed": all_ok,
        "sep_ids": sep_ids,
        "close_ids": close_ids,
        "sep_text": tokenizer.decode(sep_ids, skip_special_tokens=False),
        "close_text": tokenizer.decode(close_ids, skip_special_tokens=False),
        "construction": construction,
        "lead": (
            "Inspect complete rendered sequences, including special tokens. "
            "Current should show YouP. Separator should break that. Turn close should "
            "place <|im_end|> before B's user turn. Result IDs stay frozen."
        ),
    }
    jsonl_path = out_root / "inspect.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in inspect_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(out_root / "inspect.json", summary)
    _write_json(out_root / "check_summary.json", {"all_passed": all_ok, "n": summary["n"], "phase": "inspect"})
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: summary[k] for k in ("n", "all_passed", "sep_text", "close_text")}, indent=2))
    if inspect_rows:
        sample = next(row for row in inspect_rows if row["layout"] == "current")
        print("current prefix head:", sample["prefix_text"][:180])
        sep_row = next(row for row in inspect_rows if row["layout"] == "sep" and row["a_state_key"] == sample["a_state_key"])
        print("sep prefix head:", sep_row["prefix_text"][:180])
        close_row = next(
            row for row in inspect_rows if row["layout"] == "close" and row["a_state_key"] == sample["a_state_key"]
        )
        print("close prefix tail:", close_row["prefix_text"][-120:])
    return 0 if all_ok else 1


def _run_reconstruct(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    inspect_path = out_root / "inspect.json"
    inspect = json.loads(inspect_path.read_text(encoding="utf-8")) if inspect_path.is_file() else None
    if (inspect is None or not inspect.get("all_passed")) and not args.force_compare:
        print("Run --phase inspect first. Pass --force-compare only to skip that gate.")
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
    print(f"Rebuilding {len(needed)} unique A caches from recorded token IDs", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bundle, rebuild_s = _rebuild_a(runner, rec, src)
        spans = _compact_spans(
            rec["full_ids"],
            n_prompt=int(rec["n_prompt_positions"]),
            n_full=int(rec["n_kv_positions"]),
            decode=decode,
        )
        pack = _layouts_for_spans(spans, sep_ids=sep_ids, close_ids=close_ids)
        recon = {}
        times = {}
        for layout in LAYOUTS:
            recon[layout], times[layout], _composed = _build_recon(runner, pack, layout)
        states[key] = {
            "bundle": bundle,
            "meta": rec,
            "rebuild_time_s": rebuild_s,
            "spans": spans,
            "pack": pack,
            "recon": recon,
            "times": times,
        }
        print(f"  rebuilt {key} n_current={pack['composed']['current']['n']}", flush=True)
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [c for c in CONDS if (ex.example_id, c) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        pack_state = states[a_state_key(ex)]
        extra_a = _a_extra(pack_state)
        extra_a["rebuild_time_s"] = float(pack_state["rebuild_time_s"])
        extra_a["cache_source"] = "prefill_recorded_ids"
        extra_a["ids_source"] = pack_state["meta"].get("ids_source")
        extra_a["reused_a_from"] = (
            pack_state["meta"]["a_state_key"] if pack_state["meta"]["a_state_key"] != ex.example_id else None
        )
        for layout in remaining:
            rec_out = _run_recon_one(
                runner,
                ex,
                extra_a,
                pack_state["spans"],
                pack_state["pack"],
                layout,
                args,
                bundle=runner.clone_sender(pack_state["recon"][layout]),
                wall=pack_state["times"][layout],
                composed=pack_state["pack"]["composed"][layout],
            )
            rows.append(rec_out)
            done.add((ex.example_id, layout))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(
                f"  {layout}: pred={rec_out.get('pred_carrier')} correct={rec_out.get('correct')} "
                f"elig={rec_out.get('plan_all_status_correct')} parse={rec_out.get('parse_ok')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    summary = _summarize(rows, inspect=inspect)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:12000])
    print(json.dumps(summary.get("carrier_label_errors"), indent=2))
    print(json.dumps(summary.get("qualifies_vs_text"), indent=2))
    return 0


def _run_transfer(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    layout = args.layout
    if layout not in LAYOUTS:
        print("Transfer needs --layout current|sep|close|both after the reconstruct comparison.")
        return 2
    inspect_path = out_root / "inspect.json"
    inspect = json.loads(inspect_path.read_text(encoding="utf-8")) if inspect_path.is_file() else None
    if (inspect is None or not inspect.get("all_passed")) and not args.force_compare:
        print("Run --phase inspect, then reconstruct, then transfer. Pass --force-compare only to skip inspect.")
        return 2
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    by_id = {ex.example_id: ex for ex in examples}
    jsonl_path = out_root / "transfer_evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    conds = ("transfer", "reconstruct")
    done = {(row["example_id"], row["condition"]) for row in rows if row.get("layout") == layout}
    states: dict = {}
    needed = {a_state_key(ex) for ex in examples if any((ex.example_id, c) not in done for c in conds)}
    print(f"Transfer layout={layout} rebuilding {len(needed)} unique A caches", flush=True)
    for key in sorted(needed):
        rec = artifacts[key]
        src = by_id.get(key) or next(ex for ex in examples if a_state_key(ex) == key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bundle, rebuild_s = _rebuild_a(runner, rec, src)
        spans = _compact_spans(
            rec["full_ids"],
            n_prompt=int(rec["n_prompt_positions"]),
            n_full=int(rec["n_kv_positions"]),
            decode=decode,
        )
        pack = _layouts_for_spans(spans, sep_ids=sep_ids, close_ids=close_ids)
        recon, recon_s, composed = _build_recon(runner, pack, layout)
        transfer, trans_s, info, _c = _build_transfer(runner, bundle, pack, layout)
        if [int(t) for t in recon.full_ids_cpu.view(-1).tolist()] != list(composed["ids"]):
            raise RuntimeError("Reconstruct IDs drifted from the composed layout.")
        if [int(t) for t in transfer.full_ids_cpu.view(-1).tolist()] != list(composed["ids"]):
            raise RuntimeError("Transfer IDs drifted from the composed layout.")
        states[key] = {
            "bundle": bundle,
            "meta": rec,
            "rebuild_time_s": rebuild_s,
            "spans": spans,
            "pack": pack,
            "recon": recon,
            "transfer": transfer,
            "recon_s": recon_s,
            "trans_s": trans_s,
            "info": info,
            "composed": composed,
        }
        print(f"  rebuilt {key} n={composed['n']} added={composed['n_added']}", flush=True)
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [c for c in conds if (ex.example_id, c) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        pack_state = states[a_state_key(ex)]
        extra_a = _a_extra(pack_state)
        extra_a["rebuild_time_s"] = float(pack_state["rebuild_time_s"])
        extra_a["ids_source"] = pack_state["meta"].get("ids_source")
        extra_a["reused_a_from"] = (
            pack_state["meta"]["a_state_key"] if pack_state["meta"]["a_state_key"] != ex.example_id else None
        )
        for cond in remaining:
            if cond == "transfer":
                bundle = runner.clone_sender(pack_state["transfer"])
                wall = float(pack_state["trans_s"])
                cache_source = "transfer_gather_rerotate_extend"
            else:
                bundle = runner.clone_sender(pack_state["recon"])
                wall = float(pack_state["recon_s"])
                cache_source = "reconstruct_prefill"
            rec_out = _run_kv_b(
                runner,
                ex,
                bundle,
                plan_max_new=args.plan_max_new,
                answer_max_new=args.max_new_tokens,
                plan_prompt=b_reason_kv_instruct_prompt(ex),
                make_answer_prompt=b_answer_kv_instruct_prompt,
            )
            rec_out.update(extra_a)
            rec_out["condition"] = cond
            rec_out["layout"] = layout
            rec_out["layout_separator"] = bool(pack_state["composed"]["separator"])
            rec_out["layout_turn_close"] = bool(pack_state["composed"]["turn_close"])
            rec_out["selection_time_s"] = wall
            rec_out["n_kept"] = int(pack_state["composed"]["n"])
            rec_out["n_added"] = int(pack_state["composed"]["n_added"])
            rec_out["n_result_kept"] = len(pack_state["spans"]["result_pos"])
            rec_out["n_sink_kept"] = int(pack_state["spans"].get("n_sink_extra") or 0)
            rec_out["n_full"] = int(pack_state["spans"]["n_full"])
            rec_out["selected_text"] = pack_state["spans"]["result_text"]
            rec_out["selected_token_ids"] = list(pack_state["composed"]["ids"])
            rec_out["framing"] = "instruct"
            rec_out["framing_instruction"] = FRAMING_INSTRUCTION
            rec_out["framing_bytes"] = len(FRAMING_INSTRUCTION.encode("utf-8"))
            rec_out["message_bytes"] = int(bundle.full_relay_bytes) + rec_out["framing_bytes"]
            rec_out["cache_source"] = cache_source
            rec_out["shared_across_heads"] = True
            rec_out["payload_note"] = PAYLOAD_NOTE
            rec_out["process_time_s"] = _process_time_s(rec_out)
            rec_out["peak_gpu_gb"] = peak_gpu_memory_gb()
            _annotate_plan(rec_out, ex)
            rows.append(rec_out)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec_out.get('pred_carrier')} correct={rec_out.get('correct')} "
                f"elig={rec_out.get('plan_all_status_correct')}",
                flush=True,
            )
            _persist(out_root, kaggle_dest)

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("layout") == layout:
            by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("layout") == layout:
            grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    text_ref = _frozen_text_ref()
    receiver = {cond: _cond_block(by_cond.get(cond, [])) for cond in conds}
    summary = {
        "layout": layout,
        "n_rows": sum(len(v) for v in by_cond.values()),
        "n_examples": len(grouped),
        "n_evals_expected": 90 * len(conds),
        "receiver": receiver,
        "carrier_label_errors": {cond: _carrier_block(by_cond.get(cond, [])) for cond in conds},
        "paired_receiver": {"reconstruct_vs_transfer": _paired(grouped_plain, "transfer", "reconstruct")},
        "family_clustered_eligibility": {
            "reconstruct_vs_transfer": _family_delta(
                [r for r in rows if r.get("layout") == layout],
                "transfer",
                "reconstruct",
                n_boot=N_BOOT,
                seed=BOOT_SEED,
                key="plan_all_status_correct",
            )
        },
        "frozen_text_result": text_ref,
        "qualifies_vs_text": {cond: candidate_qualifies(receiver.get(cond) or {}, text_ref) for cond in conds},
        "payload_note": PAYLOAD_NOTE,
        "interpretation_guide": INTERPRETATION_GUIDE,
        "lead": (
            f"Matched transfer vs reconstruct under layout={layout}. Same token IDs, "
            "receiver prompts, positions, masks, and decoding."
        ),
        "after_this_run": "These are development results. Do not load plan4_hold.",
    }
    _write_json(out_root / "transfer_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary["receiver"], indent=2)[:8000])
    print(json.dumps(summary.get("carrier_label_errors"), indent=2))
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="plan4_dev reconstructed compact boundary layouts, then optional matched transfer."
    )
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument(
        "--phase",
        choices=("inspect", "reconstruct", "transfer", "all", "rescore"),
        default="inspect",
    )
    parser.add_argument("--layout", default=None, help="current|sep|close|both; required for --phase transfer")
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
        print("Boundary layouts use plan4_dev only. Do not load plan4_hold or plan4_fresh.")
        return 2
    if args.phase == "transfer" and args.layout not in LAYOUTS:
        print("Transfer needs --layout current|sep|close|both after the reconstruct comparison.")
        return 2

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_boundary" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_boundary.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in (
            "evaluations.jsonl",
            "summary.json",
            "inspect.json",
            "inspect.jsonl",
            "check_summary.json",
            "transfer_evaluations.jsonl",
            "transfer_summary.json",
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

    if args.phase == "rescore":
        inspect = None
        ipath = out_root / "inspect.json"
        if ipath.is_file():
            inspect = json.loads(ipath.read_text(encoding="utf-8"))
        summary = _summarize(_load_rows(out_root / "evaluations.jsonl"), inspect=inspect)
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps({cond: summary["receiver"].get(cond) for cond in CONDS}, indent=2)[:8000])
        return 0

    policy, runner, revision = _prepare_runner(
        args, out_root, extra={"phase": args.phase, "conditions": list(CONDS), "transfer_layout": args.layout}
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    if args.phase == "inspect":
        return _run_inspect(args, out_root, kaggle_dest, examples, runner, artifacts)
    if args.phase == "reconstruct":
        return _run_reconstruct(args, out_root, kaggle_dest, examples, runner, artifacts)
    if args.phase == "transfer":
        return _run_transfer(args, out_root, kaggle_dest, examples, runner, artifacts)
    rc = _run_inspect(args, out_root, kaggle_dest, examples, runner, artifacts)
    if rc != 0 and not args.force_compare:
        print("Inspect failed; stopping before the 90-case reconstructed comparison.")
        return rc
    return _run_reconstruct(args, out_root, kaggle_dest, examples, runner, artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
