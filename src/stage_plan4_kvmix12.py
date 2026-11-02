"""Result-region key/value swap on the twelve-carrier generic-support message.

Two versions of R under the identical generic-support cache:
- transferred R: A's selected result states, already aligned to the packed slots
- fresh R: the same result tokens forwarded after that support

Four conditions replace only the result-region tensors:
- tk_tv: transferred K + transferred V (generic_r reference)
- fk_tv: fresh K + transferred V
- tk_fv: transferred K + fresh V
- fk_fv: fresh K + fresh V (generic_fresh_r reference)

Support, initial prefix and separator tensors, token IDs, packed positions,
masks, B's starting position, receiver prompts and decoding stay identical.
Keys are copied as stored; RoPE is not applied again.

Primary: the original twelve-line protocol (9 cases x 4 = 36). Secondary: the
single-carrier question (9 x 12 x 4 = 432), reusing the same constructed caches.
Frozen A outputs only. These mixed caches are diagnostic interventions, not a
repair, and computing fresh states is not an efficient method.
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
from src.plan4_task import LEVELS, a_state_key
from src.plan4_timing_task import (
    N_CARRIERS_12,
    N_TIMING_FAMILIES,
    assert_timing12_invariants,
    ensure_timing12_jsonl,
    gold_eligibility_statuses_n,
)
from src.precision import apply_precision_policy, reset_peak_memory
from src.prompts_plan4 import FRAMING_INSTRUCTION, prompt_contains_a_private
from src.prompts_plan4_query import QUERY_MAX_NEW, b_query_kv_prompt, parse_single_label
from src.span_select import caches_equal
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_boundary import PAYLOAD_NOTE, _added_token_ids
from src.stage_plan4_generic import _prefill_templates
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_query12 import (
    _build_state,
    _check_construction,
    _cluster_delta,
    _copy_first,
    _recon_refs,
)
from src.stage_plan4_query12 import _cond_block as _query_block
from src.stage_plan4_recon12 import _run_kv_n
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_support import _cond_block
from src.stage_plan4_timing import B_PLAN_MAX_NEW_12, _collect_templates_n
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

CONDS = ("tk_tv", "fk_tv", "tk_fv", "fk_fv")
SPEC = {
    "tk_tv": {"keys": "transfer", "values": "transfer", "reference": "generic_r"},
    "fk_tv": {"keys": "fresh", "values": "transfer", "reference": None},
    "tk_fv": {"keys": "transfer", "values": "fresh", "reference": None},
    "fk_fv": {"keys": "fresh", "values": "fresh", "reference": "generic_fresh_r"},
}
PAIRED = (
    ("fk_tv_vs_tk_tv", "tk_tv", "fk_tv"),
    ("tk_fv_vs_tk_tv", "tk_tv", "tk_fv"),
    ("fk_fv_vs_tk_tv", "tk_tv", "fk_fv"),
    ("fk_tv_vs_fk_fv", "fk_fv", "fk_tv"),
    ("tk_fv_vs_fk_fv", "fk_fv", "tk_fv"),
    ("fk_tv_vs_tk_fv", "tk_fv", "fk_tv"),
)
PRESERVED = (
    "results_stage_plan4_query12",
    "results_stage_plan4_recon12",
    "results_stage_plan4_timing",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
    "results_stage_plan4_kvswap",
)
FROZEN_PLACEMENT = (
    "Base cache is generic support + transferred R. Only result-region keys and/or "
    "values are replaced by the fresh-R tensors computed after that same support. "
    "Support, prefix and separator tensors, token IDs, packed positions, masks, "
    "cursor, prompts and greedy decoding are identical. Keys are copied as stored; "
    "no RoPE is applied again."
)
INTERPRETATION_GUIDE = (
    "If fresh K + transferred V substantially recovers, changing the keys is "
    "sufficient for improvement; that strengthens investigating attention or "
    "query-side adaptation while keeping transferred values. If transferred K + "
    "fresh V substantially recovers, the values deserve closer investigation, "
    "including selective reconstruction. If only replacing both works well, the "
    "key/value interaction matters and a one-component repair may be insufficient. "
    "If a hybrid works under only one prompt, the repair is sensitive to how B is "
    "asked to use the message. Mixed caches are diagnostic interventions: they do "
    "not establish that keys or values 'contain the error', and computing fresh "
    "states does not demonstrate an efficient repair. Nine cases are diagnostic "
    "examples, not independent confirmation."
)
PERSIST_NAMES = (
    "evaluations.jsonl",
    "query_evaluations.jsonl",
    "summary.json",
    "check_summary.json",
    "manifest.json",
    "a_artifacts_12.jsonl",
    "recon12_evaluations.jsonl",
    "recon12_check_summary.json",
    "query12_evaluations.jsonl",
)


def _paths():
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_kvmix12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_kvmix12")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return timing12, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in PERSIST_NAMES:
            src = out_root / name
            if src.is_file() and src.resolve() != (dest / name).resolve():
                (dest / name).write_bytes(src.read_bytes())


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _copy_inputs(out_root: Path, *, artifacts12: str | None = None) -> dict:
    recon = [REPO_ROOT / "results_stage_plan4_recon12", Path("/kaggle/working/results_stage_plan4_recon12")]
    query = [REPO_ROOT / "results_stage_plan4_query12", Path("/kaggle/working/results_stage_plan4_query12")]
    art_sources = ([Path(artifacts12)] if artifacts12 else []) + [
        *(d / "a_artifacts_12.jsonl" for d in recon + query),
        REPO_ROOT / "results_stage_plan4_timing" / "a_artifacts_12.jsonl",
    ]
    return {
        "artifacts": _copy_first(out_root, "a_artifacts_12.jsonl", art_sources, what="frozen 12-carrier A artifacts"),
        "recon_evals": _copy_first(
            out_root, "recon12_evaluations.jsonl", [d / "evaluations.jsonl" for d in recon], what="recon12 evaluations"
        ),
        "recon_check": _copy_first(
            out_root, "recon12_check_summary.json", [d / "check_summary.json" for d in recon], what="recon12 check"
        ),
        "query_evals": _copy_first(
            out_root, "query12_evaluations.jsonl", [d / "evaluations.jsonl" for d in query], what="query12 evaluations"
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
            "stage": "plan4-kvmix12",
            "do_not_overwrite": list(PRESERVED),
            "conditions": list(CONDS),
            "plan_max_new_12": args.plan_max_new_12,
            "answer_max_new": args.max_new_tokens,
            "query_max_new": QUERY_MAX_NEW,
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
    if revision.get("sha"):
        pin_model_revision(revision["sha"])
    return policy, runner


def _attach_mixes(runner, state: dict) -> dict:
    """Build the four caches and verify every unchanged tensor. Returns the check row."""
    transfer = state["mixed"]["generic_r"]
    fresh = state["mixed"]["generic_fresh_r"]
    slots = state["slots_extra"]
    result = list(slots["result"])
    support = list(range(int(slots["result_start"])))
    src = {"transfer": transfer, "fresh": fresh}
    mixes = {}
    for cond in CONDS:
        spec = SPEC[cond]
        mixes[cond] = runner.mix_sender_kv_at(
            src[spec["keys"]], src[spec["values"]], base=transfer, positions=result, source=f"kvmix12_{cond}"
        )
    ids = [int(t) for t in transfer.full_ids_cpu.view(-1).tolist()]
    n = n_kv_positions(transfer.cache_cpu)
    cursor = int(transfer.pos_cursor_cpu.view(-1)[0].item())
    mask = transfer.past_mask_cpu.view(-1).tolist()
    per = {}
    ok = True
    for cond, bundle in mixes.items():
        spec = SPEC[cond]
        sup = cache_slice_delta(bundle.cache_cpu, transfer.cache_cpu, support)
        k_ref = cache_slice_delta(bundle.cache_cpu, src[spec["keys"]].cache_cpu, result)
        v_ref = cache_slice_delta(bundle.cache_cpu, src[spec["values"]].cache_cpu, result)
        row = {
            "ids_equal": [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()] == ids,
            "n_kv_equal": n_kv_positions(bundle.cache_cpu) == n,
            "cursor_equal": int(bundle.pos_cursor_cpu.view(-1)[0].item()) == cursor,
            "mask_equal": bundle.past_mask_cpu.view(-1).tolist() == mask,
            "support_equal": bool(sup["equal"]),
            "result_keys_from_source": bool(k_ref["keys_equal"]),
            "result_values_from_source": bool(v_ref["values_equal"]),
        }
        if spec["reference"] is not None:
            row["equals_reference"] = bool(caches_equal(bundle.cache_cpu, state["mixed"][spec["reference"]].cache_cpu))
        cond_ok = all(bool(v) for v in row.values())
        row["ok"] = cond_ok
        ok = ok and cond_ok
        per[cond] = row
    k_diff = cache_slice_delta(fresh.cache_cpu, transfer.cache_cpu, result)
    state["kvmix"] = mixes
    return {
        "ok": ok,
        "n_kv": n,
        "pos_cursor": cursor,
        "result_start": int(slots["result_start"]),
        "n_result": len(result),
        "fresh_vs_transfer_result": {
            "keys_equal": bool(k_diff["keys_equal"]),
            "values_equal": bool(k_diff["values_equal"]),
            "keys_max_abs": k_diff["keys_max_abs"],
            "values_max_abs": k_diff["values_max_abs"],
            "keys_mean_abs": k_diff["keys_mean_abs"],
            "values_mean_abs": k_diff["values_mean_abs"],
        },
        "conditions": per,
    }


def _setup(runner, examples, artifacts):
    decode = _decode_fn(runner)
    sep_ids, close_ids = _added_token_ids(runner.wrapper.tokenizer)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    return decode, sep_ids, close_ids, spans_by_key, generics


def _state_for(runner, ex, artifacts, setup, recon_refs):
    decode, sep_ids, close_ids, spans_by_key, generics = setup
    key = a_state_key(ex)
    state = _build_state(runner, ex, artifacts[key], decode, sep_ids, close_ids, spans_by_key[key], generics)
    built = _check_construction(state, recon_refs.get(key))
    mix = _attach_mixes(runner, state)
    return state, built, mix


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs) -> int:
    setup = _setup(runner, examples, artifacts)
    rows = []
    for i, ex in enumerate(examples):
        state, built, mix = _state_for(runner, ex, artifacts, setup, recon_refs)
        row = {
            "a_state_key": a_state_key(ex),
            "ok": bool(built["matches_recon12"] and built["support_held"] and mix["ok"]),
            "construction": built,
            **mix,
        }
        rows.append(row)
        print(
            f"  [{i + 1}/{len(examples)}] {row['a_state_key']} ok={row['ok']} "
            f"recon12={built['matches_recon12']} mixes={mix['ok']}",
            flush=True,
        )
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_passed = bool(rows) and all(r["ok"] for r in rows)
    _write_json(
        out_root / "check_summary.json",
        {"n": len(rows), "all_passed": all_passed, "rows": rows, "frozen_placement": FROZEN_PLACEMENT},
    )
    _persist(out_root, kaggle_dest)
    print(json.dumps({"n": len(rows), "all_passed": all_passed}, indent=2))
    return 0 if all_passed else 1


def _twelve_row(runner, ex, state, cond, args) -> dict:
    base = state["kvmix"][cond]
    rec = _run_kv_n(
        runner,
        ex,
        runner.clone_sender(base),
        plan_max_new=args.plan_max_new_12,
        answer_max_new=args.max_new_tokens,
    )
    rec.update(
        {
            "condition": cond,
            "keys": SPEC[cond]["keys"],
            "values": SPEC[cond]["values"],
            "a_state_key": a_state_key(ex),
            "n_carriers": N_CARRIERS_12,
            "n_kept": n_kv_positions(base.cache_cpu),
            "n_result_kept": int(state["slots_extra"]["n_result"]),
            "result_start": int(state["slots_extra"]["result_start"]),
            "pos_cursor": int(base.pos_cursor_cpu.view(-1)[0].item()),
            "layout": LAYOUT,
            "framing": "instruct",
            "framing_instruction": FRAMING_INSTRUCTION,
            "payload_note": PAYLOAD_NOTE,
        }
    )
    return rec


def _query_row(runner, ex, state, cond, carrier, gold) -> dict:
    base = state["kvmix"][cond]
    prompt = b_query_kv_prompt(ex, carrier)
    if prompt_contains_a_private(prompt, ex):
        raise RuntimeError(f"Single-label prompt leaked A's raw sheet for {ex.example_id}.")
    text, n_new, trunc, lat = runner.generate_from_cache(
        runner.clone_sender(base), prompt, max_new_tokens=QUERY_MAX_NEW
    )
    parsed = parse_single_label(text)
    pred = parsed["pred_label"]
    return {
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "difficulty": ex.difficulty,
        "a_state_key": a_state_key(ex),
        "condition": cond,
        "keys": SPEC[cond]["keys"],
        "values": SPEC[cond]["values"],
        "carrier": carrier,
        "carrier_index": list(ex.carriers).index(carrier),
        "gold_label": gold,
        **parsed,
        "correct": pred == gold,
        "ineligible_marked_eligible": gold == "ineligible" and pred == "eligible",
        "eligible_marked_ineligible": gold == "eligible" and pred == "ineligible",
        "raw_text": text,
        "n_new_tokens": int(n_new),
        "truncated": bool(trunc),
        "max_new_tokens": QUERY_MAX_NEW,
        "latency_s": float(lat),
        "n_kv_positions": n_kv_positions(base.cache_cpu),
        "pos_cursor": int(base.pos_cursor_cpu.view(-1)[0].item()),
        "prompt": prompt,
    }


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs) -> int:
    cpath = out_root / "check_summary.json"
    check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first.")
        return 2
    setup = _setup(runner, examples, artifacts)
    twelve_path = out_root / "evaluations.jsonl"
    query_path = out_root / "query_evaluations.jsonl"
    twelve = _load_rows(twelve_path)
    query = _load_rows(query_path)
    done12 = {(r["example_id"], r["condition"]) for r in twelve}
    doneq = {(r["example_id"], r["condition"], r["carrier"]) for r in query}
    print(
        f"KV-mix12 compare: twelve-line {len(done12)}/{len(examples) * len(CONDS)}, "
        f"single-label {len(doneq)}/{len(examples) * len(CONDS) * N_CARRIERS_12}",
        flush=True,
    )
    for i, ex in enumerate(examples):
        todo12 = [c for c in CONDS if (ex.example_id, c) not in done12]
        todoq = [(c, k) for c in CONDS for k in ex.carriers if (ex.example_id, c, k) not in doneq]
        if not todo12 and not todoq:
            print(f"  [{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        state, built, mix = _state_for(runner, ex, artifacts, setup, recon_refs)
        if not (mix["ok"] and built["matches_recon12"]) and not args.force_compare:
            raise RuntimeError(f"Mixed-cache check failed for {ex.example_id}.")
        snapshots = {c: runner.clone_sender(state["kvmix"][c]) for c in CONDS}
        reset_peak_memory()
        for cond in todo12:
            rec = _twelve_row(runner, ex, state, cond, args)
            twelve.append(rec)
            done12.add((ex.example_id, cond))
            _append(twelve_path, rec)
            print(
                f"    {ex.example_id} {cond} twelve-line: labels={rec.get('plan_n_status_correct')}/12 "
                f"I->E={rec.get('plan_ineligible_marked_eligible')} E->I={rec.get('plan_eligible_marked_ineligible')} "
                f"choice={rec.get('correct')}",
                flush=True,
            )
        _persist(out_root, kaggle_dest)
        gold = gold_eligibility_statuses_n(ex)
        for cond, carrier in todoq:
            rec = _query_row(runner, ex, state, cond, carrier, gold[carrier])
            query.append(rec)
            doneq.add((ex.example_id, cond, carrier))
            _append(query_path, rec)
        for cond in CONDS:
            if not caches_equal(snapshots[cond].cache_cpu, state["kvmix"][cond].cache_cpu):
                raise RuntimeError(f"{cond} cache changed during evaluation for {ex.example_id}.")
        print(f"    {ex.example_id} single-label done ({len(todoq)} answers)", flush=True)
        _persist(out_root, kaggle_dest)
        del state, snapshots
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _summarize(
        twelve,
        query,
        recon_rows=_load_rows(out_root / "recon12_evaluations.jsonl"),
        query12_rows=_load_rows(out_root / "query12_evaluations.jsonl"),
        check=check,
    )
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"twelve_line": summary["twelve_line"]["labels"], "single_label": summary["single_label"]["labels"]}, indent=2))
    return 0


def _twelve_labels(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n_cases": 0}
    return {
        "n_cases": n,
        "n_labels": sum(len(r.get("gold_status") or {}) for r in rows),
        "labels_correct": sum(int(r.get("plan_n_status_correct") or 0) for r in rows),
        "cases_all_twelve_correct": sum(bool(r.get("plan_all_status_correct")) for r in rows),
        "ineligible_marked_eligible": sum(int(r.get("plan_ineligible_marked_eligible") or 0) for r in rows),
        "eligible_marked_ineligible": sum(int(r.get("plan_eligible_marked_ineligible") or 0) for r in rows),
        "parse_ok": sum(bool(r.get("parse_ok")) for r in rows),
        "plan_truncated": sum(bool(r.get("plan_truncated")) for r in rows),
        "answer_truncated": sum(bool(r.get("truncated")) for r in rows),
        "plan_lines_missing": sum(
            sum(v is None for v in (r.get("plan_pred_status") or {}).values()) for r in rows
        ),
    }


def _twelve_choice(rows: list[dict]) -> dict:
    return {
        "n_cases": len(rows),
        "final_choice_correct": sum(bool(r.get("correct")) for r in rows),
        "follows_trap": sum(bool(r.get("follows_trap")) for r in rows),
        "correct_with_wrong_eligibility": sum(bool(r.get("correct_with_wrong_eligibility")) for r in rows),
    }


def _case_metric_twelve(rows: list[dict], field: str) -> dict:
    return {r["example_id"]: int(r.get(field) or 0) for r in rows}


def _case_metric_query(rows: list[dict], field: str) -> dict:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[r["example_id"]] += int(bool(r[field]))
    return out


def _twelve_label_map(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        gold = r.get("gold_status") or {}
        pred = r.get("plan_pred_status") or {}
        for carrier, g in gold.items():
            out[(r["example_id"], carrier)] = (g, pred.get(carrier))
    return out


def _query_label_map(rows: list[dict]) -> dict:
    return {(r["example_id"], r["carrier"]): (r["gold_label"], r["pred_label"]) for r in rows}


def _label_crosstab(left: dict, right: dict) -> dict:
    """Carrier-level change from left to right condition."""
    tab = {"n": 0, "fixed": 0, "broken": 0, "both_correct": 0, "both_wrong": 0, "same_label": 0}
    for key, (gold, lp) in left.items():
        if key not in right:
            continue
        _g, rp = right[key]
        l_ok, r_ok = lp == gold, rp == gold
        tab["n"] += 1
        tab["fixed"] += (not l_ok) and r_ok
        tab["broken"] += l_ok and not r_ok
        tab["both_correct"] += l_ok and r_ok
        tab["both_wrong"] += (not l_ok) and (not r_ok)
        tab["same_label"] += lp == rp
    return tab


def _reproduction(new_rows, ref_rows, *, key_fields, text_fields) -> dict:
    ref = {tuple(r[k] for k in key_fields): r for r in ref_rows}
    n = matched = missing = 0
    for row in new_rows:
        other = ref.get(tuple(row[k] for k in key_fields))
        if other is None:
            missing += 1
            continue
        n += 1
        matched += all((row.get(f) or "") == (other.get(f) or "") for f in text_fields)
    return {"n": n, "identical_outputs": matched, "missing_reference": missing}


def _summarize(twelve, query, *, recon_rows, query12_rows, check=None) -> dict:
    t_by: dict[str, list[dict]] = defaultdict(list)
    q_by: dict[str, list[dict]] = defaultdict(list)
    for r in twelve:
        t_by[r["condition"]].append(r)
    for r in query:
        q_by[r["condition"]].append(r)
    t_maps = {c: _twelve_label_map(t_by[c]) for c in CONDS}
    q_maps = {c: _query_label_map(q_by[c]) for c in CONDS}
    paired_t, paired_q, cross_t, cross_q = {}, {}, {}, {}
    for name, left, right in PAIRED:
        paired_t[name] = {
            metric: _cluster_delta(_case_metric_twelve(t_by[left], field), _case_metric_twelve(t_by[right], field))
            for metric, field in (
                ("labels_correct", "plan_n_status_correct"),
                ("ineligible_marked_eligible", "plan_ineligible_marked_eligible"),
                ("eligible_marked_ineligible", "plan_eligible_marked_ineligible"),
            )
        }
        paired_q[name] = {
            metric: _cluster_delta(_case_metric_query(q_by[left], metric), _case_metric_query(q_by[right], metric))
            for metric in ("correct", "ineligible_marked_eligible", "eligible_marked_ineligible")
        }
        cross_t[name] = _label_crosstab(t_maps[left], t_maps[right])
        cross_q[name] = _label_crosstab(q_maps[left], q_maps[right])
    reproduce = {}
    for cond in CONDS:
        ref_name = SPEC[cond]["reference"]
        if ref_name is None:
            continue
        reproduce[cond] = {
            "reference": ref_name,
            "twelve_line_vs_recon12": _reproduction(
                t_by[cond],
                [r for r in recon_rows if r.get("condition") == ref_name],
                key_fields=("example_id",),
                text_fields=("plan_text", "raw_text"),
            ),
            "single_label_vs_query12": _reproduction(
                q_by[cond],
                [r for r in query12_rows if r.get("condition") == ref_name],
                key_fields=("example_id", "carrier"),
                text_fields=("raw_text",),
            ),
        }
    return {
        "n_twelve_rows": len(twelve),
        "n_twelve_expected": N_TIMING_FAMILIES * len(CONDS),
        "n_query_rows": len(query),
        "n_query_expected": N_TIMING_FAMILIES * len(CONDS) * N_CARRIERS_12,
        "conditions": list(CONDS),
        "spec": SPEC,
        "twelve_line": {
            "labels": {c: _twelve_labels(t_by[c]) for c in CONDS},
            "final_choice": {c: _twelve_choice(t_by[c]) for c in CONDS},
            "receiver_block": {c: _cond_block(t_by[c]) for c in CONDS},
            "by_difficulty": {
                lvl: {c: _twelve_labels([r for r in t_by[c] if r.get("difficulty") == lvl]) for c in CONDS}
                for lvl in LEVELS
            },
            "paired_by_case": paired_t,
            "label_crosstab": cross_t,
        },
        "single_label": {
            "labels": {c: _query_block(q_by[c]) for c in CONDS},
            "paired_by_case": paired_q,
            "label_crosstab": cross_q,
        },
        "reproduction": reproduce,
        "check": None if check is None else {"all_passed": check.get("all_passed"), "n": check.get("n")},
        "frozen_placement": FROZEN_PLACEMENT,
        "stats_note": (
            "Primary comparison is the twelve-line protocol; single-label is secondary. "
            "Bootstrap resamples the nine source cases (2000 draws); labels are not "
            "independent examples. Deltas are right minus left; cases_increased / "
            "cases_decreased give direction only, so for error counts an increase is worse. "
            "Final-choice accuracy is reported separately from label accuracy."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Result-region K/V swap on the twelve-carrier message.")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument("--phase", choices=("check", "compare", "all", "rescore"), default="check")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    parser.add_argument("--artifacts12", default=None)
    args = parser.parse_args()
    if args.split != "plan4_timing12":
        print("KV-mix12 uses only the frozen timing-12 sheets. Do not load plan4_dev or plan4_hold.")
        return 2

    timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_kvmix12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_kvmix12.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    if kaggle_dest is not None:
        for name in PERSIST_NAMES:
            src, dest = kaggle_dest / name, out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    examples = ensure_timing12_jsonl(timing12_path)
    for ex in examples:
        assert_timing12_invariants(ex)
    if len(examples) != N_TIMING_FAMILIES:
        print(f"Expected {N_TIMING_FAMILIES} twelve-carrier sheets, got {len(examples)}.")
        return 2

    if args.phase == "rescore":
        shared = REPO_ROOT / "results_stage_plan4_kvmix12"
        if not (out_root / "evaluations.jsonl").is_file() and (shared / "evaluations.jsonl").is_file():
            out_root = shared
        cpath = out_root / "check_summary.json"
        check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
        summary = _summarize(
            _load_rows(out_root / "evaluations.jsonl"),
            _load_rows(out_root / "query_evaluations.jsonl"),
            recon_rows=_load_rows(out_root / "recon12_evaluations.jsonl"),
            query12_rows=_load_rows(out_root / "query12_evaluations.jsonl"),
            check=check,
        )
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps(summary["twelve_line"]["labels"], indent=2))
        return 0

    inputs = _copy_inputs(out_root, artifacts12=args.artifacts12)
    artifacts = _load_messages(inputs["artifacts"])
    if any(a_state_key(ex) not in artifacts for ex in examples):
        print("Frozen 12-carrier artifacts are missing. Do not regenerate A.")
        return 2
    recon_refs = _recon_refs(inputs["recon_check"])

    policy, runner = _prepare_runner(args, out_root, extra={"phase": args.phase})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    if args.phase == "check":
        return _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs)
    if args.phase == "compare":
        return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs)
    rc = _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs)
    if rc != 0 and not args.force_compare:
        print("Mixed-cache check failed; stopping.")
        return rc
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs)


if __name__ == "__main__":
    raise SystemExit(main())
