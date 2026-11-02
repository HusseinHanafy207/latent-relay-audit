"""Reference-attention replay on the twelve-carrier transferred cache (oracle diagnostic).

Reference: fresh K + fresh V under the same generic support (fk_fv). Its compact
layout matches the transferred cache (tk_tv) exactly: same length, token IDs,
masks and cursor. The full A cache is not used.

Intervention: the transferred cache keeps A's result keys and values. In a
two-row forward [fk_fv, tk_tv], every attention layer gives the tk_tv row the
reference row's attention weights, applied to tk_tv's own values.

First pass is teacher-forced: every branch receives the same saved response
prefix (the reference's twelve-line plan from the K/V swap run) after the same
frozen twelve-line plan prompt. Measured: distance from the reference's output
distributions and label decisions at eligibility-label positions. These are not
free-generation accuracies.

Before interpreting, replaying a branch's own attention must reproduce its
ordinary predictions within the declared tolerance. One case at a time.
Not a deployable method. Do not train an adapter from this stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attn_replay import (
    dist_stats,
    label_decision,
    replay_attention,
    token_index_at_char,
    within_tolerance,
)
from src.cache_utils import cache_to_device, n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import LEVELS, a_state_key
from src.plan4_timing_task import (
    N_TIMING_FAMILIES,
    assert_timing12_invariants,
    ensure_timing12_jsonl,
    gold_eligibility_statuses_n,
)
from src.precision import apply_precision_policy
from src.stage_plan0 import _load_messages
from src.stage_plan4_kvmix12 import _setup, _state_for
from src.stage_plan4_query12 import _cluster_delta, _copy_first, _recon_refs
from src.stage_plan4_timing import B_PLAN_LINE_N, _prompts_for
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

REFERENCE = "fk_fv"
BRANCHES = ("tk_tv", "fk_tv", "fk_fv", "tk_tv_refattn", "tk_tv_selfattn")
BRANCH_SPEC = {
    "tk_tv": "transferred K + transferred V, ordinary attention",
    "fk_tv": "fresh K + transferred V, ordinary attention",
    "fk_fv": "fresh K + fresh V, ordinary attention (reference)",
    "tk_tv_refattn": "transferred K + transferred V, reference attention weights on its own values",
    "tk_tv_selfattn": "transferred K + transferred V, its own attention replayed (numerics control)",
}
TOL_MAX_MEAN_KL = 0.01
TOL_MIN_AGREE = 0.98
PAIRED = (
    ("tk_tv_refattn_vs_tk_tv", "tk_tv", "tk_tv_refattn"),
    ("tk_tv_refattn_vs_tk_tv_selfattn", "tk_tv_selfattn", "tk_tv_refattn"),
    ("tk_tv_refattn_vs_fk_tv", "fk_tv", "tk_tv_refattn"),
    ("tk_tv_refattn_vs_fk_fv", "fk_fv", "tk_tv_refattn"),
    ("fk_tv_vs_tk_tv", "tk_tv", "fk_tv"),
)
PRESERVED = (
    "results_stage_plan4_kvmix12",
    "results_stage_plan4_query12",
    "results_stage_plan4_recon12",
    "results_stage_plan4_timing",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
)
INTERPRETATION_GUIDE = (
    "Teacher-forced on the reference's saved plan; not free-generation accuracy. "
    "If reference attention substantially improves predictions while transferred "
    "values are retained, a small receiver-side attention adjustment or query adapter "
    "is worth investigating. If it gives little improvement, investigate selective "
    "reconstruction of result states; that would not prove every attention adjustment "
    "must fail. If it improves distribution matching but not label decisions, report "
    "the distinction and examine where generation diverges before training anything. "
    "A promising result needs a follow-up where reference and intervention share the "
    "same actual generated history. This is an oracle diagnostic, not a method."
)
PERSIST_NAMES = (
    "evaluations.jsonl",
    "summary.json",
    "check_summary.json",
    "compare_checks.json",
    "manifest.json",
    "a_artifacts_12.jsonl",
    "recon12_check_summary.json",
    "kvmix12_evaluations.jsonl",
)


def _paths():
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_refattn12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_refattn12")
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


def _copy_inputs(out_root: Path) -> dict:
    recon = [REPO_ROOT / "results_stage_plan4_recon12", Path("/kaggle/working/results_stage_plan4_recon12")]
    kvmix = [REPO_ROOT / "results_stage_plan4_kvmix12", Path("/kaggle/working/results_stage_plan4_kvmix12")]
    return {
        "artifacts": _copy_first(
            out_root,
            "a_artifacts_12.jsonl",
            [d / "a_artifacts_12.jsonl" for d in kvmix + recon],
            what="frozen 12-carrier A artifacts",
        ),
        "recon_check": _copy_first(
            out_root, "recon12_check_summary.json", [d / "check_summary.json" for d in recon], what="recon12 check"
        ),
        "kvmix_evals": _copy_first(
            out_root,
            "kvmix12_evaluations.jsonl",
            [d / "evaluations.jsonl" for d in kvmix],
            what="K/V swap twelve-line evaluations (reference plans)",
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
            "stage": "plan4-refattn12",
            "do_not_overwrite": list(PRESERVED),
            "branches": list(BRANCHES),
            "reference": REFERENCE,
            "tolerance": {"max_mean_kl": TOL_MAX_MEAN_KL, "min_argmax_agree": TOL_MIN_AGREE, "labels_equal": True},
            **extra,
        },
    )
    runner = TwoAgentInventory(
        args.model_name, args.device, latent_steps=0, kv_budget=32, sink_size=4, pca_rank=2, max_new_tokens=8
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha"):
        pin_model_revision(revision["sha"])
    return policy, runner


def _reference_plans(path: Path) -> dict:
    return {r["example_id"]: r for r in _load_rows(path) if r.get("condition") == REFERENCE}


def response_labels(plan_text: str, carriers, offsets, ids) -> list[dict]:
    """Label first-token index for each carrier's first plan line."""
    out = []
    seen = set()
    wanted = {str(c).upper() for c in carriers}
    for m in B_PLAN_LINE_N.finditer(plan_text or ""):
        carrier = m.group("carrier").upper()
        if carrier not in wanted or carrier in seen:
            continue
        seen.add(carrier)
        idx = token_index_at_char(offsets, m.start("status"))
        out.append(
            {
                "carrier": carrier,
                "ref_text_label": m.group("status").lower(),
                "token_index": idx,
                "token_id": None if idx is None else int(ids[idx]),
            }
        )
    return out


def label_tokens(labels: list[dict], tokenizer) -> tuple[int, int]:
    ids = {"eligible": set(), "ineligible": set()}
    for lab in labels:
        if lab["token_id"] is not None:
            ids[lab["ref_text_label"]].add(int(lab["token_id"]))
    picked = {}
    for name in ("eligible", "ineligible"):
        if len(ids[name]) == 1:
            picked[name] = next(iter(ids[name]))
        elif not ids[name]:
            picked[name] = int(tokenizer.encode(f" {name}", add_special_tokens=False)[0])
        else:
            raise RuntimeError(f"Label {name!r} starts with several different tokens: {sorted(ids[name])}.")
    if picked["eligible"] == picked["ineligible"]:
        raise RuntimeError("Eligible and ineligible share a first token; the binary decision is undefined.")
    return picked["eligible"], picked["ineligible"]


def _batch_logits(runner, bundles, prompt_ids, resp_ids, *, mode: str | None):
    first = bundles[0]
    ids0 = first.full_ids_cpu.view(-1).tolist()
    cursor = int(first.pos_cursor_cpu.view(-1)[0].item())
    n_past = n_kv_positions(first.cache_cpu)
    for b in bundles[1:]:
        if b.full_ids_cpu.view(-1).tolist() != ids0 or int(b.pos_cursor_cpu.view(-1)[0].item()) != cursor:
            raise RuntimeError("Batched branches must share token IDs and cursor.")
        if n_kv_positions(b.cache_cpu) != n_past:
            raise RuntimeError("Batched branches must share cache length.")
    n_layers = len(first.cache_cpu["layers"])
    layers = []
    for i in range(n_layers):
        k = torch.cat([b.cache_cpu["layers"][i][0] for b in bundles], dim=0)
        v = torch.cat([b.cache_cpu["layers"][i][1] for b in bundles], dim=0)
        layers.append((k, v))
    device = runner.device
    past = cache_to_device({"layers": layers}, device)
    n_b = len(bundles)
    cont = torch.tensor(list(prompt_ids) + list(resp_ids), dtype=torch.long, device=device).view(1, -1)
    cont = cont.expand(n_b, -1).contiguous()
    k_len = int(cont.size(1))
    full_mask = torch.ones((n_b, n_past + k_len), dtype=torch.long, device=device)
    position_ids = (cursor + torch.arange(k_len, device=device, dtype=torch.long)).view(1, -1).expand(n_b, -1)
    keep = len(resp_ids) + 1
    model = runner.wrapper.model
    ctx = replay_attention(model, mode) if mode else nullcontext(None)
    with torch.inference_mode(), ctx as ctl:
        out = model(
            input_ids=cont,
            attention_mask=full_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=False,
            return_dict=True,
            logits_to_keep=keep,
        )
        logits = out.logits[:, : len(resp_ids), :].float().cpu().contiguous()
        info = None
        if ctl is not None:
            info = {"mode": ctl.mode, "n_calls": ctl.n_calls, "n_replaced": ctl.n_replaced, "n_layers": len(ctl.layers_seen)}
    del out, past, cont, full_mask, position_ids, layers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, info


def _position_ids_match(runner, cursor: int, k_len: int) -> bool:
    attn = torch.ones((1, k_len), dtype=torch.long, device=runner.device)
    ref = runner.wrapper._build_position_ids(torch.tensor([cursor], device=runner.device, dtype=torch.long), attn)
    mine = (cursor + torch.arange(k_len, device=runner.device, dtype=torch.long)).view(1, -1)
    return bool(torch.equal(ref.to(mine.dtype).view(1, -1), mine))


def _compare(ref_logits, logits, labels, tok_e, tok_i, gold, resp_ids) -> dict:
    ds = dist_stats(ref_logits, logits)
    resp = torch.tensor(resp_ids, dtype=torch.long).view(-1, 1)
    nll = -ds["logp"].gather(-1, resp).view(-1)
    per = []
    for lab in labels:
        t = lab["token_index"]
        if t is None:
            continue
        dec = label_decision(logits[t], tok_e, tok_i)
        ref_dec = label_decision(ref_logits[t], tok_e, tok_i)
        g = gold[lab["carrier"]]
        other = "ineligible" if g == "eligible" else "eligible"
        per.append(
            {
                "carrier": lab["carrier"],
                "gold": g,
                "decision": dec["decision"],
                "ref_decision": ref_dec["decision"],
                "correct": dec["decision"] == g,
                "agrees_ref": dec["decision"] == ref_dec["decision"],
                "ineligible_marked_eligible": g == "ineligible" and dec["decision"] == "eligible",
                "eligible_marked_ineligible": g == "eligible" and dec["decision"] == "ineligible",
                "gold_margin": dec[f"logp_{g}"] - dec[f"logp_{other}"],
                "label_mass": dec["label_mass"],
                "kl": float(ds["kl"][t].item()),
            }
        )
    n = max(len(per), 1)
    return {
        "n_resp": len(resp_ids),
        "mean_kl": float(ds["kl"].mean().item()),
        "max_kl": float(ds["kl"].max().item()),
        "argmax_agree": float(ds["agree"].float().mean().item()),
        "mean_nll_saved_tokens": float(nll.mean().item()),
        "n_labels": len(per),
        "labels_correct": sum(p["correct"] for p in per),
        "labels_agree_ref": sum(p["agrees_ref"] for p in per),
        "ineligible_marked_eligible": sum(p["ineligible_marked_eligible"] for p in per),
        "eligible_marked_ineligible": sum(p["eligible_marked_ineligible"] for p in per),
        "mean_label_kl": sum(p["kl"] for p in per) / n,
        "mean_gold_margin": sum(p["gold_margin"] for p in per) / n,
        "decisions": [p["decision"] for p in per],
        "per_carrier": per,
    }


def _tol(a_logits, b_logits, labels, tok_e, tok_i, gold, resp_ids) -> dict:
    st = _compare(a_logits, b_logits, labels, tok_e, tok_i, gold, resp_ids)
    a_dec = _compare(a_logits, a_logits, labels, tok_e, tok_i, gold, resp_ids)["decisions"]
    labels_equal = st["decisions"] == a_dec
    return {
        "mean_kl": st["mean_kl"],
        "max_kl": st["max_kl"],
        "argmax_agree": st["argmax_agree"],
        "labels_equal": labels_equal,
        "ok": within_tolerance(st, max_mean_kl=TOL_MAX_MEAN_KL, min_agree=TOL_MIN_AGREE, labels_equal=labels_equal),
    }


def _case(runner, ex, artifacts, setup, recon_refs, ref_rows, *, with_intervention: bool) -> dict:
    state, built, mix = _state_for(runner, ex, artifacts, setup, recon_refs)
    if not (mix["ok"] and built["matches_recon12"]):
        raise RuntimeError(f"Mixed-cache construction failed for {ex.example_id}.")
    caches = state["kvmix"]
    tokenizer = runner.wrapper.tokenizer
    prompt = _prompts_for(ex)["kv_plan"]
    _r, input_ids, _a = runner._encode(prompt)
    prompt_ids = [int(t) for t in input_ids[0].tolist()]
    ref_row = ref_rows[ex.example_id]
    plan_text = ref_row["plan_text"]
    enc = tokenizer(plan_text, add_special_tokens=False, return_offsets_mapping=True)
    resp_ids = [int(t) for t in enc["input_ids"]]
    labels = response_labels(plan_text, ex.carriers, enc["offset_mapping"], resp_ids)
    tok_e, tok_i = label_tokens(labels, tokenizer)
    gold = gold_eligibility_statuses_n(ex)
    cursor = int(caches["tk_tv"].pos_cursor_cpu.view(-1)[0].item())
    n_layers = len(caches["tk_tv"].cache_cpu["layers"])

    ordinary = {}
    for name in ("tk_tv", "fk_tv", "fk_fv"):
        logits, _ = _batch_logits(runner, [caches[name]], prompt_ids, resp_ids, mode=None)
        ordinary[name] = logits[0]
    ref = ordinary[REFERENCE]
    self_tk, info_tk = _batch_logits(runner, [caches["tk_tv"], caches["tk_tv"]], prompt_ids, resp_ids, mode="replace")
    self_ref, info_ref = _batch_logits(runner, [caches["fk_fv"], caches["fk_fv"]], prompt_ids, resp_ids, mode="replace")
    off, info_off = _batch_logits(runner, [caches["fk_fv"], caches["tk_tv"]], prompt_ids, resp_ids, mode="off")
    args_tol = (labels, tok_e, tok_i, gold, resp_ids)
    checks = {
        "position_ids_match_wrapper": _position_ids_match(runner, cursor, len(prompt_ids) + len(resp_ids)),
        "all_labels_located": len(labels) == len(ex.carriers) and all(l["token_index"] is not None for l in labels),
        "tk_tv_self_replay": _tol(ordinary["tk_tv"], self_tk[1], *args_tol),
        "fk_fv_self_replay": _tol(ref, self_ref[1], *args_tol),
        "batched_off_reference": _tol(ref, off[0], *args_tol),
        "batched_off_transferred": _tol(ordinary["tk_tv"], off[1], *args_tol),
        "replace_every_layer": info_tk["n_replaced"] == n_layers and info_ref["n_replaced"] == n_layers
        and info_tk["n_layers"] == n_layers,
        "off_never_replaces": info_off["n_replaced"] == 0 and info_off["n_calls"] == n_layers,
    }
    branches = {
        "tk_tv": ordinary["tk_tv"],
        "fk_tv": ordinary["fk_tv"],
        "fk_fv": ref,
        "tk_tv_selfattn": self_tk[1],
    }
    if with_intervention:
        pair, info_pair = _batch_logits(runner, [caches["fk_fv"], caches["tk_tv"]], prompt_ids, resp_ids, mode="replace")
        checks["donor_row_unchanged"] = _tol(ref, pair[0], *args_tol)
        checks["intervention_replaced_every_layer"] = info_pair["n_replaced"] == n_layers
        branches["tk_tv_refattn"] = pair[1]
    ok = all(v["ok"] if isinstance(v, dict) else bool(v) for v in checks.values())
    rows = []
    if with_intervention:
        for name in BRANCHES:
            st = _compare(ref, branches[name], labels, tok_e, tok_i, gold, resp_ids)
            rows.append(
                {
                    "example_id": ex.example_id,
                    "family_id": ex.family_id,
                    "difficulty": ex.difficulty,
                    "a_state_key": a_state_key(ex),
                    "branch": name,
                    "branch_spec": BRANCH_SPEC[name],
                    "reference": REFERENCE,
                    "prefix_source": f"kvmix12 {REFERENCE} saved plan_text",
                    "reference_free_generation_labels": ref_row.get("plan_n_status_correct"),
                    "n_prompt_tokens": len(prompt_ids),
                    "n_kv_positions": n_kv_positions(caches["tk_tv"].cache_cpu),
                    "pos_cursor": cursor,
                    "label_token_eligible": tok_e,
                    "label_token_ineligible": tok_i,
                    "instrumentation_ok": ok,
                    **st,
                }
            )
    del state, caches, branches, ordinary
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"check": {"example_id": ex.example_id, "ok": ok, "checks": checks}, "rows": rows}


def _run(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, ref_rows, *, with_intervention) -> int:
    if with_intervention:
        cpath = out_root / "check_summary.json"
        check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
        if (check is None or not check.get("all_passed")) and not args.force_compare:
            print("Run --phase check first; instrumentation must reproduce ordinary predictions.")
            return 2
    setup = _setup(runner, examples, artifacts)
    eval_path = out_root / "evaluations.jsonl"
    rows = _load_rows(eval_path) if with_intervention else []
    done = {r["example_id"] for r in rows}
    checks = []
    for i, ex in enumerate(examples):
        if with_intervention and ex.example_id in done:
            print(f"  [{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        res = _case(runner, ex, artifacts, setup, recon_refs, ref_rows, with_intervention=with_intervention)
        checks.append(res["check"])
        for row in res["rows"]:
            rows.append(row)
            with eval_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        brief = {
            k: (round(v["mean_kl"], 5) if isinstance(v, dict) else v) for k, v in res["check"]["checks"].items()
        }
        print(f"  [{i + 1}/{len(examples)}] {ex.example_id} instrumentation_ok={res['check']['ok']} {brief}", flush=True)
        for row in res["rows"]:
            print(
                f"    {row['branch']}: labels={row['labels_correct']}/{row['n_labels']} "
                f"agree_ref={row['labels_agree_ref']} meanKL={row['mean_kl']:.4f}",
                flush=True,
            )
        _persist(out_root, kaggle_dest)
    name = "compare_checks.json" if with_intervention else "check_summary.json"
    all_passed = bool(checks) and all(c["ok"] for c in checks)
    _write_json(
        out_root / name,
        {
            "n": len(checks),
            "all_passed": all_passed,
            "tolerance": {"max_mean_kl": TOL_MAX_MEAN_KL, "min_argmax_agree": TOL_MIN_AGREE, "labels_equal": True},
            "rows": checks,
        },
    )
    if with_intervention:
        cpath = out_root / "check_summary.json"
        check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
        summary = _summarize(rows, check=check, compare_checks=checks)
        _write_json(out_root / "summary.json", summary)
        print(json.dumps(summary["branches"], indent=2)[:6000])
    _persist(out_root, kaggle_dest)
    print(json.dumps({"n": len(checks), "all_passed": all_passed}, indent=2))
    return 0 if all_passed else 1


def _branch_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n_cases": 0}
    return {
        "n_cases": n,
        "n_labels": sum(r["n_labels"] for r in items),
        "labels_correct": sum(r["labels_correct"] for r in items),
        "labels_agree_ref": sum(r["labels_agree_ref"] for r in items),
        "ineligible_marked_eligible": sum(r["ineligible_marked_eligible"] for r in items),
        "eligible_marked_ineligible": sum(r["eligible_marked_ineligible"] for r in items),
        "cases_all_labels_correct": sum(r["labels_correct"] == r["n_labels"] for r in items),
        "mean_kl_vs_reference": sum(r["mean_kl"] for r in items) / n,
        "mean_label_kl_vs_reference": sum(r["mean_label_kl"] for r in items) / n,
        "mean_argmax_agree_reference": sum(r["argmax_agree"] for r in items) / n,
        "mean_nll_saved_tokens": sum(r["mean_nll_saved_tokens"] for r in items) / n,
        "mean_gold_margin": sum(r["mean_gold_margin"] for r in items) / n,
        "all_instrumentation_ok": all(bool(r.get("instrumentation_ok")) for r in items),
    }


def _per_case(items: list[dict], field: str) -> dict:
    return {r["example_id"]: r[field] for r in items}


def _recovery(blocks: dict) -> dict:
    lo, hi, mid = blocks.get("tk_tv") or {}, blocks.get("fk_fv") or {}, blocks.get("tk_tv_refattn") or {}
    if not lo.get("n_cases") or not mid.get("n_cases"):
        return {}
    out = {}
    gap = hi["labels_correct"] - lo["labels_correct"]
    out["label_gap_closed"] = None if gap == 0 else (mid["labels_correct"] - lo["labels_correct"]) / gap
    kl0 = lo["mean_kl_vs_reference"]
    out["kl_reduction"] = None if kl0 == 0 else 1.0 - mid["mean_kl_vs_reference"] / kl0
    fk = blocks.get("fk_tv") or {}
    if fk.get("n_cases"):
        out["fk_tv_label_gap_closed"] = None if gap == 0 else (fk["labels_correct"] - lo["labels_correct"]) / gap
        out["fk_tv_kl_reduction"] = None if kl0 == 0 else 1.0 - fk["mean_kl_vs_reference"] / kl0
    return out


def _summarize(rows: list[dict], *, check: dict | None = None, compare_checks: list | None = None) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["branch"]].append(r)
    blocks = {b: _branch_block(by[b]) for b in BRANCHES}
    paired = {}
    for name, left, right in PAIRED:
        paired[name] = {
            "labels_correct": _cluster_delta(_per_case(by[left], "labels_correct"), _per_case(by[right], "labels_correct")),
            "ineligible_marked_eligible": _cluster_delta(
                _per_case(by[left], "ineligible_marked_eligible"), _per_case(by[right], "ineligible_marked_eligible")
            ),
            "eligible_marked_ineligible": _cluster_delta(
                _per_case(by[left], "eligible_marked_ineligible"), _per_case(by[right], "eligible_marked_ineligible")
            ),
            "mean_kl": _cluster_delta(_per_case(by[left], "mean_kl"), _per_case(by[right], "mean_kl")),
        }
    instrumentation_ok = bool(check and check.get("all_passed")) and all(
        bool(c.get("ok")) for c in (compare_checks or [])
    )
    return {
        "n_rows": len(rows),
        "n_cases": len({r["example_id"] for r in rows}),
        "n_expected_cases": N_TIMING_FAMILIES,
        "reference": REFERENCE,
        "branch_spec": BRANCH_SPEC,
        "instrumentation_ok": instrumentation_ok,
        "branches": blocks,
        "by_difficulty": {
            lvl: {b: _branch_block([r for r in by[b] if r.get("difficulty") == lvl]) for b in BRANCHES} for lvl in LEVELS
        },
        "recovery_toward_reference": _recovery(blocks),
        "paired_by_case": paired,
        "measurement_note": (
            "Teacher-forced: every branch receives the frozen twelve-line plan prompt and the "
            "reference's saved plan text, re-tokenized identically for all branches. Label "
            "decisions compare the two label first tokens at each label position. These are "
            "not free-generation accuracies. The full-A cache is not used."
        ),
        "stats_note": (
            "Bootstrap resamples the nine source cases (2000 draws). Deltas are right minus left; "
            "cases_increased / cases_decreased give direction only (for errors and KL an increase is worse)."
        ),
        "tolerance": {"max_mean_kl": TOL_MAX_MEAN_KL, "min_argmax_agree": TOL_MIN_AGREE, "labels_equal": True},
        "interpretation_guide": INTERPRETATION_GUIDE,
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Reference-attention replay on the twelve-carrier transferred cache.")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument("--phase", choices=("check", "compare", "all", "rescore"), default="check")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_timing12":
        print("Refattn12 uses only the frozen timing-12 sheets. Do not load plan4_dev or plan4_hold.")
        return 2

    timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_refattn12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_refattn12.")
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
        shared = REPO_ROOT / "results_stage_plan4_refattn12"
        if not (out_root / "evaluations.jsonl").is_file() and (shared / "evaluations.jsonl").is_file():
            out_root = shared
        cpath = out_root / "check_summary.json"
        check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
        cc = out_root / "compare_checks.json"
        compare_checks = json.loads(cc.read_text(encoding="utf-8")).get("rows") if cc.is_file() else []
        summary = _summarize(_load_rows(out_root / "evaluations.jsonl"), check=check, compare_checks=compare_checks)
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps(summary["branches"], indent=2))
        return 0

    inputs = _copy_inputs(out_root)
    artifacts = _load_messages(inputs["artifacts"])
    if any(a_state_key(ex) not in artifacts for ex in examples):
        print("Frozen 12-carrier artifacts are missing. Do not regenerate A.")
        return 2
    recon_refs = _recon_refs(inputs["recon_check"])
    ref_rows = _reference_plans(inputs["kvmix_evals"])
    if any(ex.example_id not in ref_rows for ex in examples):
        print("Missing saved reference plans from the K/V swap run.")
        return 2

    policy, runner = _prepare_runner(args, out_root, extra={"phase": args.phase})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    common = (args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, ref_rows)
    if args.phase == "check":
        return _run(*common, with_intervention=False)
    if args.phase == "compare":
        return _run(*common, with_intervention=True)
    rc = _run(*common, with_intervention=False)
    if rc != 0 and not args.force_compare:
        print("Instrumentation check failed; stopping.")
        return rc
    return _run(*common, with_intervention=True)


if __name__ == "__main__":
    raise SystemExit(main())
