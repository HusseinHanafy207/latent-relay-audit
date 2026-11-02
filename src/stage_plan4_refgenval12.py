"""Result-value edits under the reference-attention oracle, in free generation.

Extends the free-generation oracle run (refgen12) with the two value edits of the
teacher-forced value check (refval12). Same nine cases, frozen construction,
prompts, generic support, positions, greedy decoding and allowances
(384 plan / 48 answer tokens).

Branches:
- tk_tv, fk_fv: ordinary generation (reproduce refgen12)
- tk_tv_refattn: lockstep [fk_fv, tk_tv] with reference attention (reproduces refgen12)
- tk_tv_refattn_vzero: same, target result-region values set to zero
- tk_tv_refattn_vperm: same, target result-region value rows permuted with the
  derangement saved by refval12 (same for every layer and head)

Edits are applied to the target cache before B's prompt processing and are used
in both the plan and the choice pass. Tokens come from the intervention row and
are fed to both rows; no saved reference response is used. Each branch's choice
pass uses its own plan. Artificial interventions; not a method.
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

from src.attn_replay import label_decision
from src.cache_utils import n_kv_positions
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
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import FRAMING_INSTRUCTION, prompt_contains_a_private
from src.span_select import caches_equal
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_boundary import PAYLOAD_NOTE
from src.stage_plan4_kvmix12 import (
    _label_crosstab,
    _reproduction,
    _setup,
    _state_for,
    _twelve_choice,
    _twelve_label_map,
    _twelve_labels,
    _twelve_row,
)
from src.stage_plan4_kvswap import LAYOUT
from src.stage_plan4_query12 import _cluster_delta, _copy_first, _recon_refs
from src.stage_plan4_refattn12 import TOL_MAX_MEAN_KL, TOL_MIN_AGREE, _batch_logits
from src.stage_plan4_refgen12 import (
    DONOR,
    TARGET,
    _divergence_block,
    divergence,
    first_text_difference,
    history_labels,
    lockstep_generate,
    path_tolerance,
)
from src.stage_plan4_refval12 import _edited_bundle, check_value_edit, fixed_derangement
from src.stage_plan4_timing import B_PLAN_MAX_NEW_12, _prompts_for, _row_n
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

REFERENCE = "fk_fv"
ORACLE = "tk_tv_refattn"
ORDINARY = ("tk_tv", "fk_fv")
LOCKSTEP = {ORACLE: None, "tk_tv_refattn_vzero": "zero", "tk_tv_refattn_vperm": "perm"}
EDITED = tuple(b for b, kind in LOCKSTEP.items() if kind)
BRANCHES = ORDINARY + tuple(LOCKSTEP)
BRANCH_SPEC = {
    "tk_tv": "transferred K + V, ordinary generation",
    "fk_fv": "fresh K + V, ordinary generation (reference)",
    "tk_tv_refattn": "lockstep [fk_fv, tk_tv]; reference attention on intact transferred values",
    "tk_tv_refattn_vzero": "lockstep [fk_fv, tk_tv]; reference attention; target result values zeroed",
    "tk_tv_refattn_vperm": "lockstep [fk_fv, tk_tv]; reference attention; target result value rows permuted (saved)",
}
PAIRED = (
    ("vzero_vs_oracle", ORACLE, "tk_tv_refattn_vzero"),
    ("vperm_vs_oracle", ORACLE, "tk_tv_refattn_vperm"),
    ("vzero_vs_tk_tv", "tk_tv", "tk_tv_refattn_vzero"),
    ("vperm_vs_tk_tv", "tk_tv", "tk_tv_refattn_vperm"),
    ("oracle_vs_tk_tv", "tk_tv", ORACLE),
    ("fk_fv_vs_oracle", ORACLE, REFERENCE),
)
PRESERVED = (
    "results_stage_plan4_refval12",
    "results_stage_plan4_refgen12",
    "results_stage_plan4_refattn12",
    "results_stage_plan4_kvmix12",
    "results_stage_plan4_query12",
    "results_stage_plan4_recon12",
    "results_stage_plan4_timing",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
)
INTERPRETATION_GUIDE = (
    "Free generation with each branch's own history. If recovery stays high with zeroed "
    "values, the oracle can drive successful output without intact result values even "
    "without a supplied answer history, which substantially limits it as evidence of "
    "cache-content recovery. If recovery drops substantially, the supplied reference "
    "history concealed value dependence in the teacher-forced check. If generation becomes "
    "malformed (parsing, missing lines, truncation), the binary label score missed an "
    "effect on unrestricted output. Report whichever occurs and close this diagnostic; "
    "do not search permutations or prompts for a preferred outcome. Keep teacher-forced "
    "figures (refval12) separate from these free-generation figures."
)
PERSIST_NAMES = (
    "evaluations.jsonl",
    "case_checks.jsonl",
    "summary.json",
    "manifest.json",
    "a_artifacts_12.jsonl",
    "recon12_check_summary.json",
    "refgen12_evaluations.jsonl",
    "refval12_case_checks.jsonl",
    "refval12_summary.json",
)


def _paths():
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_refgenval12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_refgenval12")
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


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_inputs(out_root: Path) -> dict:
    def dirs(name):
        return [REPO_ROOT / f"results_stage_plan4_{name}", Path(f"/kaggle/working/results_stage_plan4_{name}")]

    refval, refgen, refattn, recon = dirs("refval12"), dirs("refgen12"), dirs("refattn12"), dirs("recon12")
    return {
        "artifacts": _copy_first(
            out_root,
            "a_artifacts_12.jsonl",
            [d / "a_artifacts_12.jsonl" for d in refval + refgen + refattn],
            what="frozen 12-carrier A artifacts",
        ),
        "recon_check": _copy_first(
            out_root,
            "recon12_check_summary.json",
            [d / "recon12_check_summary.json" for d in refval + refgen + refattn] + [d / "check_summary.json" for d in recon],
            what="recon12 check",
        ),
        "refgen_evals": _copy_first(
            out_root,
            "refgen12_evaluations.jsonl",
            [d / "evaluations.jsonl" for d in refgen],
            what="free-generation oracle evaluations (intact branches to reproduce)",
        ),
        "refval_checks": _copy_first(
            out_root,
            "refval12_case_checks.jsonl",
            [d / "case_checks.jsonl" for d in refval],
            what="value-check case checks (saved permutations)",
        ),
        "refval_summary": _copy_first(
            out_root, "refval12_summary.json", [d / "summary.json" for d in refval], what="value-check summary"
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
            "stage": "plan4-refgenval12",
            "do_not_overwrite": list(PRESERVED),
            "branches": list(BRANCHES),
            "reference": REFERENCE,
            "edits": {b: k for b, k in LOCKSTEP.items() if k},
            "permutation_source": "refval12 case_checks.jsonl (perm per case)",
            "plan_max_new_12": args.plan_max_new_12,
            "answer_max_new": args.max_new_tokens,
            "tolerance": {"max_mean_kl": TOL_MAX_MEAN_KL, "min_argmax_agree": TOL_MIN_AGREE, "labels_equal": True},
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


def result_lines_in_prompt(prompt: str, result_text: str, own_plan: str = "") -> list[str]:
    """A's result lines found in B's prompt, ignoring B's own generated plan."""
    body = prompt.replace(own_plan, "") if own_plan else prompt
    lines = [line.strip() for line in (result_text or "").splitlines() if len(line.strip()) >= 8]
    return [line for line in lines if line in body]


def saved_permutation(refval_checks: dict, example_id: str, n_result: int) -> tuple[list[int], dict]:
    row = refval_checks.get(example_id)
    if row is None:
        raise RuntimeError(f"No saved permutation for {example_id}.")
    perm = [int(p) for p in row["perm"]]
    check = {
        "n_result_matches_saved": int(row["n_result"]) == int(n_result),
        "is_permutation": sorted(perm) == list(range(int(n_result))),
        "equals_seeded_derangement": perm == fixed_derangement(int(n_result)),
    }
    check["ok"] = all(check.values())
    return perm, check


def label_confidence(target_logits, hist: dict) -> dict:
    """Probability mass on the two label tokens at the target's own label positions."""
    if hist["pair"] is None:
        return {"n": 0, "method": "unavailable"}
    masses = [
        label_decision(target_logits[lab["token_index"]], *hist["pair"])["label_mass"]
        for lab in hist["labels"]
        if lab["token_index"] is not None
    ]
    if not masses:
        return {"n": 0, "method": "binary"}
    return {
        "n": len(masses),
        "method": "binary",
        "mean_label_mass": sum(masses) / len(masses),
        "min_label_mass": min(masses),
        "n_below_0_5": sum(m < 0.5 for m in masses),
    }


def _lockstep_row(runner, ex, donor, target, branch, args, result_text) -> tuple[dict, dict]:
    tokenizer = runner.wrapper.tokenizer
    bundles = [donor, target]
    prompts = _prompts_for(ex)
    plan_prompt = prompts["kv_plan"]
    if prompt_contains_a_private(plan_prompt, ex):
        raise RuntimeError("KV B leaked A's raw sheet.")
    _r, plan_ids_t, _a = runner._encode(plan_prompt)
    plan_prompt_ids = [int(t) for t in plan_ids_t[0].tolist()]
    plan = lockstep_generate(runner, bundles, plan_prompt_ids, max_new_tokens=args.plan_max_new_12, mode="replace")
    prompt = prompts["kv_answer"](ex, plan["text"])
    if prompt_contains_a_private(prompt, ex):
        raise RuntimeError("KV answer pass leaked A's raw sheet.")
    _r, ans_ids_t, _a = runner._encode(prompt)
    ans_prompt_ids = [int(t) for t in ans_ids_t[0].tolist()]
    ans = lockstep_generate(runner, bundles, ans_prompt_ids, max_new_tokens=args.max_new_tokens, mode="replace")

    gold = gold_eligibility_statuses_n(ex)
    checks = {
        "no_result_text_in_plan_prompt": not result_lines_in_prompt(plan_prompt, result_text),
        "no_result_text_in_answer_prompt": not result_lines_in_prompt(prompt, result_text, own_plan=plan["text"]),
    }
    diag = {}
    for pass_name, res, p_ids in (("plan", plan, plan_prompt_ids), ("answer", ans, ans_prompt_ids)):
        hist = history_labels(tokenizer, res["ids"], ex.carriers)
        ordinary, _ = _batch_logits(runner, [donor], p_ids, res["ids"], mode=None)
        checks[f"{pass_name}_donor_vs_ordinary_{REFERENCE}"] = path_tolerance(ordinary[0], res["logits"][DONOR].float(), hist)
        full, _ = _batch_logits(runner, bundles, p_ids, res["ids"], mode="replace")
        checks[f"{pass_name}_target_vs_full_sequence_replay"] = path_tolerance(full[1], res["logits"][TARGET].float(), hist)
        checks[f"{pass_name}_instrumentation"] = {
            **res["info"],
            "ok": bool(res["info"]["every_forward_every_layer"] and res["info"].get("replaced_every_call", True)),
        }
        if pass_name == "plan":
            diag["divergence"] = divergence(
                tokenizer, res["ids"], res["logits"][DONOR].float(), res["logits"][TARGET].float(), hist, gold
            )
            diag["plan_label_confidence"] = label_confidence(res["logits"][TARGET].float(), hist)
            diag["plan_labels_located"] = len(hist["labels"])
        del ordinary, full
    extra = {
        "channel": "kv",
        "message_text": "",
        "message_bytes": int(target.full_relay_bytes),
        "n_kv_positions": plan["n_past"],
        "plan_time_s": plan["latency_s"],
        "n_plan_tokens": plan["n_new"],
        "plan_truncated": plan["truncated"],
        "plan_text": plan["text"],
        "plan_max_new": args.plan_max_new_12,
        "truncated": ans["truncated"],
        "n_new_tokens": ans["n_new"],
        "receiver_latency_s": ans["latency_s"],
        "max_new_tokens": args.max_new_tokens,
        "peak_gpu_gb": peak_gpu_memory_gb(),
    }
    rec = _row_n(ex, prompt=prompt, text=ans["text"], extra=extra)
    rec.update(
        {
            "condition": branch,
            "decoding": "lockstep",
            "lockstep_rows": [REFERENCE, "tk_tv"],
            "value_edit": LOCKSTEP[branch],
            "replay_mode": "replace",
            "token_chooser": "target row (intervention)",
            "plan_ids": plan["ids"],
            "answer_ids": ans["ids"],
            "pos_cursor": plan["pos_cursor"],
            "layout": LAYOUT,
            "framing": "instruct",
            "framing_instruction": FRAMING_INSTRUCTION,
            "payload_note": PAYLOAD_NOTE,
            **diag,
        }
    )
    del plan, ans
    return rec, checks


def _case(runner, ex, artifacts, setup, recon_refs, refval_checks, refgen_rows, args) -> dict:
    state, built, mix = _state_for(runner, ex, artifacts, setup, recon_refs)
    if not (mix["ok"] and built["matches_recon12"]):
        raise RuntimeError(f"Mixed-cache construction failed for {ex.example_id}.")
    caches = state["kvmix"]
    tk, ref_cache = caches["tk_tv"], caches[REFERENCE]
    snapshots = {c: runner.clone_sender(caches[c]) for c in ORDINARY}
    result = [int(p) for p in state["slots_extra"]["result"]]
    perm, perm_check = saved_permutation(refval_checks, ex.example_id, len(result))
    targets = {ORACLE: tk}
    construction = {}
    for branch in EDITED:
        kind = LOCKSTEP[branch]
        targets[branch] = _edited_bundle(runner, tk, result, kind, perm)
        construction[branch] = check_value_edit(tk.cache_cpu, targets[branch].cache_cpu, result, kind, perm)
    result_text = (state.get("instr") or {}).get("result_text") or ""
    reset_peak_memory()

    rows = {}
    for cond in ORDINARY:
        rec = _twelve_row(runner, ex, state, cond, args)
        rec["decoding"] = "ordinary"
        rows[cond] = rec
    checks = {
        "permutation_matches_refval12": perm_check,
        "construction": construction,
        "result_text_available": bool(result_text.strip()),
    }
    for branch, target in targets.items():
        rec, ch = _lockstep_row(runner, ex, ref_cache, target, branch, args, result_text)
        rows[branch] = rec
        for k, v in ch.items():
            checks[f"{branch}:{k}"] = v
    checks["caches_unchanged"] = all(caches_equal(snapshots[c].cache_cpu, caches[c].cache_cpu) for c in ORDINARY)
    saved = refgen_rows.get(ex.example_id) or {}
    checks["intact_branches_reproduce_refgen12"] = {
        b: bool(saved.get(b))
        and rows[b]["plan_text"] == saved[b].get("plan_text")
        and rows[b]["raw_text"] == saved[b].get("raw_text")
        for b in (*ORDINARY, ORACLE)
    }
    ok = True
    for v in checks.values():
        if isinstance(v, dict) and "ok" in v:
            ok &= bool(v["ok"])
        elif isinstance(v, dict):
            ok &= all((x["ok"] if isinstance(x, dict) else bool(x)) for x in v.values())
        else:
            ok &= bool(v)
    oracle_rec = rows[ORACLE]
    comparisons = {}
    for branch in EDITED:
        rec = rows[branch]
        comparisons[branch] = {
            "plan_identical_to_oracle": rec["plan_text"] == oracle_rec["plan_text"],
            "answer_identical_to_oracle": rec["raw_text"] == oracle_rec["raw_text"],
            "labels_equal_oracle": rec["plan_pred_status"] == oracle_rec["plan_pred_status"],
            "first_text_difference_vs_oracle": first_text_difference(oracle_rec["plan_text"], rec["plan_text"]),
        }
    for rec in rows.values():
        rec["instrumentation_ok"] = ok
    check_row = {
        "example_id": ex.example_id,
        "a_state_key": a_state_key(ex),
        "ok": ok,
        "n_result": len(result),
        "checks": checks,
        "comparisons": comparisons,
        "divergence": {b: rows[b].get("divergence") for b in LOCKSTEP},
    }
    del state, caches, snapshots, targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": [rows[b] for b in BRANCHES], "check": check_row}


def _print_case(i, n, ex, res) -> None:
    c = res["check"]
    print(f"  [{i + 1}/{n}] {ex.example_id} instrumentation_ok={c['ok']} n_result={c['n_result']}", flush=True)
    for k, v in c["checks"].items():
        if isinstance(v, dict) and "mean_kl" in v:
            print(f"    {k}: ok={v['ok']} meanKL={v['mean_kl']:.5f} agree={v['argmax_agree']:.4f} labels_equal={v['labels_equal']}")
        elif isinstance(v, dict) and "ok" in v:
            print(f"    {k}: ok={v['ok']}")
        elif isinstance(v, dict):
            print(f"    {k}:", {a: (b.get("ok") if isinstance(b, dict) else b) for a, b in v.items()})
        else:
            print(f"    {k}: {v}")
    for row in res["rows"]:
        conf = row.get("plan_label_confidence") or {}
        print(
            f"    {row['condition']}: labels={row.get('plan_n_status_correct')}/12 "
            f"I->E={row.get('plan_ineligible_marked_eligible')} E->I={row.get('plan_eligible_marked_ineligible')} "
            f"parse={row.get('parse_ok')} plan_trunc={row.get('plan_truncated')} choice={row.get('correct')}"
            + (f" min_label_mass={conf.get('min_label_mass'):.3f}" if conf.get("n") else ""),
            flush=True,
        )


def _run(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, refval_checks, refgen_rows, inputs) -> int:
    setup = _setup(runner, examples, artifacts)
    eval_path = out_root / "evaluations.jsonl"
    check_path = out_root / "case_checks.jsonl"
    checks = _load_rows(check_path)
    done = {c["example_id"] for c in checks}
    saved = _load_rows(eval_path)
    rows = [r for r in saved if r["example_id"] in done]
    if len(rows) != len(saved):
        eval_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    todo = [ex for ex in examples if ex.example_id not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Refgenval12: {len(done)}/{len(examples)} cases complete; running {len(todo)}", flush=True)
    rc = 0
    for i, ex in enumerate(todo):
        res = _case(runner, ex, artifacts, setup, recon_refs, refval_checks, refgen_rows, args)
        for row in res["rows"]:
            rows.append(row)
            _append(eval_path, row)
        checks.append(res["check"])
        _append(check_path, res["check"])
        _print_case(i, len(todo), ex, res)
        _persist(out_root, kaggle_dest)
        if not res["check"]["ok"] and not args.force:
            print("Check failed for this case; stopping. Inspect case_checks.jsonl.")
            rc = 1
            break
    summary = _summarize(rows, checks, refval_summary=_load_json(inputs["refval_summary"]))
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary["labels"], indent=2))
    return rc


def _load_json(path) -> dict | None:
    p = Path(path) if path else None
    return json.loads(p.read_text(encoding="utf-8")) if p is not None and p.is_file() else None


def _case_metric(rows: list[dict], field: str) -> dict:
    return {r["example_id"]: int(r.get(field) or 0) for r in rows}


def _malformed(rows: list[dict]) -> dict:
    return {
        "n_cases": len(rows),
        "parse_failed": sum(not r.get("parse_ok") for r in rows),
        "think_collapse": sum(bool(r.get("think_collapse")) for r in rows),
        "plan_truncated": sum(bool(r.get("plan_truncated")) for r in rows),
        "answer_truncated": sum(bool(r.get("truncated")) for r in rows),
        "cases_with_missing_lines": sum(
            any(v is None for v in (r.get("plan_pred_status") or {}).values()) for r in rows
        ),
        "mean_plan_tokens": (sum(int(r.get("n_plan_tokens") or 0) for r in rows) / len(rows)) if rows else None,
    }


def _confidence(rows: list[dict]) -> dict:
    confs = [r.get("plan_label_confidence") or {} for r in rows]
    confs = [c for c in confs if c.get("n")]
    if not confs:
        return {"n_labels": 0}
    n = sum(c["n"] for c in confs)
    return {
        "n_labels": n,
        "mean_label_mass": sum(c["mean_label_mass"] * c["n"] for c in confs) / n,
        "min_label_mass": min(c["min_label_mass"] for c in confs),
        "labels_below_0_5": sum(c["n_below_0_5"] for c in confs),
    }


def _retained(labels: dict) -> dict:
    lo = (labels.get("tk_tv") or {}).get("labels_correct")
    hi = (labels.get(ORACLE) or {}).get("labels_correct")
    if lo is None or hi is None or hi == lo:
        return {}
    return {
        b: {
            "additional_correct_vs_tk_tv": labels[b]["labels_correct"] - lo,
            "oracle_additional_correct": hi - lo,
            "label_recovery_retained": (labels[b]["labels_correct"] - lo) / (hi - lo),
        }
        for b in EDITED
        if (labels.get(b) or {}).get("n_cases")
    }


def _teacher_forced_context(refval_summary: dict | None) -> dict:
    blocks = (refval_summary or {}).get("branches") or {}
    return {
        "teacher_forced_refval12": {
            b: {k: blocks[b].get(k) for k in ("labels_correct", "n_labels", "mean_kl_vs_reference")} for b in blocks
        },
        "note": (
            "Teacher-forced label decisions on the reference's saved plan (refval12) are not "
            "free-generation accuracies. Compare them with this run only as a change of setting."
        ),
    }


def _summarize(rows: list[dict], checks: list[dict], *, refval_summary: dict | None) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    maps = {b: _twelve_label_map(by[b]) for b in BRANCHES}
    labels = {b: _twelve_labels(by[b]) for b in BRANCHES}
    paired, cross = {}, {}
    for name, left, right in PAIRED:
        paired[name] = {
            metric: _cluster_delta(_case_metric(by[left], field), _case_metric(by[right], field))
            for metric, field in (
                ("labels_correct", "plan_n_status_correct"),
                ("ineligible_marked_eligible", "plan_ineligible_marked_eligible"),
                ("eligible_marked_ineligible", "plan_eligible_marked_ineligible"),
            )
        }
        cross[name] = _label_crosstab(maps[left], maps[right])
    comp = [c.get("comparisons") or {} for c in checks]
    repro = [((c.get("checks") or {}).get("intact_branches_reproduce_refgen12") or {}) for c in checks]
    return {
        "n_rows": len(rows),
        "n_cases": len({r["example_id"] for r in rows}),
        "n_expected_cases": N_TIMING_FAMILIES,
        "reference": REFERENCE,
        "branch_spec": BRANCH_SPEC,
        "instrumentation_ok": bool(checks) and all(bool(c.get("ok")) for c in checks),
        "n_cases_instrumentation_ok": sum(bool(c.get("ok")) for c in checks),
        "intact_reproduce_refgen12": {b: sum(bool(r.get(b)) for r in repro) for b in (*ORDINARY, ORACLE)},
        "labels": labels,
        "final_choice": {b: _twelve_choice(by[b]) for b in BRANCHES},
        "malformed_output": {b: _malformed(by[b]) for b in BRANCHES},
        "plan_label_confidence": {b: _confidence(by[b]) for b in LOCKSTEP},
        "by_difficulty": {
            lvl: {b: _twelve_labels([r for r in by[b] if r.get("difficulty") == lvl]) for b in BRANCHES} for lvl in LEVELS
        },
        "recovery_retained": _retained(labels),
        "edited_vs_oracle_text": {
            b: {
                "n_cases": len(comp),
                "plan_identical": sum(bool((c.get(b) or {}).get("plan_identical_to_oracle")) for c in comp),
                "answer_identical": sum(bool((c.get(b) or {}).get("answer_identical_to_oracle")) for c in comp),
                "labels_equal": sum(bool((c.get(b) or {}).get("labels_equal_oracle")) for c in comp),
            }
            for b in EDITED
        },
        "divergence_from_reference_row": {
            b: _divergence_block([{"example_id": c["example_id"], "divergence": (c.get("divergence") or {}).get(b)} for c in checks])
            for b in LOCKSTEP
        },
        "paired_by_case": paired,
        "label_crosstab": cross,
        "context": _teacher_forced_context(refval_summary),
        "measurement_note": (
            "Free generation. Labels come from each branch's own twelve-line plan; the final "
            "choice from each branch's own answer pass on its own plan. Lockstep branches choose "
            "tokens from the intervention row; the reference row follows that history and never "
            "supplies tokens. Value edits apply to the target cache in both passes. Final-choice "
            "accuracy is reported separately."
        ),
        "stats_note": (
            "Bootstrap resamples the nine source cases (2000 draws); labels are not independent. "
            "Deltas are right minus left; cases_increased / cases_decreased give direction only "
            "(for error counts an increase is worse)."
        ),
        "tolerance": {"max_mean_kl": TOL_MAX_MEAN_KL, "min_argmax_agree": TOL_MIN_AGREE, "labels_equal": True},
        "interpretation_guide": INTERPRETATION_GUIDE,
    }


def _refgen_rows(path: Path) -> dict:
    out: dict[str, dict] = defaultdict(dict)
    for r in _load_rows(path):
        out[r["example_id"]][r["condition"]] = r
    return dict(out)


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Result-value edits under the oracle, in free generation (twelve carriers).")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument("--phase", choices=("run", "rescore"), default="run")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many remaining cases.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force", action="store_true", help="Continue after a failed check.")
    args = parser.parse_args()
    if args.split != "plan4_timing12":
        print("Refgenval12 uses only the frozen timing-12 sheets. Do not load plan4_dev or plan4_hold.")
        return 2
    if args.plan_max_new_12 != B_PLAN_MAX_NEW_12 or args.max_new_tokens != B_ANSWER_MAX_NEW:
        print("Keep the output allowances of the earlier runs (384 plan / 48 answer tokens).")
        return 2

    timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_refgenval12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_refgenval12.")
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
        shared = REPO_ROOT / "results_stage_plan4_refgenval12"
        if not (out_root / "evaluations.jsonl").is_file() and (shared / "evaluations.jsonl").is_file():
            out_root = shared
        summary = _summarize(
            _load_rows(out_root / "evaluations.jsonl"),
            _load_rows(out_root / "case_checks.jsonl"),
            refval_summary=_load_json(out_root / "refval12_summary.json"),
        )
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps(summary["labels"], indent=2))
        return 0

    inputs = _copy_inputs(out_root)
    artifacts = _load_messages(inputs["artifacts"])
    if any(a_state_key(ex) not in artifacts for ex in examples):
        print("Frozen 12-carrier artifacts are missing. Do not regenerate A.")
        return 2
    recon_refs = _recon_refs(inputs["recon_check"])
    refval_checks = {r["example_id"]: r for r in _load_rows(inputs["refval_checks"])}
    refgen_rows = _refgen_rows(inputs["refgen_evals"])
    if any(ex.example_id not in refval_checks or ex.example_id not in refgen_rows for ex in examples):
        print("Missing saved permutations (refval12) or intact free-generation rows (refgen12).")
        return 2

    policy, runner = _prepare_runner(args, out_root, extra={"phase": args.phase, "n_carriers": N_CARRIERS_12})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    return _run(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, refval_checks, refgen_rows, inputs)


if __name__ == "__main__":
    raise SystemExit(main())
