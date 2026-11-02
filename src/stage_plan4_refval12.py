"""Does the reference-attention oracle still depend on the transferred result values?

Teacher-forced, same setup as the replay run: frozen twelve-line plan prompt plus
the reference's saved plan (fk_fv from the K/V swap run), same nine cases, same
generic support, positions, masks and cache length.

Branches:
- tk_tv: transferred K + V, ordinary attention
- fk_fv: reference, ordinary attention (untouched)
- tk_tv_refattn: normal oracle; pair [fk_fv, tk_tv], reference weights on tk_tv's values
- tk_tv_refattn_vzero: same oracle; target cache has result-region values set to zero
- tk_tv_refattn_vperm: same oracle; target result-region value rows permuted with one
  fixed derangement, the same for every layer and head

Keys, support, positions, masks, token IDs and cache length are unchanged in the
edited caches, and the reference cache is never edited. Under replacement the
target row's own queries and keys do not enter its attention weights, so the
transferred cache acts on the target row only through values.
Artificial interventions; not a semantic explanation and not a method.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import n_kv_positions
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
from src.span_select import caches_equal
from src.stage_plan0 import _load_messages
from src.stage_plan4_kvmix12 import _setup, _state_for
from src.stage_plan4_query12 import _cluster_delta, _copy_first, _recon_refs
from src.stage_plan4_refattn12 import (
    TOL_MAX_MEAN_KL,
    TOL_MIN_AGREE,
    _batch_logits,
    _branch_block,
    _compare,
    _per_case,
    _reference_plans,
    _tol,
    label_tokens,
    response_labels,
)
from src.stage_plan4_timing import _prompts_for
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

REFERENCE = "fk_fv"
ORACLE = "tk_tv_refattn"
EDITS = {"tk_tv_refattn_vzero": "zero", "tk_tv_refattn_vperm": "perm"}
BRANCHES = ("tk_tv", REFERENCE, ORACLE) + tuple(EDITS)
BRANCH_SPEC = {
    "tk_tv": "transferred K + V, ordinary attention",
    "fk_fv": "fresh K + V, ordinary attention (reference, untouched)",
    "tk_tv_refattn": "transferred K + V, reference attention weights (normal oracle)",
    "tk_tv_refattn_vzero": "reference attention; target result-region values set to zero",
    "tk_tv_refattn_vperm": "reference attention; target result-region value rows permuted (fixed derangement)",
}
PERM_SEED = 2026
PAIRED = (
    ("vzero_vs_oracle", ORACLE, "tk_tv_refattn_vzero"),
    ("vperm_vs_oracle", ORACLE, "tk_tv_refattn_vperm"),
    ("vzero_vs_tk_tv", "tk_tv", "tk_tv_refattn_vzero"),
    ("vperm_vs_tk_tv", "tk_tv", "tk_tv_refattn_vperm"),
    ("oracle_vs_tk_tv", "tk_tv", ORACLE),
)
PRESERVED = (
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
    "A large drop under zeroed or permuted result values supports the view that, under "
    "the oracle, B still depends on the transferred result values. Little change means "
    "the reference's attention pattern can drive much of the answer by itself, and claims "
    "about B 'reading A's results' need to be more cautious. Both edits are artificial "
    "(zero values are off-distribution; permutation keeps the value distribution but "
    "breaks token-to-value correspondence), so neither alone is a semantic explanation. "
    "Teacher-forced; not free-generation accuracy."
)
PERSIST_NAMES = (
    "evaluations.jsonl",
    "case_checks.jsonl",
    "summary.json",
    "manifest.json",
    "a_artifacts_12.jsonl",
    "recon12_check_summary.json",
    "kvmix12_evaluations.jsonl",
    "refattn12_evaluations.jsonl",
)


def _paths():
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_refval12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_refval12")
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
    recon = [REPO_ROOT / "results_stage_plan4_recon12", Path("/kaggle/working/results_stage_plan4_recon12")]
    kvmix = [REPO_ROOT / "results_stage_plan4_kvmix12", Path("/kaggle/working/results_stage_plan4_kvmix12")]
    refattn = [REPO_ROOT / "results_stage_plan4_refattn12", Path("/kaggle/working/results_stage_plan4_refattn12")]
    return {
        "artifacts": _copy_first(
            out_root,
            "a_artifacts_12.jsonl",
            [d / "a_artifacts_12.jsonl" for d in refattn + kvmix + recon],
            what="frozen 12-carrier A artifacts",
        ),
        "recon_check": _copy_first(
            out_root,
            "recon12_check_summary.json",
            [d / "recon12_check_summary.json" for d in refattn] + [d / "check_summary.json" for d in recon],
            what="recon12 check",
        ),
        "kvmix_evals": _copy_first(
            out_root,
            "kvmix12_evaluations.jsonl",
            [d / "kvmix12_evaluations.jsonl" for d in refattn] + [d / "evaluations.jsonl" for d in kvmix],
            what="K/V swap evaluations (saved reference plans)",
        ),
        "refattn_evals": _copy_first(
            out_root,
            "refattn12_evaluations.jsonl",
            [d / "evaluations.jsonl" for d in refattn],
            what="teacher-forced replay evaluations (oracle reproduction)",
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
            "stage": "plan4-refval12",
            "do_not_overwrite": list(PRESERVED),
            "branches": list(BRANCHES),
            "reference": REFERENCE,
            "perm_seed": PERM_SEED,
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


def fixed_derangement(n: int, seed: int = PERM_SEED) -> list[int]:
    """Seeded permutation of range(n) with no fixed points (n >= 2)."""
    if n < 2:
        raise ValueError("A derangement needs at least two positions.")
    rng = random.Random(seed)
    idx = list(range(n))
    while True:
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            return list(idx)


def edit_result_values(blob: dict, positions: list[int], kind: str, perm: list[int] | None = None) -> dict:
    """New cache blob: keys copied unchanged; values edited only at `positions`."""
    idx = torch.tensor(list(positions), dtype=torch.long)
    if kind == "perm":
        if perm is None or sorted(perm) != list(range(len(positions))):
            raise ValueError("perm must be a permutation of the result positions.")
        src = idx[torch.tensor(perm, dtype=torch.long)]
    elif kind != "zero":
        raise ValueError(f"Unknown value edit {kind!r}")
    layers = []
    for k, v in blob["layers"]:
        new_v = v.detach().contiguous().clone()
        if kind == "zero":
            new_v[:, :, idx, :] = 0
        else:
            new_v[:, :, idx, :] = v[:, :, src, :]
        layers.append((k.detach().contiguous().clone(), new_v))
    return {"layers": layers}


def check_value_edit(base: dict, edited: dict, positions: list[int], kind: str, perm: list[int] | None = None) -> dict:
    n = n_kv_positions(base)
    pos = sorted(set(int(p) for p in positions))
    outside = torch.tensor([p for p in range(n) if p not in set(pos)], dtype=torch.long)
    idx = torch.tensor(list(positions), dtype=torch.long)
    src = idx[torch.tensor(perm, dtype=torch.long)] if kind == "perm" else None
    keys_equal = values_outside_equal = edited_as_intended = changed = True
    for (bk, bv), (ek, ev) in zip(base["layers"], edited["layers"]):
        keys_equal &= bool(torch.equal(bk, ek))
        values_outside_equal &= bool(torch.equal(bv[:, :, outside, :], ev[:, :, outside, :]))
        if kind == "zero":
            edited_as_intended &= bool((ev[:, :, idx, :] == 0).all())
        else:
            edited_as_intended &= bool(torch.equal(ev[:, :, idx, :], bv[:, :, src, :]))
        changed &= not bool(torch.equal(ev[:, :, idx, :], bv[:, :, idx, :]))
    row = {
        "same_length": n_kv_positions(edited) == n,
        "same_layers": len(edited["layers"]) == len(base["layers"]),
        "keys_equal": keys_equal,
        "values_outside_result_equal": values_outside_equal,
        "result_values_edited_as_intended": edited_as_intended,
        "result_values_changed": changed,
    }
    row["ok"] = all(row.values())
    return row


def _edited_bundle(runner, bundle, positions, kind, perm):
    out = runner.clone_sender(bundle)
    out.cache_cpu = edit_result_values(bundle.cache_cpu, positions, kind, perm)
    out.source = f"{bundle.source}_v{kind}"
    return out


def _decision_change(left: dict, right: dict) -> dict:
    tab = {"n": 0, "changed": 0, "fixed": 0, "broken": 0}
    lp = {p["carrier"]: p for p in left["per_carrier"]}
    for p in right["per_carrier"]:
        q = lp.get(p["carrier"])
        if q is None:
            continue
        tab["n"] += 1
        tab["changed"] += p["decision"] != q["decision"]
        tab["fixed"] += (not q["correct"]) and p["correct"]
        tab["broken"] += q["correct"] and not p["correct"]
    return tab


def _oracle_reproduction(row: dict, saved: dict | None) -> dict:
    if saved is None:
        return {"available": False, "ok": False}
    return {
        "available": True,
        "decisions_equal": row["decisions"] == saved.get("decisions"),
        "labels_correct_equal": row["labels_correct"] == saved.get("labels_correct"),
        "mean_kl_abs_diff": abs(row["mean_kl"] - float(saved.get("mean_kl") or 0.0)),
        "ok": row["decisions"] == saved.get("decisions") and row["labels_correct"] == saved.get("labels_correct"),
    }


def _case(runner, ex, artifacts, setup, recon_refs, ref_rows, oracle_rows) -> dict:
    state, built, mix = _state_for(runner, ex, artifacts, setup, recon_refs)
    if not (mix["ok"] and built["matches_recon12"]):
        raise RuntimeError(f"Mixed-cache construction failed for {ex.example_id}.")
    caches = state["kvmix"]
    tk, ref_cache = caches["tk_tv"], caches[REFERENCE]
    ref_snapshot = runner.clone_sender(ref_cache)
    tk_snapshot = runner.clone_sender(tk)
    result = [int(p) for p in state["slots_extra"]["result"]]
    perm = fixed_derangement(len(result))
    edited = {name: _edited_bundle(runner, tk, result, kind, perm) for name, kind in EDITS.items()}
    construction = {
        name: check_value_edit(tk.cache_cpu, edited[name].cache_cpu, result, kind, perm) for name, kind in EDITS.items()
    }

    tokenizer = runner.wrapper.tokenizer
    _r, input_ids, _a = runner._encode(_prompts_for(ex)["kv_plan"])
    prompt_ids = [int(t) for t in input_ids[0].tolist()]
    ref_row = ref_rows[ex.example_id]
    plan_text = ref_row["plan_text"]
    enc = tokenizer(plan_text, add_special_tokens=False, return_offsets_mapping=True)
    resp_ids = [int(t) for t in enc["input_ids"]]
    labels = response_labels(plan_text, ex.carriers, enc["offset_mapping"], resp_ids)
    tok_e, tok_i = label_tokens(labels, tokenizer)
    gold = gold_eligibility_statuses_n(ex)
    n_layers = len(tk.cache_cpu["layers"])
    args_tol = (labels, tok_e, tok_i, gold, resp_ids)

    ordinary_tk, _ = _batch_logits(runner, [tk], prompt_ids, resp_ids, mode=None)
    ordinary_ref, _ = _batch_logits(runner, [ref_cache], prompt_ids, resp_ids, mode=None)
    ref = ordinary_ref[0]
    logits = {"tk_tv": ordinary_tk[0], REFERENCE: ref}
    checks = {
        "all_labels_located": len(labels) == len(ex.carriers) and all(l["token_index"] is not None for l in labels),
        "construction": construction,
    }
    targets = {ORACLE: tk, **edited}
    for name, target in targets.items():
        pair, info = _batch_logits(runner, [ref_cache, target], prompt_ids, resp_ids, mode="replace")
        logits[name] = pair[1]
        checks[f"{name}:donor_row_unchanged"] = _tol(ref, pair[0], *args_tol)
        checks[f"{name}:replaced_every_layer"] = info["n_replaced"] == n_layers and info["n_layers"] == n_layers
    checks["reference_cache_unchanged"] = caches_equal(ref_snapshot.cache_cpu, ref_cache.cache_cpu)
    checks["transferred_cache_unchanged"] = caches_equal(tk_snapshot.cache_cpu, tk.cache_cpu)

    stats = {name: _compare(ref, logits[name], *args_tol) for name in BRANCHES}
    checks["oracle_reproduces_refattn12"] = _oracle_reproduction(stats[ORACLE], oracle_rows.get(ex.example_id))
    ok = all(
        (v["ok"] if "ok" in v else all(x["ok"] for x in v.values())) if isinstance(v, dict) else bool(v)
        for v in checks.values()
    )
    rows = []
    for name in BRANCHES:
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
                "n_prompt_tokens": len(prompt_ids),
                "n_kv_positions": n_kv_positions(tk.cache_cpu),
                "result_start": int(state["slots_extra"]["result_start"]),
                "n_result": len(result),
                "label_token_eligible": tok_e,
                "label_token_ineligible": tok_i,
                "instrumentation_ok": ok,
                "decision_change_vs_oracle": None if name == ORACLE else _decision_change(stats[ORACLE], stats[name]),
                **stats[name],
            }
        )
    check_row = {"example_id": ex.example_id, "ok": ok, "n_result": len(result), "perm": perm, "checks": checks}
    del state, caches, edited, logits, stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": rows, "check": check_row}


def _run(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, ref_rows, oracle_rows) -> int:
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
    print(f"Refval12: {len(done)}/{len(examples)} cases complete; running {len(todo)}", flush=True)
    rc = 0
    for i, ex in enumerate(todo):
        res = _case(runner, ex, artifacts, setup, recon_refs, ref_rows, oracle_rows)
        for row in res["rows"]:
            rows.append(row)
            _append(eval_path, row)
        checks.append(res["check"])
        _append(check_path, res["check"])
        c = res["check"]
        print(f"  [{i + 1}/{len(todo)}] {ex.example_id} instrumentation_ok={c['ok']} n_result={c['n_result']}", flush=True)
        for k, v in c["checks"].items():
            if k == "construction":
                print("    construction:", {n: r["ok"] for n, r in v.items()})
            elif isinstance(v, dict) and "mean_kl" in v:
                print(f"    {k}: ok={v['ok']} meanKL={v['mean_kl']:.5f} agree={v['argmax_agree']:.4f}")
            else:
                print(f"    {k}: {v}")
        for row in res["rows"]:
            print(
                f"    {row['branch']}: labels={row['labels_correct']}/{row['n_labels']} "
                f"I->E={row['ineligible_marked_eligible']} E->I={row['eligible_marked_ineligible']} "
                f"meanKL={row['mean_kl']:.4f} agree_ref={row['labels_agree_ref']}",
                flush=True,
            )
        _persist(out_root, kaggle_dest)
        if not c["ok"] and not args.force:
            print("Check failed for this case; stopping. Inspect case_checks.jsonl.")
            rc = 1
            break
    summary = _summarize(rows, checks)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({b: {k: summary["branches"][b].get(k) for k in ("labels_correct", "n_labels", "mean_kl_vs_reference")} for b in BRANCHES}, indent=2))
    return rc


def _retained(blocks: dict) -> dict:
    lo, hi = blocks.get("tk_tv") or {}, blocks.get(ORACLE) or {}
    if not lo.get("n_cases") or not hi.get("n_cases"):
        return {}
    gap = hi["labels_correct"] - lo["labels_correct"]
    kl_gap = lo["mean_kl_vs_reference"] - hi["mean_kl_vs_reference"]
    out = {}
    for name in EDITS:
        b = blocks.get(name) or {}
        if not b.get("n_cases"):
            continue
        out[name] = {
            "label_recovery_retained": None if gap == 0 else (b["labels_correct"] - lo["labels_correct"]) / gap,
            "kl_recovery_retained": None if kl_gap == 0 else (lo["mean_kl_vs_reference"] - b["mean_kl_vs_reference"]) / kl_gap,
        }
    return out


def _change_totals(items: list[dict]) -> dict:
    tab = {"n": 0, "changed": 0, "fixed": 0, "broken": 0}
    for r in items:
        for k in tab:
            tab[k] += int((r.get("decision_change_vs_oracle") or {}).get(k) or 0)
    return tab


def _summarize(rows: list[dict], checks: list[dict]) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["branch"]].append(r)
    blocks = {b: _branch_block(by[b]) for b in BRANCHES}
    paired = {}
    for name, left, right in PAIRED:
        paired[name] = {
            metric: _cluster_delta(_per_case(by[left], metric), _per_case(by[right], metric))
            for metric in ("labels_correct", "ineligible_marked_eligible", "eligible_marked_ineligible", "mean_kl")
        }
    repro = [((c.get("checks") or {}).get("oracle_reproduces_refattn12") or {}) for c in checks]
    return {
        "n_rows": len(rows),
        "n_cases": len({r["example_id"] for r in rows}),
        "n_expected_cases": N_TIMING_FAMILIES,
        "reference": REFERENCE,
        "branch_spec": BRANCH_SPEC,
        "instrumentation_ok": bool(checks) and all(bool(c.get("ok")) for c in checks),
        "n_cases_instrumentation_ok": sum(bool(c.get("ok")) for c in checks),
        "oracle_reproduces_refattn12": {
            "n": len(repro),
            "decisions_equal": sum(bool(r.get("decisions_equal")) for r in repro),
            "labels_correct_equal": sum(bool(r.get("labels_correct_equal")) for r in repro),
            "max_mean_kl_abs_diff": max((float(r.get("mean_kl_abs_diff") or 0.0) for r in repro), default=None),
        },
        "branches": blocks,
        "by_difficulty": {
            lvl: {b: _branch_block([r for r in by[b] if r.get("difficulty") == lvl]) for b in BRANCHES} for lvl in LEVELS
        },
        "recovery_retained": _retained(blocks),
        "decision_change_vs_oracle": {b: _change_totals(by[b]) for b in BRANCHES if b != ORACLE},
        "paired_by_case": paired,
        "measurement_note": (
            "Teacher-forced: every branch receives the frozen twelve-line plan prompt and the "
            "reference's saved plan. Label decisions compare the two label first tokens at each "
            "label position. Not free-generation accuracy. The reference cache is never edited; "
            "edited target caches change only result-region values."
        ),
        "mechanism_note": (
            "Under replacement the target row's attention weights come entirely from the reference "
            "row; its own queries and keys are unused. The transferred cache therefore acts on the "
            "target row only through values, including values B computes for its own tokens."
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
    parser = argparse.ArgumentParser(description="Value dependence of the reference-attention oracle (twelve carriers).")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument("--phase", choices=("run", "rescore"), default="run")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many remaining cases.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force", action="store_true", help="Continue after a failed check.")
    args = parser.parse_args()
    if args.split != "plan4_timing12":
        print("Refval12 uses only the frozen timing-12 sheets. Do not load plan4_dev or plan4_hold.")
        return 2

    timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_refval12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_refval12.")
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
        shared = REPO_ROOT / "results_stage_plan4_refval12"
        if not (out_root / "evaluations.jsonl").is_file() and (shared / "evaluations.jsonl").is_file():
            out_root = shared
        summary = _summarize(_load_rows(out_root / "evaluations.jsonl"), _load_rows(out_root / "case_checks.jsonl"))
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
    oracle_rows = {r["example_id"]: r for r in _load_rows(inputs["refattn_evals"]) if r.get("branch") == ORACLE}
    if any(ex.example_id not in ref_rows for ex in examples):
        print("Missing saved reference plans from the K/V swap run.")
        return 2

    policy, runner = _prepare_runner(args, out_root, extra={"phase": args.phase})
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    return _run(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs, ref_rows, oracle_rows)


if __name__ == "__main__":
    raise SystemExit(main())
