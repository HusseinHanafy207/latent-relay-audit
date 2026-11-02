"""Can B read one carrier's label from the complete twelve-carrier message?

Uses the nine frozen twelve-carrier sheets and A outputs from the matched
reconstruction run. Do not regenerate A. Do not train an adapter.

All twelve carriers' result states stay present. Only the requested output
changes: one question per carrier,
"According to Agent A's results, is Carrier Q eligible or ineligible?
Reply with exactly eligible or ineligible."

Four conditions, same construction, positions and boundaries as recon12:
- generic_r: generic support plus A's transferred result states
- generic_fresh_r: the same support plus result states recomputed after it
- full_kv: the rebuilt completed sender cache
- text_result: the RESULT lines as text

9 cases x 12 carriers x 4 conditions = 432 short greedy answers. Every question
starts from an independent copy of its condition's message cache.
Statistics are grouped by the nine source cases; the 108 labels are not
independent examples.
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
    N_CARRIERS_12,
    N_TIMING_FAMILIES,
    assert_timing12_invariants,
    ensure_timing12_jsonl,
    gold_eligibility_statuses_n,
)
from src.precision import apply_precision_policy
from src.prompts_plan4 import prompt_contains_a_private
from src.prompts_plan4_query import (
    QUERY_MAX_NEW,
    b_query_kv_prompt,
    b_query_text_prompt,
    parse_single_label,
)
from src.span_select import caches_equal
from src.stage_plan0 import _load_messages
from src.stage_plan4_boundary import _added_token_ids
from src.stage_plan4_generic import _prefill_templates
from src.stage_plan4_recon12 import _fresh_r_after_support
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_timing import _collect_templates_n, _pair_state_n
from src.stage_select import _load_rows
from src.two_agent import TwoAgentInventory

N_BOOT = 2000
BOOT_SEED = 2026
CONDS = ("generic_r", "generic_fresh_r", "full_kv", "text_result")
SPEC = {
    "generic_r": {"kind": "kv", "r_source": "transfer", "cache_source": "generic_instr_same_R"},
    "generic_fresh_r": {"kind": "kv", "r_source": "fresh_after_support", "cache_source": "generic_instr_fresh_R"},
    "full_kv": {"kind": "kv", "r_source": "full", "cache_source": "rebuilt_full"},
    "text_result": {"kind": "text", "r_source": "text", "cache_source": "text"},
}
PAIRED = (
    ("generic_fresh_r_vs_generic_r", "generic_r", "generic_fresh_r"),
    ("full_kv_vs_generic_r", "generic_r", "full_kv"),
    ("text_result_vs_generic_r", "generic_r", "text_result"),
    ("generic_fresh_r_vs_full_kv", "full_kv", "generic_fresh_r"),
)
PRESERVED = (
    "results_stage_plan4_recon12",
    "results_stage_plan4_timing",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
)
FROZEN_PLACEMENT = (
    "Same frozen A sheets, result tokens, cache construction, positions and boundaries "
    "as recon12. All twelve result lines stay in every message. B keeps the same "
    "private facts. Only the requested output changes to one label for one named "
    "carrier. Greedy decoding with a fixed short allowance. Each question starts "
    "from an independent copy of its message cache."
)
INTERPRETATION_GUIDE = (
    "If transferred R becomes accurate when queried one carrier at a time, the states "
    "can support correct answers under a targeted request; investigate carrier-label "
    "association and errors accumulating over the longer response. That would justify "
    "examining receiver attention, not prove attention is the sole cause. "
    "If transferred R stays poor while fresh R stays strong, the reuse problem persists "
    "under a focused request; proceed to matched K/V replacement and attention "
    "diagnostics with fresh R under the identical support as the reference. "
    "If fresh R and text also deteriorate, first inspect the single-label prompt and "
    "scoring; that would not isolate cache reuse. "
    "Reconstruction is a diagnostic reference here, not an efficient repair, and the "
    "processing totals are not a transmission or speed claim."
)


def _paths():
    timing12 = DATA_DIR / "plan4_timing12.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_query12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_query12")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return timing12, out_root, kaggle_dest


PERSIST_NAMES = (
    "evaluations.jsonl",
    "summary.json",
    "check_summary.json",
    "manifest.json",
    "a_artifacts_12.jsonl",
    "twelve_line_evaluations.jsonl",
    "recon12_check_summary.json",
)


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


def _copy_first(out_root: Path, name: str, sources: list[Path], *, what: str) -> Path:
    dest = out_root / name
    if dest.is_file():
        return dest
    for src in sources:
        if src.is_file() and src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
            print(f"Copied {what} from {src}")
            return dest
    raise FileNotFoundError(f"Need {what}: one of {[str(s) for s in sources]}.")


def _copy_inputs(out_root: Path, *, artifacts12: str | None = None) -> dict:
    recon = [REPO_ROOT / "results_stage_plan4_recon12", Path("/kaggle/working/results_stage_plan4_recon12")]
    art_sources = ([Path(artifacts12)] if artifacts12 else []) + [
        *(d / "a_artifacts_12.jsonl" for d in recon),
        REPO_ROOT / "results_stage_plan4_timing" / "a_artifacts_12.jsonl",
        Path("/kaggle/working/results_stage_plan4_timing/a_artifacts_12.jsonl"),
    ]
    return {
        "artifacts": _copy_first(out_root, "a_artifacts_12.jsonl", art_sources, what="frozen 12-carrier A artifacts"),
        "twelve_line": _copy_first(
            out_root,
            "twelve_line_evaluations.jsonl",
            [d / "evaluations.jsonl" for d in recon],
            what="recon12 twelve-line evaluations",
        ),
        "recon_check": _copy_first(
            out_root,
            "recon12_check_summary.json",
            [d / "check_summary.json" for d in recon],
            what="recon12 construction check",
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
            "stage": "plan4-query12",
            "do_not_overwrite": list(PRESERVED),
            "conditions": list(CONDS),
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
        max_new_tokens=QUERY_MAX_NEW,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha"):
        pin_model_revision(revision["sha"])
    return policy, runner


def _build_state(runner, ex, rec, decode, sep_ids, close_ids, spans, generics) -> dict:
    state = _pair_state_n(runner, rec, ex, decode, sep_ids, close_ids, spans, generics)
    fresh_r, _fresh_s, fresh_info = _fresh_r_after_support(runner, state)
    state["mixed"]["generic_fresh_r"] = fresh_r
    state["fresh_info"] = fresh_info
    return state


def _prompt_for(ex, cond: str, carrier: str, message: str) -> str:
    if SPEC[cond]["kind"] == "text":
        prompt = b_query_text_prompt(ex, message, carrier)
    else:
        prompt = b_query_kv_prompt(ex, carrier)
    if prompt_contains_a_private(prompt, ex):
        raise RuntimeError(f"Single-label prompt leaked A's raw sheet for {ex.example_id}.")
    if prompt.count(f"Carrier {carrier} eligible or ineligible") != 1:
        raise RuntimeError("Single-label prompt must name exactly one queried carrier.")
    return prompt


def _check_construction(state: dict, ref: dict | None) -> dict:
    slots = state["slots_extra"]
    got = {
        "result_start": int(slots["result_start"]),
        "n_result": int(slots["n_result"]),
        "generic_r_n": n_kv_positions(state["mixed"]["generic_r"].cache_cpu),
        "fresh_r_n": n_kv_positions(state["mixed"]["generic_fresh_r"].cache_cpu),
        "template_id": (state.get("instr") or {}).get("template_id"),
        "support_held": bool(state["fresh_info"].get("support_held")),
    }
    matches = ref is not None and all(
        got[k] == ref.get(k) for k in ("result_start", "n_result", "generic_r_n", "fresh_r_n", "template_id")
    )
    return {**got, "matches_recon12": bool(matches)}


def _recon_refs(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["a_state_key"]: row for row in data.get("rows") or []}


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs) -> int:
    decode = _decode_fn(runner)
    sep_ids, close_ids = _added_token_ids(runner.wrapper.tokenizer)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    rows = []
    for i, ex in enumerate(examples):
        key = a_state_key(ex)
        state = _build_state(runner, ex, artifacts[key], decode, sep_ids, close_ids, spans_by_key[key], generics)
        built = _check_construction(state, recon_refs.get(key))
        message = (state.get("instr") or {}).get("result_text") or ""
        for cond in CONDS:
            for carrier in ex.carriers:
                _prompt_for(ex, cond, carrier, message)
        row = {"a_state_key": key, "ok": bool(built["matches_recon12"] and built["support_held"]), **built}
        rows.append(row)
        print(f"  [{i + 1}/{len(examples)}] {key} ok={row['ok']} n_R={row['n_result']}", flush=True)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_passed = bool(rows) and all(r["ok"] for r in rows)
    _write_json(out_root / "check_summary.json", {"n": len(rows), "all_passed": all_passed, "rows": rows})
    _persist(out_root, kaggle_dest)
    print(json.dumps({"n": len(rows), "all_passed": all_passed}, indent=2))
    return 0 if all_passed else 1


def _query_row(ex, cond, carrier, gold, prompt, text, n_new, truncated, latency, extra) -> dict:
    parsed = parse_single_label(text)
    pred = parsed["pred_label"]
    return {
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "difficulty": ex.difficulty,
        "a_state_key": a_state_key(ex),
        "condition": cond,
        "cache_source": SPEC[cond]["cache_source"],
        "r_source": SPEC[cond]["r_source"],
        "carrier": carrier,
        "carrier_index": list(ex.carriers).index(carrier),
        "gold_label": gold,
        **parsed,
        "correct": pred == gold,
        "ineligible_marked_eligible": gold == "ineligible" and pred == "eligible",
        "eligible_marked_ineligible": gold == "eligible" and pred == "ineligible",
        "raw_text": text,
        "n_new_tokens": int(n_new),
        "truncated": bool(truncated),
        "max_new_tokens": QUERY_MAX_NEW,
        "latency_s": float(latency),
        "prompt": prompt,
        **extra,
    }


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts) -> int:
    cpath = out_root / "check_summary.json"
    check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
    if (check is None or not check.get("all_passed")) and not args.force_compare:
        print("Run --phase check first.")
        return 2
    decode = _decode_fn(runner)
    sep_ids, close_ids = _added_token_ids(runner.wrapper.tokenizer)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    jsonl_path = out_root / "evaluations.jsonl"
    rows = _load_rows(jsonl_path)
    done = {(r["example_id"], r["condition"], r["carrier"]) for r in rows}
    total = len(examples) * N_CARRIERS_12 * len(CONDS)
    print(f"Query12 compare n_cases={len(examples)} n_answers={total} done={len(done)}", flush=True)
    for i, ex in enumerate(examples):
        todo = [(c, k) for c in CONDS for k in ex.carriers if (ex.example_id, c, k) not in done]
        if not todo:
            print(f"  [{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        key = a_state_key(ex)
        state = _build_state(runner, ex, artifacts[key], decode, sep_ids, close_ids, spans_by_key[key], generics)
        gold = gold_eligibility_statuses_n(ex)
        message = (state.get("instr") or {}).get("result_text") or ""
        for cond in CONDS:
            carriers = [k for c, k in todo if c == cond]
            if not carriers:
                continue
            base = None if SPEC[cond]["kind"] == "text" else state["mixed"][cond]
            snapshot = None if base is None else runner.clone_sender(base)
            for carrier in carriers:
                prompt = _prompt_for(ex, cond, carrier, message)
                if base is None:
                    text, n_new, trunc, lat = runner.generate_text_message(prompt, max_new_tokens=QUERY_MAX_NEW)
                    extra = {"channel": "text", "message_bytes": len(message.encode("utf-8")), "n_kv_positions": 0}
                else:
                    copy = runner.clone_sender(base)
                    text, n_new, trunc, lat = runner.generate_from_cache(copy, prompt, max_new_tokens=QUERY_MAX_NEW)
                    extra = {
                        "channel": "kv",
                        "message_bytes": int(base.full_relay_bytes),
                        "n_kv_positions": n_kv_positions(base.cache_cpu),
                        "pos_cursor": int(base.pos_cursor_cpu.view(-1)[0].item()),
                        "result_start": int(state["slots_extra"]["result_start"]),
                    }
                    del copy
                row = _query_row(ex, cond, carrier, gold[carrier], prompt, text, n_new, trunc, lat, extra)
                rows.append(row)
                done.add((ex.example_id, cond, carrier))
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"    {ex.example_id} {cond} {carrier}: gold={gold[carrier]} "
                    f"pred={row['pred_label']} raw={text.strip()[:24]!r}",
                    flush=True,
                )
            if snapshot is not None and not caches_equal(snapshot.cache_cpu, base.cache_cpu):
                raise RuntimeError(f"{cond} message cache changed during questions for {ex.example_id}.")
            del snapshot
            _persist(out_root, kaggle_dest)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _summarize(rows, _load_rows(out_root / "twelve_line_evaluations.jsonl"), check=check)
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps(summary["receiver"], indent=2)[:8000])
    return 0


def _cond_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    by_case: dict[str, int] = defaultdict(int)
    for item in items:
        by_case[item["example_id"]] += int(bool(item["correct"]))
    n_case = len(by_case)
    return {
        "n": n,
        "parse_ok": sum(bool(r["parse_ok"]) for r in items),
        "exact_raw": sum(bool(r["exact_raw"]) for r in items),
        "truncated": sum(bool(r["truncated"]) for r in items),
        "correct": sum(bool(r["correct"]) for r in items),
        "ineligible_marked_eligible": sum(bool(r["ineligible_marked_eligible"]) for r in items),
        "eligible_marked_ineligible": sum(bool(r["eligible_marked_ineligible"]) for r in items),
        "unparsed": sum(not r["parse_ok"] for r in items),
        "n_gold_ineligible": sum(r["gold_label"] == "ineligible" for r in items),
        "n_gold_eligible": sum(r["gold_label"] == "eligible" for r in items),
        "n_cases": n_case,
        "cases_all_twelve_correct": sum(v == N_CARRIERS_12 for v in by_case.values()),
        "correct_by_case": dict(sorted(by_case.items())),
        "mean_latency_s": sum(float(r.get("latency_s") or 0.0) for r in items) / n,
    }


def _line_labels(twelve_line_rows: list[dict]) -> dict:
    out = {}
    for row in twelve_line_rows:
        gold = row.get("gold_status") or {}
        pred = row.get("plan_pred_status") or {}
        for carrier, g in gold.items():
            out[(row["example_id"], row["condition"], carrier)] = {"gold": g, "pred": pred.get(carrier)}
    return out


def _versus_line(items: list[dict], line: dict) -> dict:
    tab = {"n": 0, "both_correct": 0, "query_only_correct": 0, "line_only_correct": 0, "both_wrong": 0,
           "same_label": 0, "line_ie": 0, "query_ie": 0, "ie_fixed_by_query": 0, "ie_new_in_query": 0,
           "missing_line": 0}
    for r in items:
        ref = line.get((r["example_id"], r["condition"], r["carrier"]))
        if ref is None:
            tab["missing_line"] += 1
            continue
        q_ok = bool(r["correct"])
        l_ok = ref["pred"] == ref["gold"]
        l_ie = ref["gold"] == "ineligible" and ref["pred"] == "eligible"
        q_ie = bool(r["ineligible_marked_eligible"])
        tab["n"] += 1
        tab["both_correct"] += q_ok and l_ok
        tab["query_only_correct"] += q_ok and not l_ok
        tab["line_only_correct"] += l_ok and not q_ok
        tab["both_wrong"] += not q_ok and not l_ok
        tab["same_label"] += r["pred_label"] == ref["pred"]
        tab["line_ie"] += l_ie
        tab["query_ie"] += q_ie
        tab["ie_fixed_by_query"] += l_ie and not q_ie
        tab["ie_new_in_query"] += q_ie and not l_ie
    return tab


def _by_index(items: list[dict], line: dict) -> dict:
    out = {}
    for idx in range(N_CARRIERS_12):
        sel = [r for r in items if int(r["carrier_index"]) == idx]
        refs = [line.get((r["example_id"], r["condition"], r["carrier"])) for r in sel]
        out[str(idx)] = {
            "n": len(sel),
            "query_correct": sum(bool(r["correct"]) for r in sel),
            "line_correct": sum(1 for ref in refs if ref is not None and ref["pred"] == ref["gold"]),
        }
    return out


def _case_counts(items: list[dict], metric: str) -> dict:
    out: dict[str, int] = defaultdict(int)
    for r in items:
        out[r["example_id"]] += int(bool(r[metric]))
    return out


def _cluster_delta(left: dict, right: dict, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    cases = sorted(set(left) & set(right))
    if not cases:
        return {"n_cases": 0}
    deltas = [right[c] - left[c] for c in cases]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        draw = [deltas[rng.randrange(len(deltas))] for _ in cases]
        boots.append(sum(draw) / len(draw))
    boots.sort()
    return {
        "n_cases": len(cases),
        "total_delta": sum(deltas),
        "mean_delta_per_case": sum(deltas) / len(deltas),
        "ci95_per_case": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
        "cases_increased": sum(d > 0 for d in deltas),
        "cases_decreased": sum(d < 0 for d in deltas),
        "cases_equal": sum(d == 0 for d in deltas),
        "per_case": dict(zip(cases, deltas)),
    }


def _line_case_counts(line: dict, cond: str, *, metric: str) -> dict:
    out: dict[str, int] = defaultdict(int)
    for (eid, c, _carrier), ref in line.items():
        if c != cond:
            continue
        if metric == "correct":
            out[eid] += int(ref["pred"] == ref["gold"])
        else:
            out[eid] += int(ref["gold"] == "ineligible" and ref["pred"] == "eligible")
    return out


def _summarize(rows: list[dict], twelve_line_rows: list[dict], *, check: dict | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    line = _line_labels(twelve_line_rows)
    paired = {}
    for name, left, right in PAIRED:
        paired[name] = {
            "correct": _cluster_delta(_case_counts(by_cond[left], "correct"), _case_counts(by_cond[right], "correct")),
            "ineligible_marked_eligible": _cluster_delta(
                _case_counts(by_cond[left], "ineligible_marked_eligible"),
                _case_counts(by_cond[right], "ineligible_marked_eligible"),
            ),
        }
    query_vs_line = {}
    for cond in CONDS:
        query_vs_line[cond] = {
            "crosstab": _versus_line(by_cond[cond], line),
            "correct_query_minus_line": _cluster_delta(
                _line_case_counts(line, cond, metric="correct"), _case_counts(by_cond[cond], "correct")
            ),
            "ie_query_minus_line": _cluster_delta(
                _line_case_counts(line, cond, metric="ie"), _case_counts(by_cond[cond], "ineligible_marked_eligible")
            ),
            "by_carrier_index": _by_index(by_cond[cond], line),
        }
    return {
        "n_rows": len(rows),
        "n_expected": N_TIMING_FAMILIES * N_CARRIERS_12 * len(CONDS),
        "n_cases": len({r["example_id"] for r in rows}),
        "conditions": list(CONDS),
        "query_max_new": QUERY_MAX_NEW,
        "receiver": {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {
            lvl: {cond: _cond_block([r for r in by_cond.get(cond, []) if r["difficulty"] == lvl]) for cond in CONDS}
            for lvl in LEVELS
        },
        "paired_by_case": paired,
        "query_vs_twelve_line": query_vs_line,
        "check": None if check is None else {"all_passed": check.get("all_passed"), "n": check.get("n")},
        "spec": SPEC,
        "frozen_placement": FROZEN_PLACEMENT,
        "stats_note": (
            "Bootstrap resamples the nine source cases (2000 draws). The 108 labels per "
            "condition are not independent examples; nine clusters give wide intervals. "
            "Deltas are right minus left. cases_increased / cases_decreased count the "
            "direction of the change in that metric; for error counts an increase is worse."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Single-carrier label questions on the twelve-carrier message.")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument("--phase", choices=("check", "compare", "all", "rescore"), default="check")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--force-compare", action="store_true")
    parser.add_argument("--artifacts12", default=None)
    args = parser.parse_args()
    if args.split != "plan4_timing12":
        print("Query12 uses only the frozen timing-12 sheets. Do not load plan4_dev or plan4_hold.")
        return 2

    timing12_path, out_root, kaggle_dest = _paths()
    if "stage_plan4_query12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_query12.")
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
        eval_path = out_root / "evaluations.jsonl"
        shared = REPO_ROOT / "results_stage_plan4_query12"
        if not eval_path.is_file() and (shared / "evaluations.jsonl").is_file():
            out_root = shared
        cpath = out_root / "check_summary.json"
        check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
        summary = _summarize(
            _load_rows(out_root / "evaluations.jsonl"),
            _load_rows(out_root / "twelve_line_evaluations.jsonl"),
            check=check,
        )
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps(summary["receiver"], indent=2)[:8000])
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
        return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)
    rc = _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, recon_refs)
    if rc != 0 and not args.force_compare:
        print("Construction check failed; stopping.")
        return rc
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
