"""Complete public task prefix versus the instruction slice (twelve carriers).

Question: does keeping the complete public task prefix help B use A's selected
result states, compared with keeping only the instruction slice?

Same nine timing12 cases and frozen A outputs, same checkpoint, T4, FP16,
greedy decoding, 384 plan / 48 answer tokens, B prompts and framing sentence.
No A regeneration, no added reasons or calculations, no training. These cases
were already inspected: this is a development diagnostic, not confirmation.

Conditions:
1. compact_baseline: short prefix, generic instruction slice, newline, R (tk_tv)
2. task_prefix_r: complete public task prefix, newline, transferred R
3. woodland_prefix_r: frozen woodland prefix + same instruction, newline, R
4. library_prefix_r: frozen library prefix + same instruction, newline, R
5. task_prefix_fresh_r: prefix as in 2, R token IDs processed again after it
6. full_kv: the sender's complete available cache
7. text_result: the same selected result lines as text

Transferred R is gathered from A's frozen cache, values unchanged, keys moved
once from their original sender positions to the new packed positions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.boundary_layout import EXTRA_SEP_KIND, render_handoff_view
from src.cache_utils import cache_slice_delta, kv_blob_size_bytes, n_kv_positions
from src.manifest import load_pinned
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import LEVELS, a_state_key
from src.plan4_timing_task import (
    N_CARRIERS_12,
    N_TIMING_FAMILIES,
    assert_timing12_invariants,
    load_timing12_jsonl,
)
from src.precision import reset_peak_memory
from src.public_prefix import (
    CONTEXTS,
    boundary_record,
    neutral_prefixes,
    packed_result_slots,
    private_prefix_hits,
)
from src.rope_rerotate import rerotate_cache_keys
from src.span_select import caches_equal, gather_cache_positions
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_boundary import _added_token_ids
from src.stage_plan4_generic import _prefill_templates
from src.stage_plan4_kvmix12 import (
    _append,
    _label_crosstab,
    _prepare_runner,
    _twelve_choice,
    _twelve_label_map,
    _twelve_labels,
)
from src.stage_plan4_query12 import N_BOOT, BOOT_SEED, _cluster_delta, _copy_first
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_timing import (
    B_PLAN_MAX_NEW_12,
    _collect_templates_n,
    _pair_state_n,
    _prompts_for,
    _run_kv_b_timed,
    _run_text_b_timed,
)
from src.stage_select import _load_rows

BASELINE = "compact_baseline"
CANDIDATE = "task_prefix_r"
FRESH = "task_prefix_fresh_r"
CONDS = (
    BASELINE,
    CANDIDATE,
    "woodland_prefix_r",
    "library_prefix_r",
    FRESH,
    "full_kv",
    "text_result",
)
SPEC = {
    BASELINE: {"kind": "kv", "context": None, "results": "transferred", "layout": "short prefix + generic instruction slice + newline + R"},
    CANDIDATE: {"kind": "kv", "context": "task", "results": "transferred", "layout": "complete public task prefix + newline + R"},
    "woodland_prefix_r": {"kind": "kv", "context": "woodland", "results": "transferred", "layout": "woodland prefix + instruction + newline + R"},
    "library_prefix_r": {"kind": "kv", "context": "library", "results": "transferred", "layout": "library prefix + instruction + newline + R"},
    FRESH: {"kind": "kv", "context": "task", "results": "fresh", "layout": "complete public task prefix + newline + R processed again"},
    "full_kv": {"kind": "kv", "context": None, "results": "full", "layout": "sender's complete cache"},
    "text_result": {"kind": "text", "context": None, "results": "text", "layout": "result lines as text"},
}
LONG_TRANSFERRED = (CANDIDATE, "woodland_prefix_r", "library_prefix_r")
LONG = LONG_TRANSFERRED + (FRESH,)
REFERENCE_CONDS = {BASELINE: "generic_r", "full_kv": "full_kv", "text_result": "text_result"}
PAIRED = (
    ("task_vs_baseline", BASELINE, CANDIDATE),
    ("task_vs_woodland", "woodland_prefix_r", CANDIDATE),
    ("task_vs_library", "library_prefix_r", CANDIDATE),
    ("woodland_vs_baseline", BASELINE, "woodland_prefix_r"),
    ("library_vs_baseline", BASELINE, "library_prefix_r"),
    ("fresh_vs_transferred_task_prefix", CANDIDATE, FRESH),
    ("full_kv_vs_task", CANDIDATE, "full_kv"),
    ("text_vs_task", CANDIDATE, "text_result"),
)
CONT_LEN = 64
PRESERVED = (
    "results_stage_plan4_timing",
    "results_stage_plan4_recon12",
    "results_stage_plan4_query12",
    "results_stage_plan4_kvmix12",
    "results_stage_plan4_refattn12",
    "results_stage_plan4_refgen12",
    "results_stage_plan4_refval12",
    "results_stage_plan4_refgenval12",
    "results_stage_plan4_nbound",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
)
PERSIST_NAMES = (
    "manifest.json",
    "construction.json",
    "check_summary.json",
    "evaluations.jsonl",
    "summary.json",
    "a_artifacts_12.jsonl",
    "recon12_evaluations.jsonl",
    "recon12_check_summary.json",
    "nbound_construction.json",
)
FROZEN_PLACEMENT = (
    "Public prefixes are prefilled once per template and context, then one fixed "
    "newline is processed after them; no closing token or new framing is added and "
    "the old five-token prefix is not prepended. Transferred R is gathered from A's "
    "frozen cache: values unchanged, keys moved once from the original sender "
    "positions to consecutive packed positions after the newline. Task and neutral "
    "conditions share lengths, chat header, instruction IDs and positions, "
    "separator, result slots, masks and B's starting position."
)
INTERPRETATION_GUIDE = {
    "task_beats_both_neutrals": (
        "Retaining the omitted public task context is a promising repair. Supports "
        "a context-content explanation without identifying the attention mechanism."
    ),
    "task_and_neutrals_similar": (
        "A longer, internally consistent prefix may help. Do not claim the task "
        "wording is responsible."
    ),
    "fresh_works_transferred_poor": (
        "This public-context repair is insufficient; the remaining difficulty is "
        "reuse of the transferred result states under these layouts."
    ),
    "fresh_unreliable": (
        "The new layout is not a healthy reference. Investigate it before "
        "attributing the outcome to transferred states."
    ),
    "candidate_improves_below_references": "Report a partial repair, not recovery.",
    "recovery_standard": (
        "Within five percentage points of full KV's 9/9 all-label cases requires "
        "9/9 at the point estimate; even that is a promising diagnostic, not "
        "evidence of generalization."
    ),
}
COST_NOTE = (
    "Public-prefix states can be prepared once on B's side and need not be sent "
    "per example, but they occupy memory and have a setup cost; they are not free. "
    "Resident MiB are packed-cache sizes, not measured traffic. This run settles "
    "reliability; a repeated, counterbalanced timing comparison comes later."
)


def _paths():
    out_root = RESULTS_DIR / "stage_plan4_publicprefix12" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_publicprefix12")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return out_root, kaggle_dest


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


def _mib(blob: dict) -> float:
    return kv_blob_size_bytes(blob) / (1024 * 1024)


def _copy_inputs(out_root: Path) -> dict:
    dirs = [REPO_ROOT, Path("/kaggle/working")]
    recon = [d / "results_stage_plan4_recon12" for d in dirs]
    return {
        "artifacts": _copy_first(
            out_root,
            "a_artifacts_12.jsonl",
            [d / "a_artifacts_12.jsonl" for d in recon]
            + [d / "results_stage_plan4_refval12" / "a_artifacts_12.jsonl" for d in dirs],
            what="frozen 12-carrier A artifacts",
        ),
        "recon_evals": _copy_first(
            out_root, "recon12_evaluations.jsonl", [d / "evaluations.jsonl" for d in recon], what="recon12 evaluations"
        ),
        "recon_check": _copy_first(
            out_root, "recon12_check_summary.json", [d / "check_summary.json" for d in recon], what="recon12 check"
        ),
        "nbound": _copy_first(
            out_root,
            "nbound_construction.json",
            [d / "results_stage_plan4_nbound" / "construction.json" for d in dirs],
            what="frozen woodland/library token sequences (nbound construction)",
        ),
    }


# ---------------------------------------------------------------- construction


def _prepare_heads(runner, prefixes: dict, sep_ids: list[int], generics: dict) -> dict:
    heads: dict[str, dict] = {}
    print("Preparing complete public prefixes (once per template and context)", flush=True)
    for template, entry in prefixes.items():
        heads[template] = {}
        for ctx in CONTEXTS:
            ids = entry[ctx]["ids"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pre, pre_s = runner.prefill_token_sequence(ids, source=f"public_{ctx}_{template}")
            head, sep_s = runner.extend_cache_with_tokens(pre, sep_ids, source=f"public_{ctx}_{template}_sep")
            rec = {
                "bundle": head,
                "ids": [int(t) for t in head.full_ids_cpu.view(-1).tolist()],
                "n_prefix": len(ids),
                "n_head": n_kv_positions(head.cache_cpu),
                "prefill_s": float(pre_s),
                "separator_s": float(sep_s),
                "prepare_s": float(pre_s + sep_s),
                "resident_mib": _mib(head.cache_cpu),
                "prefix_held": bool(cache_slice_delta(head.cache_cpu, pre.cache_cpu, list(range(len(ids))))["equal"]),
            }
            if ctx == "task":
                rec["prefill_equals_generic_template"] = bool(caches_equal(pre.cache_cpu, generics[template]["bundle"].cache_cpu))
            heads[template][ctx] = rec
            print(
                f"  {template} {ctx} n_prefix={rec['n_prefix']} n_head={rec['n_head']} "
                f"prepare={rec['prepare_s']:.2f}s {rec['resident_mib']:.2f} MiB",
                flush=True,
            )
    return heads


def _setup(runner, examples, artifacts, nbound_contexts):
    decode = _decode_fn(runner)
    tokenizer = runner.wrapper.tokenizer
    sep_ids, close_ids = _added_token_ids(tokenizer)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    generics = _prefill_templates(runner, templates)
    prefixes = neutral_prefixes(nbound_contexts, {t: rec["ids"] for t, rec in templates.items()})
    heads = _prepare_heads(runner, prefixes, list(sep_ids), generics)
    return {
        "decode": decode,
        "tokenizer": tokenizer,
        "sep_ids": list(sep_ids),
        "close_ids": list(close_ids),
        "templates": templates,
        "spans_by_key": spans_by_key,
        "generics": generics,
        "prefixes": prefixes,
        "heads": heads,
    }


def _result_piece(composed: dict) -> dict:
    pieces = [p for p in composed.get("pieces") or [] if p.get("kind") == "result"]
    if len(pieces) != 1:
        raise RuntimeError("Compact layout must have exactly one result piece.")
    return pieces[0]


def _case_state(runner, ex, rec, setup) -> dict:
    key = a_state_key(ex)
    spans = setup["spans_by_key"][key]
    t0 = time.perf_counter()
    pair = _pair_state_n(
        runner, rec, ex, setup["decode"], setup["sep_ids"], setup["close_ids"], spans, setup["generics"]
    )
    rebuild_s = time.perf_counter() - t0
    template = spans["template_id"]
    piece = _result_piece(pair["composed_extra"])
    result_pos = [int(p) for p in spans["result_pos"]]
    r_ids = [int(t) for t in piece["ids"]]
    caches = {BASELINE: pair["mixed"]["generic_r"], "full_kv": pair["bundle"]}
    assemble_s = {BASELINE: float(pair["times_by"].get("generic_r") or 0.0), "full_kv": 0.0, "text_result": 0.0}
    heads = setup["heads"][template]
    for cond in LONG_TRANSFERRED:
        head = heads[SPEC[cond]["context"]]["bundle"]
        t1 = time.perf_counter()
        caches[cond] = runner.concat_gathered_block(
            head, pair["bundle"], src_pos=result_pos, token_ids=r_ids, source=f"publicprefix_{cond}"
        )
        assemble_s[cond] = time.perf_counter() - t1
    fresh, fresh_s = runner.extend_cache_with_tokens(heads["task"]["bundle"], r_ids, source=f"publicprefix_{FRESH}")
    caches[FRESH] = fresh
    assemble_s[FRESH] = float(fresh_s)
    return {
        "key": key,
        "pair": pair,
        "spans": spans,
        "template": template,
        "result_pos": result_pos,
        "r_ids": r_ids,
        "piece": piece,
        "caches": caches,
        "assemble_s": assemble_s,
        "rebuild_s": rebuild_s,
    }


# ---------------------------------------------------------------- checks


def _selection_check(state, rec, recon_ref: dict | None) -> dict:
    full_ids = [int(t) for t in rec["full_ids"]]
    pos = state["result_pos"]
    row = {
        "n_result": len(pos),
        "ids_match_record": state["r_ids"] == [full_ids[p] for p in pos],
        "source_pos_match_layout": [int(p) for p in state["piece"]["source_pos"]] == pos,
        "positions_increasing": pos == sorted(set(pos)),
        "result_text_nonempty": bool((state["spans"].get("result_text") or "").strip()),
        "n_result_matches_recon12": recon_ref is not None and int(recon_ref.get("n_result") or -1) == len(pos),
        "baseline_result_start_70": int(state["pair"]["slots_extra"]["result_start"]) == 70,
    }
    row["ok"] = all(row.values())
    return row


def _bundle_layout(bundle) -> dict:
    return {
        "ids": [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()],
        "n_kv": n_kv_positions(bundle.cache_cpu),
        "cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
        "mask": [int(t) for t in bundle.past_mask_cpu.view(-1).tolist()],
    }


def _matched_check(state, setup) -> dict:
    template = state["template"]
    heads = setup["heads"][template]
    prefixes = setup["prefixes"][template]
    n_head = heads["task"]["n_head"]
    n_r = len(state["r_ids"])
    slots = packed_result_slots(n_head, n_r)
    lay = {c: _bundle_layout(state["caches"][c]) for c in LONG}
    sep = setup["sep_ids"]
    task_ids = lay[CANDIDATE]["ids"]
    p = prefixes["task"]
    row = {
        "n_head": {c: heads[SPEC[c]["context"]]["n_head"] for c in LONG},
        "lengths_equal": len({lay[c]["n_kv"] for c in LONG}) == 1 and lay[CANDIDATE]["n_kv"] == n_head + n_r,
        "cursors_equal": len({lay[c]["cursor"] for c in LONG}) == 1 and lay[CANDIDATE]["cursor"] == n_head + n_r,
        "masks_all_ones_equal": all(lay[c]["mask"] == [1] * (n_head + n_r) for c in LONG),
        "result_ids_equal": all(lay[c]["ids"][n_head:] == state["r_ids"] for c in LONG),
        "separator_equal": all(lay[c]["ids"][n_head - len(sep) : n_head] == sep for c in LONG),
        "header_equal": all(lay[c]["ids"][: p["header_len"]] == task_ids[: p["header_len"]] for c in LONG),
        "instruction_equal": all(
            lay[c]["ids"][p["instruction_start"] : p["instruction_end"]]
            == task_ids[p["instruction_start"] : p["instruction_end"]]
            for c in LONG
        ),
        "task_prefix_is_a_prefix": task_ids[: p["instruction_end"]] == p["ids"] and lay[FRESH]["ids"] == task_ids,
        "single_chat_header": all(lay[c]["ids"].count(task_ids[0]) == 1 for c in LONG),
        "baseline_separator_equal": [
            int(t)
            for piece in state["pair"]["composed_extra"]["pieces"]
            if piece.get("kind") == EXTRA_SEP_KIND
            for t in piece["ids"]
        ]
        == sep,
        "result_start": n_head,
        "result_slots": [slots[0], slots[-1]] if slots else [],
    }
    row["ok"] = all(v for k, v in row.items() if k not in ("n_head", "result_start", "result_slots"))
    return row


def _transferred_check(runner, state) -> dict:
    a_cache = state["pair"]["bundle"].cache_cpu
    pos = state["result_pos"]
    n_head = n_kv_positions(state["caches"][CANDIDATE].cache_cpu) - len(pos)
    slots = packed_result_slots(n_head, len(pos))
    src = gather_cache_positions(a_cache, pos)
    expected = rerotate_cache_keys(src, old_positions=pos, new_positions=slots, model=runner.wrapper.model)
    per = {}
    for cond in LONG_TRANSFERRED:
        got = gather_cache_positions(state["caches"][cond].cache_cpu, slots)
        per[cond] = {
            "values_equal_sender": all(torch.equal(gv, sv) for (_gk, gv), (_sk, sv) in zip(got["layers"], src["layers"])),
            "keys_equal_single_rotation": all(
                torch.equal(gk, ek) for (gk, _gv), (ek, _ev) in zip(got["layers"], expected["layers"])
            ),
        }
    base = state["caches"][BASELINE]
    base_slots = [int(s) for s in state["pair"]["slots_extra"]["result"]]
    base_r = gather_cache_positions(base.cache_cpu, base_slots)
    task_r = gather_cache_positions(state["caches"][CANDIDATE].cache_cpu, slots)
    key_shift = max(
        float((bk.float() - tk.float()).abs().max().item()) for (bk, _bv), (tk, _tv) in zip(base_r["layers"], task_r["layers"])
    )
    pair_eq = {
        f"{a}_vs_{b}": bool(cache_slice_delta(state["caches"][a].cache_cpu, state["caches"][b].cache_cpu, slots)["equal"])
        for a, b in ((CANDIDATE, "woodland_prefix_r"), (CANDIDATE, "library_prefix_r"))
    }
    fresh_vs = cache_slice_delta(state["caches"][FRESH].cache_cpu, state["caches"][CANDIDATE].cache_cpu, slots)
    row = {
        "per_condition": per,
        "task_neutral_result_identical": pair_eq,
        "values_equal_baseline": all(
            torch.equal(bv, tv) for (_bk, bv), (_tk, tv) in zip(base_r["layers"], task_r["layers"])
        ),
        "keys_max_abs_vs_baseline": key_shift,
        "fresh_vs_transferred": {
            "equal": bool(fresh_vs["equal"]),
            "keys_max_abs": fresh_vs["keys_max_abs"],
            "values_max_abs": fresh_vs["values_max_abs"],
        },
    }
    row["ok"] = bool(
        all(all(v.values()) for v in per.values()) and all(pair_eq.values()) and row["values_equal_baseline"]
    )
    return row


def _support_check(state, setup) -> dict:
    heads = setup["heads"][state["template"]]
    out = {}
    for cond in LONG:
        head = heads[SPEC[cond]["context"]]
        n = head["n_head"]
        out[cond] = bool(cache_slice_delta(state["caches"][cond].cache_cpu, head["bundle"].cache_cpu, list(range(n)))["equal"])
    return {"head_held": out, "ok": all(out.values())}


def _cont_ids(runner, ex) -> list[int]:
    _rendered, input_ids, _mask = runner._encode(_prompts_for(ex)["kv_plan"])
    ids = [int(t) for t in input_ids.view(-1).tolist()]
    return ids[: min(CONT_LEN, len(ids))]


def _logit_check(runner, bundle, cont: list[int]) -> dict:
    ids = [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()]
    res = runner.compare_prefix_logits(
        torch.tensor(ids + cont, dtype=torch.long),
        prefix_len=len(ids),
        continuation_len=len(cont),
        completed_cache=bundle.cache_cpu,
    )
    keep = ("repeat", "ordinary_cached_vs_uncached", "completed_gather_vs_reencode", "passed", "prefix_len", "continuation_len")
    return {k: res.get(k) for k in keep}


def _render(setup, bundle, b_turn_ids: list[int]) -> dict:
    view = render_handoff_view(setup["tokenizer"], bundle.full_ids_cpu.view(-1).tolist(), b_turn_ids)
    return {k: view[k] for k in ("prefix_text", "b_turn_text", "n_prefix", "n_b_turn", "junction")}


def _template_checks(runner, setup, examples, artifacts) -> dict:
    out = {}
    first = {}
    for ex in examples:
        first.setdefault(setup["spans_by_key"][a_state_key(ex)]["template_id"], ex)
    decode = setup["decode"]
    for template, entry in setup["prefixes"].items():
        ex = first[template]
        a_ids = artifacts[a_state_key(ex)]["full_ids"]
        cont = _cont_ids(runner, ex)
        rows = {}
        for ctx in CONTEXTS:
            ids = entry[ctx]["ids"]
            text = decode(ids)
            head = setup["heads"][template][ctx]
            rows[ctx] = {
                "n_prefix": len(ids),
                "decoded_prefix": text,
                "tokens": [[int(t), decode([int(t)])] for t in ids],
                "private_hits": private_prefix_hits(text, examples),
                "prefix_reuse": _logit_check(runner, head["bundle"], cont),
                "prepare_s": head["prepare_s"],
                "resident_mib": head["resident_mib"],
                "prefix_held": head["prefix_held"],
            }
            if ctx == "task":
                rows[ctx]["boundary"] = boundary_record(ids, a_ids, decode)
                rows[ctx]["prefill_equals_generic_template"] = head.get("prefill_equals_generic_template")
            else:
                rows[ctx]["matched"] = entry[ctx]["matched"]
                rows[ctx]["decoded_join"] = entry[ctx].get("decoded_join")
        task_b = rows["task"]["boundary"]
        ok = (
            all(not r["private_hits"] for r in rows.values())
            and all(bool(r["prefix_reuse"]["passed"]) for r in rows.values())
            and all(r["prefix_held"] for r in rows.values())
            and task_b["prefix_matches_a"]
            and task_b["ends_with_paragraph_break"]
            and task_b["no_merge_across_boundary"]
            and all(rows[c]["matched"]["ok"] for c in ("woodland", "library"))
        )
        out[template] = {"ok": bool(ok), "reuse_example": ex.example_id, "contexts": rows}
        print(
            f"  template {template}: ok={ok} private={[r['private_hits'] for r in rows.values()]} "
            f"reuse={[r['prefix_reuse']['passed'] for r in rows.values()]} next_A={task_b['next_a_text']!r}",
            flush=True,
        )
    return out


def _case_check(runner, ex, rec, state, setup, recon_ref) -> dict:
    sel = _selection_check(state, rec, recon_ref)
    matched = _matched_check(state, setup)
    transferred = _transferred_check(runner, state)
    support = _support_check(state, setup)
    cont = _cont_ids(runner, ex)
    fresh_ref = _logit_check(runner, state["caches"][FRESH], cont)
    _rendered, b_ids, _m = runner._encode(_prompts_for(ex)["kv_plan"])
    b_turn = [int(t) for t in b_ids.view(-1).tolist()]
    renders = {c: _render(setup, state["caches"][c], b_turn) for c in (BASELINE,) + LONG}
    junction_ok = all(not r["junction"]["has_im_end"] and r["junction"]["starts_im_start_user"] for r in renders.values())
    text_prompt = _prompts_for(ex)["text_plan"](ex, state["spans"].get("result_text") or "")
    renders["text_result"] = {"rendered_plan_prompt": runner._encode(text_prompt)[0]}
    ok = bool(sel["ok"] and matched["ok"] and transferred["ok"] and support["ok"] and fresh_ref["passed"] and junction_ok)
    return {
        "a_state_key": state["key"],
        "template_id": state["template"],
        "ok": ok,
        "selection": sel,
        "matched": matched,
        "transferred": transferred,
        "support": support,
        "fresh_reference": fresh_ref,
        "junction_ok": junction_ok,
        "renders": renders,
        "n_kv": {c: n_kv_positions(state["caches"][c].cache_cpu) for c in state["caches"]},
        "resident_mib": {c: _mib(state["caches"][c].cache_cpu) for c in state["caches"]},
        "assemble_s": state["assemble_s"],
        "rebuild_s": state["rebuild_s"],
    }


def _slim_heads(heads: dict) -> dict:
    return {
        t: {c: {k: v for k, v in rec.items() if k not in ("bundle", "ids")} for c, rec in ctxs.items()}
        for t, ctxs in heads.items()
    }


def _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, nbound, recon_refs) -> int:
    setup = _setup(runner, examples, artifacts, nbound)
    template_rows = _template_checks(runner, setup, examples, artifacts)
    rows = []
    for i, ex in enumerate(examples):
        rec = artifacts[a_state_key(ex)]
        state = _case_state(runner, ex, rec, setup)
        row = _case_check(runner, ex, rec, state, setup, recon_refs.get(state["key"]))
        rows.append(row)
        print(
            f"  [{i + 1}/{len(examples)}] {row['a_state_key']} ok={row['ok']} sel={row['selection']['ok']} "
            f"matched={row['matched']['ok']} R={row['transferred']['ok']} support={row['support']['ok']} "
            f"fresh_ref={row['fresh_reference']['passed']} junction={row['junction_ok']} "
            f"n_kv={row['n_kv'][CANDIDATE]} (baseline {row['n_kv'][BASELINE]})",
            flush=True,
        )
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_passed = bool(rows) and all(r["ok"] for r in rows) and all(t["ok"] for t in template_rows.values())
    construction = {
        "frozen_placement": FROZEN_PLACEMENT,
        "separator_ids": setup["sep_ids"],
        "templates": template_rows,
        "heads": _slim_heads(setup["heads"]),
        "cases": rows,
    }
    _write_json(out_root / "construction.json", construction)
    summary = {
        "n": len(rows),
        "all_passed": all_passed,
        "templates_ok": {t: r["ok"] for t, r in template_rows.items()},
        "cases": [
            {
                "a_state_key": r["a_state_key"],
                "ok": r["ok"],
                "failed": [
                    k for k in ("selection", "matched", "transferred", "support") if not r[k]["ok"]
                ]
                + ([] if r["fresh_reference"]["passed"] else ["fresh_reference"])
                + ([] if r["junction_ok"] else ["junction"]),
                "n_kv": r["n_kv"],
                "keys_max_abs_vs_baseline": r["transferred"]["keys_max_abs_vs_baseline"],
                "fresh_vs_transferred": r["transferred"]["fresh_vs_transferred"],
            }
            for r in rows
        ],
        "frozen_placement": FROZEN_PLACEMENT,
    }
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({"n": summary["n"], "all_passed": all_passed, "templates_ok": summary["templates_ok"]}, indent=2))
    return 0 if all_passed else 1


# ---------------------------------------------------------------- compare


def _row_meta(rec: dict, ex, state, setup, cond: str, bundle) -> dict:
    spec = SPEC[cond]
    template = state["template"]
    head = setup["heads"][template].get(spec["context"]) if spec["context"] else None
    rec.update(
        {
            "condition": cond,
            "context": spec["context"],
            "results": spec["results"],
            "layout_description": spec["layout"],
            "a_state_key": state["key"],
            "template_id": template,
            "n_carriers": N_CARRIERS_12,
            "n_result": len(state["r_ids"]),
            "assemble_s": float(state["assemble_s"].get(cond) or 0.0),
            "a_rebuild_s": float(state["rebuild_s"]),
            "prefix_prepare_s": None if head is None else head["prepare_s"],
            "prefix_resident_mib": None if head is None else head["resident_mib"],
        }
    )
    if bundle is not None:
        rec.update(
            {
                "n_kv_positions": n_kv_positions(bundle.cache_cpu),
                "resident_mib": _mib(bundle.cache_cpu),
                "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
                "result_start": (
                    int(state["pair"]["slots_extra"]["result_start"])
                    if cond == BASELINE
                    else (head["n_head"] if head is not None else None)
                ),
            }
        )
    else:
        rec.update({"n_kv_positions": 0, "resident_mib": 0.0, "message_text": state["spans"].get("result_text") or ""})
    return rec


def _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, nbound, recon_refs) -> int:
    cpath = out_root / "check_summary.json"
    check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
    if check is None or not check.get("all_passed"):
        print("Run --phase check first; it must pass before any answers are generated.")
        return 2
    setup = _setup(runner, examples, artifacts, nbound)
    head_snap = {(t, c): runner.clone_sender(r["bundle"]) for t, ctxs in setup["heads"].items() for c, r in ctxs.items()}
    path = out_root / "evaluations.jsonl"
    rows = _load_rows(path)
    done = {(r["example_id"], r["condition"]) for r in rows}
    todo = [ex for ex in examples if any((ex.example_id, c) not in done for c in CONDS)]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]
    print(f"Compare: {len(done)}/{len(examples) * len(CONDS)} done, {len(todo)} cases to run", flush=True)
    for i, ex in enumerate(todo):
        rec = artifacts[a_state_key(ex)]
        state = _case_state(runner, ex, rec, setup)
        quick = (_selection_check(state, rec, recon_refs.get(state["key"])), _matched_check(state, setup))
        if not all(q["ok"] for q in quick) or not _transferred_check(runner, state)["ok"]:
            raise RuntimeError(f"Construction check failed for {ex.example_id}.")
        snaps = {c: runner.clone_sender(b) for c, b in state["caches"].items()}
        reset_peak_memory()
        for cond in CONDS:
            if (ex.example_id, cond) in done:
                continue
            if SPEC[cond]["kind"] == "text":
                out = _run_text_b_timed(
                    runner,
                    ex,
                    message=state["spans"].get("result_text") or "",
                    plan_max_new=args.plan_max_new_12,
                    answer_max_new=args.max_new_tokens,
                )
                bundle = None
            else:
                bundle = state["caches"][cond]
                out = _run_kv_b_timed(
                    runner,
                    ex,
                    runner.clone_sender(bundle),
                    plan_max_new=args.plan_max_new_12,
                    answer_max_new=args.max_new_tokens,
                )
            out = _row_meta(out, ex, state, setup, cond, bundle)
            rows.append(out)
            done.add((ex.example_id, cond))
            _append(path, out)
            print(
                f"  [{i + 1}/{len(todo)}] {ex.example_id} {cond}: labels={out.get('plan_n_status_correct')}/12 "
                f"all12={out.get('plan_all_status_correct')} I->E={out.get('plan_ineligible_marked_eligible')} "
                f"E->I={out.get('plan_eligible_marked_ineligible')} choice={out.get('correct')} "
                f"parse={out.get('parse_ok')}",
                flush=True,
            )
        for c, snap in snaps.items():
            if not caches_equal(snap.cache_cpu, state["caches"][c].cache_cpu):
                raise RuntimeError(f"{c} cache changed during generation for {ex.example_id}.")
        for (t, c), snap in head_snap.items():
            if not caches_equal(snap.cache_cpu, setup["heads"][t][c]["bundle"].cache_cpu):
                raise RuntimeError(f"Prepared {c} prefix for {t} changed during generation.")
        _persist(out_root, kaggle_dest)
        del state, snaps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _summarize(rows, check=check, recon_rows=_load_rows(out_root / "recon12_evaluations.jsonl"))
    _write_json(out_root / "summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({c: summary["labels"][c] for c in CONDS}, indent=2))
    return 0


# ---------------------------------------------------------------- summary


def _case_counts(rows: list[dict], field: str) -> dict:
    return {r["example_id"]: int(r.get(field) or 0) for r in rows}


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _costs(rows: list[dict]) -> dict:
    return {
        "mean_n_kv_positions": _mean(r.get("n_kv_positions") for r in rows),
        "mean_resident_mib": _mean(r.get("resident_mib") for r in rows),
        "mean_assemble_s": _mean(r.get("assemble_s") for r in rows),
        "mean_plan_prefill_s": _mean(r.get("plan_prefill_s") for r in rows),
        "mean_plan_decode_s": _mean(r.get("plan_decode_s") for r in rows),
        "mean_answer_prefill_s": _mean(r.get("answer_prefill_s") for r in rows),
        "mean_answer_decode_s": _mean(r.get("answer_decode_s") for r in rows),
        "mean_receiver_prefill_s": _mean(r.get("b_prefill_s") for r in rows),
        "mean_receiver_decode_s": _mean(r.get("b_decode_s") for r in rows),
        "prefix_prepare_s_by_template": {
            r.get("template_id"): r.get("prefix_prepare_s") for r in rows if r.get("prefix_prepare_s") is not None
        },
        "prefix_resident_mib_by_template": {
            r.get("template_id"): r.get("prefix_resident_mib") for r in rows if r.get("prefix_resident_mib") is not None
        },
    }


def _reproduction(rows: list[dict], recon_rows: list[dict]) -> dict:
    ref = {(r["example_id"], r["condition"]): r for r in recon_rows}
    out = {}
    for cond, ref_cond in REFERENCE_CONDS.items():
        mine = [r for r in rows if r["condition"] == cond]
        pairs = [(r, ref.get((r["example_id"], ref_cond))) for r in mine]
        pairs = [(a, b) for a, b in pairs if b is not None]
        out[cond] = {
            "reference": f"recon12 {ref_cond}",
            "n_compared": len(pairs),
            "plan_text_equal": sum((a.get("plan_text") or "") == (b.get("plan_text") or "") for a, b in pairs),
            "answer_text_equal": sum((a.get("raw_text") or "") == (b.get("raw_text") or "") for a, b in pairs),
            "labels_here": sum(int(a.get("plan_n_status_correct") or 0) for a, _ in pairs),
            "labels_reference": sum(int(b.get("plan_n_status_correct") or 0) for _, b in pairs),
            "all12_here": sum(bool(a.get("plan_all_status_correct")) for a, _ in pairs),
            "all12_reference": sum(bool(b.get("plan_all_status_correct")) for _, b in pairs),
        }
    return out


def _summarize(rows: list[dict], *, check: dict | None = None, recon_rows: list[dict] | None = None) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    maps = {c: _twelve_label_map(by_cond[c]) for c in CONDS}
    paired = {}
    for name, left, right in PAIRED:
        paired[name] = {
            "left": left,
            "right": right,
            "all_twelve_correct": _cluster_delta(
                _case_counts(by_cond[left], "plan_all_status_correct"),
                _case_counts(by_cond[right], "plan_all_status_correct"),
            ),
            "labels_correct": _cluster_delta(
                _case_counts(by_cond[left], "plan_n_status_correct"),
                _case_counts(by_cond[right], "plan_n_status_correct"),
            ),
            "label_changes": _label_crosstab(maps[left], maps[right]),
        }
    return {
        "n_rows": len(rows),
        "n_cases": len({r["example_id"] for r in rows}),
        "n_expected_rows": N_TIMING_FAMILIES * len(CONDS),
        "conditions": SPEC,
        "primary": "all twelve eligibility labels correct per case (out of 9)",
        "labels": {c: _twelve_labels(by_cond[c]) for c in CONDS},
        "final_choice": {c: _twelve_choice(by_cond[c]) for c in CONDS},
        "vs_baseline_label_changes": {c: _label_crosstab(maps[BASELINE], maps[c]) for c in CONDS if c != BASELINE},
        "paired_by_case": paired,
        "by_difficulty": {
            lvl: {c: _twelve_labels([r for r in by_cond[c] if r.get("difficulty") == lvl]) for c in CONDS} for lvl in LEVELS
        },
        "costs": {c: _costs(by_cond[c]) for c in CONDS},
        "reproduction_vs_recon12": _reproduction(rows, recon_rows or []),
        "meets_recovery_standard": {
            c: _twelve_labels(by_cond[c]).get("cases_all_twelve_correct") == N_TIMING_FAMILIES for c in LONG
        },
        "check": None if check is None else {"all_passed": check.get("all_passed"), "n": check.get("n")},
        "stats_note": (
            f"Paired bootstrap over the nine cases ({N_BOOT} draws, seed {BOOT_SEED}); each case's twelve "
            "labels stay together. The 108 labels are not independent examples."
        ),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "cost_note": COST_NOTE,
        "scope_note": (
            "Nine already-inspected cases: a development diagnostic, not independent confirmation. "
            "Final choices are secondary; B makes financial-decision errors even when full KV gives every label."
        ),
        "frozen_placement": FROZEN_PLACEMENT,
    }


# ---------------------------------------------------------------- main


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Complete public task prefix versus the instruction slice.")
    parser.add_argument("--split", default="plan4_timing12")
    parser.add_argument("--phase", choices=("check", "compare", "all", "rescore"), default="check")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.split != "plan4_timing12":
        print("This diagnostic uses only the nine frozen timing12 cases.")
        return 2
    if args.max_new_tokens != 48 or args.plan_max_new_12 != 384:
        print("Decoding allowances are frozen at 384 plan / 48 answer tokens.")
        return 2

    out_root, kaggle_dest = _paths()
    if "stage_plan4_publicprefix12" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_publicprefix12.")
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    for folder in [d for d in (kaggle_dest, REPO_ROOT / "results_stage_plan4_publicprefix12") if d is not None]:
        for name in PERSIST_NAMES:
            src, dest = folder / name, out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    examples = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    for ex in examples:
        assert_timing12_invariants(ex)
    if len(examples) != N_TIMING_FAMILIES:
        print(f"Expected {N_TIMING_FAMILIES} twelve-carrier sheets, got {len(examples)}.")
        return 2

    if args.phase == "rescore":
        cpath = out_root / "check_summary.json"
        check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
        summary = _summarize(
            _load_rows(out_root / "evaluations.jsonl"),
            check=check,
            recon_rows=_load_rows(out_root / "recon12_evaluations.jsonl"),
        )
        _write_json(out_root / "summary.json", summary)
        _persist(out_root, kaggle_dest)
        print(json.dumps({c: summary["labels"][c] for c in CONDS}, indent=2))
        return 0

    inputs = _copy_inputs(out_root)
    artifacts = _load_messages(inputs["artifacts"])
    if any(a_state_key(ex) not in artifacts for ex in examples):
        print("Frozen 12-carrier A artifacts are missing. Do not regenerate A.")
        return 2
    nbound = json.loads(Path(inputs["nbound"]).read_text(encoding="utf-8"))["contexts"]
    recon_check = json.loads(Path(inputs["recon_check"]).read_text(encoding="utf-8"))
    recon_refs = {r["a_state_key"]: r for r in recon_check.get("rows") or []}

    policy, runner = _prepare_runner(
        args,
        out_root,
        extra={
            "stage": "plan4-publicprefix12",
            "phase": args.phase,
            "conditions": list(CONDS),
            "do_not_overwrite": list(PRESERVED),
            "frozen_placement": FROZEN_PLACEMENT,
            "interpretation_guide": INTERPRETATION_GUIDE,
            "cost_note": COST_NOTE,
            "protocol": pinned.get("study_plan4_publicprefix12") or {},
        },
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase} n={len(examples)}")
    if args.phase == "check":
        return _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, nbound, recon_refs)
    if args.phase == "compare":
        return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, nbound, recon_refs)
    rc = _run_check(args, out_root, kaggle_dest, examples, runner, artifacts, nbound, recon_refs)
    if rc != 0:
        print("Construction check failed; stopping before any answers are generated.")
        return rc
    return _run_compare(args, out_root, kaggle_dest, examples, runner, artifacts, nbound, recon_refs)


if __name__ == "__main__":
    raise SystemExit(main())
