"""Query-adapter pilot, preparation only (no adapter is trained here).

Pilot question: can a small receiver-side query adjustment improve B's use of
the transferred cache when no reference model supplies attention or answers?
This stage prepares the data the later stages need, under the frozen handoff
construction (generic support, result selection, positions, prompts):

- data:    write train / dev / eval families (new seeds, disjoint from the nine
           diagnostic cases); no model needed
- write:   A's ordinary written pass for train and dev sheets only. Eval A
           sheets are not generated until a candidate is frozen.
- check:   templates must equal the frozen timing12 templates; every sheet must
           build generic support + transferred R (tk_tv) and the reconstructed
           reference (fk_fv) with unchanged support, IDs, positions and masks
- teacher: on train base sheets, the reference fk_fv plan + answer (training
           targets, unfiltered) and ordinary tk_tv (the gap before training)

The reference is used only to make training targets. The nine timing12 cases
stay diagnostic examples and are never trained on or used for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cache_utils import cache_slice_delta, n_kv_positions
from src.manifest import load_pinned
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_qadapt_task import (
    QADAPT_SPLITS,
    assert_disjoint,
    changed_carrier,
    ensure_split,
    file_sha256,
    mismatch_partner,
    split_path,
)
from src.plan4_task import LEVELS, a_state_key
from src.plan4_timing_task import N_CARRIERS_12, gold_eligibility_statuses_n, load_timing12_jsonl
from src.precision import reset_peak_memory
from src.prompts_plan4 import ordinary_sender_prompt, prompt_contains_b_private
from src.prompts_plan4_timing import encode_prompt_for, written_result_handoff_n
from src.span_select import caches_equal
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages
from src.stage_plan4_boundary import _added_token_ids
from src.stage_plan4_generic import _prefill_templates
from src.stage_plan4_kvmix12 import (
    _append,
    _attach_mixes,
    _prepare_runner,
    _twelve_choice,
    _twelve_labels,
    _twelve_row,
)
from src.stage_plan4_query12 import _build_state, _check_construction, _copy_first
from src.stage_plan4_span import _decode_fn
from src.stage_plan4_timing import (
    A_WRITTEN_MAX_NEW_12,
    B_PLAN_MAX_NEW_12,
    RESULT_LINE_N,
    _collect_templates_n,
    _score_sender_n,
    _write_artifacts12,
    prompt_contains_solver_labels_n,
)
from src.parse_plan4 import written_working_metrics
from src.stage_select import _load_rows

STUDENT = "tk_tv"
TEACHER = "fk_fv"
A_SPLITS = ("train", "dev")
PRESERVED = (
    "results_stage_plan4_timing",
    "results_stage_plan4_recon12",
    "results_stage_plan4_query12",
    "results_stage_plan4_kvmix12",
    "results_stage_plan4_refattn12",
    "results_stage_plan4_refgen12",
    "results_stage_plan4_refval12",
    "results_stage_plan4_refgenval12",
    "results_stage_plan4_confirm",
    "results_stage_plan4_hold",
)
DO_NOT_LOAD = (
    "data/plan4_dev.jsonl",
    "data/plan_dev.jsonl",
    "data/plan4_fresh.jsonl",
    "data/plan4_hold.jsonl",
    "data/test.jsonl",
)
PERSIST_NAMES = (
    "manifest.json",
    "data_summary.json",
    "a_artifacts_train.jsonl",
    "a_artifacts_dev.jsonl",
    "timing12_a_artifacts_12.jsonl",
    "check_summary.json",
    "teacher_train.jsonl",
    "teacher_summary.json",
)
ROLE_NOTE = (
    "Preparation only. fk_fv rows are training targets for the adapter (reference "
    "used during training only); tk_tv rows show the gap on training sheets before "
    "any training. Neither is a pilot result, and dev/eval sheets are not run here."
)
SUPPORT_NOTE = (
    "Support token IDs must be identical within a template (required). Support "
    "tensors are compared with the template's first sheet and reported only: the "
    "opening positions are re-prefilled per sheet inside sequences of different "
    "lengths, so FP16 kernels may round differently. Mismatched R is built from "
    "each sheet's own support plus the partner's R, so it does not rely on "
    "byte-identical support across sheets."
)


def _paths():
    out_root = RESULTS_DIR / "stage_plan4_qadapt_prep" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_qadapt_prep")
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


def _load_splits() -> dict:
    return {name: ensure_split(name) for name in QADAPT_SPLITS}


def _data_summary(splits: dict, timing12) -> dict:
    assert_disjoint(splits, timing12)
    out = {"disjoint_from_each_other_and_timing12": True, "splits": {}}
    for name, examples in splits.items():
        bases = [ex for ex in examples if ex.variant == "base"]
        flips = [ex for ex in examples if ex.variant == "a_flip"]
        by_id = {ex.family_id: ex for ex in bases}
        out["splits"][name] = {
            "file": str(split_path(name).relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": file_sha256(split_path(name)),
            "seed": QADAPT_SPLITS[name]["seed"],
            "n_examples": len(examples),
            "n_base": len(bases),
            "n_a_flip": len(flips),
            "by_level": {lvl: sum(ex.difficulty == lvl for ex in bases) for lvl in LEVELS},
            "a_change": {
                kind: sum(ex.a_change == kind for ex in flips) for kind in ("make_eligible", "make_ineligible")
            },
            "a_flip_changed_carrier": {
                ex.example_id: changed_carrier(by_id[ex.family_id], ex) for ex in flips
            },
            "mismatch_partner": mismatch_partner(examples) if flips else {},
            "a_sheets_written_now": name in A_SPLITS,
        }
    return out


def registered_hash_mismatches(summary: dict, protocol: dict) -> list[str]:
    registered = (protocol.get("data") or {}) if protocol else {}
    return [
        name
        for name, block in summary["splits"].items()
        if name in registered and registered[name].get("sha256") != block["sha256"]
    ]


def _run_data(out_root, kaggle_dest, splits, timing12, protocol) -> int:
    summary = _data_summary(splits, timing12)
    summary["registered_hash_mismatches"] = registered_hash_mismatches(summary, protocol)
    _write_json(out_root / "data_summary.json", summary)
    _persist(out_root, kaggle_dest)
    for name, block in summary["splits"].items():
        print(name, {k: block[k] for k in ("n_examples", "n_base", "n_a_flip", "by_level", "a_change", "sha256")})
    if summary["registered_hash_mismatches"]:
        print(f"Split files differ from the registered hashes: {summary['registered_hash_mismatches']}. Do not regenerate.")
        return 2
    return 0


def _a_record(ex, text, n_new, truncated, gen_s, bundle, ordinary) -> dict:
    return {
        "example_id": ex.example_id,
        "family_id": ex.family_id,
        "variant": ex.variant,
        "difficulty": ex.difficulty,
        "a_state_key": a_state_key(ex),
        "n_carriers": N_CARRIERS_12,
        "ordinary_prompt": ordinary,
        "written": text,
        "written_n_new": n_new,
        "written_truncated": truncated,
        "written_s": gen_s,
        "n_kv_positions": n_kv_positions(bundle.cache_cpu),
        "n_prompt_positions": int(bundle.prompt_mask_cpu.sum().item()),
        "full_relay_bytes": int(bundle.full_relay_bytes),
        "latent_steps": 0,
        "ids_source": bundle.ids_source,
        "token_align": dict(bundle.token_align or {}),
        "full_ids": bundle.full_ids_cpu.tolist(),
        "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
        "handoff": "\n".join(m.group(0).strip() for m in RESULT_LINE_N.finditer(text or "")),
        "written_score": _score_sender_n(text, ex),
        "written_working": written_working_metrics(text),
    }


def _run_write(args, out_root, kaggle_dest, splits, runner) -> int:
    for name in A_SPLITS:
        path = out_root / f"a_artifacts_{name}.jsonl"
        artifacts = _load_messages(path)
        todo = [ex for ex in splits[name] if a_state_key(ex) not in artifacts]
        if args.limit is not None:
            todo = todo[: max(0, args.limit)]
        print(f"A write {name}: {len(artifacts)} written, {len(todo)} to do", flush=True)
        for i, ex in enumerate(todo):
            key = a_state_key(ex)
            ordinary = ordinary_sender_prompt(encode_prompt_for(ex), written_result_handoff_n(ex.carriers))
            if prompt_contains_b_private(ordinary, ex) or prompt_contains_solver_labels_n(ordinary, ex):
                raise RuntimeError(f"A prompt leaked private fields for {key}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            text, n_new, truncated, gen_s, bundle = runner.generate_text_keep_cache(
                ordinary, max_new_tokens=A_WRITTEN_MAX_NEW_12, source="written"
            )
            rec = _a_record(ex, text, n_new, truncated, gen_s, bundle, ordinary)
            artifacts[key] = rec
            _write_artifacts12(path, artifacts)
            _persist(out_root, kaggle_dest)
            print(
                f"  [{i + 1}/{len(todo)}] {key} trunc={truncated} n_new={n_new} "
                f"status={rec['written_score']['n_status_correct']}/{N_CARRIERS_12} s={gen_s:.1f}",
                flush=True,
            )
            del bundle
    return 0


def _support_sha(bundle, n_support: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([int(t) for t in bundle.full_ids_cpu.view(-1).tolist()[:n_support]]).encode("utf-8"))
    for k, v in bundle.cache_cpu["layers"]:
        h.update(k[..., :n_support, :].contiguous().numpy().tobytes())
        h.update(v[..., :n_support, :].contiguous().numpy().tobytes())
    return h.hexdigest()


def _support_ids(bundle, n_support: int) -> list[int]:
    return [int(t) for t in bundle.full_ids_cpu.view(-1).tolist()[:n_support]]


def _support_vs_ref(bundle, n_support: int, ref: dict) -> dict:
    ids_equal = n_support == ref["result_start"] and _support_ids(bundle, n_support) == ref["ids"]
    out = {"reference": ref["a_state_key"], "ids_equal": bool(ids_equal)}
    if ids_equal:
        d = cache_slice_delta(bundle.cache_cpu, ref["cache"], list(range(n_support)))
        out.update(
            {
                "bytes_equal": bool(d["equal"]),
                "keys_max_abs": d["keys_max_abs"],
                "values_max_abs": d["values_max_abs"],
                "keys_mean_abs": d["keys_mean_abs"],
                "values_mean_abs": d["values_mean_abs"],
            }
        )
    return out


def support_within_template(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for row in rows:
        s = row["support_vs_template_first"]
        b = out.setdefault(
            row["construction"]["template_id"],
            {"n": 0, "ids_equal": 0, "bytes_equal": 0, "keys_max_abs": 0.0, "values_max_abs": 0.0},
        )
        b["n"] += 1
        b["ids_equal"] += int(s["ids_equal"])
        b["bytes_equal"] += int(bool(s.get("bytes_equal")))
        b["keys_max_abs"] = max(b["keys_max_abs"], float(s.get("keys_max_abs") or 0.0))
        b["values_max_abs"] = max(b["values_max_abs"], float(s.get("values_max_abs") or 0.0))
    return out


def compare_templates(new: dict, frozen: dict) -> dict:
    names_equal = sorted(new) == sorted(frozen)
    per = {}
    for name in sorted(set(new) | set(frozen)):
        a, b = new.get(name), frozen.get(name)
        per[name] = {
            "present_in_both": a is not None and b is not None,
            "ids_equal": a is not None and b is not None and list(a["ids"]) == list(b["ids"]),
            "instruction_equal": a is not None
            and b is not None
            and list(a["instruction_ids"]) == list(b["instruction_ids"])
            and int(a["instruction_start"]) == int(b["instruction_start"]),
            "n_new": None if a is None else a["n"],
        }
    ok = names_equal and all(r["present_in_both"] and r["ids_equal"] and r["instruction_equal"] for r in per.values())
    return {"ok": bool(ok), "names_equal": names_equal, "templates": per}


def _a_flip_sender(ex, base, rec) -> dict | None:
    if ex.variant != "a_flip":
        return None
    carrier = changed_carrier(base, ex)
    pred = None
    for m in RESULT_LINE_N.finditer(rec.get("written") or ""):
        if m.group("carrier").upper() == carrier:
            pred = m.group("status").lower()
    return {
        "changed_carrier": carrier,
        "gold": gold_eligibility_statuses_n(ex)[carrier],
        "a_wrote": pred,
        "a_correct": pred == gold_eligibility_statuses_n(ex)[carrier],
    }


def _setup(runner, examples, artifacts, frozen_artifacts, timing12):
    decode = _decode_fn(runner)
    sep_ids, close_ids = _added_token_ids(runner.wrapper.tokenizer)
    templates, spans_by_key = _collect_templates_n(examples, artifacts, decode)
    frozen, _ = _collect_templates_n(timing12, frozen_artifacts, decode)
    match = compare_templates(templates, frozen)
    if not match["ok"]:
        raise RuntimeError(f"Pilot templates differ from the frozen timing12 templates: {match}")
    generics = _prefill_templates(runner, frozen)
    return decode, sep_ids, close_ids, spans_by_key, generics, match


def _state(runner, ex, artifacts, setup):
    decode, sep_ids, close_ids, spans_by_key, generics, _match = setup
    key = a_state_key(ex)
    t0 = time.perf_counter()
    state = _build_state(runner, ex, artifacts[key], decode, sep_ids, close_ids, spans_by_key[key], generics)
    built = _check_construction(state, None)
    built.pop("matches_recon12", None)
    mix = _attach_mixes(runner, state)
    return state, built, mix, time.perf_counter() - t0


def _examples_with_a(splits) -> list:
    return [ex for name in A_SPLITS for ex in splits[name]]


def _all_artifacts(out_root) -> dict:
    out = {}
    for name in A_SPLITS:
        out.update(_load_messages(out_root / f"a_artifacts_{name}.jsonl"))
    return out


def _run_check(args, out_root, kaggle_dest, splits, runner, frozen_artifacts, timing12) -> int:
    examples = _examples_with_a(splits)
    artifacts = _all_artifacts(out_root)
    missing = [a_state_key(ex) for ex in examples if a_state_key(ex) not in artifacts]
    if missing:
        print(f"{len(missing)} A sheets are missing (first: {missing[0]}). Run --phase write first.")
        return 2
    setup = _setup(runner, examples, artifacts, frozen_artifacts, timing12)
    bases = {ex.family_id: ex for ex in examples if ex.variant == "base"}
    split_of = {ex.example_id: name for name in A_SPLITS for ex in splits[name]}
    template_ref: dict[str, dict] = {}
    rows = []
    for i, ex in enumerate(examples):
        key = a_state_key(ex)
        state, built, mix, build_s = _state(runner, ex, artifacts, setup)
        rs = int(state["slots_extra"]["result_start"])
        rec = artifacts[key]
        student = state["kvmix"][STUDENT]
        ref = template_ref.setdefault(
            built["template_id"],
            {"a_state_key": key, "result_start": rs, "ids": _support_ids(student, rs), "cache": student.cache_cpu},
        )
        row = {
            "a_state_key": key,
            "split": split_of[ex.example_id],
            "variant": ex.variant,
            "difficulty": ex.difficulty,
            "build_s": build_s,
            "construction": built,
            "mixes_ok": bool(mix["ok"]),
            "n_kv": mix["n_kv"],
            "result_start": rs,
            "n_result": mix["n_result"],
            "support_sha256": {c: _support_sha(state["kvmix"][c], rs) for c in (STUDENT, TEACHER)},
            "support_vs_template_first": _support_vs_ref(student, rs, ref),
            "result_tail": mix["n_kv"] == rs + mix["n_result"],
            "a_written_status_correct": rec["written_score"]["n_status_correct"],
            "a_written_truncated": bool(rec["written_truncated"]),
            "a_flip_sender": _a_flip_sender(ex, bases[ex.family_id], rec),
        }
        row["ok"] = bool(
            built["support_held"]
            and mix["ok"]
            and row["result_tail"]
            and mix["n_result"] > 0
            and row["support_sha256"][STUDENT] == row["support_sha256"][TEACHER]
            and row["support_vs_template_first"]["ids_equal"]
        )
        rows.append(row)
        print(
            f"  [{i + 1}/{len(examples)}] {key} ok={row['ok']} n_kv={row['n_kv']} n_R={row['n_result']} "
            f"A={row['a_written_status_correct']}/12 build={build_s:.1f}s",
            flush=True,
        )
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    flips = [r["a_flip_sender"] for r in rows if r["a_flip_sender"]]
    summary = {
        "n": len(rows),
        "all_passed": bool(rows) and all(r["ok"] for r in rows),
        "templates_match_timing12": setup[-1],
        "support_within_template": support_within_template(rows),
        "support_note": SUPPORT_NOTE,
        "a_written": {
            name: {
                "n_sheets": sum(r["split"] == name for r in rows),
                "status_correct": sum(r["a_written_status_correct"] for r in rows if r["split"] == name),
                "n_status": N_CARRIERS_12 * sum(r["split"] == name for r in rows),
                "truncated": sum(r["a_written_truncated"] for r in rows if r["split"] == name),
            }
            for name in A_SPLITS
        },
        "a_flip_changed_carrier_written_correctly": {"n": len(flips), "correct": sum(f["a_correct"] for f in flips)},
        "rows": rows,
    }
    _write_json(out_root / "check_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0 if summary["all_passed"] else 1


def _teacher_summary(rows: list[dict]) -> dict:
    out = {"role_note": ROLE_NOTE}
    for cond in (STUDENT, TEACHER):
        sub = [r for r in rows if r["condition"] == cond]
        out[cond] = {
            "labels": _twelve_labels(sub),
            "final_choice": _twelve_choice(sub),
            "plan_time_s": sum(float(r.get("plan_time_s") or 0) for r in sub),
            "answer_time_s": sum(float(r.get("receiver_latency_s") or 0) for r in sub),
        }
    teach = {r["example_id"]: r for r in rows if r["condition"] == TEACHER}
    stud = {r["example_id"]: r for r in rows if r["condition"] == STUDENT}
    both = sorted(set(teach) & set(stud))
    out["paired"] = {
        "n_cases": len(both),
        "teacher_minus_student_labels": sum(
            int(teach[k].get("plan_n_status_correct") or 0) - int(stud[k].get("plan_n_status_correct") or 0)
            for k in both
        ),
        "cases_teacher_better": sum(
            int(teach[k].get("plan_n_status_correct") or 0) > int(stud[k].get("plan_n_status_correct") or 0)
            for k in both
        ),
        "cases_student_better": sum(
            int(teach[k].get("plan_n_status_correct") or 0) < int(stud[k].get("plan_n_status_correct") or 0)
            for k in both
        ),
    }
    out["teacher_targets"] = {
        "n": len(teach),
        "plan_truncated": sum(bool(r.get("plan_truncated")) for r in teach.values()),
        "answer_truncated": sum(bool(r.get("truncated")) for r in teach.values()),
        "plan_tokens": sum(int(r.get("n_plan_tokens") or 0) for r in teach.values()),
        "answer_tokens": sum(int(r.get("n_new_tokens") or 0) for r in teach.values()),
        "filtered": False,
    }
    return out


def _run_teacher(args, out_root, kaggle_dest, splits, runner, frozen_artifacts, timing12) -> int:
    cpath = out_root / "check_summary.json"
    check = json.loads(cpath.read_text(encoding="utf-8")) if cpath.is_file() else None
    if check is None or not check.get("all_passed"):
        print("Run --phase check first (it must pass).")
        return 2
    examples = _examples_with_a(splits)
    artifacts = _all_artifacts(out_root)
    setup = _setup(runner, examples, artifacts, frozen_artifacts, timing12)
    path = out_root / "teacher_train.jsonl"
    rows = _load_rows(path)
    done = {(r["example_id"], r["condition"]) for r in rows}
    train = [ex for ex in splits["train"] if ex.variant == "base"]
    todo = [ex for ex in train if any((ex.example_id, c) not in done for c in (TEACHER, STUDENT))]
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]
    print(f"Teacher: {len(done)} rows done, {len(todo)} train sheets to do", flush=True)
    for i, ex in enumerate(todo):
        state, built, mix, build_s = _state(runner, ex, artifacts, setup)
        if not (mix["ok"] and built["support_held"]):
            raise RuntimeError(f"Construction failed for {ex.example_id}.")
        snapshots = {c: runner.clone_sender(state["kvmix"][c]) for c in (TEACHER, STUDENT)}
        reset_peak_memory()
        for cond in (TEACHER, STUDENT):
            if (ex.example_id, cond) in done:
                continue
            rec = _twelve_row(runner, ex, state, cond, args)
            rec.update({"split": "train", "role": "target" if cond == TEACHER else "before_training", "build_s": build_s})
            rows.append(rec)
            done.add((ex.example_id, cond))
            _append(path, rec)
            print(
                f"  [{i + 1}/{len(todo)}] {ex.example_id} {cond} labels={rec.get('plan_n_status_correct')}/12 "
                f"I->E={rec.get('plan_ineligible_marked_eligible')} E->I={rec.get('plan_eligible_marked_ineligible')} "
                f"choice={rec.get('correct')}",
                flush=True,
            )
        for cond, snap in snapshots.items():
            if not caches_equal(snap.cache_cpu, state["kvmix"][cond].cache_cpu):
                raise RuntimeError(f"{cond} cache changed during generation for {ex.example_id}.")
        _persist(out_root, kaggle_dest)
        del state, snapshots
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _teacher_summary(rows)
    _write_json(out_root / "teacher_summary.json", summary)
    _persist(out_root, kaggle_dest)
    print(json.dumps({k: v for k, v in summary.items() if k != "role_note"}, indent=2))
    return 0


def _frozen_inputs(out_root: Path) -> Path:
    sources = [
        REPO_ROOT / d / "a_artifacts_12.jsonl"
        for d in ("results_stage_plan4_recon12", "results_stage_plan4_refval12", "results_stage_plan4_kvmix12")
    ] + [
        Path("/kaggle/working") / d / "a_artifacts_12.jsonl"
        for d in ("results_stage_plan4_recon12", "results_stage_plan4_refval12")
    ]
    return _copy_first(out_root, "timing12_a_artifacts_12.jsonl", sources, what="frozen timing12 A artifacts")


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Query-adapter pilot: data, A sheets, construction check, teacher targets.")
    parser.add_argument("--phase", choices=("data", "write", "check", "teacher", "all"), default="data")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--plan_max_new_12", type=int, default=B_PLAN_MAX_NEW_12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.max_new_tokens != 48 or args.plan_max_new_12 != 384:
        print("Decoding allowances are frozen at 384 plan / 48 answer tokens.")
        return 2

    out_root, kaggle_dest = _paths()
    if "stage_plan4_qadapt_prep" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_qadapt_prep.")
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    restore = [d for d in (kaggle_dest, REPO_ROOT / "results_stage_plan4_qadapt_prep") if d is not None]
    for folder in restore:
        for name in PERSIST_NAMES:
            src, dest = folder / name, out_root / name
            if src.is_file() and not dest.is_file():
                dest.write_bytes(src.read_bytes())

    splits = _load_splits()
    timing12 = load_timing12_jsonl(DATA_DIR / "plan4_timing12.jsonl")
    protocol = pinned.get("study_plan4_qadapt") or {}
    rc = _run_data(out_root, kaggle_dest, splits, timing12, protocol)
    if args.phase == "data" or rc != 0:
        return rc

    frozen_artifacts = _load_messages(_frozen_inputs(out_root))
    if any(a_state_key(ex) not in frozen_artifacts for ex in timing12):
        print("Frozen timing12 A artifacts are incomplete. Do not regenerate them.")
        return 2
    policy, runner = _prepare_runner(
        args,
        out_root,
        extra={
            "stage": "plan4-qadapt-prep",
            "phase": args.phase,
            "do_not_overwrite": list(PRESERVED),
            "do_not_load": list(DO_NOT_LOAD),
            "a_written_max_new_12": A_WRITTEN_MAX_NEW_12,
            "a_sheets_for": list(A_SPLITS),
            "student": STUDENT,
            "teacher": TEACHER,
            "protocol": protocol,
        },
    )
    _persist(out_root, kaggle_dest)
    print(f"device={policy.device_name} phase={args.phase}")
    if args.phase in ("write", "all"):
        rc = _run_write(args, out_root, kaggle_dest, splits, runner)
        if args.phase == "write" or rc != 0:
            return rc
    if args.phase in ("check", "all"):
        rc = _run_check(args, out_root, kaggle_dest, splits, runner, frozen_artifacts, timing12)
        if args.phase == "check" or rc != 0:
            return rc
    return _run_teacher(args, out_root, kaggle_dest, splits, runner, frozen_artifacts, timing12)


if __name__ == "__main__":
    raise SystemExit(main())
