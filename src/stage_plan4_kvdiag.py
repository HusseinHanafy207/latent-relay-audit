"""Small sparse-cache continuation diagnostic. Three frozen families.

Keep A's written work and B's instructions fixed. Compare:
- kv: full completed cache
- kv_identity: the same cache gathered through keep_shared_positions at every index
- kv_result: RESULT-line KV only
- kv_result_sink: RESULT-line KV plus the first four original positions
- text_result: those RESULT lines as text

plan4_dev subset only. Do not resume the span folder. Do not retune prompts.
Do not add compression methods. Do not reset retained key positions.
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

from src.cache_utils import n_kv_positions
from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse_plan4 import (
    handoff_from_written,
    is_think_collapse,
    score_b_plan,
    score_sender_message,
    written_working_metrics,
)
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    PLAN4_KVDIAG_FAMILIES,
    SINK_INITIAL,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
    subset_plan4_kvdiag,
)
from src.precision import apply_precision_policy, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    ordinary_sender_prompt,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.span_select import (
    decode_positions,
    line_spans,
    positions_from_spans,
    result_spans,
    with_initial_positions,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _load_messages, _subset
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _run_text_b
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist, _process_time_s
from src.stage_plan4_span import _decode_fn
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

CONDS = ("kv", "kv_identity", "kv_result", "kv_result_sink", "text_result")
PRESERVED = (
    "results_stage_plan0",
    "results_stage_plan0_corrected",
    "results_stage_plan0_diag",
    "results_stage_plan0_e2e",
    "results_stage_study1",
    "results_stage_plan4_cal",
    "results_stage_plan4_diag",
    "results_stage_plan4_cachecheck",
    "results_stage_plan4_bprompt",
    "results_stage_plan4_chain",
    "results_stage_plan4_comm",
    "results_stage_plan4_compress",
    "results_stage_plan4_probe",
    "results_stage_plan4_budget",
    "results_stage_plan4_fresh",
    "results_stage_plan4_span",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_kvdiag" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_kvdiag")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _plan_fields(rec: dict, ex) -> None:
    plan = score_b_plan(
        str(rec.get("plan_text") or ""),
        gold_status=gold_eligibility_statuses(ex),
        prices={c: int(ex.b_facts["prices"][c]) for c in ("P", "Q", "R", "S")},
        budget=int(ex.b_facts["budget"]),
        pred_carrier=rec.get("pred_carrier"),
    )
    rec["plan_all_status_correct"] = plan["all_status_correct"]
    rec["plan_all_price_correct"] = plan["all_price_correct"]
    rec["plan_choice"] = plan["choice"]
    rec["plan_choice_matches_gold"] = plan["choice_matches_gold"]
    rec["plan_answer_agrees"] = plan["plan_answer_agrees"]
    rec["think_collapse"] = plan["think_collapse"] or is_think_collapse(str(rec.get("raw_text") or ""))


def _cond_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "correct": sum(bool(item.get("correct")) for item in items),
        "parse_ok": sum(bool(item.get("parse_ok")) for item in items),
        "think_collapse": sum(bool(item.get("think_collapse")) for item in items),
        "plan_all_status_correct": sum(bool(item.get("plan_all_status_correct")) for item in items),
        "plan_answer_agrees": sum(item.get("plan_answer_agrees") is True for item in items),
        "gathered_equals_source": sum(bool(item.get("gathered_equals_source")) for item in items),
        "mean_n_kept": sum(int(item.get("n_kept") or item.get("n_kv_positions") or 0) for item in items) / n,
        "mean_n_sink_extra": sum(int(item.get("n_sink_extra") or 0) for item in items) / n,
        "mean_message_bytes": sum(int(item.get("message_bytes") or 0) for item in items) / n,
        "ids_source": sorted({str(item.get("ids_source") or "") for item in items}),
        "sequences_eq_retokenize": sum(item.get("sequences_eq_retokenize") is True for item in items),
        "truncated": sum(bool(item.get("truncated") or item.get("plan_truncated")) for item in items),
    }


def _summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    kv_rows = by_cond.get("kv", [])
    unique_a = [r for r in kv_rows if r.get("variant") != "b_flip"]
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": len({row["family_id"] for row in rows}) if rows else 0,
        "families": list(PLAN4_KVDIAG_FAMILIES),
        "unit_note": (
            "Three predetermined families, one per difficulty, with three variants each. "
            "Not 45 independent scenarios. Small continuation diagnostic, not a ranking."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "identity_vs_kv": _paired(grouped_plain, "kv", "kv_identity"),
            "result_vs_kv": _paired(grouped_plain, "kv", "kv_result"),
            "result_sink_vs_result": _paired(grouped_plain, "kv_result", "kv_result_sink"),
            "text_result_vs_kv_result": _paired(grouped_plain, "kv_result", "text_result"),
        },
        "lead": (
            "If keeping every position changes B's answers, fix gather/continuation "
            "before interpreting compressed KV. If adding the first four positions "
            "restores result-KV generation, that is a structural-support finding, "
            "not yet a text-versus-latent conclusion."
        ),
        "decision_rule": [
            "Keeping every position changes behavior: fix the implementation before interpreting compressed results.",
            "Adding the initial positions restores result-KV generation: validate on development data, and add an initial-positions-only control.",
            "Result KV remains unusable despite verified alignment and continuation: investigate the sparse-cache interface before expanding.",
        ],
        "b_prompt": "frozen line-by-line; KV uses the same procedure over relayed memory",
    }
    for difficulty in ("L1", "L2", "L3"):
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _cond_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Three-family sparse-cache continuation diagnostic.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("KV diagnostic uses three plan4_dev families only.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_kvdiag" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_kvdiag.")
        return 2
    if not path.is_file():
        print(f"Missing {path}.")
        return 2
    examples = subset_plan4_kvdiag(load_plan4_jsonl(path))
    by_id = {ex.example_id: ex for ex in examples}
    for ex in examples:
        if ex.split != "plan4_dev":
            print(f"Refusing {ex.example_id}.")
            return 2
        if ex.family_id not in PLAN4_KVDIAG_FAMILIES:
            print(f"Refusing unexpected family {ex.family_id}.")
            return 2
        assert_plan4_invariants(ex)
    if len(examples) != 9:
        print(f"Expected 9 examples (3 families x 3 variants), got {len(examples)}.")
        return 2

    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    artifacts_path = out_root / "a_artifacts.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json", "a_artifacts.jsonl"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    artifacts = _load_messages(artifacts_path)
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-kvdiag",
            "split": "plan4_dev",
            "families": list(PLAN4_KVDIAG_FAMILIES),
            "n_examples": len(examples),
            "n_families": 3,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "sink_initial": SINK_INITIAL,
            "compressor": "none",
            "obf": False,
            "a_path": "ordinary written calculations; not latent-40",
            "b_prompt": "frozen line-by-line; KV uses relayed-memory wording",
            "held_out": "plan4_fresh and the rest of plan4_dev are not loaded",
            "do_not_resume": list(PRESERVED),
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=0,
        kv_budget=256,
        sink_size=SINK_INITIAL,
        pca_rank=2,
        max_new_tokens=args.max_new_tokens,
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha") and not pinned.get("model", {}).get("revision"):
        pin_model_revision(revision["sha"])
    extra_manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8")).get("extra", {})
    extra_manifest["model_revision"] = revision.get("sha")
    extra_manifest["model_revision_source"] = revision.get("source")
    write_manifest(out_root / "manifest.json", policy=policy, extra=extra_manifest)
    _persist(out_root, kaggle_dest)

    decode = _decode_fn(runner)
    states: dict[str, dict] = {}

    def ensure_a(ex) -> dict:
        key = a_state_key(ex)
        if key in states:
            return states[key]
        src_ex = by_id[key]
        encode_prompt = a_local_task_prompt(src_ex)
        written = written_result_handoff()
        ordinary = ordinary_sender_prompt(encode_prompt, written)
        if prompt_contains_b_private(ordinary, src_ex):
            raise RuntimeError("A prompt leaked B's prices.")
        if prompt_contains_solver_labels(ordinary, src_ex):
            raise RuntimeError("A prompt contained solver q/eligibility lines.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text, n_new, truncated, gen_s, bundle = runner.generate_text_keep_cache(
            ordinary, max_new_tokens=args.written_max_new, source="written"
        )
        if int(bundle.latent_steps) != 0:
            raise RuntimeError("Written cache must have zero latent steps.")
        if bundle.full_ids_cpu is None:
            raise RuntimeError("Written cache is missing aligned token ids.")
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
        if int(bundle.full_ids_cpu.numel()) != n_full:
            raise RuntimeError("Aligned token ids do not match cache length.")
        token_align = dict(bundle.token_align or {})
        rec = {
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "variant": src_ex.variant,
            "difficulty": src_ex.difficulty,
            "a_state_key": key,
            "ordinary_prompt": ordinary,
            "written": text,
            "written_n_new": n_new,
            "written_truncated": truncated,
            "written_s": gen_s,
            "handoff": handoff_from_written(text),
            "n_kv_positions": n_full,
            "n_prompt_positions": n_prompt,
            "full_relay_bytes": int(bundle.full_relay_bytes),
            "latent_steps": 0,
            "ids_source": bundle.ids_source,
            "token_align": token_align,
            "full_ids": bundle.full_ids_cpu.tolist(),
            "pos_cursor": int(bundle.pos_cursor_cpu.view(-1)[0].item()),
            "written_score": score_sender_message(
                text, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_working": written_working_metrics(text),
        }
        artifacts[key] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        states[key] = {"bundle": bundle, "meta": rec}
        return states[key]

    def ensure_spans(pack: dict, ex) -> None:
        if pack.get("result_pos") is not None:
            return
        bundle = pack["bundle"]
        full_ids = bundle.full_ids_cpu.tolist()
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
        written = line_spans(full_ids, start=n_prompt, end=n_full, decode=decode)
        selected_result = result_spans(written)
        fallback = False
        if not selected_result:
            selected_result = written[-4:] if len(written) >= 4 else list(written)
            fallback = True
        result_pos = positions_from_spans(selected_result)
        if not result_pos:
            raise RuntimeError(f"No complete spans in the written region for {ex.example_id}.")
        sink_pos, extra = with_initial_positions(result_pos, n_full=n_full, n_initial=SINK_INITIAL)
        pack["result_spans"] = selected_result
        pack["result_pos"] = result_pos
        pack["result_text"] = decode_positions(full_ids, result_pos, decode)
        pack["result_fallback"] = fallback
        pack["sink_pos"] = sink_pos
        pack["sink_extra"] = extra
        pack["n_full"] = n_full
        pack["all_pos"] = list(range(n_full))

    needed = set()
    for ex in examples:
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if remaining:
            needed.add(a_state_key(ex))
    print(f"Encoding written A caches for {len(needed)} unique operational sheets", flush=True)
    for eid in sorted(needed):
        src = by_id.get(eid) or next(ex for ex in examples if a_state_key(ex) == eid)
        ensure_a(src)
        _persist(out_root, kaggle_dest)

    for i, ex in enumerate(examples):
        remaining = [job["condition"] for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        pack = ensure_a(ex)
        extra_a = _a_extra(pack)
        ensure_spans(pack, ex)
        token_align = dict(pack["meta"].get("token_align") or {})

        for cond in remaining:
            if cond == "kv":
                rec = _run_kv_b(
                    runner,
                    ex,
                    pack["bundle"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["n_kept"] = int(rec.get("n_kv_positions") or 0)
                rec["positions"] = pack["all_pos"]
                rec["n_sink_extra"] = 0
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
            elif cond in ("kv_identity", "kv_result", "kv_result_sink"):
                if cond == "kv_identity":
                    positions = pack["all_pos"]
                    extra_n = 0
                elif cond == "kv_result":
                    positions = pack["result_pos"]
                    extra_n = 0
                else:
                    positions = pack["sink_pos"]
                    extra_n = len(pack["sink_extra"])
                compressed, wall, info = runner.keep_shared_positions(pack["bundle"], positions)
                rec = _run_kv_b(
                    runner,
                    ex,
                    compressed,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["handoff"] = cond
                rec["selection_time_s"] = float(wall)
                rec["compression_time_s"] = float(wall)
                rec["n_sink_extra"] = extra_n
                rec.update(info)
                rec["n_kept"] = int(info["n_kept"])
                rec["positions"] = list(info.get("positions") or positions)
            elif cond == "text_result":
                rec = _run_text_b(
                    runner,
                    ex,
                    message=pack["result_text"],
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                    allow_a_source=True,
                )
                rec.update(extra_a)
                rec["handoff"] = cond
                rec["n_kept"] = len(pack["result_pos"])
                rec["n_kv_positions"] = 0
                rec["positions"] = list(pack["result_pos"])
                rec["n_sink_extra"] = 0
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
                rec["shared_across_heads"] = True
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rec["condition"] = cond
            rec["selector"] = {
                "kv": "none",
                "kv_identity": "all",
                "kv_result": "result",
                "kv_result_sink": "result_sink",
                "text_result": "result",
            }[cond]
            rec["selected_text"] = pack["result_text"] if cond != "kv" else pack["meta"].get("written")
            rec["result_fallback"] = bool(pack.get("result_fallback"))
            rec["n_full"] = int(pack["n_full"])
            rec["ids_source"] = pack["meta"].get("ids_source")
            rec["sequences_eq_retokenize"] = token_align.get("sequences_eq_retokenize")
            rec["pos_cursor"] = token_align.get("pos_cursor")
            rec["shared_across_heads"] = True
            rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            rec["process_time_s"] = _process_time_s(rec)
            _plan_fields(rec, ex)
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: pred={rec.get('pred_carrier')} correct={rec.get('correct')} "
                f"think={rec.get('think_collapse')} kept={rec.get('n_kept')} "
                f"equal={rec.get('gathered_equals_source')}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Interpret continuation first. Do not resume the span folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
