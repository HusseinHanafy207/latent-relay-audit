"""Initial-four-positions-only control on the same three families.

Keep A's written work and B's instructions fixed. Compare:
- kv: full completed cache
- kv_result: RESULT-line KV only
- kv_result_sink: RESULT-line KV plus the first four original positions
- kv_sink: the first four original positions only
- text_result: those RESULT lines as text

Token ids come from the decode loop's forwarded input_ids, not model.generate().
EOS counted in gen_lens/pos_cursor but not written into KV is recorded, not assumed
to explain the earlier breakdown.

plan4_dev subset only. Do not resume the kvdiag folder.
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
    RESULT_LINE,
    handoff_from_written,
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
from src.stage_plan4_kvdiag import _plan_fields
from src.stage_plan4_probe import A_WRITTEN_MAX_NEW, B_PLAN_MAX_NEW, _persist, _process_time_s
from src.stage_plan4_span import _decode_fn
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

CONDS = ("kv", "kv_result", "kv_result_sink", "kv_sink", "text_result")
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
    "results_stage_plan4_kvdiag",
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_initctrl" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_initctrl")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _checked_bool(items: list[dict], key: str) -> dict:
    values = [item.get(key) for item in items]
    return {
        "n_true": sum(v is True for v in values),
        "n_false": sum(v is False for v in values),
        "n_not_checked": sum(v is None for v in values),
    }


def _cond_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    bytes_ = [int(item.get("message_bytes") or 0) for item in items]
    mean_bytes = sum(bytes_) / n
    return {
        "n": n,
        "parse_ok": sum(bool(item.get("parse_ok")) for item in items),
        "think_collapse": sum(bool(item.get("think_collapse")) for item in items),
        "truncated": sum(bool(item.get("truncated") or item.get("plan_truncated")) for item in items),
        "plan_all_status_correct": sum(bool(item.get("plan_all_status_correct")) for item in items),
        "correct": sum(bool(item.get("correct")) for item in items),
        "mean_n_kept": sum(int(item.get("n_kept") or item.get("n_kv_positions") or 0) for item in items) / n,
        "mean_n_result_kept": sum(int(item.get("n_result_kept") or 0) for item in items) / n,
        "mean_n_sink_kept": sum(int(item.get("n_sink_kept") or 0) for item in items) / n,
        "mean_message_bytes": mean_bytes,
        "mean_relay_mb": mean_bytes / (1024 * 1024),
        "ids_source": sorted({str(item.get("ids_source") or "") for item in items if item.get("ids_source")}),
        "loop_eq_retokenize": _checked_bool(items, "loop_eq_retokenize"),
        "loop_len_eq_n_kv": _checked_bool(items, "loop_len_eq_n_kv"),
        "eos_counted_not_cached": _checked_bool(items, "eos_counted_not_cached"),
        "result_spans_from_loop_ids": _checked_bool(items, "result_spans_from_loop_ids"),
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
            "Not 45 independent scenarios. Checks whether result KV adds information "
            "beyond the first four cache positions that restore generation."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": {cond: _cond_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "result_sink_vs_sink": _paired(grouped_plain, "kv_sink", "kv_result_sink"),
            "result_sink_vs_result": _paired(grouped_plain, "kv_result", "kv_result_sink"),
            "sink_vs_result": _paired(grouped_plain, "kv_result", "kv_sink"),
            "text_vs_result_sink": _paired(grouped_plain, "kv_result_sink", "text_result"),
            "result_sink_vs_kv": _paired(grouped_plain, "kv", "kv_result_sink"),
        },
        "lead": (
            "Parseability, all-four eligibility, final-answer accuracy, and payload "
            "are reported separately. kv_sink asks whether the first four positions "
            "alone restore generation without the result states. "
            "loop_eq_retokenize null means not checked, not unequal."
        ),
        "decision_rule": [
            "If decode-loop ids do not match cache length, fix alignment before expanding.",
            "If result KV plus four positions does not beat four positions alone on eligibility or final answers, the result states are not yet shown to carry the useful information.",
            "If alignment checks pass and result+four beats four-only, expand that fixed comparison to the 90 development examples, then freeze before a new held-out evaluation.",
        ],
        "kvdiag_reference": (
            "Full KV 7/9, identity 7/9, result KV 0/9, result+four 5/9 with eligibility 3/9, "
            "result text 8/9. Identity gather matched tensors and answers."
        ),
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
    parser = argparse.ArgumentParser(
        description="Same nine examples: result KV vs initial-four-positions-only."
    )
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Init-control diagnostic uses three plan4_dev families only.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_initctrl" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_initctrl.")
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
            "stage": "plan4-initctrl",
            "split": "plan4_dev",
            "families": list(PLAN4_KVDIAG_FAMILIES),
            "n_examples": len(examples),
            "n_families": 3,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "sink_initial": SINK_INITIAL,
            "compressor": "none",
            "obf": False,
            "token_ids": "decode-loop forwarded input_ids; EOS counted but not cached",
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
        sink_only = list(range(min(SINK_INITIAL, n_full)))
        sink_pos, extra = with_initial_positions(result_pos, n_full=n_full, n_initial=SINK_INITIAL)
        result_text = decode_positions(full_ids, result_pos, decode)
        from_loop = pack["meta"].get("ids_source") == "decode_loop"
        pack["result_spans"] = selected_result
        pack["result_pos"] = result_pos
        pack["result_text"] = result_text
        pack["result_fallback"] = fallback
        pack["result_spans_from_loop_ids"] = (
            bool(RESULT_LINE.search(result_text or "")) if from_loop else None
        )
        pack["sink_only_pos"] = sink_only
        pack["sink_pos"] = sink_pos
        pack["sink_extra"] = extra
        pack["sink_text"] = decode_positions(full_ids, sink_only, decode)
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
                rec["positions"] = pack["all_pos"]
                rec["n_kept"] = int(rec.get("n_kv_positions") or 0)
                rec["n_result_kept"] = 0
                rec["n_sink_kept"] = 0
                rec["selected_text"] = pack["meta"].get("written")
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
            elif cond in ("kv_result", "kv_result_sink", "kv_sink"):
                if cond == "kv_result":
                    positions = pack["result_pos"]
                    n_result = len(positions)
                    n_sink = 0
                    selected_text = pack["result_text"]
                elif cond == "kv_result_sink":
                    positions = pack["sink_pos"]
                    n_result = len(pack["result_pos"])
                    n_sink = len(pack["sink_extra"])
                    selected_text = pack["result_text"]
                else:
                    positions = pack["sink_only_pos"]
                    n_result = 0
                    n_sink = len(positions)
                    selected_text = pack["sink_text"]
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
                rec.update(info)
                rec["n_kept"] = int(info["n_kept"])
                rec["n_result_kept"] = n_result
                rec["n_sink_kept"] = n_sink
                rec["n_sink_extra"] = n_sink if cond == "kv_result_sink" else 0
                rec["positions"] = list(info.get("positions") or positions)
                rec["selected_text"] = selected_text
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
                rec["n_result_kept"] = len(pack["result_pos"])
                rec["n_sink_kept"] = 0
                rec["positions"] = list(pack["result_pos"])
                rec["selected_text"] = pack["result_text"]
                rec["selection_time_s"] = 0.0
                rec["compression_time_s"] = 0.0
                rec["shared_across_heads"] = True
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rec["condition"] = cond
            rec["selector"] = {
                "kv": "none",
                "kv_result": "result",
                "kv_result_sink": "result_sink",
                "kv_sink": "sink",
                "text_result": "result",
            }[cond]
            rec["result_fallback"] = bool(pack.get("result_fallback"))
            rec["result_spans_from_loop_ids"] = pack.get("result_spans_from_loop_ids")
            rec["n_full"] = int(pack["n_full"])
            rec["ids_source"] = pack["meta"].get("ids_source")
            rec["loop_eq_retokenize"] = token_align.get("loop_eq_retokenize")
            rec["loop_len_eq_n_kv"] = token_align.get("loop_len_eq_n_kv")
            rec["eos_counted_not_cached"] = token_align.get("eos_counted_not_cached")
            rec["n_generated_counted"] = token_align.get("n_generated_counted")
            rec["n_cached_generated"] = token_align.get("n_cached_generated")
            rec["n_generated_not_cached"] = token_align.get("n_generated_not_cached")
            rec["pos_cursor"] = token_align.get("pos_cursor")
            rec["pos_cursor_minus_n_kv"] = token_align.get("pos_cursor_minus_n_kv")
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
                f"elig={rec.get('plan_all_status_correct')} parse={rec.get('parse_ok')} "
                f"think={rec.get('think_collapse')} kept={rec.get('n_kept')} "
                f"mb={float(rec.get('message_bytes') or 0) / (1024 * 1024):.2f}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Compare result+four vs four-only. Do not resume kvdiag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
