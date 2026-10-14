"""Six-family diagnostic after the first four-carrier calibration.

Fixes the written/compact prompt conflict, states the three-part eligibility
rule, asks B to name the cheapest feasible price, and compares ordinary
one-turn generation with cached continuation.

Uses plan4_dev families in PLAN4_DIAG_FAMILIES only. Preserves
results_stage_plan4_cal. No compressor. No six-condition study.
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

from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.parse_plan4 import score_sender_message, written_working_metrics
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    PLAN4_DIAG_FAMILIES,
    assert_plan4_invariants,
    a_state_key,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
    oracle_intermediate_message,
    subset_plan4_diag,
)
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    b_answer_after_message_plan_prompt,
    b_reason_with_message_prompt,
    compact_result_handoff,
    ordinary_sender_prompt,
    prompt_contains_a_private,
    prompt_contains_b_private,
    prompt_contains_solver_labels,
    written_result_handoff,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_cal import (
    A_CONDS,
    B_COND,
    CONDS,
    LATENT_STEPS_C40,
    _a_block,
    _a_row,
    _b_block,
    _b_row,
    _jobs,
    _process_time_s,
)
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

A_COMPACT_MAX_NEW = 256
A_WRITTEN_MAX_NEW = 768
B_PLAN_MAX_NEW = 192
PRESERVED = (
    "results_stage_plan0",
    "results_stage_plan0_corrected",
    "results_stage_plan0_diag",
    "results_stage_plan0_e2e",
    "results_stage_study1",
    "results_stage_plan4_cal",
)


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_diag" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_diag")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_core(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("a_artifacts.jsonl", "cache_compare.jsonl"):
            src = out_root / name
            if src.is_file():
                (dest / name).write_bytes(src.read_bytes())


def _compare_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "compact_texts_equal": sum(bool(r.get("compact_texts_equal")) for r in items) / n,
        "written_texts_equal": sum(bool(r.get("written_texts_equal")) for r in items) / n,
        "mean_ordinary_compact_n_prompt_tokens": sum(
            int(r.get("ordinary_compact_n_prompt_tokens") or 0) for r in items
        )
        / n,
        "mean_cache_encode_n_tokens": sum(int(r.get("cache_encode_n_tokens") or 0) for r in items) / n,
        "mean_cache_handoff_n_tokens": sum(
            int(r.get("cache_compact_handoff_n_tokens") or 0) for r in items
        )
        / n,
        "handoff_is_new_chat": all(r.get("handoff_is_new_chat") for r in items),
    }


def _summarize(rows: list[dict], compares: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["example_id"]][row["condition"]] = row
    grouped_plain = {eid: cells for eid, cells in grouped.items()}
    n_fam = len({row["family_id"] for row in rows}) if rows else 0
    unique_a = [r for r in by_cond.get("a_direct", []) if r.get("variant") != "b_flip"]
    out = {
        "n_rows": len(rows),
        "n_examples": len(grouped),
        "n_families": n_fam,
        "n_unique_a_sheets": len(unique_a),
        "unit_note": (
            "Six families, two per level, three variants each (18 examples). "
            "Base and B-change reuse the same A states (12 unique operational sheets). "
            "Not 18 independent sender tests. Written is scored from ordinary one-turn "
            "generation. Direct and latent40 stay on cached continuation."
        ),
        "sender": {cond: _a_block(by_cond.get(cond, [])) for cond in A_CONDS},
        "sender_unique_sheets": {
            cond: _a_block(
                [r for r in by_cond.get(cond, []) if r.get("variant") != "b_flip"]
            )
            for cond in A_CONDS
        },
        "receiver_oracle": _b_block(by_cond.get(B_COND, [])),
        "by_difficulty": {},
        "paired_sender": {
            "written_vs_direct": _paired(grouped_plain, "a_direct", "a_written"),
            "latent40_vs_direct": _paired(grouped_plain, "a_direct", "a_latent40"),
            "written_vs_latent40": _paired(grouped_plain, "a_latent40", "a_written"),
        },
        "flip_tracking": {B_COND: _flip_tracking(rows, B_COND)},
        "cache_compare": _compare_block(compares),
        "lead": (
            "Six-family diagnostic after plan4 calibration. "
            "Written working is separate from a RESULTS block. "
            "Eligibility is the three-part rule. "
            "B must list feasible prices and name the cheapest. "
            "Ordinary one-turn generation is compared with cached continuation. "
            "Preserve results_stage_plan4_cal. Do not start six-condition or compression."
        ),
        "timing_note": (
            "Written uses concatenated ordinary generation (max 768). "
            "Direct and latent40 use cached continuation (max 256). "
            "Those paths are not compute-matched. Network transmission is not measured."
        ),
        "deferred": {
            "repeat_full_plan4_dev": "After this diagnostic shows written working and interpretable B.",
            "six_condition": "After sender computation and oracle-B are interpretable.",
            "compression": "After full-relay behavior on this task is interpretable.",
        },
        "families": list(PLAN4_DIAG_FAMILIES),
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "sender": {
                cond: _a_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in A_CONDS
            },
            "receiver_oracle": _b_block(_subset(by_cond.get(B_COND, []), difficulty=difficulty)),
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _rewrite_jsonl(path: Path, records: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in records.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="Six-family four-carrier diagnostic. Preserves results_stage_plan4_cal."
    )
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--sender_text_max_new", type=int, default=A_COMPACT_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Diagnostic uses plan4_dev families only. Do not load plan_dev or a plan_test file.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_diag" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_diag.")
        return 2
    if "stage_plan4_cal" in str(out_root) and "diag" not in str(out_root):
        print("Refusing to resume or overwrite results_stage_plan4_cal.")
        return 2
    if not path.is_file():
        print(f"Missing {path}.")
        return 2
    all_examples = load_plan4_jsonl(path)
    examples = subset_plan4_diag(all_examples)
    if len({ex.family_id for ex in examples}) != 6:
        print("Diagnostic requires exactly six families.")
        return 2
    for ex in examples:
        if ex.split != "plan4_dev":
            print(f"Refusing {ex.example_id} on plan4 diagnostic.")
            return 2
        assert_plan4_invariants(ex)

    if args.fresh and out_root.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_root = out_root.parent / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    artifacts_path = out_root / "a_artifacts.jsonl"
    compare_path = out_root / "cache_compare.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json", "a_artifacts.jsonl", "cache_compare.jsonl"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["example_id"], row["condition"]) for row in rows}
    artifacts = _load_messages(artifacts_path)
    compares = {item["a_state_key"]: item for item in _load_rows(compare_path) if item.get("a_state_key")}
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-diag",
            "split": "plan4_dev",
            "families": list(PLAN4_DIAG_FAMILIES),
            "n_examples": len(examples),
            "n_families": 6,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "compressor": "none",
            "channel": "sender diagnostic plus oracle-to-B; cache vs ordinary compare",
            "written_path": "ordinary concatenated generation",
            "direct_latent_path": "cached continuation (new chat turn for the handoff)",
            "latent_steps_c40": LATENT_STEPS_C40,
            "sender_text_max_new": args.sender_text_max_new,
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "deferred": ["repeat full plan4_dev", "six-condition communication", "compressed KV"],
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=LATENT_STEPS_C40,
        kv_budget=32,
        sink_size=4,
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

    by_id = {ex.example_id: ex for ex in all_examples}
    states: dict[str, dict] = {}

    def ensure_a(ex) -> dict:
        key = a_state_key(ex)
        if key in states:
            return states[key]
        if key in artifacts:
            states[key] = artifacts[key]
            return states[key]
        src_ex = by_id[key]
        encode_prompt = a_local_task_prompt(src_ex)
        if prompt_contains_b_private(encode_prompt, src_ex):
            raise RuntimeError("A local-task prompt leaked B's prices.")
        if prompt_contains_solver_labels(encode_prompt, src_ex):
            raise RuntimeError("A local-task prompt contained solver q/eligibility lines.")
        if "Now write" in encode_prompt or "Now compute" in encode_prompt:
            raise RuntimeError("C0 encode asked A to write the result.")
        if "Write exactly four lines" in encode_prompt:
            raise RuntimeError("C0 encode asked for four result lines.")
        compact = compact_result_handoff()
        written = written_result_handoff()
        if "Write exactly four lines" in written:
            raise RuntimeError("Written handoff still opens as a four-line-only request.")
        if compact == written:
            raise RuntimeError("Written and compact handoffs must differ.")
        ordinary_written = ordinary_sender_prompt(encode_prompt, written)
        ordinary_compact = ordinary_sender_prompt(encode_prompt, compact)
        encode_rendered, encode_n = runner.render_chat(encode_prompt)
        compact_handoff_rendered, compact_handoff_n = runner.render_chat(compact)
        written_handoff_rendered, written_handoff_n = runner.render_chat(written)
        ordinary_compact_rendered, ordinary_compact_n = runner.render_chat(ordinary_compact)
        ordinary_written_rendered, ordinary_written_n = runner.render_chat(ordinary_written)
        wall0 = time.time()
        c0 = runner.encode_prompt(encode_prompt, source="plan4_diag_c0", latent_steps=0)
        c40 = runner.continue_latent(c0, LATENT_STEPS_C40)
        direct, n_d, d_trunc, d_s = runner.generate_from_cache(
            c0, compact, max_new_tokens=args.sender_text_max_new
        )
        lat, n_l, l_trunc, l_s = runner.generate_from_cache(
            c40, compact, max_new_tokens=args.sender_text_max_new
        )
        writ_cached, n_wc, wc_trunc, wc_s = runner.generate_from_cache(
            c0, written, max_new_tokens=args.written_max_new
        )
        encode_c0_s = c0.sender_latency_s
        rollout_c40_s = c40.sender_latency_s
        del c0, c40
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        writ, n_w, w_trunc, w_s = runner.generate_text_message(
            ordinary_written, max_new_tokens=args.written_max_new
        )
        ord_compact, n_oc, oc_trunc, oc_s = runner.generate_text_message(
            ordinary_compact, max_new_tokens=args.sender_text_max_new
        )
        rec = {
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "variant": src_ex.variant,
            "difficulty": src_ex.difficulty,
            "a_state_key": key,
            "encode_prompt": encode_prompt,
            "compact_handoff": compact,
            "written_handoff": written,
            "ordinary_written_prompt": ordinary_written,
            "ordinary_compact_prompt": ordinary_compact,
            "encode_c0_s": encode_c0_s,
            "rollout_c40_s": rollout_c40_s,
            "cache_prep_s": time.time() - wall0,
            "direct": direct,
            "direct_n_new": n_d,
            "direct_truncated": d_trunc,
            "direct_s": d_s,
            "direct_bytes": len(direct.encode("utf-8")),
            "latent40": lat,
            "latent40_n_new": n_l,
            "latent40_truncated": l_trunc,
            "latent40_s": l_s,
            "latent40_bytes": len(lat.encode("utf-8")),
            "written": writ,
            "written_n_new": n_w,
            "written_truncated": w_trunc,
            "written_s": w_s,
            "written_bytes": len(writ.encode("utf-8")),
            "written_path": "ordinary",
            "written_cached": writ_cached,
            "written_cached_n_new": n_wc,
            "written_cached_truncated": wc_trunc,
            "written_cached_s": wc_s,
            "ordinary_compact": ord_compact,
            "ordinary_compact_n_new": n_oc,
            "ordinary_compact_truncated": oc_trunc,
            "ordinary_compact_s": oc_s,
            "direct_score": score_sender_message(
                direct, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "latent40_score": score_sender_message(
                lat, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_score": score_sender_message(
                writ, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_working": written_working_metrics(writ),
        }
        compare = {
            "a_state_key": key,
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "difficulty": src_ex.difficulty,
            "handoff_is_new_chat": True,
            "cache_encode_rendered": encode_rendered,
            "cache_encode_n_tokens": encode_n,
            "cache_compact_handoff_rendered": compact_handoff_rendered,
            "cache_compact_handoff_n_tokens": compact_handoff_n,
            "cache_written_handoff_rendered": written_handoff_rendered,
            "cache_written_handoff_n_tokens": written_handoff_n,
            "ordinary_compact_rendered": ordinary_compact_rendered,
            "ordinary_compact_n_prompt_tokens": ordinary_compact_n,
            "ordinary_written_rendered": ordinary_written_rendered,
            "ordinary_written_n_prompt_tokens": ordinary_written_n,
            "compact_texts_equal": (ord_compact or "").strip() == (direct or "").strip(),
            "written_texts_equal": (writ or "").strip() == (writ_cached or "").strip(),
            "ordinary_compact": ord_compact,
            "cached_compact": direct,
            "ordinary_written": writ,
            "cached_written": writ_cached,
            "ordinary_compact_n_new": n_oc,
            "cached_compact_n_new": n_d,
            "ordinary_written_n_new": n_w,
            "cached_written_n_new": n_wc,
        }
        artifacts[key] = rec
        compares[key] = compare
        _rewrite_jsonl(artifacts_path, artifacts)
        _rewrite_jsonl(compare_path, compares)
        states[key] = rec
        return rec

    needed = set()
    for ex in examples:
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        kinds = {job["condition"] for job in remaining}
        if kinds & set(A_CONDS):
            needed.add(a_state_key(ex))
    print(
        f"Encoding A states for {len(needed)} unique operational sheets "
        f"({len(PLAN4_DIAG_FAMILIES)} families)",
        flush=True,
    )
    for eid in sorted(needed):
        ensure_a(by_id[eid] if eid in by_id else next(ex for ex in all_examples if a_state_key(ex) == eid))
        _persist(out_root, kaggle_dest)

    mode_field = {
        "a_direct": ("direct", "c0"),
        "a_written": ("written", "ordinary"),
        "a_latent40": ("latent40", "c40"),
    }

    for i, ex in enumerate(examples):
        remaining = [job for job in _jobs() if (ex.example_id, job["condition"]) not in done]
        if not remaining:
            print(f"[{i + 1}/{len(examples)}] {ex.example_id} already complete", flush=True)
            continue
        print(f"[{i + 1}/{len(examples)}] {ex.example_id} remaining={len(remaining)}", flush=True)
        reset_peak_memory()
        for job in remaining:
            cond = job["condition"]
            extra = {
                "encode_c0_s": 0.0,
                "rollout_c40_s": 0.0,
                "sender_generation_time_s": 0.0,
                "sender_truncated": False,
                "n_sender_new_tokens": 0,
                "message_bytes": 0,
                "message_text": "",
                "reused_a_from": None,
                "cache_point": None,
                "plan_time_s": 0.0,
                "n_plan_tokens": 0,
                "plan_truncated": False,
                "plan_text": "",
                "max_new_tokens": args.max_new_tokens,
            }
            if cond in A_CONDS:
                pack = ensure_a(ex)
                field, cache_point = mode_field[cond]
                text = pack[field]
                extra["reused_a_from"] = pack["a_state_key"] if pack["a_state_key"] != ex.example_id else None
                extra["cache_point"] = cache_point
                extra["encode_c0_s"] = pack["encode_c0_s"]
                extra["rollout_c40_s"] = pack["rollout_c40_s"] if cache_point == "c40" else 0.0
                extra["sender_generation_time_s"] = pack[f"{field}_s"]
                extra["sender_truncated"] = pack[f"{field}_truncated"]
                extra["n_sender_new_tokens"] = pack[f"{field}_n_new"]
                extra["n_new_tokens"] = pack[f"{field}_n_new"]
                extra["truncated"] = pack[f"{field}_truncated"]
                extra["message_bytes"] = pack[f"{field}_bytes"]
                extra["message_text"] = text
                extra["cache_prep_s"] = pack["cache_prep_s"]
                extra["receiver_latency_s"] = 0.0
                extra["peak_gpu_gb"] = peak_gpu_memory_gb()
                extra["process_time_s"] = _process_time_s(extra)
                extra["written_path"] = pack.get("written_path") if cond == "a_written" else None
                prompt = pack["written_handoff"] if cond == "a_written" else pack["compact_handoff"]
                rec = _a_row(ex, condition=cond, prompt=prompt, text=text, extra=extra)
            elif cond == B_COND:
                oracle = oracle_intermediate_message(ex)
                plan_prompt = b_reason_with_message_prompt(ex, oracle)
                if prompt_contains_a_private(plan_prompt, ex):
                    raise RuntimeError("Oracle B leaked A's raw sheet.")
                if "Answer: <letter>) <carrier>" in plan_prompt:
                    raise RuntimeError("Oracle reasoning pass still asks for a one-line final answer.")
                plan_text, n_plan, plan_trunc, plan_time = runner.generate_text_message(
                    plan_prompt, max_new_tokens=args.plan_max_new
                )
                prompt = b_answer_after_message_plan_prompt(ex, oracle, plan_text)
                extra["message_text"] = oracle
                extra["message_bytes"] = len(oracle.encode("utf-8"))
                extra["plan_time_s"] = plan_time
                extra["n_plan_tokens"] = n_plan
                extra["plan_truncated"] = plan_trunc
                extra["plan_text"] = plan_text
                extra["plan_max_new"] = args.plan_max_new
                text, n_new, truncated, latency = runner.generate_text_message(
                    prompt, max_new_tokens=args.max_new_tokens
                )
                extra["truncated"] = truncated
                extra["n_new_tokens"] = n_new
                extra["receiver_latency_s"] = latency
                extra["peak_gpu_gb"] = peak_gpu_memory_gb()
                extra["process_time_s"] = _process_time_s({**extra, "receiver_latency_s": latency})
                rec = _b_row(ex, prompt=prompt, text=text, extra=extra)
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if cond in A_CONDS:
                print(
                    f"  {cond}: four={rec['all_four_correct']} q={rec['n_q_correct']} "
                    f"status={rec['n_status_correct']} work={rec.get('produced_working')} "
                    f"tok={rec['n_new_tokens']} trunc={rec['truncated']}",
                    flush=True,
                )
            else:
                print(
                    f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                    f"correct={rec['correct']} trap={rec['follows_trap']} "
                    f"plan_agree={rec.get('plan_answer_agrees')}",
                    flush=True,
                )
            (out_root / "summary.json").write_text(
                json.dumps(_summarize(rows, list(compares.values())), indent=2), encoding="utf-8"
            )
            _persist(out_root, kaggle_dest)

    report = _summarize(rows, list(compares.values()))
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Check that written outputs contain working, then decide whether to repeat plan4_dev.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
