"""Same selected spans: direct KV versus the same text tokens.

A writes calculations once. A shared position set is chosen once, then
branched into gathered KV versus decoded text of those exact indices.
The uncompressed full cache checks that masks and positions still work.

Selectors:
- result: complete RESULT lines from A's written region
- probe: complete lines ranked by shared (layer/head-mean) probe scores,
  until at least as many tokens as the result spans

plan4_dev only. Do not resume the frozen fresh run or earlier plan4 folders.
Do not retune prompts. Do not add budgets 32/64 or OBF.
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
from src.parse_plan4 import handoff_from_written, score_sender_message, written_working_metrics
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import (
    LEVELS,
    a_state_key,
    assert_plan4_invariants,
    gold_eligibility_statuses,
    gold_quantities,
    load_plan4_jsonl,
)
from src.precision import apply_precision_policy, reset_peak_memory
from src.prompts_plan4 import (
    a_local_task_prompt,
    b_need_probe_prompt,
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
    select_spans_by_score,
    shared_position_scores,
)
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_comm import _a_extra, _run_kv_b, _sender_block
from src.stage_plan4_compress import _family_delta, _run_text_b
from src.stage_plan4_probe import (
    A_WRITTEN_MAX_NEW,
    B_PLAN_MAX_NEW,
    _persist,
    _process_time_s,
    _probe_block,
)
from src.stage_select import _load_rows, _paired
from src.two_agent import TwoAgentInventory

CONDS = ("kv", "kv_result", "text_result", "kv_probe", "text_probe")
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
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_span" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_span")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _decode_fn(runner):
    tokenizer = runner.wrapper.tokenizer

    def decode(ids):
        return tokenizer.decode(list(ids), skip_special_tokens=False)

    return decode


def _score_text(text: str, ex) -> dict:
    scored = score_sender_message(text, gold_quantities(ex), gold_eligibility_statuses(ex))
    return {
        "all_four": bool(scored["all_four_correct"]),
        "n_q_correct": int(scored["n_q_correct"]),
        "n_status_correct": int(scored["n_status_correct"]),
        "n_carriers_parsed": int(scored["n_carriers_parsed"]),
        "malformed": bool(scored["malformed"]),
    }


def _recovery_block(items: list[dict], prefix: str) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "all_four": sum(bool(item.get(f"{prefix}_all_four")) for item in items),
        "mean_n_q_correct": sum(int(item.get(f"{prefix}_n_q_correct") or 0) for item in items) / n,
        "mean_n_status_correct": sum(int(item.get(f"{prefix}_n_status_correct") or 0) for item in items) / n,
        "mean_n_carriers_parsed": sum(int(item.get(f"{prefix}_n_carriers_parsed") or 0) for item in items) / n,
    }


def _span_block(items: list[dict]) -> dict:
    out = _probe_block(items)
    n = len(items)
    if not n:
        return out
    out["selected_recovery"] = _recovery_block(items, "selected")
    out["plan_recovery"] = _recovery_block(items, "plan")
    out["mean_n_kept"] = sum(int(item.get("n_kept") or item.get("n_kv_positions") or 0) for item in items) / n
    out["mean_selected_bytes"] = sum(int(item.get("selected_bytes") or 0) for item in items) / n
    out["mean_n_spans"] = sum(int(item.get("n_spans") or 0) for item in items) / n
    out["mean_selection_time_s"] = sum(float(item.get("selection_time_s") or 0) for item in items) / n
    out["shared_across_heads"] = all(bool(item.get("shared_across_heads", True)) for item in items)
    return out


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
        "n_unique_a_sheets": len({r.get("a_state_key") for r in kv_rows if r.get("a_state_key")}),
        "unit_note": (
            "30 families with three versions each, not 90 independent scenarios. "
            "Same written A on every branch. KV and text of a pair use the same "
            "shared positions. Uncompressed kv verifies the continuation path. "
            "Matched positions are not informationally identical: old KV can retain "
            "context from discarded tokens."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": {cond: _span_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "text_result_vs_kv_result": _paired(grouped_plain, "kv_result", "text_result"),
            "text_probe_vs_kv_probe": _paired(grouped_plain, "kv_probe", "text_probe"),
            "kv_result_vs_kv": _paired(grouped_plain, "kv", "kv_result"),
            "text_result_vs_kv": _paired(grouped_plain, "kv", "text_result"),
            "kv_probe_vs_kv": _paired(grouped_plain, "kv", "kv_probe"),
            "kv_probe_vs_kv_result": _paired(grouped_plain, "kv_result", "kv_probe"),
            "text_probe_vs_text_result": _paired(grouped_plain, "text_result", "text_probe"),
        },
        "family_clustered_delta": {
            "text_result_vs_kv_result": _family_delta(rows, "kv_result", "text_result"),
            "text_probe_vs_kv_probe": _family_delta(rows, "kv_probe", "text_probe"),
            "kv_result_vs_kv": _family_delta(rows, "kv", "kv_result"),
            "kv_probe_vs_kv": _family_delta(rows, "kv", "kv_probe"),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "lead": (
            "When the same complete spans of A's completed work are selected, "
            "how does directly transferring their KV states differ from sending "
            "those same spans as text? Shared positions across heads. "
            "Uncompressed full-KV control. Frozen line-B."
        ),
        "selector_note": (
            "result keeps complete RESULT lines. probe keeps complete lines by "
            "shared mean attention until at least as many tokens as the result "
            "spans. No extra sink. No headwise union for text."
        ),
        "representation_note": (
            "Matching selected positions does not make the representations "
            "informationally identical. Old KV states can retain context from "
            "discarded positions. That difference is part of the comparison."
        ),
        "timing_note": (
            "mean_process_time_s averages recorded process_time_s "
            "(written A, probe ranking where used, gather, and B). "
            "Network is not measured."
        ),
        "b_prompt": "frozen line-by-line; KV uses the same procedure over relayed memory",
        "possible_outcomes": [
            "Text succeeds while KV fails on the same passages: selection alone does not explain the failure.",
            "Both fail, but both improve when complete result passages are retained: selected material was a bottleneck.",
            "Both recover the results but still choose incorrectly: remaining problem is downstream in B.",
            "They perform similarly: representation is less important here; selection and cost remain the main considerations.",
        ],
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _span_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _attach_recovery(rec: dict, *, selected_text: str, ex) -> None:
    selected = _score_text(selected_text, ex)
    plan = _score_text(str(rec.get("plan_text") or ""), ex)
    rec["selected_text"] = selected_text
    rec["selected_bytes"] = len(selected_text.encode("utf-8"))
    rec["selected_all_four"] = selected["all_four"]
    rec["selected_n_q_correct"] = selected["n_q_correct"]
    rec["selected_n_status_correct"] = selected["n_status_correct"]
    rec["selected_n_carriers_parsed"] = selected["n_carriers_parsed"]
    rec["selected_malformed"] = selected["malformed"]
    rec["plan_all_four"] = plan["all_four"]
    rec["plan_n_q_correct"] = plan["n_q_correct"]
    rec["plan_n_status_correct"] = plan["n_status_correct"]
    rec["plan_n_carriers_parsed"] = plan["n_carriers_parsed"]
    rec["plan_malformed"] = plan["malformed"]


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(
        description="plan4_dev: same selected spans as shared-position KV versus text."
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
        print("Span comparison uses plan4_dev only. Do not load plan4_fresh, plan_dev, or a plan_test file.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_span" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_span.")
        return 2
    if not path.is_file():
        print(f"Missing {path}.")
        return 2
    examples = load_plan4_jsonl(path)
    by_id = {ex.example_id: ex for ex in examples}
    for ex in examples:
        if ex.split != "plan4_dev":
            print(f"Refusing {ex.example_id}.")
            return 2
        assert_plan4_invariants(ex)

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
            "stage": "plan4-span",
            "split": "plan4_dev",
            "n_examples": len(examples),
            "n_families": 30,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "selectors": ["result", "probe"],
            "shared_across_heads": True,
            "extra_sink": False,
            "compressor": "none",
            "obf": False,
            "a_path": "ordinary written calculations on every branch; not latent-40",
            "b_prompt": "frozen line-by-line; KV uses relayed-memory wording",
            "probe": "capacity/eligibility/conditions request; no prices or budget",
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan4_fresh, plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
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
        if n_full < n_prompt:
            raise RuntimeError("Written cache is shorter than the prompt.")
        rec = {
            "example_id": src_ex.example_id,
            "family_id": src_ex.family_id,
            "variant": src_ex.variant,
            "difficulty": src_ex.difficulty,
            "a_state_key": key,
            "encode_prompt": encode_prompt,
            "written_handoff": written,
            "ordinary_prompt": ordinary,
            "written": text,
            "written_n_new": n_new,
            "written_truncated": truncated,
            "written_s": gen_s,
            "written_bytes": len(text.encode("utf-8")),
            "handoff": handoff_from_written(text),
            "n_kv_positions": n_full,
            "n_prompt_positions": n_prompt,
            "full_relay_bytes": int(bundle.full_relay_bytes),
            "latent_steps": 0,
            "written_score": score_sender_message(
                text, gold_quantities(src_ex), gold_eligibility_statuses(src_ex)
            ),
            "written_working": written_working_metrics(text),
        }
        artifacts[key] = rec
        with artifacts_path.open("w", encoding="utf-8") as handle:
            for item in artifacts.values():
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        states[key] = {
            "bundle": bundle,
            "meta": rec,
            "attentions": None,
            "probe_s": 0.0,
            "n_probe": 0,
        }
        return states[key]

    def ensure_probe(pack: dict, ex) -> None:
        if pack["attentions"] is not None:
            return
        probe_text = b_need_probe_prompt(ex)
        if prompt_contains_b_private(probe_text, ex):
            raise RuntimeError("Probe request leaked B's prices.")
        attn, probe_s, n_probe = runner.probe_completed_cache_attentions(pack["bundle"], probe_text)
        cache_len = n_kv_positions(pack["bundle"].cache_cpu)
        if int(attn[0][0].shape[-1]) != cache_len:
            raise RuntimeError("Probe scores are not the completed-cache length.")
        pack["attentions"] = attn
        pack["probe_s"] = probe_s
        pack["n_probe"] = n_probe

    def ensure_spans(pack: dict, ex, *, need_probe: bool) -> None:
        bundle = pack["bundle"]
        full_ids = bundle.full_ids_cpu.tolist()
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
        pack["n_full"] = n_full
        if pack.get("result_pos") is None:
            written = line_spans(full_ids, start=n_prompt, end=n_full, decode=decode)
            selected_result = result_spans(written)
            fallback = False
            if not selected_result:
                selected_result = written[-4:] if len(written) >= 4 else list(written)
                fallback = True
            result_pos = positions_from_spans(selected_result)
            if not result_pos:
                raise RuntimeError(f"No complete spans in the written region for {ex.example_id}.")
            pack["result_spans"] = selected_result
            pack["result_pos"] = result_pos
            pack["result_text"] = decode_positions(full_ids, result_pos, decode)
            pack["result_fallback"] = fallback
        if need_probe and pack.get("probe_pos") is None:
            ensure_probe(pack, ex)
            scores = shared_position_scores(pack["attentions"])
            if len(scores) != n_full:
                raise RuntimeError("Shared probe scores are not the completed-cache length.")
            all_lines = line_spans(full_ids, start=0, end=n_full, decode=decode)
            probe_selected = select_spans_by_score(
                all_lines, scores, target_n=max(len(pack["result_pos"]), 1)
            )
            probe_pos = positions_from_spans(probe_selected)
            if not probe_pos:
                raise RuntimeError(f"Probe selector returned no positions for {ex.example_id}.")
            pack["probe_spans"] = probe_selected
            pack["probe_pos"] = probe_pos
            pack["probe_text"] = decode_positions(full_ids, probe_pos, decode)

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
        needs_spans = any(cond != "kv" for cond in remaining)
        need_probe = any(cond.endswith("probe") for cond in remaining)
        if needs_spans:
            ensure_spans(pack, ex, need_probe=need_probe)

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
                rec["condition"] = cond
                rec["selector"] = "none"
                rec["channel"] = "kv"
                rec["shared_across_heads"] = True
                rec["n_kept"] = int(rec.get("n_kv_positions") or 0)
                rec["n_spans"] = 0
                rec["compression_time_s"] = 0.0
                rec["probe_time_s"] = 0.0
                rec["selection_time_s"] = 0.0
                _attach_recovery(rec, selected_text=pack["meta"].get("written") or "", ex=ex)
            elif cond in ("kv_result", "text_result", "kv_probe", "text_probe"):
                selector = "result" if cond.endswith("result") else "probe"
                positions = pack["result_pos"] if selector == "result" else pack["probe_pos"]
                selected_text = pack["result_text"] if selector == "result" else pack["probe_text"]
                n_spans = len(pack["result_spans"] if selector == "result" else pack["probe_spans"])
                probe_s = float(pack["probe_s"] or 0.0) if selector == "probe" else 0.0
                if cond.startswith("kv_"):
                    compressed, wall, info = runner.keep_shared_positions(pack["bundle"], positions)
                    if int(info["n_kept"]) != len(set(positions)):
                        raise RuntimeError("Gathered KV length does not match the shared position set.")
                    rec = _run_kv_b(
                        runner,
                        ex,
                        compressed,
                        plan_max_new=args.plan_max_new,
                        answer_max_new=args.max_new_tokens,
                    )
                    rec.update(extra_a)
                    rec["channel"] = "kv"
                    rec["handoff"] = cond
                    rec["selection_time_s"] = float(wall)
                    rec["compression_time_s"] = float(wall) + probe_s
                    rec.update({k: v for k, v in info.items() if k != "positions"})
                    rec["n_kept"] = int(info["n_kept"])
                    rec["shared_across_heads"] = True
                else:
                    rec = _run_text_b(
                        runner,
                        ex,
                        message=selected_text,
                        plan_max_new=args.plan_max_new,
                        answer_max_new=args.max_new_tokens,
                        allow_a_source=True,
                    )
                    rec.update(extra_a)
                    rec["channel"] = "text"
                    rec["handoff"] = cond
                    rec["selection_time_s"] = 0.0
                    rec["compression_time_s"] = probe_s
                    rec["n_kept"] = len(set(positions))
                    rec["n_kv_positions"] = 0
                    rec["shared_across_heads"] = True
                rec["condition"] = cond
                rec["selector"] = selector
                rec["n_spans"] = n_spans
                rec["result_fallback"] = bool(pack.get("result_fallback"))
                rec["probe_time_s"] = probe_s
                rec["n_probe_tokens"] = int(pack["n_probe"] or 0) if selector == "probe" else 0
                rec["n_full"] = int(pack.get("n_full") or n_kv_positions(pack["bundle"].cache_cpu))
                _attach_recovery(rec, selected_text=selected_text, ex=ex)
            else:
                raise RuntimeError(f"Unknown condition {cond}")

            rec["reused_a_from"] = pack["meta"]["a_state_key"] if pack["meta"]["a_state_key"] != ex.example_id else None
            rec["process_time_s"] = _process_time_s(rec)
            rows.append(rec)
            done.add((ex.example_id, cond))
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cond}: kind={rec['kind']} pred={rec['pred_carrier']} "
                f"correct={rec['correct']} selected_four={rec.get('selected_all_four')} "
                f"plan_four={rec.get('plan_all_four')} bytes={rec.get('message_bytes')} "
                f"kept={rec.get('n_kept')}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Interpret same-span KV versus text on plan4_dev. Do not resume the fresh folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
