"""Question-conditioned whole-cache selection, then OBF on the same budgets.

A writes calculations once. Probe request names capacity, eligibility, and
conditions; it does not include B's prices or budget. Ranking scores the
completed cache, including written tokens, and discards probe keys.

- kv: full completed cache
- probe_32 / probe_64: headwise top-k on those probe scores
- obf_32 / obf_64: same scores and budgets, then residual injection

OBF here is an adaptation to whole-cache selection. It is not a replication
of protected-reasoning (reserved latent tail).

plan4_dev only. Frozen line-B. Do not resume plan4_compress or earlier folders.
Small budgets stayed poor. The next GPU run is the predeclared budget-256
probe-only check. Use --rescore to rewrite summary.json from recorded rows
without a GPU pass. The summary averages saved process_time_s values.
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
from src.stage_plan0 import B_ANSWER_MAX_NEW, _flip_tracking, _load_messages, _subset
from src.stage_plan4_cal import _process_time_s as _base_process_time_s
from src.stage_plan4_comm import _a_extra, _comm_block, _run_kv_b, _sender_block
from src.stage_plan4_compress import BUDGET_SCOPE, SINK_SIZE, _family_delta
from src.stage_select import _load_rows, _paired, _persist as _persist_core
from src.two_agent import TwoAgentInventory

CONDS = ("kv", "probe_32", "probe_64", "obf_32", "obf_64")
COMPRESS_CONDS = {
    "probe_32": ("headwise", 32),
    "probe_64": ("headwise", 64),
    "obf_32": ("hobf_fast", 32),
    "obf_64": ("hobf_fast", 64),
}
A_WRITTEN_MAX_NEW = 768
B_PLAN_MAX_NEW = 128
PREDECLARED_LARGER_BUDGET = 256
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
)


def _jobs():
    return [{"condition": name} for name in CONDS]


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_probe" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_probe")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_core(out_root, extra_dest)
    dests = [out_root.parent / "latest"]
    if extra_dest is not None:
        dests.append(extra_dest)
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        src = out_root / "a_artifacts.jsonl"
        if src.is_file():
            (dest / "a_artifacts.jsonl").write_bytes(src.read_bytes())


def _process_time_s(row: dict) -> float:
    """Charge written A, probe ranking, compression/OBF, and B's two passes."""
    return _base_process_time_s(row) + float(row.get("compression_time_s") or 0.0)


def _probe_block(items: list[dict]) -> dict:
    out = _comm_block(items)
    n = len(items)
    if not n:
        return out
    recorded = [float(item["process_time_s"]) for item in items if item.get("process_time_s") is not None]
    if recorded:
        out["mean_process_time_s"] = sum(recorded) / len(recorded)
    out["mean_probe_time_s"] = sum(item.get("probe_time_s") or 0 for item in items) / n
    out["mean_compression_core_s"] = sum(item.get("compression_core_s") or 0 for item in items) / n
    out["mean_n_probe_tokens"] = sum(int(item.get("n_probe_tokens") or 0) for item in items) / n
    out["mean_n_kept"] = sum(int(item.get("n_kept") or item.get("n_kv_positions") or 0) for item in items) / n
    written_sel = [item.get("n_written_selected") for item in items if item.get("n_written_selected") is not None]
    out["mean_n_written_selected"] = (sum(written_sel) / len(written_sel)) if written_sel else None
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
            "Same written A on every branch. Probe scores the completed cache, "
            "including written tokens, and discards probe keys. "
            "Dummy-token compress reference: kv_32 28/90, kv_64 29/90, full KV 82/90."
        ),
        "sender_unique_sheets": _sender_block(unique_a),
        "receiver": {cond: _probe_block(by_cond.get(cond, [])) for cond in CONDS},
        "by_difficulty": {},
        "paired_receiver": {
            "probe_32_vs_kv": _paired(grouped_plain, "kv", "probe_32"),
            "probe_64_vs_kv": _paired(grouped_plain, "kv", "probe_64"),
            "obf_32_vs_probe_32": _paired(grouped_plain, "probe_32", "obf_32"),
            "obf_64_vs_probe_64": _paired(grouped_plain, "probe_64", "obf_64"),
            "obf_32_vs_kv": _paired(grouped_plain, "kv", "obf_32"),
            "obf_64_vs_kv": _paired(grouped_plain, "kv", "obf_64"),
        },
        "family_clustered_delta": {
            "probe_32_vs_kv": _family_delta(rows, "kv", "probe_32"),
            "probe_64_vs_kv": _family_delta(rows, "kv", "probe_64"),
            "obf_32_vs_probe_32": _family_delta(rows, "probe_32", "obf_32"),
            "obf_64_vs_probe_64": _family_delta(rows, "probe_64", "obf_64"),
        },
        "flip_tracking": {cond: _flip_tracking(rows, cond) for cond in CONDS},
        "lead": (
            "Question-conditioned whole-cache headwise at 32 and 64, then OBF on "
            "the same scores and budgets. Frozen line-B. No dummy-token ranking. "
            "OBF is adapted to whole-cache selection, not protected latent tails."
        ),
        "budget_note": BUDGET_SCOPE,
        "probe_note": (
            "Probe request asks for capacity, eligibility, and shipment conditions. "
            "It does not include prices or budget. Ranking uses completed-cache keys; "
            "probe_attentions_to_steps(prompt_len=original prompt) is not used."
        ),
        "obf_note": (
            "hobf_fast uses the same probe scores and top-k budget as probe_32/64, "
            "then injects discarded-token residual into kept values. pca_rank=2. "
            "Not a faithful copy of the inventory protected-reasoning setup."
        ),
        "timing_note": (
            "mean_process_time_s averages the recorded process_time_s on each row "
            "(written A, probe ranking, compression/OBF, and B's plan and answer). "
            "It does not recompute those totals through the older helper that omitted "
            "probe and compression. probe_time_s and compression_core_s are separate. "
            "Network is not measured."
        ),
        "deferred": {
            "larger_budget": (
                f"Both 32 and 64 stayed poor. Next is the predeclared "
                f"budget-{PREDECLARED_LARGER_BUDGET} probe-only check (no OBF)."
            ),
            "fresh_split": "After the budget-256 check.",
        },
        "b_prompt": "frozen line-by-line; KV uses the same procedure over relayed memory",
    }
    for difficulty in LEVELS:
        out["by_difficulty"][difficulty] = {
            "receiver": {
                cond: _probe_block(_subset(by_cond.get(cond, []), difficulty=difficulty)) for cond in CONDS
            }
        }
    peaks = [row.get("peak_gpu_gb") or 0 for row in rows]
    out["peak_gpu_gb_max_over_run"] = max(peaks) if peaks else None
    return out


def _result_dirs() -> list[Path]:
    return [
        Path("/kaggle/working/results_stage_plan4_probe"),
        REPO_ROOT / "results_stage_plan4_probe",
        RESULTS_DIR / "stage_plan4_probe" / "run",
        RESULTS_DIR / "stage_plan4_probe" / "latest",
    ]


def rescore() -> int:
    """Rewrite summary.json from saved evaluations. Does not load the model."""
    found = False
    for dest in _result_dirs():
        jsonl = dest / "evaluations.jsonl"
        if not jsonl.is_file():
            continue
        rows = _load_rows(jsonl)
        if not rows:
            continue
        report = _summarize(rows)
        (dest / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Rewrote {dest / 'summary.json'} from {len(rows)} recorded rows")
        found = True
    if not found:
        print("No saved plan4_probe evaluations.jsonl to rescore.")
        return 2
    return 0


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="plan4_dev question-conditioned budgets and OBF.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=B_ANSWER_MAX_NEW)
    parser.add_argument("--written_max_new", type=int, default=A_WRITTEN_MAX_NEW)
    parser.add_argument("--plan_max_new", type=int, default=B_PLAN_MAX_NEW)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Rewrite summary.json from recorded process_time_s. No GPU.",
    )
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Probe comparison uses plan4_dev only. Do not load plan_dev or a plan_test file.")
        return 2
    if args.rescore:
        return rescore()

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    path, out_root, kaggle_dest = _paths()
    if "stage_plan4_probe" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_probe.")
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
            "stage": "plan4-probe",
            "split": "plan4_dev",
            "n_examples": len(examples),
            "n_families": 30,
            "n_evals": len(examples) * len(CONDS),
            "conditions": list(CONDS),
            "compressor": "headwise then hobf_fast on the same probe scores",
            "kv_budgets": [32, 64],
            "sink_size": SINK_SIZE,
            "budget_scope": BUDGET_SCOPE,
            "predeclared_larger_budget": PREDECLARED_LARGER_BUDGET,
            "a_path": "ordinary written calculations on every branch; not latent-40",
            "b_prompt": "frozen line-by-line; KV uses relayed-memory wording",
            "probe": "capacity/eligibility/conditions request; no prices or budget",
            "written_max_new": args.written_max_new,
            "plan_max_new": args.plan_max_new,
            "reuse_a_for_b_change": True,
            "held_out": "plan_dev, plan_fresh, test.jsonl, hop_test, and plan_test are not loaded",
            "do_not_resume": list(PRESERVED),
            "deferred": [
                f"budget-{PREDECLARED_LARGER_BUDGET} only if both 32 and 64 stay poor",
                "fresh split after the procedure is settled",
            ],
            "resume_completed": len(done),
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name,
        args.device,
        latent_steps=0,
        kv_budget=32,
        sink_size=SINK_SIZE,
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
        n_full = n_kv_positions(bundle.cache_cpu)
        n_prompt = int(bundle.prompt_mask_cpu.sum().item())
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
        states[key] = {"bundle": bundle, "meta": rec, "attentions": None, "probe_s": 0.0, "n_probe": 0}
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
        n_prompt = int(pack["bundle"].prompt_mask_cpu.sum().item())
        if cache_len <= n_prompt:
            raise RuntimeError("Completed cache has no written tokens to score.")
        pack["attentions"] = attn
        pack["probe_s"] = probe_s
        pack["n_probe"] = n_probe

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
                rec["compression_time_s"] = 0.0
                rec["probe_time_s"] = 0.0
            elif cond in COMPRESS_CONDS:
                relay, budget = COMPRESS_CONDS[cond]
                ensure_probe(pack, ex)
                compressed, wall, core, info = runner.compress_sender(
                    pack["bundle"],
                    kv_budget=budget,
                    attentions=pack["attentions"],
                    sink_size=SINK_SIZE,
                    relay=relay,
                )
                rec = _run_kv_b(
                    runner,
                    ex,
                    compressed,
                    plan_max_new=args.plan_max_new,
                    answer_max_new=args.max_new_tokens,
                )
                rec.update(extra_a)
                rec["condition"] = cond
                rec["handoff"] = cond
                rec["probe_time_s"] = float(pack["probe_s"] or 0.0)
                rec["n_probe_tokens"] = int(pack["n_probe"] or 0)
                rec["compression_core_s"] = core
                rec["selection_time_s"] = float(wall)
                rec.update(info)
                rec["budget_scope"] = BUDGET_SCOPE
                rec["compression_time_s"] = float(wall) + float(pack["probe_s"] or 0.0)
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
                f"correct={rec['correct']} bytes={rec.get('message_bytes')} "
                f"kv={rec.get('n_kv_positions')} kept={rec.get('n_kept')} "
                f"probe_s={rec.get('probe_time_s')} core_s={rec.get('compression_core_s')}",
                flush=True,
            )
            (out_root / "summary.json").write_text(json.dumps(_summarize(rows), indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    report = _summarize(rows)
    (out_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _persist(out_root, kaggle_dest)
    print(json.dumps(report, indent=2))
    print(f"Wrote {jsonl_path} ({len(rows)} rows)")
    print("Interpret probe vs dummy vs full KV, and OBF recovery, before a larger budget or fresh split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
