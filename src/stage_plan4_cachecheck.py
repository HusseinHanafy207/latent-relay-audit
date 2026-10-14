"""Tiny identical-token cache check. Not generate_from_cache's new-chat handoff.

Prefill a prefix of one tokenized prompt, then continue with the remaining
ids. Compare with generating from the full ids. Uses two compact concatenated
prompts from plan4_dev plus one short dummy. Preserves plan4_cal and plan4_diag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifest import load_pinned, write_manifest
from src.model_revision import pin_model_revision, resolve_model_revision
from src.paths import DATA_DIR, RESULTS_DIR
from src.plan4_task import load_plan4_jsonl
from src.precision import apply_precision_policy, peak_gpu_memory_gb, reset_peak_memory
from src.prompts_plan4 import a_local_task_prompt, compact_result_handoff, ordinary_sender_prompt
from src.stage_select import _load_rows, _persist as _persist_core
from src.two_agent import TwoAgentInventory

DUMMY = "Reply with exactly this sentence: ping ping."
PROMPT_IDS = ("plan4_dev-fam-006-base", "plan4_dev-fam-000-base")
SPLITS = (0.5,)
MAX_NEW = 64
LEAD = (
    "Identical token prefix vs full generate. Both paths used the same input tokens; "
    "no extra chat turn was inserted. A text mismatch would warrant investigation; "
    "numerical differences can also change generated output. Matching text does not "
    "mean the answers were correct, and does not validate the 40-step latent rollout, "
    "compression, or the full agent-to-agent handoff. Differing outputs in the earlier "
    "plan4_diag ordinary-versus-cached comparison used differently formatted prompts, "
    "so they were not evidence of cache corruption."
)
PRESERVED = (
    "results_stage_plan4_cal",
    "results_stage_plan4_diag",
    "results_stage_study1",
)


def _paths():
    data = DATA_DIR / "plan4_dev.jsonl"
    out_root = RESULTS_DIR / "stage_plan4_cachecheck" / "run"
    kaggle = Path("/kaggle/working/results_stage_plan4_cachecheck")
    kaggle_dest = kaggle if Path("/kaggle/working").exists() else None
    return data, out_root, kaggle_dest


def _persist(out_root: Path, extra_dest: Path | None = None) -> None:
    _persist_core(out_root, extra_dest)


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Identical-token prefix vs full generate.")
    parser.add_argument("--split", default="plan4_dev")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.split != "plan4_dev":
        print("Cache check uses two plan4_dev prompts only.")
        return 2

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)
    if policy.device_name and "T4" not in policy.device_name and policy.device_name != "cpu":
        print(f"Warning: expected the locked Tesla T4, got {policy.device_name}. Do not mix GPU types.")

    data, out_root, kaggle_dest = _paths()
    if "stage_plan4_cachecheck" not in str(out_root):
        print("Refusing to write outside results_stage_plan4_cachecheck.")
        return 2
    if args.fresh and out_root.exists():
        out_root = out_root.parent / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "evaluations.jsonl"
    if kaggle_dest is not None and (kaggle_dest / "evaluations.jsonl").is_file() and not jsonl_path.is_file():
        jsonl_path.write_bytes((kaggle_dest / "evaluations.jsonl").read_bytes())
        for name in ("summary.json", "manifest.json"):
            src = kaggle_dest / name
            if src.is_file():
                (out_root / name).write_bytes(src.read_bytes())

    rows = _load_rows(jsonl_path)
    done = {(row["prompt_id"], row["split_kind"]) for row in rows}
    write_manifest(
        out_root / "manifest.json",
        policy=policy,
        extra={
            "stage": "plan4-cachecheck",
            "do_not_resume": list(PRESERVED),
            "identical_tokens": True,
            "new_chat_handoff": False,
            "deferred": ["B prompt comparison", "full plan4_dev calibration", "compression"],
        },
    )
    _persist(out_root, kaggle_dest)

    runner = TwoAgentInventory(
        args.model_name, args.device, latent_steps=0, kv_budget=32, sink_size=4, pca_rank=2, max_new_tokens=MAX_NEW
    )
    revision = resolve_model_revision(args.model_name, model=runner.wrapper.model)
    if revision.get("mismatch"):
        print(f"Warning: pinned snapshot {revision['pinned']} != loaded {revision['loaded']}")
    if revision.get("sha") and not pinned.get("model", {}).get("revision"):
        pin_model_revision(revision["sha"])

    examples = {ex.example_id: ex for ex in load_plan4_jsonl(data)}
    jobs = [("dummy", DUMMY)]
    for eid in PROMPT_IDS:
        ex = examples[eid]
        jobs.append((eid, ordinary_sender_prompt(a_local_task_prompt(ex), compact_result_handoff())))

    for prompt_id, prompt in jobs:
        for split in SPLITS:
            key = (prompt_id, str(split))
            if key in done:
                print(f"{prompt_id} split={split} already complete", flush=True)
                continue
            print(f"{prompt_id} split={split}", flush=True)
            reset_peak_memory()
            rec = runner.compare_identical_token_prefix(prompt, split=split, max_new_tokens=MAX_NEW)
            rec.update(
                {
                    "prompt_id": prompt_id,
                    "split_kind": str(split),
                    "peak_gpu_gb": peak_gpu_memory_gb(),
                }
            )
            rows.append(rec)
            done.add(key)
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  equal={rec['texts_equal']} n_prompt={rec['n_prompt_tokens']} cut={rec['split_at']}", flush=True)
            summary = {
                "n": len(rows),
                "n_equal": sum(bool(r.get("texts_equal")) for r in rows),
                "n_truncated": sum(
                    bool(r.get("ordinary_truncated") or r.get("cached_truncated")) for r in rows
                ),
                "rows": [
                    {
                        "prompt_id": r["prompt_id"],
                        "split_kind": r["split_kind"],
                        "texts_equal": r["texts_equal"],
                        "n_prompt_tokens": r["n_prompt_tokens"],
                        "split_at": r["split_at"],
                        "ordinary_truncated": r.get("ordinary_truncated"),
                        "cached_truncated": r.get("cached_truncated"),
                    }
                    for r in rows
                ],
                "lead": LEAD,
            }
            (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            _persist(out_root, kaggle_dest)

    print(json.dumps(json.loads((out_root / "summary.json").read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
