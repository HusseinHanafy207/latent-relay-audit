"""Stage A: confirm the released pipeline fits and runs on the locked T4 policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifest import load_pinned, write_manifest
from src.paths import RESULTS_DIR, UPSTREAM_DIR
from src.precision import (
    apply_precision_policy,
    peak_gpu_memory_gb,
    reset_peak_memory,
)


def _probe_alignment(model_name: str, device: str) -> dict:
    """Load the wrapper and build the FP32 realignment matrix. Do not skip this check."""
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))

    import torch
    from models import ModelWrapper

    class _Args:
        latent_space_realign = False

    reset_peak_memory()
    report: dict = {"ok": False, "error": None, "peak_gpu_gb_after_load": None, "peak_gpu_gb_after_align": None}
    try:
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        wrapper = ModelWrapper(model_name, device_obj, args=_Args())
        report["model_param_dtype"] = str(next(wrapper.model.parameters()).dtype)
        report["peak_gpu_gb_after_load"] = peak_gpu_memory_gb()
        wrapper._ensure_latent_realign_matrix(wrapper.model, device_obj)
        matrix, target_norm = wrapper._latent_realign_matrices[id(wrapper.model)]
        report["align_matrix_dtype"] = str(matrix.dtype)
        report["target_norm_dtype"] = str(target_norm.dtype)
        report["peak_gpu_gb_after_align"] = peak_gpu_memory_gb()
        report["ok"] = True
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["peak_gpu_gb_after_align"] = peak_gpu_memory_gb()
    return report


def _run_compressor(compressor: str, args: argparse.Namespace, out_root: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "src.run_upstream",
        "--method",
        "latent_mas",
        "--prompt",
        "sequential",
        "--model_name",
        args.model_name,
        "--task",
        args.task,
        "--compressor",
        compressor,
        "--max_samples",
        str(args.max_samples),
        "--generate_bs",
        "1",
        "--latent_steps",
        str(args.latent_steps),
        "--kv_budget",
        str(args.kv_budget),
        "--sink_size",
        str(args.sink_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        "0.0",
        "--top_p",
        "1.0",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output_folder",
        str(out_root / compressor),
    ]
    if compressor == "hobf_fast":
        cmd.extend(["--pca_rank", str(args.pca_rank)])

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        "compressor": compressor,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 2),
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    pinned = load_pinned()
    parser = argparse.ArgumentParser(description="Stage A smoke: full / headwise / hobf_fast on the T4 policy.")
    parser.add_argument("--model_name", default=pinned["model"]["name"])
    parser.add_argument("--task", default=pinned["stage_a"]["task"])
    parser.add_argument("--max_samples", type=int, default=pinned["stage_a"]["max_samples"])
    parser.add_argument("--latent_steps", type=int, default=pinned["stage_a"]["latent_steps"])
    parser.add_argument("--kv_budget", type=int, default=pinned["stage_a"]["kv_budget"])
    parser.add_argument("--sink_size", type=int, default=pinned["stage_a"]["sink_size"])
    parser.add_argument("--pca_rank", type=int, default=pinned["stage_a"]["pca_rank_hobf"])
    parser.add_argument("--max_new_tokens", type=int, default=pinned["stage_a"]["max_new_tokens"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_alignment_probe", action="store_true")
    parser.add_argument("--compressors", nargs="+", default=pinned["stage_a"]["compressors"])
    args = parser.parse_args()

    os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
    policy = apply_precision_policy(enable_thinking=False)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = RESULTS_DIR / "stage_a" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "phase": "A",
        "output_dir": str(out_root),
        "policy": {
            "inference_dtype": policy.inference_dtype,
            "alignment_dtype": policy.alignment_dtype,
            "compression_dtype": policy.compression_dtype,
            "device_name": policy.device_name,
            "device_capability": policy.device_capability,
            "source": policy.source,
        },
        "alignment_probe": None,
        "compressors": [],
    }

    write_manifest(out_root / "manifest.json", policy=policy, extra={"stage": "A"})

    if not args.skip_alignment_probe:
        print("Stage A: probing model load + FP32 alignment matrix...")
        summary["alignment_probe"] = _probe_alignment(args.model_name, args.device)
        print(json.dumps(summary["alignment_probe"], indent=2))
        (out_root / "alignment_probe.json").write_text(
            json.dumps(summary["alignment_probe"], indent=2), encoding="utf-8"
        )
        if not summary["alignment_probe"]["ok"]:
            print(
                "Alignment probe failed. Continue compressor smokes only if the error is "
                "alignment-specific; a model-load OOM means Qwen3-4B does not fit."
            )

    for compressor in args.compressors:
        print(f"Stage A: smoking compressor={compressor}")
        result = _run_compressor(compressor, args, out_root)
        summary["compressors"].append(result)
        (out_root / f"{compressor}_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  returncode={result['returncode']} seconds={result['seconds']}")
        if result["returncode"] != 0:
            print(result["stderr_tail"])

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Stage A summary written to {out_root / 'summary.json'}")
    failed = [c for c in summary["compressors"] if c["returncode"] != 0]
    probe_failed = summary["alignment_probe"] is not None and not summary["alignment_probe"]["ok"]
    if failed or probe_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
