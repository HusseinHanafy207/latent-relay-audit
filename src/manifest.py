"""Run-manifest helpers. Pin versions before Stage A and copy them with results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.paths import PINNED_PATH, REPO_ROOT, UPSTREAM_DIR
from src.precision import PrecisionPolicy, policy_dict


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(cwd), stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.decode("utf-8").strip() or None


def load_pinned() -> dict[str, Any]:
    return json.loads(PINNED_PATH.read_text(encoding="utf-8"))


def collect_versions(*, policy: PrecisionPolicy, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    import torch

    try:
        import transformers
        transformers_version = getattr(transformers, "__version__", None)
    except ImportError:
        transformers_version = None

    model_revision = None
    pinned = load_pinned()
    if extra and extra.get("model_revision"):
        model_revision = extra["model_revision"]
    elif pinned.get("model", {}).get("revision"):
        model_revision = pinned["model"]["revision"]
    else:
        try:
            from huggingface_hub import snapshot_download

            model_name = pinned["model"]["name"]
            cache = snapshot_download(model_name, local_files_only=True)
            refs = Path(cache) / "refs" / "main"
            if refs.is_file():
                model_revision = refs.read_text(encoding="utf-8").strip()
        except Exception:
            model_revision = None

    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "repo_commit": _run_git(["rev-parse", "HEAD"], REPO_ROOT),
        "upstream_commit_expected": load_pinned()["upstream"]["commit"],
        "upstream_commit_actual": _run_git(["rev-parse", "HEAD"], UPSTREAM_DIR) if UPSTREAM_DIR.is_dir() else None,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "transformers": transformers_version,
        "precision": policy_dict(policy),
        "env": {
            "LATENT_RELAY_INFER_DTYPE": os.environ.get("LATENT_RELAY_INFER_DTYPE"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        },
        "model_revision": model_revision,
        "model_name": load_pinned()["model"]["name"],
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_manifest(path: Path, *, policy: PrecisionPolicy, extra: Optional[dict[str, Any]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_versions(policy=policy, extra=extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
