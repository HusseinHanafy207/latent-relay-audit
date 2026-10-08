"""Resolve and pin the Hugging Face snapshot id actually loaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.paths import PINNED_PATH
from src.manifest import load_pinned


def revision_from_model(model: Any) -> Optional[str]:
    cfg = getattr(model, "config", None)
    commit = getattr(cfg, "_commit_hash", None) if cfg is not None else None
    if isinstance(commit, str) and len(commit) >= 7:
        return commit
    return None


def revision_from_hub(model_name: str) -> Optional[str]:
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha:
            return sha
    except Exception:
        return None
    return None


def revision_from_local_cache(model_name: str) -> Optional[str]:
    try:
        from huggingface_hub import snapshot_download

        cache = Path(snapshot_download(model_name, local_files_only=True))
    except Exception:
        return None
    for cand in (cache / "refs" / "main", cache.parent / "refs" / "main"):
        if cand.is_file():
            text = cand.read_text(encoding="utf-8").strip()
            if text:
                return text
    snaps = cache.parent / "snapshots"
    if snaps.is_dir():
        hashes = sorted(p.name for p in snaps.iterdir() if p.is_dir() and len(p.name) >= 16)
        if len(hashes) == 1:
            return hashes[0]
    return None


def resolve_model_revision(model_name: str, *, model: Any = None) -> dict[str, Any]:
    pinned = load_pinned().get("model", {})
    pinned_sha = pinned.get("revision")
    found: list[tuple[str, str]] = []
    if model is not None:
        sha = revision_from_model(model)
        if sha:
            found.append((sha, "model.config._commit_hash"))
    hub = revision_from_hub(model_name)
    if hub:
        found.append((hub, "huggingface_hub.model_info"))
    local = revision_from_local_cache(model_name)
    if local:
        found.append((local, "local_cache"))
    loaded = found[0][0] if found else None
    source = found[0][1] if found else "unresolved"
    mismatch = bool(pinned_sha and loaded and pinned_sha != loaded)
    sha = pinned_sha or loaded
    return {
        "sha": sha,
        "pinned": pinned_sha,
        "loaded": loaded,
        "source": "pinned.json" if pinned_sha else source,
        "candidates": [{"sha": s, "source": src} for s, src in found],
        "mismatch": mismatch,
    }


def pin_model_revision(sha: str) -> None:
    pinned = load_pinned()
    if pinned.get("model", {}).get("revision") == sha:
        return
    pinned.setdefault("model", {})["revision"] = sha
    PINNED_PATH.write_text(json.dumps(pinned, indent=2) + "\n", encoding="utf-8")
