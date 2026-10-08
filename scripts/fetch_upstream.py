"""Clone or verify the pinned compression repository."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED = json.loads((ROOT / "configs" / "pinned.json").read_text(encoding="utf-8"))
DEST = ROOT / "third_party" / "When-Less-Latent-Leads-to-Better-Relay"
COMMIT = PINNED["upstream"]["commit"]
URL = PINNED["upstream"]["repo"]


def _git(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd) if cwd else None).decode().strip()


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not (DEST / ".git").exists():
        print(f"Cloning {URL} into {DEST}")
        subprocess.check_call(["git", "clone", URL, str(DEST)])
    actual = _git(["rev-parse", "HEAD"], cwd=DEST)
    if actual != COMMIT:
        print(f"Checking out pinned commit {COMMIT} (was {actual})")
        subprocess.check_call(["git", "fetch", "--depth", "1", "origin", COMMIT], cwd=DEST)
        subprocess.check_call(["git", "checkout", COMMIT], cwd=DEST)
        actual = _git(["rev-parse", "HEAD"], cwd=DEST)
    if actual != COMMIT:
        print(f"ERROR: upstream HEAD {actual} != pinned {COMMIT}", file=sys.stderr)
        return 1
    print(f"Upstream pinned at {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
