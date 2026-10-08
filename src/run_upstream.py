"""Apply the T4 precision policy, then run the upstream LatentMAS CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import UPSTREAM_DIR  # noqa: E402
from src.precision import apply_precision_policy  # noqa: E402

if not UPSTREAM_DIR.is_dir():
    raise SystemExit(
        f"Upstream repo missing at {UPSTREAM_DIR}. Run: python scripts/fetch_upstream.py"
    )

# Precision override must happen before ModelWrapper loads the model.
os.environ.setdefault("LATENT_RELAY_INFER_DTYPE", "fp16")
apply_precision_policy(enable_thinking=False)

if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))
os.chdir(UPSTREAM_DIR)

from run import main  # noqa: E402

if __name__ == "__main__":
    main()
