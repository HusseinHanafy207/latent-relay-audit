"""Repository paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
PINNED_PATH = CONFIGS_DIR / "pinned.json"
PROPOSAL_DIR = REPO_ROOT / "proposal"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
UPSTREAM_DIR = THIRD_PARTY_DIR / "When-Less-Latent-Leads-to-Better-Relay"
