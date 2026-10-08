"""Precision-policy unit tests. No GPU required."""

from __future__ import annotations

import os

import torch

from src.precision import select_inference_dtype


def test_env_fp16_overrides_autodetect():
    os.environ["LATENT_RELAY_INFER_DTYPE"] = "fp16"
    try:
        dtype, source = select_inference_dtype()
        assert dtype is torch.float16
        assert source == "env:fp16"
    finally:
        os.environ.pop("LATENT_RELAY_INFER_DTYPE", None)


def test_env_rejects_unknown():
    os.environ["LATENT_RELAY_INFER_DTYPE"] = "fp8"
    try:
        try:
            select_inference_dtype()
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        os.environ.pop("LATENT_RELAY_INFER_DTYPE", None)
