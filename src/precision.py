"""T4 precision policy for the released LatentMAS + OBF pipeline.

Inference tensors use float16 on Turing (T4). Alignment-matrix construction
and OBF residual algebra stay in float32. That split is already present in the
upstream code; this module makes the inference-dtype override explicit and
applies it *before* the first model load.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch


_PATCHED = False


@dataclass(frozen=True)
class PrecisionPolicy:
    inference_dtype: str
    alignment_dtype: str = "float32"
    compression_dtype: str = "float32"
    enable_thinking: bool = False
    device_name: str = ""
    device_capability: str = ""
    source: str = ""

    def torch_inference_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.inference_dtype]


def _capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def select_inference_dtype() -> tuple[torch.dtype, str]:
    """Choose inference dtype. Env LATENT_RELAY_INFER_DTYPE overrides autodetection."""
    env = os.environ.get("LATENT_RELAY_INFER_DTYPE", "").strip().lower()
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if env:
        if env not in aliases:
            raise ValueError(f"Unknown LATENT_RELAY_INFER_DTYPE={env!r}")
        return aliases[env], f"env:{env}"

    if not torch.cuda.is_available():
        return torch.float32, "cpu"

    major, _minor = _capability()
    # Ampere (sm80) and newer have native bfloat16. T4 is Turing sm75.
    if major >= 8:
        return torch.bfloat16, "auto:ampere+"
    return torch.float16, "auto:pre-ampere"


def current_policy(*, enable_thinking: bool = False) -> PrecisionPolicy:
    dtype, source = select_inference_dtype()
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    major, minor = _capability()
    cap = f"{major}.{minor}" if torch.cuda.is_available() else "cpu"
    return PrecisionPolicy(
        inference_dtype=str(dtype).replace("torch.", ""),
        enable_thinking=enable_thinking,
        device_name=name,
        device_capability=cap,
        source=source,
    )


def apply_precision_policy(*, enable_thinking: bool = False) -> PrecisionPolicy:
    """Patch upstream model loading. Call before constructing ModelWrapper.

    - Replaces the hardcoded bfloat16 in models.py from_pretrained.
    - Leaves _build_latent_realign_matrix on float32 (upstream already does this).
    - Leaves HOBF/HOBFFast QR-SVD-injection on float32 (upstream already does this).
    - Pins Qwen3 chat-template thinking off unless enable_thinking is True.
    """
    global _PATCHED
    from transformers import AutoModelForCausalLM

    policy = current_policy(enable_thinking=enable_thinking)
    infer_dtype = policy.torch_inference_dtype()

    if not _PATCHED:
        _orig_from_pretrained = AutoModelForCausalLM.from_pretrained

        def _from_pretrained(*args: Any, **kwargs: Any):
            # Upstream models.py hardcodes bfloat16. Leave other callers alone.
            if kwargs.get("torch_dtype") is torch.bfloat16:
                kwargs["torch_dtype"] = infer_dtype
            return _orig_from_pretrained(*args, **kwargs)

        AutoModelForCausalLM.from_pretrained = _from_pretrained  # type: ignore[method-assign]
        _PATCHED = True

    _patch_chat_template(enable_thinking=enable_thinking)
    return policy


def _patch_chat_template(*, enable_thinking: bool) -> None:
    """Pin Qwen3 thinking. Harmless if the tokenizer has no such kwarg."""
    from transformers import PreTrainedTokenizerBase

    orig = PreTrainedTokenizerBase.apply_chat_template
    if getattr(orig, "_latent_relay_thinking_patched", False):
        return

    def wrapped(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("enable_thinking", enable_thinking)
        try:
            return orig(self, *args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return orig(self, *args, **kwargs)

    wrapped._latent_relay_thinking_patched = True  # type: ignore[attr-defined]
    PreTrainedTokenizerBase.apply_chat_template = wrapped  # type: ignore[method-assign]


def peak_gpu_memory_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def policy_dict(policy: PrecisionPolicy) -> dict[str, Any]:
    return asdict(policy)
