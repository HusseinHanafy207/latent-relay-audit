"""Single-carrier eligibility questions for the twelve-carrier query check.

B keeps the same private facts (shipment request, prices, budget). Only the
requested output changes: one label for one named carrier. The frozen
four-carrier and twelve-carrier plan prompts stay unchanged.
"""

from __future__ import annotations

import re

from src.prompts_plan4 import FRAMING_INSTRUCTION

LABELS = ("eligible", "ineligible")
QUERY_MAX_NEW = 8
_EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*", re.IGNORECASE)
_WRAP = "*\"'`"


def single_label_question(carrier: str) -> str:
    return (
        f"According to Agent A's results, is Carrier {carrier} eligible or ineligible? "
        "Reply with exactly eligible or ineligible."
    )


def b_query_kv_prompt(ex, carrier: str) -> str:
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent (Agent A) already processed operational information. "
            "Use the relayed memory if it helps.",
            FRAMING_INSTRUCTION,
            ex.b_text,
            single_label_question(carrier),
        ]
    )


def b_query_text_prompt(ex, message: str, carrier: str) -> str:
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent (Agent A) sent the message below.",
            message.strip(),
            ex.b_text,
            single_label_question(carrier),
        ]
    )


def parse_single_label(text: str) -> dict:
    """Whole-answer match. "ineligible" can never be read as "eligible"."""
    raw = text or ""
    body = _EMPTY_THINK.sub("", raw).strip()
    exact_raw = body in LABELS
    norm = body.strip(_WRAP).strip()
    if norm.endswith("."):
        norm = norm[:-1].strip()
    norm = norm.strip(_WRAP).strip().lower()
    label = norm if norm in LABELS else None
    return {
        "pred_label": label,
        "parse_ok": label is not None,
        "exact_raw": exact_raw,
        "normalized_answer": norm,
    }
