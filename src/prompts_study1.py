"""Study 1 prompts: matched intermediate-result handoff vs source-evidence.

A's encode prompt is the local task only. Message format is added later as a
handoff continuation from a saved cache copy. Do not retune wording to chase
a single development error. Do not import the hop inventory retriever.
"""

from __future__ import annotations

import re

from src.plan_task import CARRIERS, QUESTION, PlanExample
from src.prompts_plan import SHORT_ANSWER_RULE


def _question_block(ex: PlanExample) -> str:
    return f"{QUESTION}\nChoices:\n{ex.choices_text}"

INTERMEDIATE_HANDOFF = (
    "Now write a message for Agent B.\n"
    "For each carrier, state your derived eligibility judgment and the conditions "
    "that support it.\n"
    "Use this format, one line per carrier:\n"
    "P: <eligible or ineligible> | <supporting conditions>\n"
    "Q: <eligible or ineligible> | <supporting conditions>\n"
    "R: <eligible or ineligible> | <supporting conditions>\n"
    "Replace each status placeholder with eligible or ineligible after using the "
    "operational facts. Do not copy the placeholders. "
    "Do not assume P is eligible, Q is ineligible, or R is eligible.\n"
    "After those three lines you may add a short explanation of the supporting "
    "conditions. Do not invent prices. Do not choose a final carrier."
)

SOURCE_EVIDENCE_HANDOFF = (
    "Now write a message for Agent B.\n"
    "Select and report the operational facts B needs to judge each carrier. "
    "Leave the eligibility judgment to B.\n"
    "Do not write eligible or ineligible. Do not write a final booking.\n"
    "Cover each carrier. Include the facts needed to check capacity, refrigeration, "
    "and driver availability, or whether the carrier is cleared, as they appear in "
    "your information.\n"
    "You may add brief supporting conditions. Do not invent prices."
)

KV_ELIGIBILITY_CONSTRAINT = (
    "Do not book a carrier that fails operational requirements."
)


def a_local_task_prompt(ex: PlanExample) -> str:
    """C0 encode: private operations and A's local task. No message format yet."""
    return "\n\n".join(
        [
            "You are Agent A. You have operational information only. "
            "You do not see prices or budget.",
            "Your local task is to determine which carriers can handle this shipment: "
            "whether each carrier satisfies the operational requirements "
            "(capacity, refrigeration, and driver or clearance, as given).",
            "Do not write a message yet. Do not invent prices. Do not choose a final carrier.",
            ex.a_text,
            f"Shipment: {ex.request_line}",
            f"The eventual booking decision will be: {QUESTION}",
        ]
    )


def intermediate_handoff_prompt() -> str:
    """Matched continuation for text from C0 and from C40."""
    return INTERMEDIATE_HANDOFF


def source_evidence_handoff_prompt() -> str:
    """A selects source facts. Not a hardcoded inventory retriever."""
    return SOURCE_EVIDENCE_HANDOFF


def b_kv_reason_prompt(ex: PlanExample) -> str:
    """Same reason-then-answer procedure as text, with relayed memory instead of a message."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent already processed operational information. "
            "Use the relayed memory if it helps.",
            KV_ELIGIBILITY_CONSTRAINT,
            ex.b_text,
            _question_block(ex),
            "Write a brief plan (at most a few sentences) that uses both the "
            "relayed operational information and the prices. "
            "Do not book a carrier that fails operational requirements. "
            "Do not write an Answer line yet.",
        ]
    )


def b_kv_answer_prompt(ex: PlanExample, plan_text: str) -> str:
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent already processed operational information. "
            "Use the relayed memory if it helps.",
            KV_ELIGIBILITY_CONSTRAINT,
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def source_fact_coverage(message: str, ex: PlanExample) -> dict:
    """Diagnostic: did the source-evidence message keep the necessary facts?"""
    raw = message or ""
    text = raw.lower()
    wrote_label = bool(re.search(r"\b(eligible|ineligible)\b", text))
    per: dict[str, dict] = {}
    n_named = 0
    n_complete = 0
    for carrier in CARRIERS:
        facts = ex.a_facts[carrier]
        named = bool(re.search(rf"\bcarrier\s+{carrier}\b", text, re.IGNORECASE)) or bool(
            re.search(rf"\b{carrier}\b", raw)
        )
        has_capacity = str(int(facts["capacity"])) in raw
        has_fridge = "refrigerat" in text or "fridge" in text
        has_driver = "driver" in text
        has_cleared = "cleared" in text
        if ex.difficulty == "easy":
            complete = bool(named and has_cleared)
        else:
            complete = bool(named and has_capacity and has_fridge and has_driver)
        if named:
            n_named += 1
        if complete:
            n_complete += 1
        per[carrier] = {
            "named": named,
            "has_capacity": has_capacity,
            "has_fridge": has_fridge,
            "has_driver": has_driver,
            "has_cleared": has_cleared,
            "complete": complete,
        }
    return {
        "n_carriers_named": n_named,
        "n_carriers_complete": n_complete,
        "all_carriers_complete": n_complete == 3,
        "wrote_eligibility_label": wrote_label,
        "by_carrier": per,
    }
