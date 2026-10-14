"""Prompts for Study 0 delivery planning.

Frozen after the format × procedure diagnostic. Do not retune wording to
chase single development errors. Ordinary prompts receive documents and (for
collaboration) a generated message. They do not receive evaluator eligibility
or the gold carrier.
"""

from __future__ import annotations

from src.plan_task import QUESTION, PlanExample

SHORT_ANSWER_RULE = (
    "Reply with one line only, in this exact format:\n"
    "Answer: <letter>) <carrier>\n"
    "Carriers are P, Q, or R. Do not write any other text. Stop after that line."
)

A_MESSAGE_FORMAT = (
    "Write exactly three lines, one per carrier, in this format:\n"
    "P: <status determined from P's operational facts> | <reason>\n"
    "Q: <status determined from Q's operational facts> | <reason>\n"
    "R: <status determined from R's operational facts> | <reason>\n"
    "Replace each status placeholder with eligible or ineligible after reading "
    "that carrier's operational facts. Do not copy the placeholders. "
    "Do not assume P is eligible, Q is ineligible, or R is eligible. "
    "One reason per carrier. No other sentences."
)

ELIGIBILITY_CONSTRAINT = (
    "Treat operational eligibility in that message as a mandatory constraint: "
    "do not book a carrier marked ineligible."
)


def _question_block(ex: PlanExample) -> str:
    return f"{QUESTION}\nChoices:\n{ex.choices_text}"


def agent_a_message_prompt(ex: PlanExample) -> str:
    """A writes an eligibility message. Uses A's document only."""
    return "\n\n".join(
        [
            "You are Agent A. You have operational information only. "
            "You do not see prices or budget. "
            "Agent B will choose the lowest-cost carrier that satisfies all "
            "operational requirements and does not exceed the budget. "
            "Your job is only to judge operational eligibility.",
            A_MESSAGE_FORMAT,
            "Do not choose a final carrier. Do not invent prices.",
            ex.a_text,
            f"Decision B must make: {QUESTION}",
        ]
    )


def a_alone_prompt(ex: PlanExample) -> str:
    return "\n\n".join(
        [
            "You are Agent A. You have operational information only. "
            "You do not see prices or budget. Choose a carrier anyway.",
            ex.a_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def b_alone_prompt(ex: PlanExample) -> str:
    return "\n\n".join(
        [
            "You are Agent B. You have financial information only. "
            "You do not see operational constraints. Choose a carrier.",
            ex.b_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def b_planning_prompt(ex: PlanExample) -> str:
    """First extra-compute pass: notes only, no final Answer line."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information only. "
            "You do not see operational constraints. "
            "Write a brief plan (at most a few sentences) for this decision. "
            "Note what the prices and budget tell you, and what operational "
            "information you still lack. Do not write an Answer line yet.",
            ex.b_text,
            _question_block(ex),
        ]
    )


def b_answer_after_plan_prompt(ex: PlanExample, plan_text: str) -> str:
    """Second extra-compute pass: final answer after B's own notes."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information only. "
            "You do not see operational constraints. "
            "Use your earlier notes if they help, then choose a carrier.",
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def b_with_message_prompt(ex: PlanExample, message: str) -> str:
    """Shared wrapper for collaboration and oracle. Only the message body changes."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent sent the message below.",
            message.strip(),
            ELIGIBILITY_CONSTRAINT,
            ex.b_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def b_reason_with_message_prompt(ex: PlanExample, message: str) -> str:
    """Diagnostic: B plans using the eligibility message and prices. Not b_extra_compute."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent sent the message below.",
            message.strip(),
            ELIGIBILITY_CONSTRAINT,
            ex.b_text,
            _question_block(ex),
            "Write a brief plan (at most a few sentences) that uses both the "
            "eligibility message and the prices. "
            "Do not book a carrier that fails operational requirements. "
            "Do not write an Answer line yet.",
        ]
    )


def b_answer_after_message_plan_prompt(ex: PlanExample, message: str, plan_text: str) -> str:
    """Diagnostic: final answer after a plan that saw the eligibility message."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent sent the message below.",
            message.strip(),
            ELIGIBILITY_CONSTRAINT,
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def collab_b_prompt(ex: PlanExample, message: str) -> str:
    return b_with_message_prompt(ex, message)


def oracle_to_b_prompt(ex: PlanExample, oracle_message: str) -> str:
    return b_with_message_prompt(ex, oracle_message)


def pooled_prompt(ex: PlanExample) -> str:
    return "\n\n".join(
        [
            "You are a single planner. You see both operational and financial information. "
            "Choose the lowest-cost carrier that satisfies all operational requirements "
            "and does not exceed the budget.",
            "Operational information:",
            ex.a_text,
            "Financial information:",
            ex.b_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def prompt_contains_a_private(prompt: str, ex: PlanExample) -> bool:
    rest = ex.a_text.replace(ex.request_line, "").strip()
    if not rest:
        return False
    return rest in prompt


def prompt_contains_b_private(prompt: str, ex: PlanExample) -> bool:
    rest = ex.b_text.replace(ex.request_line, "").strip()
    if not rest:
        return False
    return rest in prompt
