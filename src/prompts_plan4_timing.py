"""Receiver and sender prompts for the 12-carrier timing split only.

The frozen four-carrier B line-by-line prompts in prompts_plan4 stay unchanged.
These functions list N carriers. Do not import them from confirmation or
development stages.
"""

from __future__ import annotations

from src.prompts_plan4 import (
    ELIGIBILITY_CONSTRAINT,
    ELIGIBILITY_RULE,
    FRAMING_INSTRUCTION,
    RESULTS_MARK,
    _question_block,
    a_local_task_prompt,
)


def result_line_format(carriers: tuple[str, ...]) -> str:
    lines = [
        f"{c}: q=<effective capacity in boxes> | <eligible or ineligible> | <short reason>"
        for c in carriers
    ]
    lines.append(
        "Replace each placeholder after computing from the operational facts. "
        "Do not copy the placeholders. Do not assume any carrier is eligible."
    )
    return "\n".join(lines)


def written_result_handoff_n(carriers: tuple[str, ...]) -> str:
    n = len(carriers)
    order = ", ".join(carriers)
    return "\n".join(
        [
            "Now compute the result for Agent B.",
            "First write the working for each carrier. Use as many lines as you need. "
            f"Do not start with the {n} result lines.",
            "For each carrier, compute in this order:",
            "- productive hours, if travel time or break time appear in the facts "
            "(available working time minus travel minus breaks, not below 0); "
            "if productive hours are already given, copy that number; "
            "if the sheet gives vehicle capacity with no hours, skip this step",
            "- effective capacity from the rules above, showing the arithmetic with the "
            "carrier's numbers",
            "- eligibility from the three-part rule (capacity, refrigeration, and driver)",
            f"After the working, write a block that begins with the exact line {RESULTS_MARK} "
            f"and then contains exactly {n} result lines, one per carrier, in this format:",
            result_line_format(carriers),
            ELIGIBILITY_RULE,
            "Do not invent prices. Do not choose a final carrier.",
            f"Carrier order is {order}.",
        ]
    )


def _plan_lines(carriers: tuple[str, ...], *, source: str) -> list[str]:
    return [
        f"{c}: <eligible or ineligible from the {source}> | $<price> | <candidate or dropped>"
        for c in carriers
    ]


def b_reason_kv_instruct_prompt_n(ex) -> str:
    carriers = tuple(ex.carriers)
    n = len(carriers)
    order = ", ".join(carriers)
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent already processed operational information. "
            "Use the relayed memory if it helps.",
            FRAMING_INSTRUCTION,
            "Treat operational eligibility in the relayed memory as a mandatory constraint: "
            "do not book a carrier marked ineligible.",
            ex.b_text,
            _question_block(ex),
            f"Write exactly {n} short lines, one per carrier, then one Choice line. Use:",
            *_plan_lines(carriers, source="relayed memory"),
            "Choice: <carrier> at $<price>",
            "A candidate is marked eligible in the relayed memory and costs at most the budget. "
            "Otherwise write dropped. "
            f"The Choice carrier must be the lowest-price candidate. "
            f"If two candidate prices tie, choose the earlier carrier in {order}. "
            "Do not book a dropped carrier. Do not write an Answer line yet.",
        ]
    )


def b_answer_kv_instruct_prompt_n(ex, plan_text: str) -> str:
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent already processed operational information. "
            "Use the relayed memory if it helps.",
            FRAMING_INSTRUCTION,
            "Treat operational eligibility in the relayed memory as a mandatory constraint: "
            "do not book a carrier marked ineligible.",
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            "The Answer line must be the Choice carrier from those notes. "
            "If the notes named a dropped carrier, correct to the lowest-price candidate.",
            _short_answer_rule_n(ex),
        ]
    )


def b_reason_with_message_prompt_n(ex, message: str) -> str:
    carriers = tuple(ex.carriers)
    n = len(carriers)
    order = ", ".join(carriers)
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent sent the message below.",
            message.strip(),
            ELIGIBILITY_CONSTRAINT,
            ex.b_text,
            _question_block(ex),
            f"Write exactly {n} short lines, one per carrier, then one Choice line. Use:",
            *_plan_lines(carriers, source="message"),
            "Choice: <carrier> at $<price>",
            "A candidate is marked eligible in the message and costs at most the budget. "
            "Otherwise write dropped. "
            f"The Choice carrier must be the lowest-price candidate. "
            f"If two candidate prices tie, choose the earlier carrier in {order}. "
            "Do not book a dropped carrier. Do not write an Answer line yet.",
        ]
    )


def b_answer_after_message_plan_prompt_n(ex, message: str, plan_text: str) -> str:
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
            "The Answer line must be the Choice carrier from those notes. "
            "If the notes named a dropped carrier, correct to the lowest-price candidate.",
            _short_answer_rule_n(ex),
        ]
    )


def _short_answer_rule_n(ex) -> str:
    carriers = tuple(ex.carriers)
    letters = ", ".join(ex.letter_of[c] for c in carriers)
    return (
        "Reply with one line only, in this exact format:\n"
        "Answer: <letter>) <carrier>\n"
        f"Carriers are {', '.join(carriers)}. Letters are {letters}. "
        "Do not write any other text. Stop after that line."
    )


def encode_prompt_for(ex) -> str:
    return a_local_task_prompt(ex)
