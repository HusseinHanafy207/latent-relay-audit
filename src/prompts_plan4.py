"""Prompts for four-carrier sender calibration.

Ordinary A prompts receive operational facts and the computation rules.
They do not receive solver quantities, eligibility labels, prices, or the gold carrier.

The written handoff asks for free working, then a marked RESULTS block.
The compact handoff asks for four result lines only. Those two must stay distinct.

B's line-by-line plan is frozen after the plan4_bprompt comparison. Do not
retune it to chase remaining cheapest-eligible misses. The KV receiver uses
that same line procedure with relayed memory instead of a pasted message.
"""

from __future__ import annotations

from src.plan4_task import CARRIERS, QUESTION, Plan4Example

SHORT_ANSWER_RULE = (
    "Reply with one line only, in this exact format:\n"
    "Answer: <letter>) <carrier>\n"
    "Carriers are P, Q, R, or S. Do not write any other text. Stop after that line."
)

ELIGIBILITY_RULE = (
    "A carrier is eligible only if all three are true: "
    "effective capacity is at least the shipment size, "
    "refrigeration is present whenever the shipment requires it, "
    "and a driver is available. "
    "If any one of those fails, the carrier is ineligible. "
    "Reporting the capacity number is not the same as checking it against the shipment size."
)

RESULT_LINE_FORMAT = (
    "P: q=<effective capacity in boxes> | <eligible or ineligible> | <short reason>\n"
    "Q: q=<effective capacity in boxes> | <eligible or ineligible> | <short reason>\n"
    "R: q=<effective capacity in boxes> | <eligible or ineligible> | <short reason>\n"
    "S: q=<effective capacity in boxes> | <eligible or ineligible> | <short reason>\n"
    "Replace each placeholder after computing from the operational facts. "
    "Do not copy the placeholders. Do not assume any carrier is eligible."
)

ELIGIBILITY_CONSTRAINT = (
    "Treat operational eligibility in that message as a mandatory constraint: "
    "do not book a carrier marked ineligible."
)

L2_RULE = (
    "Effective capacity for a carrier is the minimum of its vehicle capacity and "
    "floor(loading rate times productive hours). Use integer boxes."
)

L3_RULE = (
    "Productive hours equal max(0, available working time minus travel time minus break time). "
    "Effective capacity is the minimum of vehicle capacity and floor(loading rate times productive hours). "
    "Use integer hours and integer boxes."
)

RESULTS_MARK = "RESULTS"


def _question_block(ex: Plan4Example) -> str:
    return f"{QUESTION}\nChoices:\n{ex.choices_text}"


def _rules_for(ex: Plan4Example) -> str:
    if ex.difficulty == "L1":
        return (
            "Capacity, refrigeration, and driver availability are given directly. "
            "Effective capacity is the given vehicle capacity."
        )
    if ex.difficulty == "L2":
        return L2_RULE
    return L3_RULE


def a_local_task_prompt(ex: Plan4Example) -> str:
    """C0 encode: operations and A's local computation task. No result format yet."""
    return "\n\n".join(
        [
            "You are Agent A. You have operational information only. "
            "You do not see prices or budget.",
            "Your local task is to determine, for each carrier, the effective capacity "
            "in boxes and whether that carrier can handle this shipment "
            "(effective capacity, refrigeration, and driver).",
            _rules_for(ex),
            ELIGIBILITY_RULE,
            "Do not write a message yet. Do not invent prices. Do not choose a final carrier.",
            ex.a_text,
            f"Shipment: {ex.request_line}",
            f"The eventual booking decision will be: {QUESTION}",
        ]
    )


def compact_result_handoff() -> str:
    """Matched result request for no extra rollout and for the 40-step continuation."""
    return "\n".join(
        [
            "Now write the result lines for Agent B.",
            "Write exactly four lines, one per carrier, and nothing else.",
            RESULT_LINE_FORMAT,
            ELIGIBILITY_RULE,
            "Do not invent prices. Do not choose a final carrier.",
        ]
    )


def written_result_handoff() -> str:
    """Free working, then a marked four-line block. Must not open with a four-line-only rule."""
    return "\n".join(
        [
            "Now compute the result for Agent B.",
            "First write the working for each carrier. Use as many lines as you need. "
            "Do not start with the four result lines.",
            "For each carrier, compute in this order:",
            "- productive hours, if travel time or break time appear in the facts "
            "(available working time minus travel minus breaks, not below 0); "
            "if productive hours are already given, copy that number; "
            "if the sheet gives vehicle capacity with no hours, skip this step",
            "- effective capacity from the rules above, showing the arithmetic with the "
            "carrier's numbers",
            "- eligibility from the three-part rule (capacity, refrigeration, and driver)",
            f"After the working, write a block that begins with the exact line {RESULTS_MARK} "
            "and then contains exactly four result lines, one per carrier, in this format:",
            RESULT_LINE_FORMAT,
            ELIGIBILITY_RULE,
            "Do not invent prices. Do not choose a final carrier.",
        ]
    )


def b_need_probe_prompt(ex: Plan4Example) -> str:
    """Receiver request for cache ranking. No prices, budget, or gold carrier."""
    return "\n".join(
        [
            "I need each carrier's effective capacity, eligibility, "
            "and any conditions affecting this shipment.",
            f"Shipment: {ex.request_line}",
            "Do not invent prices. Do not choose a final carrier.",
        ]
    )


def fuller_text_handoff(ex: Plan4Example, written: str) -> str:
    """A's operational sheet plus the already-written calculations. No extra A generation."""
    return "\n\n".join(
        [
            "Operational information:",
            (ex.a_text or "").strip(),
            "Written calculations:",
            (written or "").strip(),
        ]
    )


def ordinary_sender_prompt(encode_prompt: str, handoff: str) -> str:
    """Single user turn: local task plus the handoff. Used to check cached continuation."""
    return encode_prompt.rstrip() + "\n\n" + handoff.lstrip()


def _b_front(ex: Plan4Example, message: str) -> list[str]:
    return [
        "You are Agent B. You have financial information. "
        "You do not see the operational sheet.",
        "A previous agent sent the message below.",
        message.strip(),
        ELIGIBILITY_CONSTRAINT,
        ex.b_text,
        _question_block(ex),
    ]


def b_reason_brief_prompt(ex: Plan4Example, message: str) -> str:
    """Previous receiver plan (first calibration). Comparison baseline. Do not retune on six families."""
    return "\n\n".join(
        [
            *_b_front(ex, message),
            "Write a brief plan (at most a few sentences) that uses both the "
            "eligibility message and the prices. "
            "Do not book a carrier that fails operational requirements. "
            "Do not write an Answer line yet.",
        ]
    )


def b_answer_after_brief_plan_prompt(ex: Plan4Example, message: str, plan_text: str) -> str:
    """Previous receiver answer pass. Pairs with b_reason_brief_prompt."""
    return "\n\n".join(
        [
            *_b_front(ex, message)[:4],
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def b_reason_with_message_prompt(ex: Plan4Example, message: str) -> str:
    """Frozen after plan4_bprompt: one short line per carrier, then a Choice."""
    return "\n\n".join(
        [
            *_b_front(ex, message),
            "Write exactly four short lines, one per carrier, then one Choice line. Use:",
            "P: <eligible or ineligible from the message> | $<price> | <candidate or dropped>",
            "Q: <eligible or ineligible from the message> | $<price> | <candidate or dropped>",
            "R: <eligible or ineligible from the message> | $<price> | <candidate or dropped>",
            "S: <eligible or ineligible from the message> | $<price> | <candidate or dropped>",
            "Choice: <carrier> at $<price>",
            "A candidate is marked eligible in the message and costs at most the budget. "
            "Otherwise write dropped. "
            "The Choice carrier must be the lowest-price candidate. "
            "If two candidate prices tie, choose the earlier carrier in P, Q, R, S. "
            "Do not book a dropped carrier. Do not write an Answer line yet.",
        ]
    )


def b_answer_after_message_plan_prompt(ex: Plan4Example, message: str, plan_text: str) -> str:
    return "\n\n".join(
        [
            *_b_front(ex, message)[:4],
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            "The Answer line must be the Choice carrier from those notes. "
            "If the notes named a dropped carrier, correct to the lowest-price candidate.",
            SHORT_ANSWER_RULE,
        ]
    )


def b_reason_kv_prompt(ex: Plan4Example) -> str:
    """Frozen line plan over relayed memory. Same procedure as the text message plan."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent already processed operational information. "
            "Use the relayed memory if it helps.",
            "Treat operational eligibility in the relayed memory as a mandatory constraint: "
            "do not book a carrier marked ineligible.",
            ex.b_text,
            _question_block(ex),
            "Write exactly four short lines, one per carrier, then one Choice line. Use:",
            "P: <eligible or ineligible from the relayed memory> | $<price> | <candidate or dropped>",
            "Q: <eligible or ineligible from the relayed memory> | $<price> | <candidate or dropped>",
            "R: <eligible or ineligible from the relayed memory> | $<price> | <candidate or dropped>",
            "S: <eligible or ineligible from the relayed memory> | $<price> | <candidate or dropped>",
            "Choice: <carrier> at $<price>",
            "A candidate is marked eligible in the relayed memory and costs at most the budget. "
            "Otherwise write dropped. "
            "The Choice carrier must be the lowest-price candidate. "
            "If two candidate prices tie, choose the earlier carrier in P, Q, R, S. "
            "Do not book a dropped carrier. Do not write an Answer line yet.",
        ]
    )


def b_answer_kv_prompt(ex: Plan4Example, plan_text: str) -> str:
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet.",
            "A previous agent already processed operational information. "
            "Use the relayed memory if it helps.",
            "Treat operational eligibility in the relayed memory as a mandatory constraint: "
            "do not book a carrier marked ineligible.",
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            "The Answer line must be the Choice carrier from those notes. "
            "If the notes named a dropped carrier, correct to the lowest-price candidate.",
            SHORT_ANSWER_RULE,
        ]
    )


def b_reason_no_message_prompt(ex: Plan4Example) -> str:
    """No eligibility message. Same line format; candidates are budget-only."""
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet. "
            "No previous agent sent an eligibility message.",
            ex.b_text,
            _question_block(ex),
            "Write exactly four short lines, one per carrier, then one Choice line. Use:",
            "P: unknown | $<price> | <candidate or dropped>",
            "Q: unknown | $<price> | <candidate or dropped>",
            "R: unknown | $<price> | <candidate or dropped>",
            "S: unknown | $<price> | <candidate or dropped>",
            "Choice: <carrier> at $<price>",
            "There is no eligibility information. A candidate costs at most the budget. "
            "Otherwise write dropped. "
            "The Choice carrier must be the lowest-price candidate. "
            "If two candidate prices tie, choose the earlier carrier in P, Q, R, S. "
            "Do not write an Answer line yet.",
        ]
    )


def b_answer_no_message_prompt(ex: Plan4Example, plan_text: str) -> str:
    return "\n\n".join(
        [
            "You are Agent B. You have financial information. "
            "You do not see the operational sheet. "
            "No previous agent sent an eligibility message.",
            "Your earlier notes:",
            plan_text.strip(),
            ex.b_text,
            _question_block(ex),
            "The Answer line must be the Choice carrier from those notes.",
            SHORT_ANSWER_RULE,
        ]
    )


def prompt_contains_a_private(prompt: str, ex: Plan4Example) -> bool:
    rest = ex.a_text.replace(ex.request_line, "").strip()
    if not rest:
        return False
    return rest in prompt


def prompt_contains_b_private(prompt: str, ex: Plan4Example) -> bool:
    rest = ex.b_text.replace(ex.request_line, "").strip()
    if not rest:
        return False
    return rest in prompt


def prompt_contains_solver_labels(prompt: str, ex: Plan4Example) -> bool:
    oracle_bits = [f"{c}: q={int(ex.intermediates[c]['q'])} |" for c in CARRIERS]
    return any(bit in prompt for bit in oracle_bits)
