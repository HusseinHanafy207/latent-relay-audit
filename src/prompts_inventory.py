"""Prompts for the two-agent inventory task."""

from __future__ import annotations

from src.inventory import InventoryExample

SHORT_ANSWER_RULE = (
    "Reply with one line only, in this exact format:\n"
    "Answer: <letter>) locker <number>\n"
    "Do not write any other text. Stop after that line."
)


def _question_block(ex: InventoryExample) -> str:
    return f"{ex.question}\nChoices:\n{ex.choices_text}"


def sender_prompt(inventory_text: str, ex: InventoryExample) -> str:
    return (
        "You are Agent A. Read the inventory and the question. "
        "Identify which locker holds the queried package. "
        "Do not write an answer; your next steps are silent.\n\n"
        f"Inventory:\n{inventory_text}\n\n"
        f"{_question_block(ex)}"
    )


def sender_prompt_inventory_only(inventory_text: str) -> str:
    """A reads the inventory and never sees the eventual question or choices."""
    return (
        "You are Agent A. Read the inventory. "
        "Do not write an answer; your next steps are silent.\n\n"
        f"Inventory:\n{inventory_text}"
    )


def receiver_prompt(
    ex: InventoryExample,
    *,
    inventory_text: str | None,
    filler_text: str | None,
    has_relay: bool,
) -> str:
    parts = ["You are Agent B."]
    if has_relay:
        parts.append("A previous agent already read an inventory. Use the relayed memory if it helps.")
    if inventory_text is not None:
        parts.append("Use the inventory below.")
        parts.append(f"Inventory:\n{inventory_text}")
    elif filler_text is not None:
        parts.append("The document below is NOT an inventory of packages and lockers.")
        parts.append(f"Warehouse notes:\n{filler_text}")
    parts.append(_question_block(ex))
    parts.append(SHORT_ANSWER_RULE)
    return "\n\n".join(parts)


def retrieved_sentence_prompt(ex: InventoryExample, sentence: str) -> str:
    """B gets retrieved records as text. No sender cache."""
    return text_message_prompt(ex, sentence, source="iterative")


def text_message_prompt(ex, message: str, *, source: str) -> str:
    """B gets a text payload. Never the complete inventory."""
    if source == "oracle":
        intro = "These supporting records are provided for a diagnostic check. Use them if they help."
    elif source == "sender":
        intro = "A previous agent sent the message below. Use it if it helps."
    else:
        intro = "A retrieved these records from an inventory. Use them if they help."
    return "\n\n".join(
        [
            "You are Agent B.",
            intro,
            message.strip(),
            _question_block(ex),
            SHORT_ANSWER_RULE,
        ]
    )


def sender_helpful_message_prompt(inventory_text: str, ex) -> str:
    """A writes a short message after the question arrives. No latent rollout."""
    return (
        "You are Agent A. You have read the inventory below. "
        "Agent B must answer the question and cannot see the inventory. "
        "Write a short helpful message for B. You may include the answer if you can determine it.\n\n"
        f"Inventory:\n{inventory_text}\n\n"
        f"{_question_block(ex)}"
    )


def probe_prompt(ex: InventoryExample) -> str:
    """Short question pass for selection. No inventory, filler, or answer."""
    return _question_block(ex)


def generic_probe_prompt(ex: InventoryExample) -> str:
    """Question-shaped probe with no package name."""
    return f"Which locker is the queried package in?\nChoices:\n{ex.choices_text}"


def mismatch_probe_prompt(package_name: str, choices_text: str) -> str:
    """Ask about a different package. B's actual question stays unchanged."""
    return f"Which locker is package {package_name} in?\nChoices:\n{choices_text}"


def true_inventory_absent_from_receiver_prompt(prompt: str, ex: InventoryExample) -> bool:
    """Sender-only prompts must not contain the true evidence line."""
    evidence = f"Package {ex.queried_package} is in locker {ex.true_locker}."
    return evidence not in prompt
