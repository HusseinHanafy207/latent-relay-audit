"""Matched-length neutral prefixes for the eligibility-instruction states.

Both passages were chosen before any receiver evaluation. They are ordinary
running text: no shipment facts, answers, eligibility guidance, or pad/mask
tokens. Special chat-header tokens stay in place; the 64-token eligibility
paragraph stays at its original positions.
"""

from __future__ import annotations

from typing import Callable, Sequence

from src.generic_select import CARRIER_HEADING
from src.prompts_plan4 import ELIGIBILITY_RULE, L2_RULE, L3_RULE

CHAT_HEADER = "<|im_start|>user\n"
CONTEXT_TASK = "task"
CONTEXT_NEUTRAL_1 = "neutral_1"
CONTEXT_NEUTRAL_2 = "neutral_2"
CONTEXT_KINDS = (CONTEXT_TASK, CONTEXT_NEUTRAL_1, CONTEXT_NEUTRAL_2)

# Chosen before examining results. Do not edit after looking at scores.
NEUTRAL_PASSAGES = {
    CONTEXT_NEUTRAL_1: (
        "Along the north ridge a mixed woodland follows a slow creek that gathers "
        "rain from mossy slopes. In early spring, ferns unfurl beside fallen logs, "
        "and thrushes search the leaf litter after dawn. Summer heat thins the "
        "understory, so sunlight reaches patches of wild strawberry and the pale "
        "trunks of birch. Autumn turns the canopy copper, and wind carries dry "
        "leaves across the stone path that follows the water. Winter is quieter: "
        "ice forms at the edges of still pools, and deer tracks mark the snow "
        "between the pines.\n\n"
        "Village walkers use the path on weekends, pausing at a wooden bridge "
        "where the creek drops over a ledge. The bridge is old but sound, repaired "
        "with local timber after a storm years ago. From the ridge one can see a "
        "lake in the valley and, beyond it, fields that change color with the "
        "season. A bench sits under a maple near the last switchback. People eat "
        "bread there and watch clouds move over the far hills.\n\n"
        "Birds nest in the hawthorn. After rain, the soil smells of mushrooms and "
        "wet bark. It is only a landscape that a reader might hold in mind while "
        "walking, then forget by evening. The same path returns in later years, "
        "with new moss on the stones and the same view of the water."
    ),
    CONTEXT_NEUTRAL_2: (
        "The old town library occupies a brick building with tall windows and a "
        "reading room that smells of paper and polish. Catalog drawers still stand "
        "along one wall, though most records now live in a modest computer at the "
        "desk. On clear evenings the astronomy club meets on the roof, where a "
        "small telescope points toward familiar constellations. Members take turns "
        "naming stars, then go downstairs for tea and a talk about a recent book.\n\n"
        "In the garden, roses and a row of herbs grow beside a stone bench. "
        "Children visit on Saturdays to hear a story and to borrow illustrated "
        "volumes about insects, islands, and musical instruments. The clock above "
        "the door keeps ordinary time. A volunteer mends torn pages with tissue "
        "and wheat paste. In winter the radiators tick, and in summer the windows "
        "open onto the square.\n\n"
        "Downstairs, maps of distant coasts fill a shallow cabinet. A second room "
        "holds sheet music and a quiet piano that visitors may play after noon. "
        "It is a civic room for reading, looking up, and sitting still for an hour. "
        "On some nights the roof talk is only about the moon, and then everyone "
        "walks home along the lit street."
    ),
}

BANNED_PHRASES = (
    ELIGIBILITY_RULE,
    L2_RULE,
    L3_RULE,
    "Shipment:",
    "Agent A",
    "Agent B",
    "effective capacity",
    "productive hours",
    "vehicle capacity",
    "loading rate",
    "loading capacity",
    "eligible",
    "ineligible",
    "refrigeration",
    "You are Agent",
)


def _as_int_ids(raw) -> list[int]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return [int(t) for t in raw]


QWEN_CHAT_HEADER_IDS = (151644, 872, 198)


def special_header_len(ids: Sequence[int], decode: Callable) -> int:
    """Number of leading tokens that render as the chat header."""
    ids = [int(t) for t in ids]
    if not ids:
        raise RuntimeError("Empty prefix has no chat header.")
    for n in range(1, min(8, len(ids)) + 1):
        if decode(ids[:n]) == CHAT_HEADER:
            return n
    acc = ""
    for i, tok in enumerate(ids):
        acc += decode([tok])
        if acc == CHAT_HEADER:
            return i + 1
        if not CHAT_HEADER.startswith(acc):
            break
    if tuple(ids[:3]) == QWEN_CHAT_HEADER_IDS:
        return 3
    raise RuntimeError("Causal prefix does not start with the chat header <|im_start|>user\\n.")


def assert_clean_text(text: str, *, where: str) -> None:
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lowered:
            raise RuntimeError(f"{where} contains task-related text: {phrase!r}.")
    for heading in CARRIER_HEADING:
        if heading.lower() in lowered:
            raise RuntimeError(f"{where} contains a carrier heading: {heading!r}.")
    if "<|im_start|>" in text or "<|im_end|>" in text:
        raise RuntimeError(f"{where} contains special chat tokens.")


def encode_ordinary(
    encode: Callable,
    text: str,
    *,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
) -> list[int]:
    ids = _as_int_ids(encode(text))
    forbidden = {int(x) for x in (pad_token_id, mask_token_id) if x is not None and int(x) >= 0}
    if any(i in forbidden for i in ids):
        raise RuntimeError("Neutral encoding used pad or mask tokens.")
    return ids


def crop_ordinary_body(
    ids: Sequence[int],
    n: int,
    *,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
) -> list[int]:
    """Take the first n ordinary tokens. Do not pad or mask to reach the length."""
    if n < 1:
        raise RuntimeError("Neutral body length must be positive.")
    ids = [int(t) for t in ids]
    if len(ids) < n:
        raise RuntimeError(f"Neutral passage encoded to {len(ids)} tokens; need {n}. Do not pad.")
    body = ids[:n]
    forbidden = {int(x) for x in (pad_token_id, mask_token_id) if x is not None and int(x) >= 0}
    if any(i in forbidden for i in body):
        raise RuntimeError("Cropped neutral body contains pad or mask tokens.")
    return body


def splice_neutral_prefix(
    original_ids: Sequence[int],
    *,
    instruction_start: int,
    instruction_end: int,
    passage: str,
    encode: Callable,
    decode: Callable,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
    passage_id: str | None = None,
) -> dict:
    """Replace the body between the chat header and the eligibility paragraph."""
    ids = [int(t) for t in original_ids]
    start = int(instruction_start)
    end = int(instruction_end)
    if not (0 <= start < end <= len(ids)):
        raise RuntimeError(f"Instruction span [{start}, {end}) falls outside {len(ids)} tokens.")
    header_n = special_header_len(ids, decode)
    if header_n > start:
        raise RuntimeError("Chat header overlaps the eligibility instruction.")
    header = ids[:header_n]
    instr = ids[start:end]
    n_body = start - header_n
    assert_clean_text(passage, where="predeclared neutral passage")
    encoded = encode_ordinary(
        encode, passage, pad_token_id=pad_token_id, mask_token_id=mask_token_id
    )
    body = crop_ordinary_body(
        encoded, n_body, pad_token_id=pad_token_id, mask_token_id=mask_token_id
    )
    body_text = decode(body)
    assert_clean_text(body_text, where="cropped neutral body")
    spliced = header + body + instr + ids[end:]
    if len(spliced) != len(ids):
        raise RuntimeError("Spliced prefix length drifted from the original causal prefix.")
    if spliced[:header_n] != header:
        raise RuntimeError("Spliced prefix did not keep the chat header.")
    if spliced[start:end] != instr:
        raise RuntimeError("Spliced prefix moved the eligibility instruction tokens.")
    if spliced == ids:
        raise RuntimeError("Spliced prefix is identical to the task-context prefix.")
    return {
        "ids": spliced,
        "n": len(spliced),
        "header_len": header_n,
        "header_ids": header,
        "body_ids": body,
        "n_body": n_body,
        "instruction_start": start,
        "instruction_end": end,
        "instruction_ids": instr,
        "instruction_pos": list(range(start, end)),
        "passage_id": passage_id,
        "body_text": body_text,
        "header_equal": True,
        "instruction_equal": True,
        "length_equal": True,
        "used_pad": False,
    }
