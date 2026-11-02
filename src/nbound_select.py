"""Complete matched-length neutrals that keep the original instruction join.

The first length-matched run cropped mid-sentence. This construction uses the
same two topics, writes complete sentences, and copies the original
punctuation-plus-paragraph suffix so the eligibility instruction stays at the
same positions. Wording is frozen from these construction rules, not from
receiver scores.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from src.neutral_select import (
    CONTEXT_NEUTRAL_1,
    CONTEXT_NEUTRAL_2,
    assert_clean_text,
    encode_ordinary,
    special_header_len,
)

JOIN_RE = re.compile(r"[.!?]\s*\n+\s*A carrier")
STUCK_RE = re.compile(r"[A-Za-z0-9]A carrier")
SUFFIX_RE = re.compile(r"(?:[.!?]+[ \t]*)?\n+\Z")
SENTENCE_END_RE = re.compile(r"[.!?]\s*$")

# Same two topics as the unmatched-boundary run. Extra complete sentences only
# fill the token budget; they are not a search over new themes.
NEUTRAL_SENTENCES = {
    CONTEXT_NEUTRAL_1: (
        "Along the north ridge a mixed woodland follows a slow creek that gathers rain from mossy slopes.",
        "In early spring, ferns unfurl beside fallen logs, and thrushes search the leaf litter after dawn.",
        "Summer heat thins the understory, so sunlight reaches patches of wild strawberry and the pale trunks of birch.",
        "Autumn turns the canopy copper, and wind carries dry leaves across the stone path that follows the water.",
        "Winter is quieter: ice forms at the edges of still pools, and deer tracks mark the snow between the pines.",
        "Village walkers use the path on weekends, pausing at a wooden bridge where the creek drops over a ledge.",
        "The bridge is old but sound, repaired with local timber after a storm years ago.",
        "From the ridge one can see a lake in the valley and, beyond it, fields that change color with the season.",
        "A bench sits under a maple near the last switchback.",
        "People eat bread there and watch clouds move over the far hills.",
        "Birds nest in the hawthorn.",
        "After rain, the soil smells of mushrooms and wet bark.",
        "It is only a landscape that a reader might hold in mind while walking, then forget by evening.",
        "The same path returns in later years, with new moss on the stones and the same view of the water.",
        "Moss covers the stones after rain.",
        "A hawk circles above the ridge.",
        "The lake reflects the late sky.",
        "Footsteps fade along the switchback.",
        "The maple drops its last leaves.",
        "Steam rises from the wet earth.",
        "Crows call from the far pines.",
        "The ledge holds a thin sheet of ice.",
        "Clouds drag shadows over the fields.",
        "The creek sounds louder at night.",
        "Ferns close again by dusk.",
        "The wooden rail is smooth from years of hands.",
        "No one stays on the bench after dark.",
        "The hills go blue in the distance.",
        "A fox crosses the path and is gone.",
        "Rain ticks on the dry leaves.",
        "The valley holds its quiet.",
        "It is a long walk home.",
        "The stones keep the day's warmth.",
        "Owls start from the dark wood.",
        "The wet stones shine.",
        "A jay calls once and goes.",
        "The last light leaves the ridge.",
        "Water beads on the maple.",
        "The path is empty now.",
        "Fog sits in the low field.",
        "The creek runs brown after rain.",
        "A fallen branch blocks the switchback.",
        "The far shore is only a line.",
        "Crows have the tallest pine.",
        "The grass holds a frost.",
        "A stone skips and sinks.",
        "The trail turns toward home.",
    ),
    CONTEXT_NEUTRAL_2: (
        "The old town library occupies a brick building with tall windows and a reading room that smells of paper and polish.",
        "Catalog drawers still stand along one wall, though most records now live in a modest computer at the desk.",
        "On clear evenings the astronomy club meets on the roof, where a small telescope points toward familiar constellations.",
        "Members take turns naming stars, then go downstairs for tea and a talk about a recent book.",
        "In the garden, roses and a row of herbs grow beside a stone bench.",
        "Children visit on Saturdays to hear a story and to borrow illustrated volumes about insects, islands, and musical instruments.",
        "The clock above the door keeps ordinary time.",
        "A volunteer mends torn pages with tissue and wheat paste.",
        "In winter the radiators tick, and in summer the windows open onto the square.",
        "Downstairs, maps of distant coasts fill a shallow cabinet.",
        "A second room holds sheet music and a quiet piano that visitors may play after noon.",
        "It is a civic room for reading, looking up, and sitting still for an hour.",
        "On some nights the roof talk is only about the moon, and then everyone walks home along the lit street.",
        "The reading lamps click on at dusk.",
        "A clock ticks above the stair.",
        "Dust floats in the tall window light.",
        "Someone turns a page in the far corner.",
        "The piano lid stays closed until noon.",
        "Tea cools on the side table.",
        "A moth taps the roof glass.",
        "The garden smell comes in with the air.",
        "Maps curl at the corners in summer.",
        "The brick holds the day's heat.",
        "Children whisper near the picture books.",
        "The square grows loud, then still.",
        "A bicycle leans by the door.",
        "The roof is dark except for the small dome.",
        "Stars show between the moving clouds.",
        "The paste dries on a mended spine.",
        "Chairs scrape, then the room is quiet.",
        "It is enough to sit and look.",
        "The herbs smell sharper after rain.",
        "Night walks home are slow and lit.",
        "The hall is almost empty.",
        "A chair faces the tall window.",
        "The desk lamp makes a small circle.",
        "Ink dries on the blotter.",
        "The stair creaks on the third step.",
        "A coat hangs by the side door.",
        "The catalog smells of old paper.",
        "One window is cracked open.",
        "The moon is a thin coin.",
        "The last visitor has gone.",
        "A pencil rests in the gutter.",
        "The binding is soft from use.",
        "The porch light is already on.",
    ),
}

NEUTRAL_CLOSERS = {
    CONTEXT_NEUTRAL_1: (
        "Rain falls.",
        "Dusk comes.",
        "Moss grows.",
        "It is still.",
        "The path waits.",
        "Quiet holds.",
        "Birds lift.",
        "The ridge darkens.",
        "Water sounds.",
        "Leaves settle.",
        "Ice remains.",
        "The lake stills.",
        "Night is near.",
        "The wood listens.",
        "Wind passes.",
        "The bench cools.",
        "Pines lean.",
        "The valley rests.",
        "No one speaks.",
        "The walk ends.",
    ),
    CONTEXT_NEUTRAL_2: (
        "Lamps glow.",
        "Pages turn.",
        "Dust settles.",
        "It is quiet.",
        "The clock ticks.",
        "Chairs rest.",
        "Tea cools.",
        "Stars show.",
        "The square stills.",
        "Doors close.",
        "Maps wait.",
        "The roof darkens.",
        "Roses nod.",
        "Night comes.",
        "The room listens.",
        "Glass reflects.",
        "Steps fade.",
        "Books remain.",
        "The garden rests.",
        "The street is lit.",
    ),
}


def extract_join_suffix(body_ids: Sequence[int], decode: Callable) -> dict:
    """Trailing punctuation and paragraph break from the original task body."""
    ids = [int(t) for t in body_ids]
    if not ids:
        raise RuntimeError("Empty body has no join suffix.")
    best = None
    for n in range(1, min(8, len(ids)) + 1):
        text = decode(ids[-n:])
        if SUFFIX_RE.search(text) and not re.search(r"[A-Za-z0-9]", text):
            best = {"n": n, "ids": ids[-n:], "text": text}
    if best is None:
        raise RuntimeError(
            "Original task body does not end with punctuation and a paragraph break: "
            f"{decode(ids[-12:])!r}."
        )
    return best


def assert_clean_join(body_text: str, instr_text: str, *, where: str) -> None:
    glued = f"{body_text}{instr_text}"
    if STUCK_RE.search(glued):
        raise RuntimeError(
            f"{where} concatenates the instruction onto the preceding word: {glued[-90:]!r}."
        )
    if not re.search(r"[.!?]\s*\n+\s*$", body_text):
        raise RuntimeError(
            f"{where} does not end with sentence punctuation and a paragraph break: "
            f"{body_text[-80:]!r}."
        )
    if not JOIN_RE.search(glued):
        raise RuntimeError(
            f"{where} does not show a paragraph break before the eligibility instruction: "
            f"{glued[-90:]!r}."
        )


def _assemble(sentences: Sequence[str], *, drop_final_period: bool) -> str:
    parts = [s.strip() for s in sentences if s.strip()]
    if not parts:
        raise RuntimeError("No sentences to assemble.")
    if drop_final_period:
        parts = list(parts)
        parts[-1] = SENTENCE_END_RE.sub("", parts[-1]).rstrip()
        if not parts[-1]:
            raise RuntimeError("Last sentence vanished after dropping its period.")
    return " ".join(parts)


def _complete_end(text: str) -> bool:
    return bool(SENTENCE_END_RE.search(text.rstrip()))


def fit_complete_content(
    *,
    sentences: Sequence[str],
    closers: Sequence[str],
    n_content: int,
    encode: Callable,
    decode: Callable,
    drop_final_period: bool,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
    max_extra: int = 3,
) -> dict:
    """Assemble complete sentences to an exact token count. Do not crop or pad."""
    if n_content < 1:
        raise RuntimeError("Neutral content length must be positive.")
    encoded_cache: dict[str, list[int]] = {}

    def encoded_ids(parts: Sequence[str]) -> tuple[str, list[int]]:
        text = _assemble(parts, drop_final_period=drop_final_period)
        ids = encoded_cache.get(text)
        if ids is None:
            ids = encode_ordinary(
                encode, text, pad_token_id=pad_token_id, mask_token_id=mask_token_id
            )
            encoded_cache[text] = ids
        return text, ids

    def ok_end(ids: list[int], text: str) -> bool:
        if len(ids) != n_content:
            return False
        decoded = decode(ids)
        if drop_final_period:
            if _complete_end(decoded) or not re.search(r"[A-Za-z0-9]$", decoded.rstrip()):
                return False
        elif not _complete_end(decoded):
            return False
        return True

    def fill(parts: list[str], depth: int, fillers: Sequence[str], allow_repeat: bool):
        if not parts:
            if depth <= 0:
                return None
            for sentence in fillers:
                hit = fill([sentence], depth - 1, fillers, allow_repeat)
                if hit is not None:
                    return hit
            return None
        text, ids = encoded_ids(parts)
        n = len(ids)
        if n == n_content:
            return (ids, text, list(parts)) if ok_end(ids, text) else None
        if n > n_content or depth <= 0:
            return None
        for sentence in fillers:
            if not allow_repeat and sentence == parts[-1]:
                continue
            hit = fill(parts + [sentence], depth - 1, fillers, allow_repeat)
            if hit is not None:
                return hit
        return None

    chosen: list[str] = []
    for sentence in sentences:
        trial = chosen + [sentence]
        _text, ids = encoded_ids(trial)
        if len(ids) <= n_content:
            chosen = trial
        elif chosen:
            break

    base = list(chosen)
    hit = None
    while True:
        unused = [s for s in sentences if s not in set(base)]
        fillers = unused + list(closers)
        hit = None
        if base:
            text, ids = encoded_ids(base)
            if ok_end(ids, text):
                hit = (ids, text, list(base))
        if hit is None:
            start = list(base)
            for allow_repeat in (False, True):
                hit = fill(start, max_extra, fillers, allow_repeat)
                if hit is not None:
                    break
        if hit is not None:
            break
        if not base:
            have = 0
            if chosen:
                _have_text, have_ids = encoded_ids(chosen)
                have = len(have_ids)
            raise RuntimeError(
                f"Could not assemble a complete passage of {n_content} tokens "
                f"(had {have} from {len(chosen)} sentences). Do not crop or pad."
            )
        base.pop()
    ids, text, used = hit
    return {"ids": ids, "text": text, "sentences": used, "n": len(ids)}


def splice_complete_neutral_prefix(
    original_ids: Sequence[int],
    *,
    instruction_start: int,
    instruction_end: int,
    passage_id: str,
    encode: Callable,
    decode: Callable,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
) -> dict:
    """Replace the task body with a complete same-topic passage and the original join suffix."""
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
    body_orig = ids[header_n:start]
    n_body = len(body_orig)
    suffix = extract_join_suffix(body_orig, decode)
    n_content = n_body - int(suffix["n"])
    drop_final_period = bool(re.search(r"[.!?]", suffix["text"]))
    fitted = fit_complete_content(
        sentences=NEUTRAL_SENTENCES[passage_id],
        closers=NEUTRAL_CLOSERS[passage_id],
        n_content=n_content,
        encode=encode,
        decode=decode,
        drop_final_period=drop_final_period,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
    )
    assert_clean_text(fitted["text"], where="complete neutral content")
    body = list(fitted["ids"]) + list(suffix["ids"])
    if len(body) != n_body:
        raise RuntimeError("Complete neutral body length drifted from the original body.")
    body_text = decode(body)
    instr_text = decode(instr)
    assert_clean_text(body_text, where="complete neutral body")
    assert_clean_join(body_text, instr_text, where="complete neutral join")
    spliced = header + body + instr + ids[end:]
    if len(spliced) != len(ids):
        raise RuntimeError("Spliced prefix length drifted from the original causal prefix.")
    if spliced[:header_n] != header:
        raise RuntimeError("Spliced prefix did not keep the chat header.")
    if spliced[start:end] != instr:
        raise RuntimeError("Spliced prefix moved the eligibility instruction tokens.")
    if spliced == ids:
        raise RuntimeError("Spliced prefix is identical to the task-context prefix.")
    decoded_prefix = decode(spliced)
    decoded_join = decode(body[-16:] + instr[:16])
    original_join = decode(body_orig[-16:] + instr[:16])
    return {
        "ids": spliced,
        "n": len(spliced),
        "header_len": header_n,
        "header_ids": header,
        "body_ids": body,
        "n_body": n_body,
        "n_content": n_content,
        "n_suffix": int(suffix["n"]),
        "suffix_ids": list(suffix["ids"]),
        "suffix_text": suffix["text"],
        "instruction_start": start,
        "instruction_end": end,
        "instruction_ids": instr,
        "instruction_pos": list(range(start, end)),
        "passage_id": passage_id,
        "body_text": body_text,
        "assembled_text": fitted["text"],
        "used_sentences": list(fitted["sentences"]),
        "decoded_prefix": decoded_prefix,
        "decoded_join": decoded_join,
        "original_join": original_join,
        "join_ok": True,
        "cropped_mid_sentence": False,
        "header_equal": True,
        "instruction_equal": True,
        "length_equal": True,
        "used_pad": False,
    }
