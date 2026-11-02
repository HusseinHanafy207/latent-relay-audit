"""Frozen instruction and calculation spans, plus equal-size random controls.

Spans come from A's recorded token sequence. They do not use gold labels, B's
prices or budget, or which cases previously failed.
"""

from __future__ import annotations

import random
from typing import Sequence

from src.parse_plan4 import RESULT_LINE, RESULTS_MARK
from src.plan4_task import SINK_INITIAL
from src.prompts_plan4 import ELIGIBILITY_RULE
from src.span_select import (
    DecodeOne,
    TokenSpan,
    contiguous_runs,
    decode_positions,
    line_spans,
    positions_from_spans,
    result_spans,
    with_initial_positions,
)

RANDOM_SEEDS = (2101, 2102, 2103)
INSTRUCTION_RANDOM_SEEDS = (1, 2, 3)
INSTRUCTION_OCCURRENCE = "first"
CARRIER_HEADING = ("Carrier P", "Carrier Q", "Carrier R", "Carrier S")


def merge_positions(*groups: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for group in groups:
        for pos in group:
            idx = int(pos)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return sorted(out)


def find_first_text_span(
    token_ids: Sequence[int],
    needle: str,
    decode: DecodeOne,
    *,
    start: int = 0,
    end: int | None = None,
) -> TokenSpan | None:
    """Map the first exact occurrence of `needle` onto token indices."""
    start = int(start)
    end = int(len(token_ids) if end is None else end)
    if not needle or end <= start:
        return None
    ids = [int(t) for t in token_ids[start:end]]
    full = decode(ids)
    idx = full.find(needle)
    if idx < 0:
        return None
    idx_end = idx + len(needle)
    lengths: list[int] = []
    for i in range(len(ids)):
        lengths.append(len(decode(ids[: i + 1])))
    tok_start = next((i for i, n in enumerate(lengths) if n > idx), None)
    tok_end = next((i for i, n in enumerate(lengths) if n >= idx_end), None)
    if tok_start is None or tok_end is None:
        return None
    abs_start = start + int(tok_start)
    abs_end = start + int(tok_end) + 1
    text = decode([int(t) for t in token_ids[abs_start:abs_end]])
    return TokenSpan(abs_start, abs_end, text)


def results_marker_span(spans: Sequence[TokenSpan]) -> TokenSpan | None:
    for span in spans:
        if RESULTS_MARK.search((span.text or "").strip()) or RESULTS_MARK.search(span.text or ""):
            return span
    return None


def _rng(seed: int, a_state_key: str) -> random.Random:
    return random.Random(f"{int(seed)}:{a_state_key}")


def sample_equal_random(
    *,
    n_full: int,
    baseline: Sequence[int],
    extra_count: int,
    extra_runs: Sequence[tuple[int, int]],
    seed: int,
    a_state_key: str,
) -> dict:
    """Sample M extra positions from A's remaining cache. May overlap the support."""
    n_full = int(n_full)
    extra_count = int(extra_count)
    baseline_set = {int(p) for p in baseline if 0 <= int(p) < n_full}
    remaining = [p for p in range(n_full) if p not in baseline_set]
    if extra_count < 0:
        raise ValueError("extra_count must be non-negative")
    if extra_count > len(remaining):
        raise RuntimeError(
            f"Need {extra_count} random positions but only {len(remaining)} remain outside the baseline."
        )
    rng = _rng(seed, a_state_key)
    sampling_mode = "empty"
    selected: list[int] = []
    if extra_count == 0:
        selected = []
        sampling_mode = "empty"
    else:
        lengths = [int(end) - int(start) for start, end in extra_runs if int(end) > int(start)]
        if sum(lengths) != extra_count:
            lengths = [extra_count]
        block = _block_matched_sample(remaining, lengths, rng)
        if block is not None:
            selected = block
            sampling_mode = "block_matched"
        else:
            selected = sorted(rng.sample(remaining, extra_count))
            sampling_mode = "uniform"
    ordered = sorted({int(p) for p in selected})
    if len(ordered) != extra_count:
        raise RuntimeError("Random control did not keep exactly M extra positions.")
    if any(p in baseline_set for p in ordered):
        raise RuntimeError("Random control overlapped the baseline.")
    return {
        "seed": int(seed),
        "a_state_key": a_state_key,
        "sampling_mode": sampling_mode,
        "positions": ordered,
        "n": len(ordered),
        "runs": contiguous_runs(ordered),
        "requested_runs": [(int(a), int(b)) for a, b in extra_runs],
    }


def _block_matched_sample(
    remaining: Sequence[int],
    lengths: Sequence[int],
    rng: random.Random,
) -> list[int] | None:
    pool = set(int(p) for p in remaining)
    chosen: list[int] = []
    for length in lengths:
        length = int(length)
        if length <= 0:
            continue
        options = []
        for start, end in contiguous_runs(sorted(pool)):
            run_len = end - start
            if run_len >= length:
                options.extend(range(start, end - length + 1))
        if not options:
            return None
        begin = int(rng.choice(options))
        block = list(range(begin, begin + length))
        chosen.extend(block)
        pool.difference_update(block)
    return sorted(chosen)


def _span_record(span: TokenSpan | None) -> dict | None:
    if span is None:
        return None
    return {
        "start": int(span.start),
        "end": int(span.end),
        "n_tokens": int(span.n_tokens),
        "text": span.text,
    }


def extract_support_pack(
    token_ids: Sequence[int],
    *,
    n_prompt: int,
    n_full: int,
    decode: DecodeOne,
    a_state_key: str,
    a_error_recorded: bool = False,
) -> dict:
    """Instruction paragraph, calculation block, result lines, and random controls."""
    ids = [int(t) for t in token_ids]
    n_full = int(n_full)
    n_prompt = int(n_prompt)
    failures: list[str] = []
    if len(ids) != n_full:
        failures.append(f"token id length {len(ids)} != n_full {n_full}")
    if n_prompt <= 0 or n_prompt >= n_full:
        failures.append(f"n_prompt {n_prompt} is not inside (0, {n_full})")

    instruction = find_first_text_span(
        ids, ELIGIBILITY_RULE, decode, start=0, end=n_prompt if n_prompt > 0 else n_full
    )
    if instruction is None:
        instruction = find_first_text_span(ids, ELIGIBILITY_RULE, decode, start=0, end=n_full)
    if instruction is None:
        failures.append("eligibility instruction paragraph not found")
    elif ELIGIBILITY_RULE not in (instruction.text or ""):
        failures.append("decoded instruction span does not contain the eligibility rule")

    written = line_spans(ids, start=n_prompt, end=n_full, decode=decode)
    marker = results_marker_span(written)
    if marker is None:
        failures.append("RESULTS marker not found in the written region")
        calc = None
        after_marker = written
    else:
        calc = TokenSpan(n_prompt, marker.start, decode(ids[n_prompt:marker.start]))
        after_marker = [span for span in written if span.start >= marker.end]
        if marker.start <= n_prompt:
            failures.append("RESULTS marker is not after the prompt")

    selected_result = result_spans(after_marker if marker is not None else written)
    if len(selected_result) != 4:
        failures.append(f"expected four result lines after RESULTS, found {len(selected_result)}")
    result_pos = positions_from_spans(selected_result)
    result_text = decode_positions(ids, result_pos, decode) if result_pos else ""
    carriers = []
    for span in selected_result:
        match = RESULT_LINE.search(span.text or "")
        if match:
            carriers.append(match.group("carrier").upper())
    if sorted(carriers) != ["P", "Q", "R", "S"] and selected_result:
        failures.append(f"result carriers {carriers}")

    instr_pos = list(range(instruction.start, instruction.end)) if instruction is not None else []
    calc_pos = list(range(calc.start, calc.end)) if calc is not None else []
    if calc is not None:
        if not any(name in (calc.text or "") for name in CARRIER_HEADING):
            failures.append("calculation block is missing carrier headings")
        if RESULTS_MARK.search(calc.text or ""):
            failures.append("calculation block includes the RESULTS marker")
        if any(p in set(result_pos) for p in calc_pos):
            failures.append("calculation block overlaps result-line positions")

    baseline_pos, sink_extra = with_initial_positions(
        result_pos, n_full=n_full, n_initial=SINK_INITIAL
    )
    if n_full >= SINK_INITIAL and baseline_pos[:SINK_INITIAL] != list(range(SINK_INITIAL)):
        if any(i not in set(baseline_pos) for i in range(SINK_INITIAL)):
            failures.append("first four positions are missing from the baseline")
    first_four = [p for p in baseline_pos if p < SINK_INITIAL]
    if len(first_four) != len(set(first_four)):
        failures.append("first four positions were duplicated")

    support_pos = merge_positions(baseline_pos, instr_pos, calc_pos)
    extra_pos = [p for p in support_pos if p not in set(baseline_pos)]
    extra_runs = contiguous_runs(extra_pos)
    n_extra = len(extra_pos)
    if n_extra + len(baseline_pos) != len(support_pos):
        failures.append("relevant extra positions are not disjoint from the baseline")

    randoms = {}
    for seed in RANDOM_SEEDS:
        rec = sample_equal_random(
            n_full=n_full,
            baseline=baseline_pos,
            extra_count=n_extra,
            extra_runs=extra_runs,
            seed=seed,
            a_state_key=a_state_key,
        )
        rand_union = merge_positions(baseline_pos, rec["positions"])
        rec["union_positions"] = rand_union
        rec["n_union"] = len(rand_union)
        rec["overlap_instruction"] = len(set(rec["positions"]) & set(instr_pos))
        rec["overlap_calc"] = len(set(rec["positions"]) & set(calc_pos))
        rec["overlap_support_extra"] = len(set(rec["positions"]) & set(extra_pos))
        rec["overlap_support_extra_frac"] = (
            rec["overlap_support_extra"] / n_extra if n_extra else 0.0
        )
        rec["n_from_prompt"] = sum(p < n_prompt for p in rec["positions"])
        rec["n_from_written"] = sum(p >= n_prompt for p in rec["positions"])
        if len(rand_union) != len(support_pos):
            failures.append(f"seed {seed} union size {len(rand_union)} != relevant {len(support_pos)}")
        randoms[str(seed)] = rec

    supported_text = "\n\n".join(
        piece.strip()
        for piece in (
            instruction.text if instruction is not None else "",
            calc.text if calc is not None else "",
            result_text,
        )
        if piece and piece.strip()
    )
    ok = not failures
    return {
        "ok": ok,
        "failures": failures,
        "a_state_key": a_state_key,
        "a_error_recorded": bool(a_error_recorded),
        "n_full": n_full,
        "n_prompt": n_prompt,
        "instruction_occurrence": INSTRUCTION_OCCURRENCE,
        "instruction": _span_record(instruction),
        "instruction_pos": instr_pos,
        "calc": _span_record(calc),
        "calc_pos": calc_pos,
        "results_marker": _span_record(marker),
        "result_spans": [_span_record(span) for span in selected_result],
        "result_pos": result_pos,
        "result_text": result_text,
        "result_fallback": False,
        "result_carriers": carriers,
        "baseline_pos": baseline_pos,
        "n_sink_extra": len(sink_extra),
        "sink_extra": list(sink_extra),
        "support_extra_pos": extra_pos,
        "support_extra_runs": extra_runs,
        "n_extra": n_extra,
        "support_pos": support_pos,
        "random": randoms,
        "supported_text": supported_text,
        "first_four_once": first_four == list(range(min(SINK_INITIAL, n_full)))
        or set(first_four) == set(range(min(SINK_INITIAL, n_full))),
        "equal_random_counts": all(int(randoms[str(seed)]["n"]) == n_extra for seed in RANDOM_SEEDS),
        "used_gold_or_b_private": False,
    }


def extract_instruction_pack(
    token_ids: Sequence[int],
    *,
    n_prompt: int,
    n_full: int,
    decode: DecodeOne,
    a_state_key: str,
    a_error_recorded: bool = False,
) -> dict:
    """First eligibility-instruction paragraph plus equal-size random controls.

    Extra count matches the instruction span only. Random samples exclude the
    retained first four positions and result lines, but may overlap instructions
    or calculations. Seeds 1/2/3 are keyed by a_state_key.
    """
    pack = extract_support_pack(
        token_ids,
        n_prompt=n_prompt,
        n_full=n_full,
        decode=decode,
        a_state_key=a_state_key,
        a_error_recorded=a_error_recorded,
    )
    failures = list(pack.get("failures") or [])
    ids = [int(t) for t in token_ids]
    instr_pos = list(pack.get("instruction_pos") or [])
    calc_pos = list(pack.get("calc_pos") or [])
    retained = list(pack.get("baseline_pos") or [])
    n_instruction = len(instr_pos)
    if n_instruction <= 0:
        failures.append("instruction paragraph has no tokens")
    if set(instr_pos) & set(retained):
        failures.append("instruction paragraph overlaps retained compact positions")

    prompt_text = decode(ids[: int(n_prompt)]) if int(n_prompt) > 0 else ""
    ship_at = prompt_text.find("Shipment:")
    rule_at = prompt_text.find(ELIGIBILITY_RULE)
    instruction_in_prompt = bool(instr_pos) and instr_pos[-1] < int(n_prompt)
    instruction_before_facts = instruction_in_prompt
    if ship_at >= 0 and rule_at >= 0:
        instruction_before_facts = instruction_in_prompt and rule_at < ship_at
    if not instruction_before_facts:
        failures.append("instruction paragraph is not before the shipment facts")

    extra_runs = contiguous_runs(instr_pos) if instr_pos else []
    randoms = {}
    for seed in INSTRUCTION_RANDOM_SEEDS:
        rec = sample_equal_random(
            n_full=int(n_full),
            baseline=retained,
            extra_count=n_instruction,
            extra_runs=extra_runs,
            seed=seed,
            a_state_key=a_state_key,
        )
        rec["union_positions"] = merge_positions(retained, rec["positions"])
        rec["n_union"] = len(rec["union_positions"])
        rec["overlap_instruction"] = len(set(rec["positions"]) & set(instr_pos))
        rec["overlap_calc"] = len(set(rec["positions"]) & set(calc_pos))
        rec["n_from_prompt"] = sum(p < int(n_prompt) for p in rec["positions"])
        rec["n_from_written"] = sum(p >= int(n_prompt) for p in rec["positions"])
        if set(rec["positions"]) & set(retained):
            failures.append(f"seed {seed} overlapped retained compact positions")
        if len(rec["positions"]) != n_instruction:
            failures.append(f"seed {seed} extra count {len(rec['positions'])} != instruction {n_instruction}")
        randoms[str(seed)] = rec

    pack["instruction_random"] = randoms
    pack["n_instruction"] = n_instruction
    pack["retained_pos"] = retained
    pack["instruction_in_prompt"] = instruction_in_prompt
    pack["instruction_before_facts"] = instruction_before_facts
    pack["instruction_contains_rule"] = ELIGIBILITY_RULE in ((pack.get("instruction") or {}).get("text") or "")
    pack["equal_instruction_random_counts"] = all(
        int(randoms[str(seed)]["n"]) == n_instruction for seed in INSTRUCTION_RANDOM_SEEDS
    )
    pack["failures"] = failures
    pack["ok"] = not failures
    return pack
