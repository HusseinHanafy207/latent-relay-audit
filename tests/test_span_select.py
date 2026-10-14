"""Shared-position complete-span selection. No GPU."""

from __future__ import annotations

import torch

from src.span_select import (
    TokenSpan,
    caches_equal,
    contiguous_runs,
    decode_positions,
    gather_cache_positions,
    line_spans,
    mean_span_score,
    positions_from_spans,
    result_spans,
    select_spans_by_score,
    shared_position_scores,
    with_initial_positions,
)
from src.two_agent import (
    _align_written_ids,
    _capture_model_input_ids,
    _ids_from_decode_loop,
    eos_cache_alignment,
)


def _decode(ids):
    return "".join("\n" if int(i) == 10 else chr(int(i)) for i in ids)


def test_line_spans_keep_complete_newline_lines():
    ids = [ord(c) if c != "\n" else 10 for c in "aa\nbb\ncc"]
    spans = line_spans(ids, start=0, end=len(ids), decode=_decode)
    assert [span.text for span in spans] == ["aa\n", "bb\n", "cc"]
    assert positions_from_spans(spans) == list(range(len(ids)))


def test_result_spans_keep_carrier_result_lines_only():
    lines = [
        TokenSpan(0, 4, "Working 12\n"),
        TokenSpan(4, 20, "P: q = 12 | eligible\n"),
        TokenSpan(20, 38, "Q: q = 8 | ineligible\n"),
        TokenSpan(38, 42, "done"),
    ]
    kept = result_spans(lines)
    assert [span.text.strip() for span in kept] == [
        "P: q = 12 | eligible",
        "Q: q = 8 | ineligible",
    ]


def test_select_spans_does_not_truncate_a_line():
    spans = [
        TokenSpan(0, 10, "short\n"),
        TokenSpan(10, 40, "a much longer complete line\n"),
        TokenSpan(40, 45, "tiny\n"),
    ]
    scores = [0.1] * 10 + [0.9] * 30 + [0.2] * 5
    chosen = select_spans_by_score(spans, scores, target_n=12)
    assert chosen == [spans[1]]
    assert chosen[0].n_tokens == 30


def test_shared_scores_average_every_head():
    high = torch.zeros(1, 2, 4)
    high[0, 0, 1] = 1.0
    high[0, 1, 2] = 1.0
    scores = shared_position_scores([[high]])
    assert scores[1] == scores[2]
    assert scores[1] > scores[0]
    assert scores[1] > scores[3]


def test_gather_keeps_the_same_indices_on_every_head():
    layers = []
    for _ in range(2):
        key = torch.zeros(1, 3, 8, 4)
        value = torch.zeros(1, 3, 8, 4)
        for pos in range(8):
            key[:, :, pos, :] = pos
            value[:, :, pos, :] = pos + 10
        layers.append((key, value))
    blob = {"layers": layers}
    gathered = gather_cache_positions(blob, [1, 4, 6])
    for key, value in gathered["layers"]:
        assert tuple(key.shape) == (1, 3, 3, 4)
        assert key[0, 0, :, 0].tolist() == [1, 4, 6]
        assert key[0, 2, :, 0].tolist() == [1, 4, 6]
        assert value[0, 1, :, 0].tolist() == [11, 14, 16]


def test_text_decode_uses_the_same_positions_as_kv():
    ids = [ord(c) if c != "\n" else 10 for c in "keep\ndrop\nkeep"]
    positions = list(range(0, 5)) + list(range(10, 14))
    text = decode_positions(ids, positions, _decode)
    assert text == "keep\nkeep"
    assert contiguous_runs(positions) == [(0, 5), (10, 14)]
    assert set(positions) == set(positions_from_spans([TokenSpan(0, 5, "keep\n"), TokenSpan(10, 14, "keep")]))


def test_mean_span_score_is_the_token_average():
    span = TokenSpan(1, 4, "abc")
    assert mean_span_score([0.0, 1.0, 3.0, 5.0], span) == 3.0


def test_align_written_ids_matches_cache_length():
    class DummyTok:
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
            ids = [ord(c) for c in text]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    exact = _align_written_ids(DummyTok(), prompt_ids=prompt, written_text="ab", n_kv=5)
    assert exact.tolist() == [1, 2, 3, ord("a"), ord("b")]
    with_eos = _align_written_ids(DummyTok(), prompt_ids=prompt, written_text="ab", n_kv=6)
    assert with_eos.tolist() == [1, 2, 3, ord("a"), ord("b"), 99]


def test_identity_gather_preserves_tensors():
    key = torch.arange(24, dtype=torch.float32).view(1, 2, 6, 2)
    value = key + 100
    blob = {"layers": [(key.clone(), value.clone())]}
    gathered = gather_cache_positions(blob, list(range(6)))
    assert caches_equal(blob, gathered)
    extra, added = with_initial_positions([10, 11, 12], n_full=20, n_initial=4)
    assert added == [0, 1, 2, 3]
    assert extra[:4] == [0, 1, 2, 3]
    assert extra[-3:] == [10, 11, 12]
    already, none = with_initial_positions([0, 1, 2, 3, 8], n_full=10, n_initial=4)
    assert none == []
    assert already == [0, 1, 2, 3, 8]


def test_eos_counted_not_cached_matches_the_decode_loop():
    info = eos_cache_alignment(n_prompt=757, n_kv=1182, n_new=426, pos_cursor=1183)
    assert info["n_cached_generated"] == 425
    assert info["n_generated_counted"] == 426
    assert info["n_generated_not_cached"] == 1
    assert info["pos_cursor_minus_n_kv"] == 1
    assert info["eos_counted_not_cached"] is True


def test_decode_loop_ids_are_the_source_of_truth():
    class DummyTok:
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
            return {"input_ids": torch.tensor([[ord("x"), ord("y")]], dtype=torch.long)}

    prompt = torch.tensor([1, 2], dtype=torch.long)
    recorded = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([7], dtype=torch.long),
        torch.tensor([8], dtype=torch.long),
    ]
    full, info = _ids_from_decode_loop(
        DummyTok(),
        prompt_ids=prompt,
        written_text="xy",
        n_kv=4,
        n_new=3,
        pos_cursor=5,
        recorded_ids=recorded,
    )
    assert info["ids_source"] == "decode_loop"
    assert full.tolist() == [1, 2, 7, 8]
    assert info["loop_eq_retokenize"] is False
    assert info["loop_len_eq_n_kv"] is True
    assert info["eos_counted_not_cached"] is True
    assert info["n_cached_generated"] == 2
    assert info["n_generated_counted"] == 3
    assert info["n_generated_not_cached"] == 1


def test_missing_decode_capture_is_not_checked():
    class DummyTok:
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
            return {"input_ids": torch.tensor([[ord("a"), ord("b")]], dtype=torch.long)}

    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    full, info = _ids_from_decode_loop(
        DummyTok(),
        prompt_ids=prompt,
        written_text="ab",
        n_kv=5,
        n_new=3,
        pos_cursor=6,
        recorded_ids=[],
    )
    assert info["ids_source"] == "retokenize"
    assert full.tolist() == [1, 2, 3, ord("a"), ord("b")]
    assert info["loop_eq_retokenize"] is None
    assert info["loop_len_eq_n_kv"] is None


def test_capture_hooks_forward_not_generate():
    class DummyModel:
        def forward(self, *args, **kwargs):
            return "ok"

        def generate(self, *args, **kwargs):
            raise AssertionError("generate_text_batch does not call model.generate()")

    model = DummyModel()
    with _capture_model_input_ids(model) as recorded:
        assert "generate" not in model.__dict__
        assert "forward" in model.__dict__
        model.forward(input_ids=torch.tensor([[4, 5, 6]], dtype=torch.long))
        try:
            model.generate()
        except AssertionError as err:
            assert "model.generate()" in str(err)
        else:
            raise AssertionError("capture must not replace generate")
    assert recorded[0].tolist() == [4, 5, 6]
    assert "forward" not in model.__dict__
