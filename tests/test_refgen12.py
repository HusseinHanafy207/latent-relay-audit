"""Reference attention under free generation (lockstep decoding)."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

import src.stage_plan4_refgen12 as rg
from src.stage_plan4_refgen12 import (
    BRANCHES,
    INTERPRETATION_GUIDE,
    LOCKSTEP,
    PERSIST_NAMES,
    PRESERVED,
    REFERENCE,
    _summarize,
    divergence,
    first_text_difference,
    gen_char_offsets,
    lockstep_generate,
    main,
    path_tolerance,
)
from src.stage_select import _load_rows

V = 12
EOS = 11


class _Tok:
    eos_token_id = EOS

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + int(i)) for i in ids if not (skip_special_tokens and int(i) == EOS))


class _Model:
    """Row 0 prefers last+2, row 1 prefers last+1 (mod 11); EOS once the input reaches `stop`."""

    def __init__(self, ctl, n_layers, stop=None):
        self.ctl, self.n_layers, self.stop, self.calls = ctl, n_layers, stop, []

    def __call__(self, *, input_ids, attention_mask, position_ids, past_key_values, use_cache, return_dict, logits_to_keep=None):
        self.calls.append(
            {"ids": input_ids.clone(), "mask_len": int(attention_mask.size(1)), "pos": position_ids.clone(), "keep": logits_to_keep}
        )
        self.ctl.n_calls += self.n_layers
        if self.ctl.mode == "replace":
            self.ctl.n_replaced += self.n_layers
        self.ctl.layers_seen.update(range(self.n_layers))
        last = input_ids[:, -1]
        logits = torch.zeros(input_ids.size(0), 1, V, dtype=torch.float16)
        for r in range(input_ids.size(0)):
            nxt = EOS if self.stop is not None and int(last[r]) == self.stop else (int(last[r]) + (2 if r == 0 else 1)) % 11
            logits[r, 0, nxt] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _fake_runner(monkeypatch, stop=None, n_layers=3):
    ctl = SimpleNamespace(mode="off", n_calls=0, n_replaced=0, layers_seen=set())

    @contextmanager
    def fake_replay(model, mode, donor=0, target=1):
        ctl.mode, ctl.n_calls, ctl.n_replaced, ctl.layers_seen = mode, 0, 0, set()
        yield ctl

    monkeypatch.setattr(rg, "replay_attention", fake_replay)
    monkeypatch.setattr(rg, "cache_to_device", lambda blob, device: blob)
    model = _Model(ctl, n_layers, stop)
    runner = SimpleNamespace(wrapper=SimpleNamespace(model=model, tokenizer=_Tok()), device="cpu")
    return runner, model


def _bundle(n_past=5, n_layers=3, cursor=7, fill=0.0):
    k = torch.full((1, 2, n_past, 4), fill)
    return SimpleNamespace(
        full_ids_cpu=torch.arange(n_past).view(1, -1),
        pos_cursor_cpu=torch.tensor([cursor]),
        past_mask_cpu=torch.ones((1, n_past), dtype=torch.long),
        cache_cpu={"layers": [(k.clone(), k.clone()) for _ in range(n_layers)]},
    )


def test_lockstep_feeds_the_intervention_token_to_both_rows(monkeypatch):
    runner, model = _fake_runner(monkeypatch)
    a, b = _bundle(fill=1.0), _bundle(fill=2.0)
    before = a.cache_cpu["layers"][0][0].clone()
    res = lockstep_generate(runner, [a, b], [3, 4, 0], max_new_tokens=4, mode="replace")
    assert res["ids"] == [1, 2, 3, 4]
    assert res["n_new"] == 4 and res["truncated"] is True
    assert res["text"] == "bcde"
    first = model.calls[0]
    assert first["keep"] == 1 and first["mask_len"] == 5 + 3
    assert first["pos"].tolist() == [[7, 8, 9], [7, 8, 9]]
    for step, call in enumerate(model.calls[1:]):
        assert call["ids"].view(-1).tolist() == [res["ids"][step]] * 2
        assert call["pos"].view(-1).tolist() == [7 + 3 + step] * 2
        assert call["mask_len"] == 5 + 3 + step + 1
    assert len(model.calls) == 4
    donor, target = res["logits"]
    assert donor.shape == (4, V) and target.shape == (4, V)
    assert donor.argmax(-1).tolist() == [2, 3, 4, 5]
    assert target.argmax(-1).tolist() == res["ids"]
    info = res["info"]
    assert info["n_forwards"] == 4 and info["every_forward_every_layer"] and info["replaced_every_call"]
    assert torch.equal(a.cache_cpu["layers"][0][0], before)


def test_lockstep_stops_at_eos_counted_not_fed(monkeypatch):
    runner, model = _fake_runner(monkeypatch, stop=2)
    res = lockstep_generate(runner, [_bundle(), _bundle()], [0], max_new_tokens=10, mode="replace")
    assert res["ids"] == [1, 2, EOS]
    assert res["n_new"] == 3 and res["truncated"] is False
    assert res["text"] == "bc"
    assert len(model.calls) == 3
    assert all(int(c["ids"].view(-1)[0]) != EOS for c in model.calls)


def test_lockstep_rejects_unmatched_rows(monkeypatch):
    runner, _model = _fake_runner(monkeypatch)
    try:
        lockstep_generate(runner, [_bundle(cursor=7), _bundle(cursor=8)], [0], max_new_tokens=2, mode="replace")
    except RuntimeError as err:
        assert "cursor" in str(err)
    else:
        raise AssertionError("expected a mismatch error")


def test_offsets_tolerance_and_divergence():
    tok = _Tok()
    assert gen_char_offsets(tok, [0, 1, EOS]) == [(0, 1), (1, 2), (2, 2)]
    hist = {"raw": "ab", "offsets": [(0, 1), (1, 2)], "labels": [], "pair": None}
    x = torch.randn(2, V)
    st = path_tolerance(x, x.clone(), hist)
    assert st["ok"] and st["mean_kl"] == 0.0 and st["argmax_agree"] == 1.0
    y = x.clone()
    y[:, 0] = 100.0
    assert not path_tolerance(x, y, hist)["ok"]

    text = "P: eligible\nQ: ineligible\n"
    ids = list(range(len(text)))
    offsets = [(i, i + 1) for i in range(len(text))]
    e_pos, i_pos = text.index("eligible"), text.index("ineligible")
    labels = [
        {"carrier": "P", "ref_text_label": "eligible", "token_index": e_pos, "token_id": 100},
        {"carrier": "Q", "ref_text_label": "ineligible", "token_index": i_pos, "token_id": 101},
    ]
    vocab = 120
    chooser = torch.zeros(len(ids), vocab)
    donor = torch.zeros(len(ids), vocab)
    gen = [5] * len(ids)
    gen[e_pos], gen[i_pos] = 100, 101
    for t, g in enumerate(gen):
        chooser[t, g] = 5.0
        donor[t, g] = 5.0
    donor[i_pos, 101], donor[i_pos, 100] = 0.0, 5.0
    hist = {"raw": text, "offsets": offsets, "labels": labels, "pair": (100, 101)}
    d = divergence(SimpleNamespace(decode=lambda ids: str(ids)), gen, donor, chooser, hist, {"P": "eligible", "Q": "eligible"})
    assert d["first_disagree_step"] == i_pos and d["n_disagree_steps"] == 1
    assert d["at_label_position"] and d["line_carrier"] == "Q" and d["line_index"] == 1
    assert d["labels_reference_agrees"] == 1 and d["labels_reference_correct_on_history"] == 2
    assert d["first_label_error_carrier"] == "Q" and d["disagreement_precedes_first_label_error"] is False


def test_first_text_difference():
    assert first_text_difference("a\nP: eligible", "a\nP: eligible") is None
    d = first_text_difference("a\nP: eligible | $5", "a\nP: ineligible | $5")
    assert d["line_index"] == 1 and d["carrier"] == "P"


def test_stage_scope_and_guards(monkeypatch):
    assert REFERENCE == "fk_fv"
    assert BRANCHES == ("tk_tv", "fk_tv", "fk_fv", "tk_tv_refattn", "tk_tv_selfattn")
    assert LOCKSTEP["tk_tv_refattn"]["rows"] == ("fk_fv", "tk_tv")
    assert LOCKSTEP["tk_tv_selfattn"]["rows"] == ("tk_tv", "tk_tv")
    assert "results_stage_plan4_refattn12" in PRESERVED and "results_stage_plan4_kvmix12" in PRESERVED
    assert "case_checks.jsonl" in PERSIST_NAMES
    assert "96/108" in INTERPRETATION_GUIDE and "94/108" in INTERPRETATION_GUIDE
    source = inspect.getsource(rg)
    assert "_reference_plans" not in source
    assert "full_kv" not in source
    assert "argmax(logits[chooser])" in source
    monkeypatch.setattr("sys.argv", ["stage_plan4_refgen12", "--split", "plan4_dev"])
    assert main() == 2
    monkeypatch.setattr("sys.argv", ["stage_plan4_refgen12", "--plan_max_new_12", "256"])
    assert main() == 2


def _row(eid, cond, n_correct, ie, ei, plan="p", raw="r"):
    return {
        "example_id": eid,
        "condition": cond,
        "difficulty": "L1",
        "gold_status": {str(i): "eligible" for i in range(12)},
        "plan_pred_status": {str(i): "eligible" for i in range(12)},
        "plan_n_status_correct": n_correct,
        "plan_all_status_correct": n_correct == 12,
        "plan_ineligible_marked_eligible": ie,
        "plan_eligible_marked_ineligible": ei,
        "parse_ok": True,
        "plan_truncated": False,
        "truncated": False,
        "correct": n_correct == 12,
        "plan_text": plan,
        "raw_text": raw,
    }


def test_summary_counts_and_keeps_teacher_forced_separate():
    vals = {"tk_tv": (7, 5, 0), "fk_tv": (10, 2, 0), "fk_fv": (12, 0, 0), "tk_tv_refattn": (11, 1, 0), "tk_tv_selfattn": (7, 5, 0)}
    rows = [_row(f"e{i}", b, *v) for i in range(9) for b, v in vals.items()]
    checks = [
        {
            "example_id": f"e{i}",
            "ok": True,
            "comparisons": {"selfattn_plan_identical_to_ordinary_tk_tv": True, "selfattn_labels_equal_ordinary_tk_tv": True},
            "divergence": {
                "n_steps": 100,
                "n_disagree_steps": 2,
                "first_disagree_step": 40,
                "at_label_position": True,
                "n_labels_located": 12,
                "labels_reference_agrees": 11,
                "labels_generated_correct": 11,
                "labels_reference_correct_on_history": 12,
                "first_label_error_step": 40,
                "first_label_error_carrier": "Q",
                "disagreement_precedes_first_label_error": False,
            },
        }
        for i in range(9)
    ]
    kvmix = [_row(f"e{i}", c, *vals.get(c, (6, 6, 0))) for i in range(9) for c in ("tk_tv", "fk_tv", "tk_fv", "fk_fv")]
    refattn = {"branches": {"fk_tv": {"labels_correct": 96, "n_labels": 108}}}
    s = _summarize(rows, checks, kvmix_rows=kvmix, refattn_summary=refattn)
    assert s["instrumentation_ok"] is True
    assert s["labels"]["tk_tv_refattn"]["labels_correct"] == 99
    assert s["labels"]["tk_tv_refattn"]["ineligible_marked_eligible"] == 9
    d = s["paired_by_case"]["tk_tv_refattn_vs_tk_tv"]["labels_correct"]
    assert d["total_delta"] == 36 and d["cases_increased"] == 9
    assert s["reproduction_vs_kvmix12"]["tk_tv"]["identical_outputs"] == 9
    assert s["self_replay_free_generation"]["plan_identical_to_ordinary_tk_tv"] == 9
    assert s["divergence"]["cases_with_label_error"] == 9
    assert s["divergence"]["first_disagree_at_label_position"] == 9
    ctx = s["context"]
    assert ctx["teacher_forced_on_reference_plan_refattn12"]["fk_tv"]["labels_correct"] == 96
    assert ctx["free_generation_kvmix12"]["fk_tv"]["labels_correct"] == 90
    assert "Free generation" in s["measurement_note"]


def test_saved_kvmix_reference_rows_exist():
    rows = _load_rows(Path("results_stage_plan4_refattn12/kvmix12_evaluations.jsonl"))
    assert {r["condition"] for r in rows} >= {"tk_tv", "fk_tv", "fk_fv"}
    assert sum(r["condition"] == "fk_tv" and 1 or 0 for r in rows) == 9
