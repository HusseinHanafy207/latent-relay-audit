"""Receiver query adapter: mechanics on a small fake attention model."""

from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.stage_plan4_qadapt as sq
from src.paths import DATA_DIR
from src.qadapt import (
    QueryAdapter,
    argmax_agreement,
    attention_implementation,
    forced_inputs,
    grad_report,
    kl_sum,
    logit_delta,
    response_logits,
    response_slice,
)

D, V, P = 8, 23, 5


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(D, D, bias=False) for _ in range(4))

    def forward(self, h, past):
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        pk, pv = past
        K, Vv = torch.cat([pk, k], dim=1), torch.cat([pv, v], dim=1)
        n_past, n = pk.shape[1], h.shape[1]
        scores = q @ K.transpose(1, 2) / math.sqrt(D)
        allowed = torch.arange(n_past + n).view(1, -1) <= (n_past + torch.arange(n)).view(-1, 1)
        scores = scores.masked_fill(~allowed, float("-inf"))
        return self.o_proj(torch.softmax(scores, dim=-1) @ Vv), (K, Vv)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = nn.Linear(D, D)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos = nn.Embedding(128, D)
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, return_dict):
        assert attention_mask.shape[1] == past_key_values[0][0].shape[1] + input_ids.shape[1]
        h = self.embed(input_ids) + self.pos(position_ids)
        for layer, past in zip(self.layers, past_key_values):
            a, _kv = layer.self_attn(h, past)
            h = h + a
            h = h + torch.tanh(layer.mlp(h))
        return SimpleNamespace(last_hidden_state=h)


class _Model(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Inner()
        self.lm_head = nn.Linear(D, V, bias=False)
        self.config = SimpleNamespace(_attn_implementation="eager")


def _past(seed):
    g = torch.Generator().manual_seed(seed)
    return {"layers": [(torch.randn(1, P, D, generator=g), torch.randn(1, P, D, generator=g)) for _ in range(2)]}


def _maker(blob):
    return lambda: [(k.clone(), v.clone()) for k, v in blob["layers"]]


def _fwd(model, blob, prompt, response):
    return response_logits(
        model, _maker(blob), prompt_ids=prompt, response_ids=response, pos_cursor=P, past_mask=torch.ones(1, P), device="cpu"
    )


def _greedy(model, blob, prompt, n):
    """Generation semantics: prefill the prompt, then feed one token at a time."""
    seq = list(prompt)
    out = []
    for _ in range(n):
        logits = response_logits(
            model, _maker(blob), prompt_ids=seq, response_ids=[0], pos_cursor=P, past_mask=torch.ones(1, P), device="cpu"
        )
        tok = int(logits.argmax(-1)[-1])
        out.append(tok)
        seq.append(tok)
    return out


def test_param_count_and_registered_size():
    model = _Model()
    ad = QueryAdapter(model, rank=2, alpha=4)
    assert len(ad.modules) == 2 and ad.n_params() == 2 * 2 * (D + D)
    assert 36 * 16 * (2560 + 4096) == 3833856
    pinned = json.loads((DATA_DIR.parent / "configs" / "pinned.json").read_text(encoding="utf-8"))
    assert pinned["study_plan4_qadapt"]["adapter"]["trainable_params"] == 3833856
    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
    with pytest.raises(RuntimeError):
        QueryAdapter(model)


def test_zero_init_reproduces_base_exactly_and_switch_defaults_off():
    model = _Model()
    blob = _past(1)
    prompt, resp = [1, 2, 3, 4], [5, 6, 7]
    with torch.no_grad():
        base = _fwd(model, blob, prompt, resp)
        ad = QueryAdapter(model, rank=2, alpha=4)
        assert not ad.switch["on"]
        off = _fwd(model, blob, prompt, resp)
        with ad.on():
            on = _fwd(model, blob, prompt, resp)
            with ad.off():
                assert not ad.switch["on"]
            assert ad.switch["on"]
    assert not ad.switch["on"]
    assert torch.equal(base, off) and torch.equal(base, on)
    assert logit_delta(off, on)["equal"]


def test_forced_logits_match_incremental_generation():
    model = _Model(seed=3)
    blob = _past(2)
    prompt = [3, 1, 4, 1, 5]
    gen = _greedy(model, blob, prompt, 6)
    with torch.no_grad():
        logits = _fwd(model, blob, prompt, gen)
    assert logits.shape == (6, V)
    assert argmax_agreement(logits, gen) == 1.0
    assert forced_inputs(prompt, gen) == prompt + gen[:-1]
    assert response_slice(5, 6) == slice(4, 10)


def test_only_adapter_gets_gradients_and_cache_is_untouched():
    model = _Model()
    teacher_blob, student_blob = _past(1), _past(2)
    before = [(k.clone(), v.clone()) for k, v in student_blob["layers"]]
    ad = QueryAdapter(model, rank=2, alpha=4)
    prompt, resp = [1, 2, 3], [4, 5, 6, 7]
    with torch.no_grad():
        t = _fwd(model, teacher_blob, prompt, resp)
    with ad.on():
        s = _fwd(model, student_blob, prompt, resp)
    loss = kl_sum(t, s) / len(resp)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    rep = grad_report(model, ad)
    assert rep["n_base_params_with_grad_or_trainable"] == 0 and rep["trainable_ids_equal_adapter"]
    assert rep["kinds"]["lora_B"] == {"n": 2, "with_grad": 2, "finite": 2, "nonzero": 2}
    assert rep["kinds"]["lora_A"]["with_grad"] == 2 and rep["kinds"]["lora_A"]["nonzero"] == 0
    for (k, v), (k0, v0) in zip(student_blob["layers"], before):
        assert torch.equal(k, k0) and torch.equal(v, v0)


def test_training_reduces_kl_and_reset_restores_init():
    model = _Model()
    teacher_blob, student_blob = _past(1), _past(2)
    ad = QueryAdapter(model, rank=4, alpha=8)
    init = ad.snapshot()
    opt = torch.optim.AdamW(ad.parameters(), lr=5e-2, weight_decay=0.0)
    prompt, resp = [1, 2, 3], [4, 5, 6, 7, 8]
    with torch.no_grad():
        t = _fwd(model, teacher_blob, prompt, resp)
    losses = []
    for _ in range(40):
        with ad.on():
            s = _fwd(model, student_blob, prompt, resp)
        loss = kl_sum(t, s) / len(resp)
        losses.append(float(loss))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < 0.5 * losses[0]
    assert not ad.b_all_zero() and not ad.matches(init)
    with torch.no_grad(), ad.on():
        trained = _fwd(model, student_blob, prompt, resp)
    with torch.no_grad():
        base = _fwd(model, student_blob, prompt, resp)
    assert not logit_delta(trained, base)["equal"]
    ad.reset()
    assert ad.matches(init) and ad.b_all_zero()
    with torch.no_grad(), ad.on():
        assert torch.equal(_fwd(model, student_blob, prompt, resp), base)


def test_kl_and_agreement_helpers():
    a = torch.randn(4, V)
    assert float(kl_sum(a, a)) == pytest.approx(0.0, abs=1e-6)
    assert float(kl_sum(a, torch.randn(4, V))) > 0
    assert argmax_agreement(a, a.argmax(-1).tolist()) == 1.0
    assert argmax_agreement(a, []) == 1.0


def test_attention_switch_restores_even_on_error():
    model = _Model()
    with attention_implementation(model, "sdpa"):
        assert model.config._attn_implementation == "sdpa"
    assert model.config._attn_implementation == "eager"
    with pytest.raises(ValueError):
        with attention_implementation(model, "sdpa"):
            raise ValueError
    assert model.config._attn_implementation == "eager"


def test_plan_leak_check():
    plan = "P: eligible - effective capacity 12 boxes\nQ: ineligible - no refrigeration available"
    assert sq.plan_leak_check("You are Agent B. Plan the eligibility.", plan)["ok"]
    leaked = sq.plan_leak_check("context ... Q: ineligible - no refrigeration available ...", plan)
    assert not leaked["ok"] and leaked["plan_lines_in_input"]
    assert not sq.plan_leak_check("x " + plan + " y", plan)["ok"]


def _fake_runner(model):
    return SimpleNamespace(wrapper=SimpleNamespace(model=model), device=torch.device("cpu"))


def _bundle(blob):
    return SimpleNamespace(cache_cpu=blob, pos_cursor_cpu=torch.tensor([P]), past_mask_cpu=torch.ones(1, P, dtype=torch.long))


def test_stage_backward_and_step_with_fake_runner(monkeypatch):
    monkeypatch.setattr(sq, "cache_to_device", lambda blob, device: [(k, v) for k, v in blob["layers"]])
    model = _Model()
    runner = _fake_runner(model)
    ad = QueryAdapter(model, rank=2, alpha=4)
    state = {"kvmix": {sq.TEACHER: _bundle(_past(1)), sq.STUDENT: _bundle(_past(2))}}
    snap = {c: [(k.clone(), v.clone()) for k, v in state["kvmix"][c].cache_cpu["layers"]] for c in (sq.TEACHER, sq.STUDENT)}
    targets = {
        "plan": {"prompt_ids": [1, 2, 3], "response_ids": [4, 5, 6]},
        "answer": {"prompt_ids": [1, 2, 3, 4, 5, 6, 7], "response_ids": [8, 9]},
    }
    scaler = sq.make_scaler(torch.device("cpu"))
    opt = sq.make_optimizer(ad)
    row = sq.sheet_backward(runner, ad, state, targets, scaler, weight=0.25)
    assert row["finite"] and row["n_tokens"] == 5 and row["loss"] > 0
    assert model.config._attn_implementation == "eager"
    st = sq.optimizer_step(ad, opt, scaler, inspect=lambda: grad_report(model, ad))
    assert not st["skipped"] and st["grad_norm_finite"]
    assert st["grads"]["kinds"]["lora_B"]["nonzero"] == 2 and st["grads"]["kinds"]["lora_A"]["nonzero"] == 0
    assert all(p.grad is None for p in ad.parameters())
    assert not ad.b_all_zero() and not ad.switch["on"]
    sq.sheet_backward(runner, ad, state, targets, scaler, weight=0.25)
    st2 = sq.optimizer_step(ad, opt, scaler, inspect=lambda: grad_report(model, ad))
    assert st2["grads"]["kinds"]["lora_A"]["nonzero"] == 2
    for c, layers in snap.items():
        for (k, v), (k0, v0) in zip(state["kvmix"][c].cache_cpu["layers"], layers):
            assert torch.equal(k, k0) and torch.equal(v, v0)


def _good_report():
    eq = {"equal": True, "argmax_agree": 1.0}
    return {
        "adapter": {"n_params": 3833856, "registered_params": 3833856, "n_modules": 36, "n_layers": 36},
        "sheets": [{"checks": {"ok": True}, "teacher_alignment": {"plan": 0.99, "answer": 1.0}}],
        "zero_init": {"logits": {"sdpa": eq, "eager": eq}, "generation": {"x": {"plan_text_equal": True, "answer_text_equal": True}}},
        "kernels": {"student_plan": {"argmax_agree": 0.995}, "teacher_plan": {"argmax_agree": 1.0}},
        "attention_after": "eager",
        "training": {
            "micro_steps": [{"finite": True}],
            "steps": [{"grads": {"n_base_params_with_grad_or_trainable": 0}, "skipped": False}],
            "first_step_grads_ok": True,
            "later_step_grads_ok": True,
        },
        "after_training": {
            "logits_on_vs_off": {"equal": False},
            "caches_held": {"x": {"tk_tv": True, "fk_fv": True, "a_full": True}},
            "rebuilt_state_equal": True,
        },
        "reset": {"matches_init": True, "b_all_zero": True, "logits_on_vs_off": {"equal": True}, "optimizer_state_empty": True},
    }


def test_smoke_gates():
    assert sq.smoke_passed(_good_report())["all_passed"]
    bad = _good_report()
    bad["after_training"]["caches_held"]["x"]["tk_tv"] = False
    assert not sq.smoke_passed(bad)["gates"]["received_caches_unchanged"]
    bad = _good_report()
    bad["zero_init"]["logits"]["sdpa"]["equal"] = False
    assert not sq.smoke_passed(bad)["all_passed"]
    bad = _good_report()
    bad["sheets"][0]["teacher_alignment"]["plan"] = 0.5
    assert not sq.smoke_passed(bad)["gates"]["teacher_targets_align"]
    bad = _good_report()
    bad["attention_after"] = "sdpa"
    assert not sq.smoke_passed(bad)["all_passed"]


def test_epoch_order_and_selection_rule():
    ids = [f"s{i}" for i in range(96)]
    o1, o1b, o2 = sq.epoch_order(ids, 1), sq.epoch_order(ids, 1), sq.epoch_order(ids, 2)
    assert o1 == o1b and sorted(o1) == sorted(ids) and o1 != o2 and o1 != ids
    assert sq.select_epoch({1: 100, 2: 110, 3: 110}) == 2
    assert sq.select_epoch({1: 120, 2: 110, 3: 119}) == 1
    assert sq.select_epoch({"3": 5, "1": 5, "2": 4}) == 1


def _row(ex_id, fam, cond, variant, pred, gold, *, parse=True, trunc=False):
    return {
        "example_id": ex_id,
        "family_id": fam,
        "condition": cond,
        "variant": variant,
        "plan_pred_status": pred,
        "gold_status": gold,
        "plan_n_status_correct": sum(pred.get(c) == g for c, g in gold.items()),
        "parse_ok": parse,
        "plan_truncated": trunc,
    }


def _flip(status):
    return "ineligible" if status == "eligible" else "eligible"


def _dev_fixture(*, adapted_follows=True, no_r_good=False, adapted_gain=True, pair_ok=True, malformed=0):
    carriers = [f"C{j}" for j in range(12)]
    n = 12
    gold = {}
    for i in range(n):
        gold[f"b{i}"] = {c: ("eligible" if (j + i) % 3 == 0 else "ineligible") for j, c in enumerate(carriers)}
    partners = {f"b{i}": f"b{(i + 1) % n}" for i in range(n)}
    pairs = {}
    for i in range(n):
        g = dict(gold[f"b{i}"])
        g["C0"] = _flip(g["C0"])
        gold[f"f{i}"] = g
        pairs[f"b{i}"] = (f"f{i}", "C0")
    rows = []
    for i in range(n):
        b, f, fam = f"b{i}", f"f{i}", f"fam{i}"
        weak = {c: "eligible" for c in carriers}
        rows.append(_row(b, fam, "unadapted_correct_r", "base", weak, gold[b]))
        rows.append(_row(f, fam, "unadapted_correct_r", "a_flip", weak, gold[f]))
        rows.append(_row(b, fam, "adapted_correct_r", "base", dict(gold[b]) if adapted_gain else weak, gold[b],
                         parse=i >= malformed))
        flip_pred = dict(gold[f]) if pair_ok else dict(gold[b])
        rows.append(_row(f, fam, "adapted_correct_r", "a_flip", flip_pred, gold[f]))
        rows.append(_row(b, fam, "adapted_no_r", "base", dict(gold[b]) if no_r_good else weak, gold[b]))
        p = partners[b]
        rows.append(_row(b, fam, "adapted_mismatched_r", "base", dict(gold[p]) if adapted_follows else dict(gold[b]), gold[b]))
        rows.append(_row(b, fam, "unadapted_mismatched_r", "base", weak, gold[b]))
    return rows, partners, pairs, gold


def test_dev_gates_pass_when_message_dependent():
    rows, partners, pairs, gold = _dev_fixture()
    out = sq.dev_gates(rows, partners, pairs, gold)
    assert out["advance"], out["gates"]
    g = out["gates"]
    assert g["beats_unadapted_kv"]["gain"] >= 12 and g["beats_unadapted_kv"]["ci95_per_family"][0] > 0
    assert g["follows_mismatched_results"]["follows_partner"] == g["follows_mismatched_results"]["n_carriers"] > 0
    assert g["a_change_pairs"]["adapted"]["both_correct"] == 12


def test_dev_gates_catch_shortcuts():
    rows, partners, pairs, gold = _dev_fixture(no_r_good=True)
    out = sq.dev_gates(rows, partners, pairs, gold)
    assert not out["gates"]["worse_without_results"]["pass"] and not out["advance"]
    rows, partners, pairs, gold = _dev_fixture(adapted_follows=False)
    out = sq.dev_gates(rows, partners, pairs, gold)
    assert not out["gates"]["follows_mismatched_results"]["pass"] and not out["advance"]
    rows, partners, pairs, gold = _dev_fixture(adapted_gain=False)
    assert not sq.dev_gates(rows, partners, pairs, gold)["gates"]["beats_unadapted_kv"]["pass"]
    rows, partners, pairs, gold = _dev_fixture(pair_ok=False)
    out = sq.dev_gates(rows, partners, pairs, gold)
    assert out["gates"]["a_change_pairs"]["adapted"]["both_correct"] == 0
    assert out["gates"]["a_change_pairs"]["unadapted"]["both_correct"] == 0
    rows, partners, pairs, gold = _dev_fixture(malformed=2)
    assert not sq.dev_gates(rows, partners, pairs, gold)["gates"]["malformed_output"]["pass"]


def _packed_runner():
    from src.two_agent import TwoAgentInventory

    return object.__new__(TwoAgentInventory)


def _packed(runner, n, seed, ids):
    g = torch.Generator().manual_seed(seed)
    blob = {"layers": [(torch.randn(1, 2, n, 4, generator=g).half(), torch.randn(1, 2, n, 4, generator=g).half()) for _ in range(2)]}
    return runner._bundle_from_packed(blob, ids, pos_cursor=n, source="t")


def test_no_r_and_mismatched_caches():
    r = _packed_runner()
    own = _packed(r, 76, 1, list(range(70)) + [900 + i for i in range(6)])
    partner = _packed(r, 79, 2, list(range(70)) + [800 + i for i in range(9)])
    no_r = sq.no_r_bundle(r, own, 70)
    mism = sq.mismatched_bundle(r, own, partner, 70, 70)
    row = sq.dev_cache_check(own, partner, no_r, mism, 70)
    assert row["ok"], row
    assert row["n_result_own"] == 6 and row["n_result_partner"] == 9
    assert mism.full_ids_cpu.view(-1).tolist()[70:] == [800 + i for i in range(9)]
    bad = sq.mismatched_bundle(r, partner, partner, 70, 70)
    assert not sq.dev_cache_check(own, partner, no_r, bad, 70)["mismatched_support_is_own"]
    with pytest.raises(RuntimeError):
        sq.mismatched_bundle(r, own, partner, 70, 71)


def test_train_summary_by_epoch():
    log = [
        {"kind": "micro", "epoch": 1, "loss": 0.4, "seconds": 2.0},
        {"kind": "micro", "epoch": 1, "loss": 0.2, "seconds": 2.0},
        {"kind": "step", "epoch": 1, "skipped": False, "grad_norm": 1.0},
        {"kind": "micro", "epoch": 2, "loss": 0.1, "seconds": 2.0},
        {"kind": "step", "epoch": 2, "skipped": True, "grad_norm": 3.0},
    ]
    s = sq._train_summary(log, {"x": 1})
    assert s["by_epoch"][1]["mean_loss"] == pytest.approx(0.3) and s["by_epoch"][1]["n_micro"] == 2
    assert s["by_epoch"][2]["skipped_steps"] == 1 and s["x"] == 1


def test_dev_conditions_cover_registered_gates():
    assert {"adapted_correct_r", "unadapted_correct_r", "adapted_no_r", "adapted_mismatched_r", "fk_fv", "text_result"} <= set(sq.DEV_CONDS)
    assert not sq.DEV_CONDS["text_result"]["adapted"] and sq.DEV_CONDS["text_result"]["cache"] is None
    assert sq.DEV_CONDS["adapted_mismatched_r"]["variants"] == ("base",)
    for name in ("train_state.pt", "adapter_epoch1.pt", "adapter_epoch3.pt", "dev_summary.json", "selection.json"):
        assert name in sq.PERSIST_NAMES


def _eval_rows():
    rows, partners, pairs, gold = _dev_fixture()
    extra = []
    for r in rows:
        if r["condition"] == "adapted_correct_r":
            for cond in ("fk_fv", "full_kv", "text_result"):
                extra.append(dict(r, condition=cond))
    rows = rows + extra
    for r in rows:
        adapted = sq.EVAL_CONDS[r["condition"]]["adapted"]
        r.update(
            {
                "plan_all_status_correct": r["plan_n_status_correct"] == 12,
                "n_plan_tokens": 100,
                "n_new_tokens": 10,
                "b_decode_s": 11.0 if adapted else 10.0,
                "b_prefill_s": 1.0,
                "n_kv_positions": 0 if r["condition"] == "text_result" else (900 if r["condition"] == "full_kv" else 100),
            }
        )
    return rows, partners, pairs, gold


def test_eval_conditions_add_full_kv_only():
    assert set(sq.EVAL_CONDS) == set(sq.DEV_CONDS) | {"full_kv"}
    assert sq.EVAL_CONDS["full_kv"] == {"cache": "full_kv", "adapted": False, "variants": ("base", "a_flip")}
    assert {k for k, s in sq.EVAL_CONDS.items() if s["adapted"]} == {"adapted_correct_r", "adapted_no_r", "adapted_mismatched_r"}
    for name in ("frozen_candidate.json", "a_artifacts_eval.jsonl", "eval_construction.json", "eval_evaluations.jsonl", "eval_summary.json"):
        assert name in sq.PERSIST_NAMES
    for _name, left, right in sq.EVAL_PAIRED:
        assert left in sq.EVAL_CONDS and right in sq.EVAL_CONDS


def test_eval_summary_comparisons_and_overhead():
    rows, partners, pairs, gold = _eval_rows()
    train = {"trainable_params": 3833856, "adapter_bytes": 15335424, "by_epoch": {"1": {"train_s": 250.0}}, "peak_gpu_gb": 12.0}
    s = sq.eval_summary(rows, partners, pairs, gold, train=train, frozen={"selected_epoch": 1, "sha256": "x"})
    assert s["message_dependence"]["all_hold"]
    assert s["labels"]["base"]["adapted_correct_r"]["labels_correct"] == 144
    comp = s["paired_by_family"]
    up = comp["adapted_vs_unadapted"]["base"]["labels"]
    assert up["n_cases"] == 12 and up["total_delta"] == s["labels"]["base"]["adapted_correct_r"]["labels_correct"] - s["labels"]["base"]["unadapted_correct_r"]["labels_correct"] > 0
    assert comp["adapted_vs_full_kv"]["base"]["labels"]["total_delta"] == 0
    assert "a_flip" not in comp["adapted_r_vs_no_r"]
    assert "a_flip" in comp["adapted_vs_text"]
    ov = s["costs"]["adapter_overhead"]
    assert ov["decode_per_token_ratio"] == pytest.approx(1.1) and ov["prefill_ratio"] == pytest.approx(1.0)
    rec = s["costs"]["receiver"]
    assert rec["full_kv"]["mean_kv_positions"] == 900 and rec["text_result"]["mean_resident_mib"] == 0
    assert rec["adapted_correct_r"]["mean_resident_mib"] == pytest.approx(100 * 0.140625)
    assert s["costs"]["one_time"]["training_s"] == 250.0
    assert "does not by itself establish a cost advantage" in s["costs"]["note"]


def _freeze_dir(tmp_path, *, advance=True):
    (tmp_path / "dev_summary.json").write_text(
        json.dumps({"advance": advance, "construction_passed": True, "gates": {"g": {"pass": advance}}}), encoding="utf-8"
    )
    (tmp_path / "selection.json").write_text(
        json.dumps({"selected_epoch": 1, "labels_by_epoch": {"1": 144, "2": 144, "3": 144}}), encoding="utf-8"
    )
    torch.save({"w": torch.zeros(2)}, tmp_path / "adapter_epoch1.pt")
    return tmp_path


def test_freeze_requires_gates_and_locks_the_checkpoint(tmp_path):
    args = SimpleNamespace(limit=None)
    run = lambda d: sq._run_freeze(args, d, None, {}, None, {}, {})  # noqa: E731
    failed = tmp_path / "a"
    failed.mkdir()
    _freeze_dir(failed, advance=False)
    assert run(failed) == 2
    assert sq._frozen(failed) is None
    d = tmp_path / "b"
    d.mkdir()
    _freeze_dir(d)
    assert run(d) == 0
    frozen = sq._frozen(d)
    assert frozen["selected_epoch"] == 1 and len(frozen["sha256"]) == 64
    assert run(d) == 0
    torch.save({"w": torch.ones(2)}, d / "adapter_epoch1.pt")
    with pytest.raises(RuntimeError):
        sq._frozen(d)
    assert run(d) == 2
    (d / "adapter_epoch1.pt").unlink()
    assert run(d) == 2


def test_eval_write_needs_freeze_and_no_adapter(tmp_path):
    args = SimpleNamespace(limit=None)
    runner = SimpleNamespace(_query_adapter=None)
    assert sq._run_eval_write(args, tmp_path, None, {"eval": []}, runner, {}, {}) == 2
    _freeze_dir(tmp_path)
    assert sq._run_freeze(args, tmp_path, None, {}, None, {}, {}) == 0
    runner._query_adapter = object()
    with pytest.raises(RuntimeError):
        sq._run_eval_write(args, tmp_path, None, {"eval": []}, runner, {}, {})
    runner._query_adapter = None
    assert sq._run_eval_write(args, tmp_path, None, {"eval": []}, runner, {}, {}) == 0
    assert sq._run_eval(args, tmp_path, None, {"eval": []}, runner, {}, {}) == 2


def test_registered_hashes_and_guards(monkeypatch):
    pinned = json.loads((DATA_DIR.parent / "configs" / "pinned.json").read_text(encoding="utf-8"))
    protocol = pinned["study_plan4_qadapt"]
    assert sq.registered_hash_mismatches(protocol) == []
    assert len(protocol["amendments_before_training"]) >= 4
    assert any("Full KV" in a for a in protocol["amendments_before_eval"])
    assert "results_stage_plan4_qadapt_prep" in sq.PRESERVED
    assert "stage_plan4_qadapt" in str(sq._paths()[0])
    monkeypatch.setattr(sys, "argv", ["stage", "--max_new_tokens", "64"])
    assert sq.main() == 2
