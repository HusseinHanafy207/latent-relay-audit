"""Merged receiver and handoff timing helpers on small fakes."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.stage_plan4_qmerge as sm
from src.qadapt import QueryAdapter, response_logits
from src.qmerge import (
    attached_as_built,
    compact_sender,
    extend,
    gather_on_device,
    lcp,
    merge_stats,
    merged_q_weights,
    q_weight_digest,
    receiver_weights,
    two_pass,
)
from src.rope_rerotate import dummy_cos_sin, rerotate_cache_keys
from src.span_select import gather_cache_positions

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


class _AttnInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.pos = nn.Embedding(128, D)
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, return_dict):
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
        self.model = _AttnInner()
        self.lm_head = nn.Linear(D, V, bias=False)


def _past(seed):
    g = torch.Generator().manual_seed(seed)
    return {"layers": [(torch.randn(1, P, D, generator=g), torch.randn(1, P, D, generator=g)) for _ in range(2)]}


def _fwd(model, blob, prompt, response):
    return response_logits(
        model,
        lambda: [(k.clone(), v.clone()) for k, v in blob["layers"]],
        prompt_ids=prompt,
        response_ids=response,
        pos_cursor=P,
        past_mask=torch.ones(1, P),
        device="cpu",
    )


def _trained(seed=0):
    model = _Model(seed)
    ad = QueryAdapter(model, rank=2, alpha=4)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for m in ad.modules:
            m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.3)
    return model, ad


def test_merged_matches_unmerged_and_unadapted_matches_off():
    model, ad = _trained()
    blob, prompt, resp = _past(3), [1, 2, 3, 4], [5, 6, 7]
    originals = [m.base.weight for m in ad.modules]
    digest = q_weight_digest(ad)
    merged = merged_q_weights(ad)
    with torch.no_grad():
        with ad.off():
            off = _fwd(model, blob, prompt, resp)
        with receiver_weights(ad, "unmerged"):
            assert ad.switch["on"]
            on = _fwd(model, blob, prompt, resp)
        with receiver_weights(ad, "merged", merged):
            assert all(layer.self_attn.q_proj is m.base for layer, m in zip(model.model.layers, ad.modules))
            mer = _fwd(model, blob, prompt, resp)
        with receiver_weights(ad, "unadapted"):
            una = _fwd(model, blob, prompt, resp)
    assert not ad.switch["on"]
    assert not torch.allclose(off, on)
    assert torch.allclose(mer, on, atol=1e-5)
    assert torch.equal(una, off)
    assert attached_as_built(ad, originals) and q_weight_digest(ad) == digest


def test_receiver_weights_restore_on_error_and_reject_bad_input():
    model, ad = _trained()
    originals = [m.base.weight for m in ad.modules]
    merged = merged_q_weights(ad)
    with pytest.raises(KeyError):
        with receiver_weights(ad, "merged", merged):
            raise KeyError("boom")
    assert attached_as_built(ad, originals)
    with pytest.raises(ValueError):
        with receiver_weights(ad, "merged", merged[:1]):
            pass
    with pytest.raises(ValueError):
        with receiver_weights(ad, "half"):
            pass
    assert attached_as_built(ad, originals)


def test_fp16_merge_rounding_is_measured_not_assumed():
    _model, ad = _trained()
    for m in ad.modules:
        m.base.weight = nn.Parameter(m.base.weight.detach().half(), requires_grad=False)
    merged = merged_q_weights(ad)
    assert all(w.dtype == torch.float16 for w in merged)
    stats = merge_stats(ad, merged)
    w = stats["worst"]
    assert stats["n_layers"] == 2
    assert w["rounding_max_abs"] > 0
    assert 0 < w["rounding_rel_to_delta"] < 0.05
    assert w["min_applied_vs_delta_cos"] > 0.99
    assert all(not torch.equal(a, m.base.weight) for a, m in zip(merged, ad.modules))


def test_lcp_and_device_gather():
    assert lcp([1, 2, 3], [1, 2, 4, 5]) == 2 and lcp([], [1]) == 0 and lcp([7, 8], [7, 8]) == 2
    g = torch.Generator().manual_seed(5)
    blob = {"layers": [(torch.randn(1, 2, 6, 4, generator=g), torch.randn(1, 2, 6, 4, generator=g)) for _ in range(2)]}
    got = gather_on_device(blob["layers"], [0, 2, 4])
    ref = gather_cache_positions(blob, [0, 2, 4])["layers"]
    assert all(torch.equal(a, c) and torch.equal(b, d) for (a, b), (c, d) in zip(got, ref))


class _Rotary(nn.Module):
    def forward(self, x, pos):
        return dummy_cos_sin(pos.view(-1).tolist(), x.shape[-1])


def test_compact_sender_matches_cpu_assembly():
    model = SimpleNamespace(model=SimpleNamespace(rotary_emb=_Rotary()))
    g = torch.Generator().manual_seed(0)
    blob = {"layers": [(torch.randn(1, 2, 9, 8, generator=g), torch.randn(1, 2, 9, 8, generator=g)) for _ in range(2)]}
    src, new = [3, 5, 6], [70, 71, 72]
    got = compact_sender(blob["layers"], src, new, model)
    ref = rerotate_cache_keys(gather_cache_positions(blob, src), old_positions=src, new_positions=new, model=model)["layers"]
    assert all(torch.allclose(a, c) and torch.equal(b, d) for (a, b), (c, d) in zip(got, ref))


# ---------------------------------------------------------------- two-pass prefix sharing

EOS, VOCAB = 0, 11


class _Cache:
    def __init__(self, tokens=()):
        self.tokens = list(tokens)

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, n):
        self.tokens = self.tokens[:n]


class _Inner(nn.Module):
    """Positions must continue the cache (cursor == cache length in these tests)."""

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, return_dict):
        n_past = past_key_values.get_seq_length()
        ids = input_ids.view(-1).tolist()
        assert attention_mask.shape[1] == n_past + len(ids)
        assert position_ids.view(-1).tolist() == list(range(n_past, n_past + len(ids)))
        past_key_values.tokens.extend(ids)
        return SimpleNamespace(last_hidden_state=None)


class _Toy(nn.Module):
    """Next token depends on the whole history, so any prefix or crop error changes the output."""

    def __init__(self):
        super().__init__()
        self.model = _Inner()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, **_kw):
        self.model(input_ids, attention_mask, position_ids, past_key_values, True, True)
        h = past_key_values.tokens
        return (sum((i + 1) * t for i, t in enumerate(h)) * 7 + len(h)) % VOCAB


class _Wrapper:
    def __init__(self, model):
        self.model = model

    def generate_text_batch(self, x, attn, *, pos_cursor, max_new_tokens, temperature, top_p, past_key_values, past_attention_mask):
        cache = past_key_values
        assert past_attention_mask.shape[1] == cache.get_seq_length()
        n0 = cache.get_seq_length()
        pos = int(pos_cursor.view(-1)[0])
        full = torch.cat([past_attention_mask, attn], dim=-1)
        tok = self.model(input_ids=x, attention_mask=full, position_ids=(pos + torch.arange(x.size(1))).view(1, -1), past_key_values=cache)
        out = []
        while len(out) < max_new_tokens:
            out.append(tok)
            if tok == EOS:
                break
            n = cache.get_seq_length()
            tok = self.model(
                input_ids=torch.tensor([[tok]]),
                attention_mask=torch.ones(1, n + 1, dtype=torch.long),
                position_ids=torch.tensor([[pos + n - n0]]),
                past_key_values=cache,
            )
        text = " ".join(str(t) for t in out if t != EOS)
        return [text], cache, None, [len(out)]


def _single(model, prefix, ids, max_new):
    cache = _Cache(prefix)
    texts, _c, _p, lens = _Wrapper(model).generate_text_batch(
        torch.tensor([ids]),
        torch.ones(1, len(ids), dtype=torch.long),
        pos_cursor=torch.tensor([len(prefix)]),
        max_new_tokens=max_new,
        temperature=0.0,
        top_p=1.0,
        past_key_values=cache,
        past_attention_mask=torch.ones(1, len(prefix), dtype=torch.long),
    )
    return texts[0], lens[0]


def test_two_pass_equals_two_independent_passes():
    model = _Toy()
    received = [4, 9, 2, 7]
    plan_ids = [1, 5, 5, 3, 8, 6, 2]
    shared = 4

    def answer_of(plan_text):
        tail = [int(t) for t in plan_text.split()][:3]
        return plan_ids[:shared] + [10] + tail + [9], f"answer after {plan_text}"

    out = two_pass(
        _Wrapper(model),
        cache=_Cache(received),
        n_past=len(received),
        cursor=len(received),
        plan_ids=plan_ids,
        start=0,
        shared=shared,
        answer_of=answer_of,
        plan_max=6,
        answer_max=5,
        device="cpu",
    )
    plan_ref, n_plan = _single(model, received, plan_ids, 6)
    ans_ids, prompt = answer_of(plan_ref)
    ans_ref, n_ans = _single(model, received, ans_ids, 5)
    assert out["plan"]["text"] == plan_ref and out["plan"]["n_new"] == n_plan
    assert out["answer"]["text"] == ans_ref and out["answer"]["n_new"] == n_ans
    assert out["answer_prompt"] == prompt and out["n_shared_tokens"] == shared
    assert out["plan"]["n_prefill_tokens"] == len(plan_ids) - shared
    assert out["answer"]["n_prefill_tokens"] == len(ans_ids) - shared


def test_two_pass_with_reused_public_header_equals_from_scratch():
    model = _Toy()
    header = [3, 1, 4]
    plan_ids = header + [1, 5, 9, 2, 6]
    cache = _Cache()
    extend(model, cache, header, n_past=0, cursor=0, device="cpu")
    out = two_pass(
        _Wrapper(model),
        cache=cache,
        n_past=len(header),
        cursor=len(header),
        plan_ids=plan_ids,
        start=len(header),
        shared=6,
        answer_of=lambda plan: (plan_ids[:6] + [7, 7], "a"),
        plan_max=4,
        answer_max=4,
        device="cpu",
    )
    ref, _n = _single(model, [], plan_ids, 4)
    ans, _m = _single(model, [], plan_ids[:6] + [7, 7], 4)
    assert out["plan"]["text"] == ref and out["answer"]["text"] == ans
    assert out["n_shared_tokens"] == 3


def test_two_pass_refuses_answer_that_breaks_the_shared_prefix():
    model = _Toy()
    with pytest.raises(RuntimeError):
        two_pass(
            _Wrapper(model),
            cache=_Cache([1, 2]),
            n_past=2,
            cursor=2,
            plan_ids=[3, 4, 5, 6],
            start=0,
            shared=3,
            answer_of=lambda plan: ([3, 9, 5, 6, 1], "a"),
            plan_max=3,
            answer_max=3,
            device="cpu",
        )
    with pytest.raises(ValueError):
        two_pass(_Wrapper(model), cache=_Cache([1]), n_past=1, cursor=1, plan_ids=[3, 4], start=0, shared=2,
                 answer_of=lambda p: ([3, 4, 5], "a"), plan_max=1, answer_max=1, device="cpu")


def test_extend_accepts_tensor_ids():
    model = _Toy()
    a, b = _Cache([1]), _Cache([1])
    extend(model, a, [4, 5], n_past=1, cursor=1, device="cpu")
    extend(model, b, torch.tensor([4, 5]), n_past=1, cursor=1, device="cpu")
    assert a.tokens == b.tokens == [1, 4, 5]


# ---------------------------------------------------------------- stage summaries

STATUS = {f"C{i}": ("eligible" if i % 2 else "ineligible") for i in range(12)}


def _rec(eid, fam, cond, rep, total, *, labels=12, correct=True, variant="base", plan="p", raw="r", **times):
    pred = dict(STATUS)
    for c in list(pred)[: 12 - labels]:
        pred[c] = "eligible" if pred[c] == "ineligible" else "ineligible"
    row = {
        "example_id": eid,
        "family_id": fam,
        "variant": variant,
        "condition": cond,
        "rep": rep,
        "handoff_total_s": total,
        "gold_status": dict(STATUS),
        "plan_pred_status": pred,
        "plan_n_status_correct": labels,
        "plan_all_status_correct": labels == 12,
        "correct": correct,
        "parse_ok": True,
        "plan_text": plan,
        "raw_text": raw,
        "n_plan_tokens": 100,
        "n_new_tokens": 10,
    }
    row.update({c: 0.0 for c in sm.COMPONENTS})
    row.update(times)
    return row


def _bench_rows(*, merged_cost=8.0, text_cost=9.0, merged_labels=12):
    rows = []
    base = {"merged_kv": merged_cost, "unmerged_kv": merged_cost + 0.4, "unadapted_kv": merged_cost - 0.1,
            "fk_fv": merged_cost + 0.2, "text_result": text_cost, "full_kv": merged_cost + 1.0}
    for f in range(6):
        for rep in (1, 2, 3):
            for cond, cost in base.items():
                labels = merged_labels if cond == "merged_kv" else 12
                rows.append(_rec(f"fam-{f:03d}-base", f"fam-{f:03d}", cond, rep, cost + 0.01 * f + 0.02 * rep,
                                 labels=labels, plan_decode_s=cost / 2, plan_prefill_s=0.1))
    return rows


def test_bench_order_is_a_seeded_permutation():
    a = sm.bench_order("e1", 1)
    assert sorted(a) == sorted(sm.CONDS) and a == sm.bench_order("e1", 1)
    orders = {tuple(sm.bench_order(f"e{i}", r)) for i in range(10) for r in (1, 2, 3)}
    assert len(orders) > 5


def test_paired_time_sign_and_repeatability():
    rows = _bench_rows()
    p = sm.paired_time(rows, "merged_kv", "text_result")
    assert p["mean_delta_per_case"] == pytest.approx(-1.0)
    assert p["repeatable_saving"] and not p["repeatable_cost"]
    assert all(v == pytest.approx(-1.0) for v in p["mean_delta_by_rep"].values())
    q = sm.paired_time(rows, "merged_kv", "unadapted_kv")
    assert q["repeatable_cost"] and not q["repeatable_saving"]


def test_decision_requires_reliability_and_a_repeatable_saving():
    good = sm.bench_summary(_bench_rows(), [], [], one_time={"merge_s": 2.0, "training_s": 98.0})
    assert good["reliability_check"]["pass"]
    assert good["decision"]["useful_practical_result"]
    assert good["decision"]["vs_text_result"]["break_even_handoffs"] == pytest.approx(100.0)
    assert good["timing"]["merged_kv"]["handoff_total_s"] < good["timing"]["text_result"]["handoff_total_s"]
    slower = sm.bench_summary(_bench_rows(text_cost=7.0), [], [])
    assert not slower["decision"]["useful_practical_result"]
    assert slower["decision"]["vs_text_result"]["repeatable_cost"]
    lossy = sm.bench_summary(_bench_rows(merged_labels=11), [], [])
    assert not lossy["reliability_check"]["pass"] and not lossy["decision"]["useful_practical_result"]
    assert "no speed result" in lossy["decision"]["statement"]


def _agree_rows(*, merged_labels=12, merged_parse=True):
    rows = []
    for i in range(4):
        rows.append({**_rec(f"e{i}", f"f{i}", "x", 1, 0.0), "mode": "unmerged"})
        m = _rec(f"e{i}", f"f{i}", "x", 1, 0.0, labels=merged_labels if i == 0 else 12)
        m.update({"mode": "merged", "parse_ok": merged_parse or i != 0})
        if i == 0 and merged_labels != 12:
            m["plan_text"] = "changed"
        rows.append(m)
    return rows


def _layers(seed, n=3):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, 2, n, 4, generator=g).half(), torch.randn(1, 2, n, 4, generator=g).half()) for _ in range(2)]


def test_r_compare_reports_key_rounding_and_value_identity():
    frozen = _layers(0)
    same = sm.r_compare(frozen, [(k.clone(), v.clone()) for k, v in frozen])
    assert same["shapes_equal"] and same["keys_bit_equal"] and same["values_equal"] and same["keys_max_abs"] == 0.0
    nudged = [(k.clone(), v.clone()) for k, v in frozen]
    nudged[1][0][0, 0, 0, 0] += 0.01
    c = sm.r_compare(frozen, nudged)
    assert not c["keys_bit_equal"] and c["values_equal"] and 0 < c["keys_max_abs"] < 0.02
    assert c["min_keys_bit_equal_fraction"] < 1.0
    bad_v = [(k.clone(), v.clone() + 1) for k, v in frozen]
    assert not sm.r_compare(frozen, bad_v)["values_equal"]
    assert not sm.r_compare(frozen, _layers(0, n=4))["shapes_equal"]


def test_move_declares_what_leaves_the_device():
    layers = _layers(1)
    t = {"movement_s": 0.0, "movement_out_s": 0.0, "movement_in_s": 0.0}
    assert sm._move(layers, "same_gpu", "cpu", t) is layers and t["movement_s"] == 0.0
    out = sm._move(layers, "device_transfer", "cpu", t)
    assert all(torch.equal(a, c) and torch.equal(b, d) for (a, b), (c, d) in zip(out, layers))
    assert t["movement_s"] == pytest.approx(t["movement_out_s"] + t["movement_in_s"])
    assert set(sm.DEPLOYMENTS) == {"same_gpu", "device_transfer"}
    assert "network" in sm.DEPLOYMENTS["device_transfer"] and "not measured" in sm.DEPLOYMENTS["device_transfer"]


def _integrate_fixture(*, actual_labels=12, direct_labels=12, values_equal=True, keys_bit_equal=False):
    rows, r_rows = [], []
    for i in range(4):
        eid = f"e{i}"
        for run, labels in (("frozen_std", 12), ("actual_std", actual_labels), ("actual_direct", direct_labels)):
            r = _rec(eid, f"f{i}", "x", 1, 0.0, labels=labels if i == 0 else 12)
            r["run"] = run
            if i == 0 and labels != 12:
                r["plan_text"] = "changed"
            rows.append(r)
        check = {"shapes_equal": True, "values_equal": values_equal, "keys_bit_equal": keys_bit_equal,
                 "keys_max_abs": 0.0 if keys_bit_equal else 0.002, "keys_mean_abs": 1e-5, "min_keys_bit_equal_fraction": 0.97}
        logits = [{"part": p, "n_tokens": 50, "argmax_agree_tokens": 50, "equal": keys_bit_equal, "max_abs": 0.01, "kl_mean": 1e-6}
                  for p in ("plan", "answer")]
        r_rows.append({"example_id": eid, "variant": "base", "r_check": check, "logits": logits})
    agree_merged = {r["example_id"]: r for r in rows if r["run"] == "frozen_std"}
    return rows, r_rows, agree_merged


def test_integrate_summary_decides_whether_the_gpu_path_feeds_b():
    ok = sm.integrate_summary(*_integrate_fixture(), {"ok": True}, 4)
    assert ok["complete"] and ok["passed"] and ok["cause"] is None
    assert ok["r_comparison"]["values_equal_all"] and ok["r_comparison"]["sheets_keys_bit_equal"] == 0
    assert ok["frozen_std_reproduces_agree_merged"]["plan_text"] == 4
    one = sm.integrate_summary(*_integrate_fixture(actual_labels=11), {"ok": True}, 4)
    assert one["passed"] and len(one["differences"]["actual_std"]) == 1
    r_bad = sm.integrate_summary(*_integrate_fixture(actual_labels=10), {"ok": True}, 4)
    assert not r_bad["passed"] and "GPU-prepared R changes" in r_bad["cause"]
    assert "refuses" in r_bad["next"]
    direct_bad = sm.integrate_summary(*_integrate_fixture(direct_labels=10), {"ok": True}, 4)
    assert not direct_bad["passed"] and "direct path" in direct_bad["cause"]
    values_bad = sm.integrate_summary(*_integrate_fixture(values_equal=False), {"ok": True}, 4)
    assert not values_bad["passed"]
    incomplete = sm.integrate_summary(*_integrate_fixture(), {"ok": True}, 5)
    assert not incomplete["complete"] and not incomplete["passed"]


def test_agreement_gates():
    logits = [{"n_tokens": 100, "argmax_agree_tokens": 100}]
    intact = {"ok": True}
    ok = sm.agree_gates(_agree_rows(), logits, intact)
    assert ok["passed"] and ok["label_predictions_changed"] == 0
    one = sm.agree_gates(_agree_rows(merged_labels=11), logits, intact)
    assert one["passed"] and one["label_predictions_changed"] == 1
    two = sm.agree_gates(_agree_rows(merged_labels=10), logits, intact)
    assert not two["gates"]["labels"]["pass"] and not two["passed"]
    assert not sm.agree_gates(_agree_rows(merged_parse=False), logits, intact)["passed"]
    assert not sm.agree_gates(_agree_rows(), [{"n_tokens": 100, "argmax_agree_tokens": 98}], intact)["passed"]
    assert not sm.agree_gates(_agree_rows(), logits, {"ok": False})["passed"]
    assert not sm.agree_gates([], logits, intact)["passed"]
