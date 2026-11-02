"""Receiver query adapter: LoRA on q_proj, switched on only for B's own tokens.

Applies Attuner's query-side repair (arXiv 2609.36722, Section 4) to the
transferred twelve-carrier cache. The received cache is never recomputed: the
adapter changes B's queries, and through them B's own later hidden states.
The sender, support construction and teacher always run with it switched off.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import torch
from torch import nn

RANK = 16
ALPHA = 32
INIT_SEED = 2026


class QueryLoRA(nn.Module):
    """base(x) + scale * B(A(x)), with A, B in fp32 and the sum cast back to the base dtype."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, switch: dict) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.switch = switch
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.switch["on"]:
            return y
        delta = (x.float() @ self.lora_A.t()) @ self.lora_B.t()
        return y + (delta * self.scale).to(y.dtype)


def decoder_layers(model) -> Sequence[nn.Module]:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if not layers:
        raise RuntimeError("Could not find decoder layers on the model.")
    return layers


class QueryAdapter:
    """Attach, switch, reset and inspect the q_proj adapters of one model."""

    def __init__(self, model, *, rank: int = RANK, alpha: float = ALPHA, seed: int = INIT_SEED) -> None:
        self.model = model
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.switch = {"on": False}
        for p in model.parameters():
            p.requires_grad_(False)
        self.modules: list[QueryLoRA] = []
        for layer in decoder_layers(model):
            attn = layer.self_attn
            if isinstance(attn.q_proj, QueryLoRA):
                raise RuntimeError("q_proj already carries an adapter.")
            wrapped = QueryLoRA(attn.q_proj, self.rank, self.alpha, self.switch)
            attn.q_proj = wrapped
            self.modules.append(wrapped)
        self.reset()

    def parameters(self) -> list[nn.Parameter]:
        return [p for m in self.modules for p in (m.lora_A, m.lora_B)]

    def named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        out = []
        for i, m in enumerate(self.modules):
            out += [(f"layers.{i}.q_proj.lora_A", m.lora_A), (f"layers.{i}.q_proj.lora_B", m.lora_B)]
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def reset(self) -> None:
        """A ~ Kaiming-uniform(a=sqrt(5)) = U(-1/sqrt(fan_in), 1/sqrt(fan_in)); B = 0."""
        gen = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            for m in self.modules:
                bound = 1.0 / math.sqrt(m.lora_A.shape[1])
                init = (torch.rand(m.lora_A.shape, generator=gen, dtype=torch.float32) * 2 - 1) * bound
                m.lora_A.copy_(init.to(m.lora_A.device))
                m.lora_B.zero_()
                m.lora_A.grad = None
                m.lora_B.grad = None
        for p in self.parameters():
            p.requires_grad_(True)

    @contextmanager
    def on(self) -> Iterator[None]:
        prev = self.switch["on"]
        self.switch["on"] = True
        try:
            yield
        finally:
            self.switch["on"] = prev

    @contextmanager
    def off(self) -> Iterator[None]:
        prev = self.switch["on"]
        self.switch["on"] = False
        try:
            yield
        finally:
            self.switch["on"] = prev

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters()}

    def load(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in self.named_parameters():
                p.copy_(state[n].to(p.device))

    def matches(self, state: dict[str, torch.Tensor]) -> bool:
        return all(torch.equal(p.detach().cpu(), state[n]) for n, p in self.named_parameters())

    def b_all_zero(self) -> bool:
        return all(bool((m.lora_B == 0).all()) for m in self.modules)


def grad_report(model, adapter: QueryAdapter) -> dict:
    """Which parameters hold gradients. Only adapter parameters may."""
    adapter_ids = {id(p) for p in adapter.parameters()}
    base_with_grad = [n for n, p in model.named_parameters() if id(p) not in adapter_ids and (p.requires_grad or p.grad is not None)]
    per = {}
    for n, p in adapter.named_parameters():
        g = p.grad
        per[n] = {
            "has_grad": g is not None,
            "finite": g is not None and bool(torch.isfinite(g).all()),
            "abs_max": None if g is None else float(g.abs().max()),
        }
    kinds = {}
    for kind in ("lora_A", "lora_B"):
        rows = [r for n, r in per.items() if n.endswith(kind)]
        kinds[kind] = {
            "n": len(rows),
            "with_grad": sum(r["has_grad"] for r in rows),
            "finite": sum(r["finite"] for r in rows),
            "nonzero": sum(bool(r["abs_max"]) for r in rows),
        }
    return {
        "base_params_with_grad_or_trainable": base_with_grad[:5],
        "n_base_params_with_grad_or_trainable": len(base_with_grad),
        "trainable_ids_equal_adapter": {id(p) for p in model.parameters() if p.requires_grad} <= adapter_ids,
        "kinds": kinds,
    }


# ---------------------------------------------------------------- teacher forcing


def response_slice(n_prompt: int, n_response: int) -> slice:
    """Hidden positions whose next-token logits predict the response tokens."""
    return slice(int(n_prompt) - 1, int(n_prompt) - 1 + int(n_response))


def forced_inputs(prompt_ids: Sequence[int], response_ids: Sequence[int]) -> list[int]:
    """The response's last token is predicted, never fed."""
    return [int(t) for t in prompt_ids] + [int(t) for t in response_ids[:-1]]


def response_logits(
    model,
    make_cache: Callable[[], object],
    *,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    pos_cursor: int,
    past_mask: torch.Tensor,
    device,
) -> torch.Tensor:
    """Logits (n_response, V) for the response tokens, continuing from a fresh cache copy.

    Positions continue from the cache cursor and the mask covers the received
    cache plus the new tokens, as in generation from a cache.
    """
    ids = forced_inputs(prompt_ids, response_ids)
    n = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    past_mask = (past_mask.to(device=device, dtype=torch.long) > 0).to(torch.long).view(1, -1)
    full_mask = torch.cat([past_mask, torch.ones((1, n), dtype=torch.long, device=device)], dim=-1)
    position_ids = (int(pos_cursor) + torch.arange(n, device=device, dtype=torch.long)).view(1, -1)
    out = model.model(
        input_ids=input_ids,
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=make_cache(),
        use_cache=True,
        return_dict=True,
    )
    hidden = out.last_hidden_state[0, response_slice(len(prompt_ids), len(response_ids))]
    return model.lm_head(hidden)


def kl_sum(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Sum over tokens of KL(p_teacher || p_student), full vocabulary, fp32 log-softmax."""
    t = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    s = torch.log_softmax(student_logits.float(), dim=-1)
    return (t.exp() * (t - s)).sum()


def argmax_agreement(logits: torch.Tensor, targets: Sequence[int]) -> float:
    if not len(targets):
        return 1.0
    pred = logits.argmax(dim=-1).view(-1).tolist()
    return sum(int(p) == int(t) for p, t in zip(pred, targets)) / len(targets)


def logit_delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs()
    return {
        "equal": bool(torch.equal(a, b)),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "argmax_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
    }


def current_attention(model) -> str | None:
    return getattr(model.config, "_attn_implementation", None)


def _set_attention(model, impl: str) -> None:
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter(impl)
        except Exception:
            pass
    if current_attention(model) != impl:
        model.config._attn_implementation = impl


@contextmanager
def attention_implementation(model, impl: str) -> Iterator[None]:
    """Temporarily switch the attention kernel (eager generation, memory-efficient training)."""
    old = current_attention(model)
    if old == impl:
        yield
        return
    _set_attention(model, impl)
    try:
        yield
    finally:
        _set_attention(model, old)
