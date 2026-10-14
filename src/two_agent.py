"""Two-agent sequential relay: one frozen model, reused sender caches."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch

from src.cache_utils import (
    attentions_to_cpu,
    attentions_to_device,
    cache_is_finite,
    cache_to_cpu,
    cache_to_device,
    clone_cache_blob,
    kv_blob_size_bytes,
    kv_size_bytes,
    metadata_is_finite,
    n_kv_positions,
)
from src.inventory import InventoryExample
from src.parse import ParsedAnswer, label_both
from src.paths import UPSTREAM_DIR
from src.precision import peak_gpu_memory_gb
from src.prompts_inventory import (
    probe_prompt,
    receiver_prompt,
    sender_prompt,
    sender_prompt_inventory_only,
    text_message_prompt,
    true_inventory_absent_from_receiver_prompt,
)


def _ensure_upstream_path() -> None:
    path = str(UPSTREAM_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _n_selected_from(meta: Any, *, start: int) -> Optional[int]:
    if not isinstance(meta, dict):
        return None
    matrix = meta.get("selected_prompt_positions_matrix")
    if not matrix:
        return None
    kept: set[int] = set()
    for layers in matrix:
        for heads in layers:
            for idxs in heads:
                for idx in idxs:
                    pos = int(idx)
                    if pos >= start:
                        kept.add(pos)
    return len(kept)


def make_compressor(name: str, *, kv_budget: int, sink_size: int, pca_rank: int):
    _ensure_upstream_path()
    if name == "full":
        from compression_methods.Full import Full

        return Full(sink_size=sink_size, kv_budget=kv_budget)
    if name == "headwise":
        from compression_methods.Headwise import Headwise

        return Headwise(sink_size=sink_size, kv_budget=kv_budget)
    if name == "hobf_fast":
        from compression_methods.HOBFFast import HOBFFast

        return HOBFFast(sink_size=sink_size, kv_budget=kv_budget, pca_rank=pca_rank, inject_mode="uniform")
    raise ValueError(f"Unknown compressor {name}")


def _retokenize_written_ids(tokenizer, *, prompt_ids: torch.Tensor, written_text: str, n_kv: int) -> torch.Tensor:
    """Prompt ids plus generated ids reconstructed from decoded text."""
    prompt_ids = prompt_ids.detach().cpu().long().view(-1)
    encoded = tokenizer(written_text or "", add_special_tokens=False, return_tensors="pt")
    written_ids = encoded["input_ids"][0].detach().cpu().long().view(-1)
    full = torch.cat([prompt_ids, written_ids], dim=0)
    if int(full.numel()) == int(n_kv):
        return full.contiguous()
    eos = tokenizer.eos_token_id
    if eos is not None and int(full.numel()) + 1 == int(n_kv):
        return torch.cat([full, torch.tensor([int(eos)], dtype=torch.long)], dim=0).contiguous()
    raise RuntimeError(
        f"Token alignment failed: prompt {int(prompt_ids.numel())} + written "
        f"{int(written_ids.numel())} != cache {int(n_kv)}."
    )


def _align_written_ids(tokenizer, *, prompt_ids: torch.Tensor, written_text: str, n_kv: int) -> torch.Tensor:
    return _retokenize_written_ids(
        tokenizer, prompt_ids=prompt_ids, written_text=written_text, n_kv=n_kv
    )


@contextmanager
def _capture_model_input_ids(model) -> Iterator[list[torch.Tensor]]:
    """Record input_ids of every forward. generate_text_batch does not call model.generate()."""
    recorded: list[torch.Tensor] = []
    orig = model.forward
    had_instance = "forward" in getattr(model, "__dict__", {})

    def hooked(*args, **kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            maybe = args[0]
            if torch.is_tensor(maybe) and maybe.dtype in (torch.int32, torch.int64, torch.long):
                ids = maybe
        if ids is not None:
            recorded.append(ids.detach().cpu().long().reshape(-1).contiguous().clone())
        return orig(*args, **kwargs)

    model.forward = hooked
    try:
        yield recorded
    finally:
        if had_instance:
            model.forward = orig
        elif "forward" in getattr(model, "__dict__", {}):
            del model.forward


def eos_cache_alignment(
    *,
    n_prompt: int,
    n_kv: int,
    n_new: int,
    pos_cursor: int,
) -> dict[str, Any]:
    """The decode loop counts EOS in gen_lens and pos_cursor, then breaks before caching it.

    Generated tokens (n_new / gen_lens) include EOS. Tokens processed into the
    cache (n_kv) do not. pos_cursor follows the counted sequence, so it is one
    past cache length when EOS stopped decoding.
    """
    n_cached_generated = int(n_kv) - int(n_prompt)
    n_generated_not_cached = int(n_new) - n_cached_generated
    pos_gap = int(pos_cursor) - int(n_kv)
    matches = n_generated_not_cached == 1 and pos_gap == 1
    return {
        "n_prompt_ids": int(n_prompt),
        "n_kv": int(n_kv),
        "n_generated_counted": int(n_new),
        "n_cached_generated": n_cached_generated,
        "n_generated_not_cached": n_generated_not_cached,
        "pos_cursor": int(pos_cursor),
        "pos_cursor_minus_n_kv": pos_gap,
        "eos_counted_not_cached": matches,
        "eos_rule": (
            "generate_text_batch appends EOS to gen_ids and advances pos_cursor, "
            "then breaks before that forward, so the KV cache does not contain EOS."
        ),
    }


def _ids_from_decode_loop(
    tokenizer,
    *,
    prompt_ids: torch.Tensor,
    written_text: str,
    n_kv: int,
    n_new: int,
    pos_cursor: int,
    recorded_ids: Optional[list[torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """IDs that were actually forwarded into the cache. Retokenized text is a check."""
    prompt_ids = prompt_ids.detach().cpu().long().view(-1)
    n_prompt = int(prompt_ids.numel())
    info = eos_cache_alignment(
        n_prompt=n_prompt, n_kv=int(n_kv), n_new=int(n_new), pos_cursor=int(pos_cursor)
    )
    try:
        retok = _retokenize_written_ids(
            tokenizer, prompt_ids=prompt_ids, written_text=written_text, n_kv=n_kv
        )
    except RuntimeError:
        retok = None
    info["n_retokenized"] = None if retok is None else int(retok.numel())
    info["ids_source"] = "retokenize"
    info["loop_eq_retokenize"] = None
    info["n_loop_ids"] = 0
    info["n_forwards"] = 0 if not recorded_ids else len(recorded_ids)
    info["loop_len_eq_n_kv"] = None
    if not recorded_ids:
        if retok is None:
            raise RuntimeError("No decode-loop ids and retokenized text does not match cache length.")
        info["n_ids_eq_n_kv"] = int(retok.numel()) == int(n_kv)
        return retok, info
    loop = _sequence_from_forwards(recorded_ids, n_kv=int(n_kv))
    info["n_loop_ids"] = 0 if loop is None else int(loop.numel())
    info["n_prefill_ids"] = int(recorded_ids[0].numel())
    info["n_decode_forwards"] = max(0, len(recorded_ids) - 1)
    if loop is None or int(loop.numel()) != int(n_kv):
        info["n_ids_eq_n_kv"] = False
        info["loop_len_eq_n_kv"] = False
        info["loop_eq_retokenize"] = False if retok is not None else None
        if retok is None:
            raise RuntimeError(
                f"Decode-loop ids {info['n_loop_ids']} and retokenized text do not match cache {int(n_kv)}."
            )
        return retok, info
    info["ids_source"] = "decode_loop"
    info["n_ids_eq_n_kv"] = True
    info["loop_len_eq_n_kv"] = True
    info["loop_eq_retokenize"] = None if retok is None else bool(torch.equal(loop, retok))
    return loop.contiguous(), info


def _sequence_from_forwards(recorded_ids: list[torch.Tensor], *, n_kv: int) -> Optional[torch.Tensor]:
    """Concatenated input_ids actually forwarded into the cache."""
    if not recorded_ids:
        return None
    full = torch.cat([chunk.view(-1) for chunk in recorded_ids], dim=0)
    if int(full.numel()) == int(n_kv):
        return full.contiguous()
    chunks = [recorded_ids[0].view(-1)]
    for chunk in recorded_ids[1:]:
        ids = chunk.view(-1)
        if int(ids.numel()) == 1:
            chunks.append(ids)
    slim = torch.cat(chunks, dim=0)
    if int(slim.numel()) == int(n_kv):
        return slim.contiguous()
    return None


class _WrapperArgs:
    latent_space_realign = False
    think = False


@dataclass
class SenderBundle:
    source: str
    prompt: str
    rendered: str
    input_ids_cpu: torch.Tensor
    prompt_mask_cpu: torch.Tensor
    pos_cursor_cpu: torch.Tensor
    past_mask_cpu: torch.Tensor
    cache_cpu: dict[str, Any]
    attentions_cpu: list[list[torch.Tensor]]
    latent_steps: int
    sender_latency_s: float
    full_relay_bytes: int
    last_hidden_cpu: Optional[torch.Tensor] = None
    full_ids_cpu: Optional[torch.Tensor] = None
    ids_source: str = "retokenize"
    token_align: Optional[dict[str, Any]] = None


@dataclass
class ReceiverResult:
    condition: str
    relay: str
    raw_text: str
    parsed: ParsedAnswer
    parsed_legacy: ParsedAnswer
    truncated: bool
    n_new_tokens: int
    n_output_lines: int
    max_new_tokens: int
    relay_bytes: int
    compression_time_s: float
    compression_core_s: float
    receiver_latency_s: float
    sender_latency_s: float
    peak_gpu_gb: Optional[float]
    cache_finite: bool
    obf_finite: bool
    leaked_true_inventory: bool
    prompt: str
    selection_meta: Optional[dict[str, Any]] = None
    probe_time_s: float = 0.0
    n_probe_tokens: int = 0


class TwoAgentInventory:
    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        latent_steps: int = 40,
        kv_budget: int = 32,
        sink_size: int = 4,
        pca_rank: int = 2,
        max_new_tokens: int = 48,
    ) -> None:
        _ensure_upstream_path()
        from models import ModelWrapper

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.wrapper = ModelWrapper(model_name, self.device, args=_WrapperArgs())
        self.latent_steps = latent_steps
        self.kv_budget = kv_budget
        self.sink_size = sink_size
        self.pca_rank = pca_rank
        self.max_new_tokens = max_new_tokens

    def loaded_revision(self) -> Optional[str]:
        cfg = getattr(self.wrapper.model, "config", None)
        commit = getattr(cfg, "_commit_hash", None) if cfg is not None else None
        return commit if isinstance(commit, str) and commit else None

    def _encode(self, text: str) -> tuple[str, torch.Tensor, torch.Tensor]:
        messages = [{"role": "user", "content": text}]
        rendered, input_ids, attention_mask, _tokens = self.wrapper.prepare_chat_input(
            messages, add_generation_prompt=True
        )
        return rendered, input_ids, attention_mask

    def run_sender(
        self,
        inventory_text: str,
        ex: InventoryExample,
        source: str,
        *,
        include_question: bool = True,
    ) -> SenderBundle:
        prompt = sender_prompt(inventory_text, ex) if include_question else sender_prompt_inventory_only(inventory_text)
        rendered, input_ids, attention_mask = self._encode(prompt)
        pos_cursor = torch.zeros((1,), device=self.device, dtype=torch.long)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        past, attentions, new_pos, past_mask = self.wrapper.generate_latent_batch(
            input_ids,
            attention_mask=attention_mask,
            pos_cursor=pos_cursor,
            latent_steps=self.latent_steps,
            past_key_values=None,
            past_attention_mask=None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        bundle = SenderBundle(
            source=source,
            prompt=prompt,
            rendered=rendered,
            input_ids_cpu=input_ids.detach().cpu().clone(),
            prompt_mask_cpu=attention_mask.detach().cpu().clone(),
            pos_cursor_cpu=new_pos.detach().cpu().clone(),
            past_mask_cpu=past_mask.detach().cpu().clone(),
            cache_cpu=cache_to_cpu(past),
            attentions_cpu=attentions_to_cpu(attentions),
            latent_steps=self.latent_steps,
            sender_latency_s=latency,
            full_relay_bytes=kv_size_bytes(past),
        )
        del past, attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bundle

    def probe_text_attentions(self, sender: SenderBundle, text: str) -> tuple[list[list[torch.Tensor]], float, int]:
        """One forward pass of `text` on A's full cache. Discard the new KV entries."""
        from src.probe import probe_attentions_to_steps

        encoded = self.wrapper.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        n_probe = int(attention_mask.sum().item())
        working = cache_to_device(sender.cache_cpu, self.device)
        past_mask = sender.past_mask_cpu.to(self.device)
        pos_cursor = sender.pos_cursor_cpu.to(self.device)
        prompt_len = int(sender.prompt_mask_cpu.shape[-1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        past_len = int(working.key_cache[0].shape[-2]) if hasattr(working, "key_cache") else int(working[0][0].shape[-2])
        if past_mask.size(1) != past_len:
            past_mask = torch.ones((1, past_len), dtype=torch.long, device=self.device)
        full_mask = torch.cat([past_mask, attention_mask.to(dtype=torch.long)], dim=-1)
        position_ids = self.wrapper._build_position_ids(pos_cursor, attention_mask)
        out = self.wrapper.model(
            input_ids=input_ids,
            attention_mask=full_mask,
            position_ids=position_ids,
            past_key_values=working,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        if out.attentions is None:
            raise RuntimeError("Probe forward did not return attentions; the model must use eager attention.")
        steps = probe_attentions_to_steps(out.attentions, prompt_len=prompt_len)
        steps_cpu = attentions_to_cpu(steps)
        del out, working, input_ids, attention_mask, full_mask, position_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return steps_cpu, elapsed, n_probe

    def generate_text_message(self, prompt: str, *, max_new_tokens: int) -> tuple[str, int, bool, float]:
        """Plain text generation. No latent rollout and no KV relay."""
        _rendered, input_ids, attention_mask = self._encode(prompt)
        pos_cursor = torch.zeros((1,), device=self.device, dtype=torch.long)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        texts, _past_out, _pos, gen_lens = self.wrapper.generate_text_batch(
            input_ids,
            attention_mask,
            pos_cursor=pos_cursor,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            past_key_values=None,
            past_attention_mask=None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        n_new = int(gen_lens[0]) if gen_lens else 0
        truncated = n_new >= max_new_tokens
        del input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return texts[0], n_new, truncated, latency

    def generate_text_keep_cache(
        self, prompt: str, *, max_new_tokens: int, source: str = "written"
    ) -> tuple[str, int, bool, float, SenderBundle]:
        """Ordinary written generation that keeps the KV of prompt plus new tokens.

        This is the completed calculation state. It is not a latent rollout.
        """
        rendered, input_ids, attention_mask = self._encode(prompt)
        pos_cursor = torch.zeros((1,), device=self.device, dtype=torch.long)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        # generate_text_batch calls the model in a for-loop. It never calls model.generate().
        with _capture_model_input_ids(self.wrapper.model) as recorded_ids:
            texts, past, new_pos, gen_lens = self.wrapper.generate_text_batch(
                input_ids,
                attention_mask,
                pos_cursor=pos_cursor,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                past_key_values=None,
                past_attention_mask=None,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        n_new = int(gen_lens[0]) if gen_lens else 0
        truncated = n_new >= max_new_tokens
        if past is None:
            raise RuntimeError("Written generation returned no KV cache.")
        cache_cpu = cache_to_cpu(past)
        n_kv = n_kv_positions(cache_cpu)
        if n_kv <= 0:
            raise RuntimeError("Written generation produced an empty KV cache.")
        pos_after = int(new_pos.detach().cpu().view(-1)[0].item())
        full_ids, token_align = _ids_from_decode_loop(
            self.wrapper.tokenizer,
            prompt_ids=input_ids[0].detach().cpu(),
            written_text=texts[0] if texts else "",
            n_kv=n_kv,
            n_new=n_new,
            pos_cursor=pos_after,
            recorded_ids=list(recorded_ids),
        )
        token_align["n_new"] = n_new
        token_align["pos_cursor"] = pos_after
        token_align["pos_cursor_eq_n_kv"] = pos_after == int(n_kv)
        bundle = SenderBundle(
            source=source,
            prompt=prompt,
            rendered=rendered,
            input_ids_cpu=input_ids.detach().cpu().clone(),
            prompt_mask_cpu=attention_mask.detach().cpu().clone(),
            pos_cursor_cpu=new_pos.detach().cpu().clone(),
            past_mask_cpu=torch.ones((1, n_kv), dtype=torch.long),
            cache_cpu=cache_cpu,
            attentions_cpu=[],
            latent_steps=0,
            sender_latency_s=latency,
            full_relay_bytes=kv_size_bytes(past),
            full_ids_cpu=full_ids,
            ids_source=str(token_align["ids_source"]),
            token_align=token_align,
        )
        del past, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return texts[0], n_new, truncated, latency, bundle

    def clone_sender(self, sender: SenderBundle) -> SenderBundle:
        """Independent copy of a saved A state. Communication branches from clones."""
        hidden = sender.last_hidden_cpu
        return SenderBundle(
            source=sender.source,
            prompt=sender.prompt,
            rendered=sender.rendered,
            input_ids_cpu=sender.input_ids_cpu.detach().contiguous().clone(),
            prompt_mask_cpu=sender.prompt_mask_cpu.detach().contiguous().clone(),
            pos_cursor_cpu=sender.pos_cursor_cpu.detach().contiguous().clone(),
            past_mask_cpu=sender.past_mask_cpu.detach().contiguous().clone(),
            cache_cpu=clone_cache_blob(sender.cache_cpu),
            attentions_cpu=attentions_to_cpu(sender.attentions_cpu) if sender.attentions_cpu else [],
            latent_steps=sender.latent_steps,
            sender_latency_s=sender.sender_latency_s,
            full_relay_bytes=sender.full_relay_bytes,
            last_hidden_cpu=None if hidden is None else hidden.detach().cpu().contiguous().clone(),
            full_ids_cpu=(
                None
                if sender.full_ids_cpu is None
                else sender.full_ids_cpu.detach().contiguous().clone()
            ),
            ids_source=sender.ids_source,
            token_align=None if sender.token_align is None else dict(sender.token_align),
        )

    def _prefill_prompt(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill only. Matches generate_latent_batch with latent_steps=0."""
        batch = input_ids.size(0)
        pos_cursor = torch.zeros((batch,), device=self.device, dtype=torch.long)
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(torch.long)
        attention_mask = (attention_mask > 0).to(torch.long).to(self.device)
        prompt_pos_ids = self.wrapper._build_position_ids(pos_cursor, attention_mask)
        with torch.inference_mode():
            out = self.wrapper.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=prompt_pos_ids,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            last_idx = self.wrapper._last_valid_indices(attention_mask)
            last_hidden = out.hidden_states[-1][torch.arange(batch, device=self.device), last_idx, :].detach()
            past = out.past_key_values
        valid_len = attention_mask.sum(dim=-1).to(torch.long)
        new_pos = pos_cursor + valid_len
        del out
        return past, last_hidden, new_pos, attention_mask

    def _run_latent_steps(
        self,
        past: Any,
        past_mask: torch.Tensor,
        pos_cursor: torch.Tensor,
        last_hidden: torch.Tensor,
        extra_steps: int,
        *,
        output_attentions: bool = False,
    ) -> tuple[Any, list[list[torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Additional latent tokens on an existing cache. Does not re-read the prompt.

        Attentions are off by default. Calibration and full-KV relay do not need them;
        storing 40 eager attention maps on T4 is enough to OOM.
        """
        batch = last_hidden.size(0)
        device = last_hidden.device
        full_mask = past_mask
        all_steps: list[list[torch.Tensor]] = []
        hidden = last_hidden
        with torch.inference_mode():
            for step in range(int(extra_steps)):
                full_mask = torch.cat(
                    [full_mask, torch.ones((batch, 1), dtype=torch.long, device=device)], dim=-1
                )
                latent_vec = self.wrapper._apply_latent_realignment(hidden, self.wrapper.model)
                latent_embed = latent_vec.unsqueeze(1)
                if hasattr(self.wrapper.model, "dtype"):
                    latent_embed = latent_embed.to(dtype=self.wrapper.model.dtype)
                latent_pos_ids = (pos_cursor + step).unsqueeze(1)
                out = self.wrapper.model(
                    inputs_embeds=latent_embed,
                    attention_mask=full_mask,
                    position_ids=latent_pos_ids,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    return_dict=True,
                )
                past = out.past_key_values
                hidden = out.hidden_states[-1][:, -1, :].detach()
                if output_attentions:
                    if out.attentions is None:
                        raise RuntimeError("Latent step did not return attentions.")
                    all_steps.append([attn[:, :, -1, :].detach() for attn in out.attentions])
                del out
        new_pos = pos_cursor + int(extra_steps)
        return past, all_steps, new_pos, full_mask, hidden

    def encode_prompt(self, prompt: str, *, source: str, latent_steps: int = 0) -> SenderBundle:
        """Save A's state after reading `prompt`. latent_steps=0 is C0."""
        rendered, input_ids, attention_mask = self._encode(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        past, last_hidden, new_pos, past_mask = self._prefill_prompt(input_ids, attention_mask)
        attentions: list[list[torch.Tensor]] = []
        if int(latent_steps) > 0:
            past, attentions, new_pos, past_mask, last_hidden = self._run_latent_steps(
                past, past_mask, new_pos, last_hidden, int(latent_steps)
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        bundle = SenderBundle(
            source=source,
            prompt=prompt,
            rendered=rendered,
            input_ids_cpu=input_ids.detach().cpu().clone(),
            prompt_mask_cpu=attention_mask.detach().cpu().clone(),
            pos_cursor_cpu=new_pos.detach().cpu().clone(),
            past_mask_cpu=past_mask.detach().cpu().clone(),
            cache_cpu=cache_to_cpu(past),
            attentions_cpu=attentions_to_cpu(attentions),
            latent_steps=int(latent_steps),
            sender_latency_s=latency,
            full_relay_bytes=kv_size_bytes(past),
            last_hidden_cpu=last_hidden.detach().cpu().contiguous().clone(),
        )
        del past, last_hidden, attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bundle

    def continue_latent(self, sender: SenderBundle, extra_steps: int) -> SenderBundle:
        """C40: 40 additional latent steps on an independent copy of C0."""
        extra = int(extra_steps)
        if extra < 0:
            raise ValueError("extra_steps must be >= 0")
        working = self.clone_sender(sender)
        if extra == 0:
            return working
        if working.last_hidden_cpu is None:
            raise RuntimeError("continue_latent needs last_hidden from encode_prompt.")
        past = cache_to_device(clone_cache_blob(working.cache_cpu), self.device)
        past_mask = working.past_mask_cpu.to(self.device)
        pos_cursor = working.pos_cursor_cpu.to(self.device)
        last_hidden = working.last_hidden_cpu.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        past, attentions, new_pos, new_mask, last_hidden = self._run_latent_steps(
            past, past_mask, pos_cursor, last_hidden, extra
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        out = SenderBundle(
            source=working.source,
            prompt=working.prompt,
            rendered=working.rendered,
            input_ids_cpu=working.input_ids_cpu,
            prompt_mask_cpu=working.prompt_mask_cpu,
            pos_cursor_cpu=new_pos.detach().cpu().clone(),
            past_mask_cpu=new_mask.detach().cpu().clone(),
            cache_cpu=cache_to_cpu(past),
            attentions_cpu=attentions_to_cpu(attentions),
            latent_steps=working.latent_steps + extra,
            sender_latency_s=latency,
            full_relay_bytes=kv_size_bytes(past),
            last_hidden_cpu=last_hidden.detach().cpu().contiguous().clone(),
        )
        del past, last_hidden, attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def render_chat(self, prompt: str) -> tuple[str, int]:
        """Chat-template rendering only. Same wrapping generate_from_cache applies to a handoff."""
        rendered, input_ids, _mask = self._encode(prompt)
        return rendered, int(input_ids.shape[-1])

    def compare_identical_token_prefix(
        self, prompt: str, *, split: float | int, max_new_tokens: int
    ) -> dict:
        """Ordinary vs cached continuation on one token sequence.

        Tokenize `prompt` once. Ordinary generation consumes the full ids.
        Cached generation prefills a prefix of those same ids, then continues
        with the remaining ids. This is not generate_from_cache, which wraps
        the handoff as a new chat turn.
        """
        rendered, input_ids, attention_mask = self._encode(prompt)
        length = int(input_ids.shape[-1])
        if isinstance(split, float):
            cut = max(1, min(length - 1, int(round(length * float(split)))))
        else:
            cut = max(1, min(length - 1, int(split)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        ord_texts, _past_a, _pos_a, ord_lens = self.wrapper.generate_text_batch(
            input_ids,
            attention_mask,
            pos_cursor=torch.zeros((1,), device=self.device, dtype=torch.long),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            past_key_values=None,
            past_attention_mask=None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ordinary_s = time.time() - t0
        prefix_ids = input_ids[:, :cut]
        prefix_mask = attention_mask[:, :cut]
        suffix_ids = input_ids[:, cut:]
        suffix_mask = attention_mask[:, cut:]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        past, _hidden, new_pos, past_mask = self._prefill_prompt(prefix_ids, prefix_mask)
        cached_texts, _past_b, _pos_b, cached_lens = self.wrapper.generate_text_batch(
            suffix_ids,
            suffix_mask,
            pos_cursor=new_pos,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            past_key_values=past,
            past_attention_mask=past_mask,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cached_s = time.time() - t1
        ordinary_text = ord_texts[0] if ord_texts else ""
        cached_text = cached_texts[0] if cached_texts else ""
        n_ord = int(ord_lens[0]) if ord_lens else 0
        n_cached = int(cached_lens[0]) if cached_lens else 0
        del past, input_ids, attention_mask, prefix_ids, suffix_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "rendered": rendered,
            "n_prompt_tokens": length,
            "split_at": cut,
            "ordinary_text": ordinary_text,
            "cached_text": cached_text,
            "texts_equal": (ordinary_text or "").strip() == (cached_text or "").strip(),
            "ordinary_n_new": n_ord,
            "cached_n_new": n_cached,
            "ordinary_truncated": n_ord >= max_new_tokens,
            "cached_truncated": n_cached >= max_new_tokens,
            "ordinary_s": ordinary_s,
            "cached_s": cached_s,
            "handoff_is_new_chat": False,
        }

    def generate_from_cache(
        self, sender: SenderBundle, prompt: str, *, max_new_tokens: int
    ) -> tuple[str, int, bool, float]:
        """Generate from an independent copy of a saved cache. Does not mutate `sender`.

        The continuation `prompt` is wrapped as a new chat user turn
        (`_encode` / apply_chat_template with add_generation_prompt=True), then
        prepended with `sender`'s KV. That is not the same token sequence as
        concatenating the original user text with the handoff in one turn.
        """
        _rendered, input_ids, attention_mask = self._encode(prompt)
        past = cache_to_device(clone_cache_blob(sender.cache_cpu), self.device)
        past_mask = sender.past_mask_cpu.to(self.device)
        pos_cursor = sender.pos_cursor_cpu.to(self.device)
        expected = n_kv_positions(sender.cache_cpu)
        if past_mask.size(1) != expected:
            past_mask = torch.ones((1, expected), dtype=torch.long, device=self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        texts, _past_out, _pos, gen_lens = self.wrapper.generate_text_batch(
            input_ids,
            attention_mask,
            pos_cursor=pos_cursor,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            past_key_values=past,
            past_attention_mask=past_mask,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0
        n_new = int(gen_lens[0]) if gen_lens else 0
        truncated = n_new >= max_new_tokens
        del past, past_mask, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return texts[0], n_new, truncated, latency

    def last_query_attentions(self, sender: SenderBundle) -> tuple[list[list[torch.Tensor]], float]:
        """One dummy-token forward on a clone of `sender`. Keep last-query attention over existing keys."""
        working = cache_to_device(clone_cache_blob(sender.cache_cpu), self.device)
        past_len = n_kv_positions(sender.cache_cpu)
        past_mask = torch.ones((1, past_len), dtype=torch.long, device=self.device)
        eos = self.wrapper.tokenizer.eos_token_id
        dummy_id = int(eos) if eos is not None else 0
        dummy = torch.tensor([[dummy_id]], device=self.device, dtype=torch.long)
        attn_mask = torch.ones((1, 1), dtype=torch.long, device=self.device)
        full_mask = torch.cat([past_mask, attn_mask], dim=-1)
        position_ids = sender.pos_cursor_cpu.to(self.device).unsqueeze(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = self.wrapper.model(
                input_ids=dummy,
                attention_mask=full_mask,
                position_ids=position_ids,
                past_key_values=working,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        if out.attentions is None:
            raise RuntimeError("Rank forward did not return attentions; the model must use eager attention.")
        step = []
        for attn in out.attentions:
            klen = min(past_len, int(attn.shape[-1]))
            step.append(attn[:, :, -1, :klen].detach())
        steps_cpu = attentions_to_cpu([step])
        del out, working, dummy, attn_mask, full_mask, position_ids, past_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return steps_cpu, elapsed

    def probe_completed_cache_attentions(
        self, sender: SenderBundle, text: str
    ) -> tuple[list[list[torch.Tensor]], float, int]:
        """One chat-wrapped probe forward on the completed cache.

        Scores prompt plus written-calculation keys. Discards newly appended
        probe keys. Does not use probe_attentions_to_steps(prompt_len=original
        prompt), which would drop the written tokens.
        """
        from src.probe import cache_keys_attentions_to_steps

        _rendered, input_ids, attention_mask = self._encode(text)
        n_probe = int(attention_mask.sum().item())
        working = cache_to_device(clone_cache_blob(sender.cache_cpu), self.device)
        cache_len = n_kv_positions(sender.cache_cpu)
        past_mask = torch.ones((1, cache_len), dtype=torch.long, device=self.device)
        full_mask = torch.cat([past_mask, attention_mask.to(dtype=torch.long)], dim=-1)
        position_ids = self.wrapper._build_position_ids(sender.pos_cursor_cpu.to(self.device), attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = self.wrapper.model(
                input_ids=input_ids,
                attention_mask=full_mask,
                position_ids=position_ids,
                past_key_values=working,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        if out.attentions is None:
            raise RuntimeError("Probe forward did not return attentions; the model must use eager attention.")
        steps = cache_keys_attentions_to_steps(out.attentions, cache_len=cache_len)
        if int(steps[0][0].shape[-1]) != cache_len:
            raise RuntimeError("Completed-cache probe did not keep the full cache key length.")
        steps_cpu = attentions_to_cpu(steps)
        del out, working, input_ids, attention_mask, full_mask, position_ids, past_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return steps_cpu, elapsed, n_probe

    def compress_sender(
        self,
        sender: SenderBundle,
        *,
        kv_budget: int,
        attentions: list[list[torch.Tensor]],
        sink_size: int = 4,
        relay: str = "headwise",
    ) -> tuple[SenderBundle, float, float, dict[str, Any]]:
        """Evict over the entire completed cache. Does not mutate `sender`.

        Budget `kv_budget` is selected positions per KV head and layer,
        plus `sink_size` initial positions. There is no reserved latent tail: this
        cache has no 40-step rollout, so the budget applies to prompt plus written
        tokens together. Report actual kept length and bytes, not the nominal budget.
        `hobf_fast` uses the same top-k rule on these scores, then injects residual
        from discarded positions. That is an adaptation to whole-cache selection,
        not a replication of OBF's protected-reasoning (latent-tail) setup.
        """
        if relay not in {"headwise", "hobf_fast"}:
            raise ValueError(f"compress_sender relay must be headwise or hobf_fast, got {relay}")
        compressor = make_compressor(
            relay, kv_budget=int(kv_budget), sink_size=int(sink_size), pca_rank=self.pca_rank
        )
        compressor.reset()
        working = cache_to_device(clone_cache_blob(sender.cache_cpu), self.device)
        n_full = n_kv_positions(sender.cache_cpu)
        prompt_mask = torch.ones((1, n_full), dtype=torch.long, device=self.device)
        attn_dev = attentions_to_device(attentions, self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        compressed, core_time, meta = compressor.compress(
            past_key_values=working,
            latent_steps=0,
            all_steps_attentions=attn_dev,
            prompt_mask=prompt_mask,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.time() - t0
        n_kept = int(compressed.key_cache[0].shape[-2]) if hasattr(compressed, "key_cache") else int(compressed[0][0].shape[-2])
        n_prompt = int(sender.prompt_mask_cpu.sum().item())
        info = {
            "kv_budget": int(kv_budget),
            "sink_size": int(sink_size),
            "budget_scope": (
                "entire completed cache (prompt + written tokens); "
                "sink_size initial positions plus kv_budget selected positions per head; "
                "no reserved latent tail"
            ),
            "n_full": n_full,
            "n_kept": n_kept,
            "n_prompt": n_prompt,
            "n_written_tail": max(0, n_full - n_prompt),
            "n_written_selected": _n_selected_from(meta, start=n_prompt),
            "cache_finite": cache_is_finite(compressed),
            "obf_finite": metadata_is_finite(meta),
            "relay": relay,
        }
        bundle = SenderBundle(
            source=sender.source,
            prompt=sender.prompt,
            rendered=sender.rendered,
            input_ids_cpu=sender.input_ids_cpu,
            prompt_mask_cpu=sender.prompt_mask_cpu,
            pos_cursor_cpu=sender.pos_cursor_cpu.detach().contiguous().clone(),
            past_mask_cpu=torch.ones((1, n_kept), dtype=torch.long),
            cache_cpu=cache_to_cpu(compressed),
            attentions_cpu=[],
            latent_steps=0,
            sender_latency_s=sender.sender_latency_s,
            full_relay_bytes=kv_size_bytes(compressed),
            last_hidden_cpu=None,
        )
        del working, compressed, attn_dev, prompt_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return bundle, wall, float(core_time) if core_time is not None else wall, info

    def keep_shared_positions(
        self, sender: SenderBundle, positions: list[int]
    ) -> tuple[SenderBundle, float, dict[str, Any]]:
        """Keep one shared position set on every head. Does not mutate `sender`."""
        from src.span_select import caches_equal, gather_cache_positions

        n_full = n_kv_positions(sender.cache_cpu)
        ordered = sorted({int(p) for p in positions if 0 <= int(p) < n_full})
        if not ordered:
            raise RuntimeError("Shared selection is empty.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        gathered = gather_cache_positions(sender.cache_cpu, ordered)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.time() - t0
        n_kept = len(ordered)
        pos_cursor = int(sender.pos_cursor_cpu.detach().cpu().view(-1)[0].item())
        identity = ordered == list(range(n_full))
        info = {
            "n_full": n_full,
            "n_kept": n_kept,
            "shared_across_heads": True,
            "positions": ordered,
            "pos_cursor": pos_cursor,
            "past_mask_len": n_kept,
            "pos_cursor_eq_n_full": pos_cursor == n_full,
            "identity_gather": identity,
            "gathered_equals_source": caches_equal(sender.cache_cpu, gathered) if identity else None,
            "cache_finite": all(
                bool(torch.isfinite(key).all().item()) and bool(torch.isfinite(value).all().item())
                for key, value in gathered["layers"]
            ),
        }
        bundle = SenderBundle(
            source=sender.source,
            prompt=sender.prompt,
            rendered=sender.rendered,
            input_ids_cpu=sender.input_ids_cpu,
            prompt_mask_cpu=sender.prompt_mask_cpu,
            pos_cursor_cpu=sender.pos_cursor_cpu.detach().contiguous().clone(),
            past_mask_cpu=torch.ones((1, n_kept), dtype=torch.long),
            cache_cpu=gathered,
            attentions_cpu=[],
            latent_steps=0,
            sender_latency_s=sender.sender_latency_s,
            full_relay_bytes=kv_blob_size_bytes(gathered),
            last_hidden_cpu=None,
            full_ids_cpu=(
                None if sender.full_ids_cpu is None else sender.full_ids_cpu.detach().contiguous().clone()
            ),
            ids_source=sender.ids_source,
            token_align=None if sender.token_align is None else dict(sender.token_align),
        )
        return bundle, wall, info

    def probe_question_attentions(self, sender: SenderBundle, ex: InventoryExample) -> tuple[list[list[torch.Tensor]], float, int]:
        """Frozen selector: B's actual question and choices. Does not generate an answer."""
        return self.probe_text_attentions(sender, probe_prompt(ex))

    def run_receiver(
        self,
        ex: InventoryExample,
        *,
        condition: str,
        relay: str,
        sender: Optional[SenderBundle],
        inventory_text: Optional[str],
        filler_text: Optional[str],
        kv_budget: Optional[int] = None,
        selection_mode: Optional[str] = None,
        force_prompt_idx: Optional[list[int]] = None,
        rng_seed: int = 2026,
        attention_override: Optional[list[list[torch.Tensor]]] = None,
        probe_time_s: float = 0.0,
        n_probe_tokens: int = 0,
        retrieved_sentence: Optional[str] = None,
        text_source: str = "iterative",
    ) -> ReceiverResult:
        if retrieved_sentence is not None:
            if sender is not None:
                raise ValueError("Text baseline must not include a sender KV cache.")
            prompt = text_message_prompt(ex, retrieved_sentence, source=text_source)
            leaked = False
        else:
            has_relay = sender is not None
            prompt = receiver_prompt(
                ex,
                inventory_text=inventory_text,
                filler_text=filler_text,
                has_relay=has_relay,
            )
            leaked = (inventory_text is None) and (not true_inventory_absent_from_receiver_prompt(prompt, ex))
        _rendered, input_ids, attention_mask = self._encode(prompt)

        past = None
        past_mask = None
        pos_cursor = torch.zeros((1,), device=self.device, dtype=torch.long)
        compression_time_s = 0.0
        compression_core_s = 0.0
        relay_bytes = 0
        cache_finite = True
        obf_finite = True

        selection_meta: Optional[dict[str, Any]] = None
        if sender is not None:
            budget = self.kv_budget if kv_budget is None else kv_budget
            if selection_mode is not None:
                from src.prompt_select import PromptSelectCompressor

                compressor = PromptSelectCompressor(
                    sink_size=self.sink_size,
                    kv_budget=budget,
                    mode=selection_mode,
                    force_prompt_idx=force_prompt_idx,
                    rng_seed=rng_seed,
                )
            else:
                compressor = make_compressor(
                    relay,
                    kv_budget=budget,
                    sink_size=self.sink_size,
                    pca_rank=self.pca_rank,
                )
            compressor.reset()
            working = cache_to_device(sender.cache_cpu, self.device)
            attn_src = sender.attentions_cpu if attention_override is None else attention_override
            attentions = attentions_to_device(attn_src, self.device)
            prompt_mask = sender.prompt_mask_cpu.to(self.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_comp = time.time()
            compressed, core_time, meta = compressor.compress(
                past_key_values=working,
                latent_steps=sender.latent_steps,
                all_steps_attentions=attentions,
                prompt_mask=prompt_mask,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compression_wall = time.time() - t_comp
            # Wall time of compress(); core_time is the operator when the
            # compressor reports it. Keep these separate from receiver decode.
            compression_time_s = compression_wall
            compression_core_s = float(core_time) if core_time is not None else compression_wall
            past = compressed
            relay_bytes = kv_size_bytes(past)
            cache_finite = cache_is_finite(past)
            obf_finite = metadata_is_finite(meta)
            if isinstance(meta, dict):
                selection_meta = meta
            cur_len = past.key_cache[0].shape[-2] if hasattr(past, "key_cache") else past[0][0].shape[-2]
            if sender.past_mask_cpu.size(1) == cur_len:
                past_mask = sender.past_mask_cpu.to(self.device)
            else:
                past_mask = None
            pos_cursor = sender.pos_cursor_cpu.to(self.device)
            del working, attentions, compressed

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        texts, _past_out, _pos, gen_lens = self.wrapper.generate_text_batch(
            input_ids,
            attention_mask,
            pos_cursor=pos_cursor,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            past_key_values=past,
            past_attention_mask=past_mask,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        receiver_latency_s = time.time() - t0
        n_new = int(gen_lens[0]) if gen_lens else 0
        truncated = n_new >= self.max_new_tokens
        n_output_lines = len([ln for ln in (texts[0] or "").splitlines() if ln.strip()])
        parsed, parsed_legacy = label_both(
            texts[0],
            gold_letter=ex.gold_letter,
            gold_locker=ex.true_locker,
            donor_letter=ex.donor_letter,
            donor_locker=ex.alt_locker,
        )
        del past, past_mask, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ReceiverResult(
            condition=condition,
            relay=relay,
            raw_text=texts[0],
            parsed=parsed,
            parsed_legacy=parsed_legacy,
            truncated=truncated,
            n_new_tokens=n_new,
            n_output_lines=n_output_lines,
            max_new_tokens=self.max_new_tokens,
            relay_bytes=relay_bytes,
            compression_time_s=compression_time_s,
            compression_core_s=compression_core_s,
            receiver_latency_s=receiver_latency_s,
            sender_latency_s=sender.sender_latency_s if sender is not None else 0.0,
            peak_gpu_gb=peak_gpu_memory_gb(),
            cache_finite=cache_finite,
            obf_finite=obf_finite,
            leaked_true_inventory=leaked,
            prompt=prompt,
            selection_meta=selection_meta,
            probe_time_s=probe_time_s,
            n_probe_tokens=n_probe_tokens,
        )
