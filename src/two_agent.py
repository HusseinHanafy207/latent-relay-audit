"""Two-agent sequential relay: one frozen model, reused sender caches."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from src.cache_utils import (
    attentions_to_cpu,
    attentions_to_device,
    cache_is_finite,
    cache_to_cpu,
    cache_to_device,
    kv_size_bytes,
    metadata_is_finite,
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
