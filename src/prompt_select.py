"""Headwise-layout prompt selection with attention, random, or forced indices.

Sink tokens and latent reasoning states stay identical to the released
Headwise compressor. Forced evidence tokens occupy part of kv_budget; they
are not extra slots.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

MODES = ("attention", "random", "force", "recency")


def head_rng_seed(base: int, layer_idx: int, head: int) -> int:
    return int((int(base) * 1_000_003 + layer_idx * 1_009 + head * 17) % (2**31 - 1))


def select_prompt_indices(
    scores: torch.Tensor,
    *,
    k_eff: int,
    sink_len: int,
    valid_len: int,
    mode: str,
    force: Optional[Sequence[int]] = None,
    rng_seed: int = 2026,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Pick k_eff prompt indices per KV head in prompt coordinates.

    scores: (H, L_prompt). Sink region [0, sink_len) is never selected here
    because those tokens are kept separately. Returned tensor: (H, k_eff),
    sorted along the last dim.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown selection mode {mode}")
    h_kv = int(scores.shape[0])
    device = scores.device
    if k_eff <= 0 or valid_len <= sink_len:
        return torch.zeros((h_kv, 0), device=device, dtype=torch.long)

    eligible_len = valid_len - sink_len
    kk = min(int(k_eff), eligible_len)
    forced = []
    if mode == "force":
        forced = sorted({int(i) for i in (force or []) if sink_len <= int(i) < valid_len})
        if len(forced) > kk:
            raise ValueError(
                f"Forced evidence needs {len(forced)} budget slots but k_eff={kk}. "
                "The complete line must fit inside the existing budget."
            )

    out = torch.empty((h_kv, kk), device=device, dtype=torch.long)
    eligible = torch.arange(sink_len, valid_len, device="cpu", dtype=torch.long)
    for head in range(h_kv):
        if mode == "random":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(head_rng_seed(rng_seed, layer_idx, head))
            perm = torch.randperm(eligible_len, generator=gen)[:kk]
            chosen = eligible[perm]
        elif mode == "attention":
            seg = scores[head, sink_len:valid_len]
            chosen = torch.topk(seg, k=kk, dim=-1).indices.to("cpu") + sink_len
        elif mode == "recency":
            chosen = eligible[-kk:]
        else:
            n_fill = kk - len(forced)
            forced_t = torch.tensor(forced, dtype=torch.long)
            if n_fill <= 0:
                chosen = forced_t
            else:
                seg = scores[head, sink_len:valid_len].detach().to("cpu").clone()
                for idx in forced:
                    seg[idx - sink_len] = -float("inf")
                fill = torch.topk(seg, k=n_fill, dim=-1).indices + sink_len
                chosen = torch.cat([forced_t, fill], dim=0)
        chosen, _ = torch.sort(chosen.to(dtype=torch.long))
        out[head] = chosen.to(device=device)
    return out


class PromptSelectCompressor:
    """Same KV layout as released Headwise; only the prompt-index rule changes."""

    def __init__(
        self,
        sink_size: int = 4,
        kv_budget: int = 32,
        *,
        mode: str = "attention",
        force_prompt_idx: Optional[Sequence[int]] = None,
        rng_seed: int = 2026,
    ):
        self.sink_size = int(sink_size)
        self.kv_budget = int(kv_budget)
        self._has_kept_sink = False
        if mode not in MODES:
            raise ValueError(f"Unknown selection mode {mode}")
        self.mode = mode
        self.force_prompt_idx = [int(i) for i in (force_prompt_idx or [])]
        self.rng_seed = int(rng_seed)

    def reset(self) -> None:
        self._has_kept_sink = False

    @staticmethod
    def _past_length(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "key_cache"):
            return int(past_key_values.key_cache[0].shape[-2]) if len(past_key_values.key_cache) > 0 else 0
        return int(past_key_values[0][0].shape[-2]) if len(past_key_values) > 0 else 0

    @staticmethod
    def _as_legacy_tuple(past_key_values: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        if past_key_values is None:
            return tuple()
        if hasattr(past_key_values, "key_cache"):
            return tuple((k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache))
        return past_key_values

    def _prompt_scores(
        self,
        *,
        all_steps_attentions: List[List[torch.Tensor]],
        layer_idx: int,
        B: int,
        H_kv: int,
        L_prompt: int,
        L_history: int,
        L_total: int,
        device: torch.device,
    ) -> torch.Tensor:
        if len(all_steps_attentions) == 0:
            return torch.zeros((B, H_kv, L_prompt), device=device, dtype=torch.float32)
        a0 = all_steps_attentions[0][layer_idx]
        H_q = int(a0.shape[1])
        max_klen = 0
        for step_data in all_steps_attentions:
            a = step_data[layer_idx]
            max_klen = max(max_klen, int(a.shape[-1]))
        agg = torch.zeros((B, H_q, max_klen), device=device, dtype=torch.float32)
        for step_data in all_steps_attentions:
            a = step_data[layer_idx].to(device=device, dtype=torch.float32, non_blocking=True)
            if int(a.shape[-1]) < max_klen:
                a = F.pad(a, (0, max_klen - int(a.shape[-1])))
            agg += a
        if max_klen < L_total:
            agg_full = F.pad(agg, (0, L_total - max_klen))
        else:
            agg_full = agg[:, :, :L_total]
        prompt_scores_q = agg_full[:, :, L_history : L_history + L_prompt]
        if H_q == H_kv:
            return prompt_scores_q
        if (H_q % H_kv) == 0:
            group = H_q // H_kv
            return prompt_scores_q.view(B, H_kv, group, L_prompt).sum(dim=2)
        if prompt_scores_q.shape[1] >= H_kv:
            return prompt_scores_q[:, :H_kv, :]
        return F.pad(prompt_scores_q, (0, 0, 0, H_kv - prompt_scores_q.shape[1]))

    @torch.no_grad()
    def compress(
        self,
        *,
        past_key_values: Any,
        latent_steps: int,
        all_steps_attentions: List[List[torch.Tensor]],
        prompt_mask: torch.Tensor,
        current_full_mask: Optional[Any] = None,
        debug: bool = False,
        **kwargs,
    ) -> Tuple[Any, float, Any]:
        t0 = time.time()
        del current_full_mask
        if past_key_values is None:
            return None, 0.0, None
        layers = self._as_legacy_tuple(past_key_values)
        if len(layers) == 0:
            return past_key_values, 0.0, None

        k0, _ = layers[0]
        B = int(k0.shape[0])
        num_layers = len(layers)
        H_kv = int(k0.shape[1])
        if prompt_mask.dim() != 2:
            raise ValueError(f"prompt_mask must be (B, L_prompt), got {prompt_mask.shape}")
        L_prompt = int(prompt_mask.shape[1])
        steps = max(0, int(latent_steps) if latent_steps is not None else 0)
        L_total = self._past_length(past_key_values)
        if steps > L_total:
            steps = L_total
        L_latent = steps
        L_history = L_total - L_prompt - L_latent
        if L_history < 0:
            raise ValueError(f"Invalid layout: L_total={L_total}, L_prompt={L_prompt}, L_latent={L_latent}")

        apply_sink = (not getattr(self, "_has_kept_sink", False)) and (self.sink_size > 0)
        new_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        selected_position_matrices: List[List[List[List[int]]]] = []
        sink_len_common = 0
        k_eff = 0

        for layer_idx in range(num_layers):
            k, v = layers[layer_idx]
            device = k.device
            D = int(k.shape[-1])
            pm = (prompt_mask > 0).to(dtype=torch.long, device=device)
            valid_prompt_len = pm.sum(dim=-1).to(torch.long)
            if apply_sink:
                min_valid = int(valid_prompt_len.min().item())
                sink_len_common = min(int(self.sink_size), min_valid)
            else:
                sink_len_common = 0
            available = (valid_prompt_len - sink_len_common).clamp(min=0)
            k_eff = min(int(self.kv_budget), int(available.min().item()))
            k_eff = max(0, k_eff)

            prompt_scores_kv = self._prompt_scores(
                all_steps_attentions=all_steps_attentions,
                layer_idx=layer_idx,
                B=B,
                H_kv=H_kv,
                L_prompt=L_prompt,
                L_history=L_history,
                L_total=L_total,
                device=device,
            )
            prompt_scores_kv = prompt_scores_kv.masked_fill(pm.unsqueeze(1) == 0, -float("inf"))
            if sink_len_common > 0:
                prompt_scores_kv[:, :, :sink_len_common] = -float("inf")

            selected_prompt_idx: List[torch.Tensor] = []
            for b in range(B):
                vp = int(valid_prompt_len[b].item())
                selected_prompt_idx.append(
                    select_prompt_indices(
                        prompt_scores_kv[b],
                        k_eff=k_eff,
                        sink_len=sink_len_common,
                        valid_len=vp,
                        mode=self.mode,
                        force=self.force_prompt_idx,
                        rng_seed=self.rng_seed,
                        layer_idx=layer_idx,
                    )
                )

            L_new = L_history + sink_len_common + k_eff + L_latent
            idx = torch.empty((B, H_kv, L_new), device=device, dtype=torch.long)
            if L_history > 0:
                hist = torch.arange(0, L_history, device=device, dtype=torch.long)
                idx[:, :, :L_history] = hist.view(1, 1, -1).expand(B, H_kv, -1)
            if sink_len_common > 0:
                sink = torch.arange(0, sink_len_common, device=device, dtype=torch.long) + L_history
                s0 = L_history
                s1 = L_history + sink_len_common
                idx[:, :, s0:s1] = sink.view(1, 1, -1).expand(B, H_kv, -1)
            if k_eff > 0:
                p0 = L_history + sink_len_common
                p1 = p0 + k_eff
                for b in range(B):
                    topk = selected_prompt_idx[b]
                    if topk.numel() == 0:
                        idx[b, :, p0:p1] = L_history + sink_len_common
                    else:
                        idx[b, :, p0:p1] = topk + L_history
            if L_latent > 0:
                lat = torch.arange(L_total - L_latent, L_total, device=device, dtype=torch.long)
                l0 = L_new - L_latent
                idx[:, :, l0:] = lat.view(1, 1, -1).expand(B, H_kv, -1)

            idx_exp = idx.unsqueeze(-1).expand(B, H_kv, L_new, D)
            new_k = torch.gather(k, dim=2, index=idx_exp)
            new_v = torch.gather(v, dim=2, index=idx_exp)
            layer_selection: List[List[List[int]]] = []
            for b in range(B):
                layer_selection.append(
                    [
                        [int(x) for x in selected_prompt_idx[b][h].detach().cpu().tolist()]
                        for h in range(H_kv)
                    ]
                )
            selected_position_matrices.append(layer_selection)
            new_layers.append((new_k, new_v))
            if debug and layer_idx == 0:
                print(
                    f"[prompt-select {self.mode}] sink={sink_len_common}, k_eff={k_eff}, "
                    f"L_history={L_history}, L_prompt={L_prompt}, L_latent={L_latent}, L_new={L_new}"
                )

        if apply_sink:
            self._has_kept_sink = True
        n_forced_budget = len([i for i in self.force_prompt_idx if i >= sink_len_common])
        selection_metadata = {
            "mode": self.mode,
            "sink_len": sink_len_common,
            "k_eff": k_eff,
            "n_forced": len(self.force_prompt_idx),
            "n_forced_in_budget": n_forced_budget,
            "force_prompt_idx": list(self.force_prompt_idx),
            "rng_seed": self.rng_seed,
            "l_history": L_history,
            "l_latent": L_latent,
            "selected_prompt_positions_matrix": [
                [selected_position_matrices[layer_idx][b] for layer_idx in range(num_layers)]
                for b in range(B)
            ],
        }
        from transformers import DynamicCache

        new_cache = DynamicCache.from_legacy_cache(tuple(new_layers))
        return new_cache, time.time() - t0, selection_metadata
