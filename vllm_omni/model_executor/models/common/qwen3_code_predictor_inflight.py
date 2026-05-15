"""Qwen3 Code Predictor — in-flight token-flat variant.

This is the next iteration past ``qwen3_code_predictor_kv_graph``.  Instead
of lock-stepping all requests through the 14 AR decode steps in one call,
this variant takes a token-flat input ``[N, H]`` where each row is one
sub-talker step for one request — possibly a *different* step than the
neighbouring rows.  Each row carries:

* a ``slot_idx`` (which persistent K/V cache slot it lives in)
* a ``cache_len`` (how many positions of that slot are already populated)
* a ``write_pos`` (cache position to write the new K/V into, == cache_len)
* a ``position_id`` (RoPE position == cache_len)

Cache is laid out as
``[max_slots, num_kv_heads, max_seq=Q+1, head_dim]``.  Each slot belongs to
one request for the duration of one sub-talker call (Q steps).  When a
request finishes its Q steps the slot is recycled.

Attention is computed per-token by gathering the row's cache slot into a
``[N, num_kv_heads, max_seq, head_dim]`` view, then SDPA with a bool
``attn_mask`` ``[N, 1, 1, max_seq]`` selecting the valid prefix (``slot[:cache_len+1]``).

The whole forward is CUDA-graph friendly: shapes are bucketed by ``N``;
slot_indices / write_positions / cache_lens / attn_mask are graph inputs
mutated host-side between replays.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorMLP,
    _RMSNorm,
    _RotaryEmbedding,
    _rotate_half,
)

# Decode: bool mask + GQA — FLASH doesn't accept that combo in PyTorch 2.11.
_SDPA_BACKENDS = [
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


@dataclass
class InflightCacheState:
    """Persistent buffers + graph-parameter tensors used by the captured graph.

    All tensors here are sized for the maximum bucket ``N`` (== ``max_slots``)
    and mutated in place between replays by the scheduler.
    """

    # Per-layer K/V cache: [max_slots, num_kv_heads, max_seq=Q+1, head_dim]
    k_caches: list[torch.Tensor] = field(default_factory=list)
    v_caches: list[torch.Tensor] = field(default_factory=list)
    # Per-token graph parameters
    slot_indices: torch.Tensor | None = None  # [N] long  — cache slot to read/write
    write_positions: torch.Tensor | None = None  # [N] long — position in the slot to write
    position_ids: torch.Tensor | None = None  # [N] long — RoPE position
    attn_mask: torch.Tensor | None = None  # [N, 1, 1, max_seq] bool
    max_slots: int = 0
    max_seq: int = 0

    def reset_slot(self, slot_id: int) -> None:
        """Zero a slot's K/V (called when a slot is recycled for a new request)."""
        for k, v in zip(self.k_caches, self.v_caches):
            k[slot_id].zero_()
            v[slot_id].zero_()

    def set_for_step(
        self,
        slot_indices: list[int],
        cache_lens: list[int],
    ) -> None:
        """Host-side update before a graph replay.

        - ``slot_indices[i]`` = cache slot row 행에 token i가 속한 slot.
        - ``cache_lens[i]``  = 그 slot에 이미 채워진 valid 위치 수.
            * 새 token은 ``write_pos = cache_lens[i]`` 에 쓰이고,
              valid attend 범위는 ``slot[:cache_lens[i]+1]`` 이 된다.
        """
        n = len(slot_indices)
        assert n == self.slot_indices.shape[0], f"expected N={self.slot_indices.shape[0]} tokens"
        si = torch.tensor(slot_indices, dtype=torch.long)
        cl = torch.tensor(cache_lens, dtype=torch.long)
        self.slot_indices.copy_(si.to(self.slot_indices.device), non_blocking=True)
        self.write_positions.copy_(cl.to(self.write_positions.device), non_blocking=True)
        self.position_ids.copy_(cl.to(self.position_ids.device), non_blocking=True)

        # Mask: row i has slot[:cache_lens[i]+1] valid.
        self.attn_mask.zero_()
        # Vectorised: range(max_seq).unsqueeze(0) <= cache_lens.unsqueeze(1)
        ar = torch.arange(self.max_seq, device=self.attn_mask.device).unsqueeze(0)  # [1, max_seq]
        valid = ar <= cl.to(self.attn_mask.device).unsqueeze(1)  # [N, max_seq]
        # attn_mask shape: [N, 1, 1, max_seq]
        self.attn_mask[:, 0, 0, :] = valid


class CodePredictorAttentionInflight(nn.Module):
    """Token-flat in-flight attention.

    ``forward(hidden_states, position_embeddings, k_cache, v_cache,
              slot_indices, write_positions, attn_mask)``

    - ``hidden_states``: ``[N, 1, H]`` (token-flat; the seq dim is always 1
      because each row is one step for one request).
    - ``position_embeddings``: ``(cos, sin)`` shaped ``[N, 1, head_dim]``.
    - ``k_cache`` / ``v_cache``: ``[max_slots, num_kv_heads, max_seq, head_dim]``.
    - ``slot_indices``: ``[N]`` long, slot row to write/read.
    - ``write_positions``: ``[N]`` long, position in the slot to write the new K/V.
    - ``attn_mask``: ``[N, 1, 1, max_seq]`` bool — True == attend.

    Writing K/V uses ``index_put_`` with two index tensors so it's
    cuda-graph-friendly.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.is_gqa = self.num_kv_heads != self.num_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.max_seq = int(config.num_code_groups) + 1

        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_indices: torch.Tensor,
        write_positions: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [N, 1, H]
        n, s, _ = hidden_states.shape
        assert s == 1
        q = self.q_norm(self.q_proj(hidden_states).view(n, 1, self.num_heads, self.head_dim)).transpose(1, 2)
        k_new = self.k_norm(self.k_proj(hidden_states).view(n, 1, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v_new = self.v_proj(hidden_states).view(n, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q, k_new, v_new shape: [N, num_heads or kv_heads, 1, head_dim]

        cos, sin = position_embeddings  # [N, 1, head_dim]
        cos = cos.unsqueeze(1)  # [N, 1, 1, head_dim]
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k_new = (k_new * cos) + (_rotate_half(k_new) * sin)

        # Write K/V at (slot_indices, write_positions) along (dim0, dim2).
        # k_cache shape: [max_slots, num_kv, max_seq, head_dim]
        # We want k_cache[slot_indices, :, write_positions, :] = k_new.squeeze(2)
        k_cache[slot_indices, :, write_positions, :] = k_new.squeeze(2)
        v_cache[slot_indices, :, write_positions, :] = v_new.squeeze(2)

        # Gather the per-token cache view.
        # k_per_token shape: [N, num_kv, max_seq, head_dim]
        k_per_token = k_cache[slot_indices]
        v_per_token = v_cache[slot_indices]

        with sdpa_kernel(_SDPA_BACKENDS):
            attn_out = F.scaled_dot_product_attention(
                q,
                k_per_token,
                v_per_token,
                attn_mask=attn_mask,
                scale=self.scaling,
                is_causal=False,
                enable_gqa=self.is_gqa,
            )
        # attn_out shape: [N, num_heads, 1, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(n, 1, -1)
        return self.o_proj(attn_out)


class CodePredictorDecoderLayerInflight(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = CodePredictorAttentionInflight(config)
        self.mlp = CodePredictorMLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h, pe, k, v, slot_indices, write_positions, attn_mask):
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn(h, pe, k, v, slot_indices, write_positions, attn_mask)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class CodePredictorBaseModelInflight(nn.Module):
    """Inner transformer; token-flat input with per-token cache slot."""

    def __init__(self, config, *, embedding_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        emb_dim = int(embedding_dim) if embedding_dim is not None else int(config.hidden_size)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
        )
        self.layers = nn.ModuleList(
            [CodePredictorDecoderLayerInflight(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def allocate_state(self, max_slots: int, device: torch.device, dtype: torch.dtype) -> InflightCacheState:
        cp = self.config
        max_seq = int(cp.num_code_groups) + 1
        head_dim = getattr(cp, "head_dim", cp.hidden_size // cp.num_attention_heads)
        num_kv = cp.num_key_value_heads
        k_caches: list[torch.Tensor] = []
        v_caches: list[torch.Tensor] = []
        for _ in self.layers:
            k_caches.append(torch.zeros(max_slots, num_kv, max_seq, head_dim, device=device, dtype=dtype))
            v_caches.append(torch.zeros(max_slots, num_kv, max_seq, head_dim, device=device, dtype=dtype))
        return InflightCacheState(
            k_caches=k_caches,
            v_caches=v_caches,
            slot_indices=torch.zeros(max_slots, device=device, dtype=torch.long),
            write_positions=torch.zeros(max_slots, device=device, dtype=torch.long),
            position_ids=torch.zeros(max_slots, device=device, dtype=torch.long),
            attn_mask=torch.zeros(max_slots, 1, 1, max_seq, device=device, dtype=torch.bool),
            max_slots=max_slots,
            max_seq=max_seq,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [N, 1, H]
        position_ids: torch.Tensor,  # [N, 1]
        slot_indices: torch.Tensor,  # [N]
        write_positions: torch.Tensor,  # [N]
        attn_mask: torch.Tensor,  # [N, 1, 1, max_seq]
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        input_dtype = inputs_embeds.dtype
        pe = self.rotary_emb(inputs_embeds, position_ids)
        h = inputs_embeds
        for layer, k, v in zip(self.layers, k_caches, v_caches):
            h = layer(h, pe, k, v, slot_indices, write_positions, attn_mask)
        h = self.norm(h)
        return h.to(input_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)
        return loaded
