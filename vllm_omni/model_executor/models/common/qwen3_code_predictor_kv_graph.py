"""Qwen3 Code Predictor — KV cache + CUDA-graph friendly variant.

Building on ``qwen3_code_predictor_kv.py`` but reshaped so the whole inner
forward can be captured into a CUDA graph and replayed every AR step.

Key differences vs ``qwen3_code_predictor_kv``:

* Decode attends over the **entire** cache window ``[B, Hkv, Q+1, D]`` with
  an ``attn_mask`` parameter; the mask is updated host-side **in place**
  between replays to expose only the slots already populated.
* Cache write uses ``Tensor.index_copy_`` with a 0-dim ``write_idx`` tensor
  so the position-to-write is a graph parameter (value updated host-side,
  shape fixed).
* Caches are passed as ``list[Tensor]`` to the forwards so we don't drag
  a Python dataclass through ``torch.compile`` / cuda graph capture.

Numerics: float32 RMSNorm, float32 RoPE, separate Q/K/V linears — identical
to the re-prefill path so audio quality stays at parity.
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

# SDPA backend selection.
#
# The combination ``enable_gqa=True`` + ``attn_mask`` blocks **all** fused
# kernels (FLASH/EFFICIENT/CUDNN) — PyTorch's GQA broadcasting path falls
# back to MATH (~30µs at bs=8 in CUDA graph). We instead expand K/V to
# match Q's head count *manually* and drop ``enable_gqa``, which lets the
# default dispatch pick EFFICIENT_ATTENTION (~10µs, 3× faster than MATH).
#
# Even faster: CUDNN_ATTENTION at ~6µs (~5× MATH), but vLLM globally
# disables cudnn_sdp at startup (platforms/cuda.py:49) for safety
# (PyTorch 2.5 cuDNN default crashed some diffusion models). A priority
# list like ``[CUDNN, EFFICIENT, MATH]`` still skips CUDNN due to that
# gate; only a single-backend ``[CUDNN]`` override bypasses it.
#
# Default: ``[EFFICIENT, MATH]`` — safe, 3× MATH baseline.
# Opt-in: ``QWEN3_TTS_CUDNN_SDPA=1`` switches decode to ``[CUDNN]`` for
# the extra ~40% on top (10µs → 6µs).
import os as _os_sdpa
_CUDNN_SDPA = _os_sdpa.environ.get("QWEN3_TTS_CUDNN_SDPA", "0") == "1"
_SDPA_PREFILL_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
if _CUDNN_SDPA:
    _SDPA_DECODE_BACKENDS = [SDPBackend.CUDNN_ATTENTION]
else:
    _SDPA_DECODE_BACKENDS = [
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]


@dataclass
class KVGraphState:
    """Per-call mutable state for the captured graphs.

    ``k_caches`` / ``v_caches`` are the list[Tensor] we pass into the graph.
    Each tensor's shape is ``[max_bsz, num_kv_heads, Q+1, head_dim]`` and
    stays fixed across replays — only the contents mutate (in-place writes
    by ``index_copy_`` inside the graph; ``reset()`` zeros from outside).
    """

    k_caches: list[torch.Tensor] = field(default_factory=list)
    v_caches: list[torch.Tensor] = field(default_factory=list)
    write_idx: torch.Tensor | None = None  # 0-dim long tensor
    attn_mask: torch.Tensor | None = None  # [1, 1, 1, Q+1]
    max_seq: int = 0

    def reset(self) -> None:
        for t in self.k_caches:
            t.zero_()
        for t in self.v_caches:
            t.zero_()
        self.write_idx.zero_()
        # Bool mask: True == attend. Start fully masked-out.
        self.attn_mask.zero_()

    def set_decode_step(self, write_index: int) -> None:
        self.write_idx.fill_(write_index)
        self.attn_mask.zero_()
        # valid slots are [0, write_index] inclusive
        self.attn_mask[..., : write_index + 1] = True


class CodePredictorAttentionKVGraph(nn.Module):
    """Self-attention with graph-friendly K/V cache write + mask."""

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
        # Fused QKV (see CodePredictorAttention for rationale: ~2.6× faster).
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _qkv_with_rope(self, hidden_states, position_embeddings):
        bsz, s, _ = hidden_states.shape
        hsq = (bsz, s, self.num_heads, self.head_dim)
        hskv = (bsz, s, self.num_kv_heads, self.head_dim)

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(hsq)).transpose(1, 2)
        k = self.k_norm(k.view(hskv)).transpose(1, 2)
        v = v.view(hskv).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k, v

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        bsz, s, _ = hidden_states.shape  # s == 2 in capture
        q, k_new, v_new = self._qkv_with_rope(hidden_states, position_embeddings)

        k_cache[:, :, :s, :] = k_new
        v_cache[:, :, :s, :] = v_new

        # Manual GQA expand (see module-level comment). Prefill has no mask;
        # FLASH_ATTENTION supports GQA via enable_gqa, so we can take either
        # path. Keep manual expand for consistency with decode.
        if self.is_gqa:
            nq_per_kv = q.shape[1] // k_new.shape[1]
            k_attn = k_new.repeat_interleave(nq_per_kv, dim=1)
            v_attn = v_new.repeat_interleave(nq_per_kv, dim=1)
        else:
            k_attn, v_attn = k_new, v_new

        with sdpa_kernel(_SDPA_PREFILL_BACKENDS):
            attn_out = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                scale=self.scaling,
                is_causal=True,
            )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, s, -1)
        return self.o_proj(attn_out)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        write_idx: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, _ = hidden_states.shape  # s == 1
        q, k_new, v_new = self._qkv_with_rope(hidden_states, position_embeddings)

        # Graph-friendly write: index_copy_ along dim 2 with 0-dim write_idx.
        idx = write_idx.view(1)
        k_cache.index_copy_(2, idx, k_new)
        v_cache.index_copy_(2, idx, v_new)

        # Manual GQA expand — see module-level comment. Without this,
        # ``enable_gqa=True`` + ``attn_mask`` blocks all fused kernels and
        # PyTorch falls back to MATH (~30µs vs ~10µs EFFICIENT at bs=8).
        if self.is_gqa:
            nq_per_kv = q.shape[1] // k_cache.shape[1]
            k_attn = k_cache.repeat_interleave(nq_per_kv, dim=1)
            v_attn = v_cache.repeat_interleave(nq_per_kv, dim=1)
        else:
            k_attn, v_attn = k_cache, v_cache

        with sdpa_kernel(_SDPA_DECODE_BACKENDS):
            attn_out = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=attn_mask,
                scale=self.scaling,
                is_causal=False,
            )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, 1, -1)
        return self.o_proj(attn_out)


class CodePredictorDecoderLayerKVGraph(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = CodePredictorAttentionKVGraph(config)
        self.mlp = CodePredictorMLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_prefill(self, h, pe, k, v):
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn.forward_prefill(h, pe, k, v)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h

    def forward_decode(self, h, pe, k, v, idx, mask):
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn.forward_decode(h, pe, k, v, idx, mask)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class CodePredictorBaseModelKVGraph(nn.Module):
    """Inner transformer with two graph-capable forwards (prefill + decode).

    Caches are passed in as ``list[Tensor]`` so the function signature is
    compile/graph-friendly (no Python dataclass crossings).
    """

    def __init__(self, config, *, embedding_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        emb_dim = int(embedding_dim) if embedding_dim is not None else int(config.hidden_size)

        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
        )
        self.layers = nn.ModuleList(
            [CodePredictorDecoderLayerKVGraph(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def allocate_state(self, max_bsz: int, device: torch.device, dtype: torch.dtype) -> KVGraphState:
        cp = self.config
        max_seq = int(cp.num_code_groups) + 1
        head_dim = getattr(cp, "head_dim", cp.hidden_size // cp.num_attention_heads)
        num_kv = cp.num_key_value_heads
        k_caches: list[torch.Tensor] = []
        v_caches: list[torch.Tensor] = []
        for _ in self.layers:
            k_caches.append(torch.zeros(max_bsz, num_kv, max_seq, head_dim, device=device, dtype=dtype))
            v_caches.append(torch.zeros(max_bsz, num_kv, max_seq, head_dim, device=device, dtype=dtype))
        write_idx = torch.zeros((), dtype=torch.long, device=device)
        # Bool mask: True == attend, False == mask out. EFFICIENT_ATTENTION
        # supports bool masks even with enable_gqa=True; float masks may fall
        # through to MATH backend with lower bf16 reduction precision.
        attn_mask = torch.zeros((1, 1, 1, max_seq), dtype=torch.bool, device=device)
        return KVGraphState(
            k_caches=k_caches,
            v_caches=v_caches,
            write_idx=write_idx,
            attn_mask=attn_mask,
            max_seq=max_seq,
        )

    def forward_prefill(
        self,
        inputs_embeds: torch.Tensor,  # [B, 2, H]
        position_ids: torch.Tensor,  # [B, 2]
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        input_dtype = inputs_embeds.dtype
        pe = self.rotary_emb(inputs_embeds, position_ids)
        h = inputs_embeds
        for layer, k, v in zip(self.layers, k_caches, v_caches):
            h = layer.forward_prefill(h, pe, k, v)
        h = self.norm(h)
        return h.to(input_dtype)

    def forward_decode(
        self,
        inputs_embeds: torch.Tensor,  # [B, 1, H]
        position_ids: torch.Tensor,  # [B, 1]
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
        write_idx: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = inputs_embeds.dtype
        pe = self.rotary_emb(inputs_embeds, position_ids)
        h = inputs_embeds
        for layer, k, v in zip(self.layers, k_caches, v_caches):
            h = layer.forward_decode(h, pe, k, v, write_idx, attn_mask)
        h = self.norm(h)
        return h.to(input_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with QKV / gate_up fusion remapping (same as base)."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        def _remap(name: str) -> tuple[str, slice] | None:
            if name.endswith(".q_proj.weight") or name.endswith(".q_proj.bias"):
                base = name.rsplit(".q_proj.", 1)[0]
                suffix = "weight" if name.endswith("weight") else "bias"
                target = f"{base}.qkv_proj.{suffix}"
                if target not in params_dict:
                    return None
                attn = self.get_submodule(base) if base else self
                return target, slice(0, attn.q_size)
            if name.endswith(".k_proj.weight") or name.endswith(".k_proj.bias"):
                base = name.rsplit(".k_proj.", 1)[0]
                suffix = "weight" if name.endswith("weight") else "bias"
                target = f"{base}.qkv_proj.{suffix}"
                if target not in params_dict:
                    return None
                attn = self.get_submodule(base) if base else self
                return target, slice(attn.q_size, attn.q_size + attn.kv_size)
            if name.endswith(".v_proj.weight") or name.endswith(".v_proj.bias"):
                base = name.rsplit(".v_proj.", 1)[0]
                suffix = "weight" if name.endswith("weight") else "bias"
                target = f"{base}.qkv_proj.{suffix}"
                if target not in params_dict:
                    return None
                attn = self.get_submodule(base) if base else self
                return target, slice(attn.q_size + attn.kv_size, attn.q_size + 2 * attn.kv_size)
            if name.endswith(".gate_proj.weight"):
                base = name.rsplit(".gate_proj.", 1)[0]
                target = f"{base}.gate_up_proj.weight"
                if target not in params_dict:
                    return None
                mlp = self.get_submodule(base) if base else self
                return target, slice(0, mlp.intermediate_size)
            if name.endswith(".up_proj.weight"):
                base = name.rsplit(".up_proj.", 1)[0]
                target = f"{base}.gate_up_proj.weight"
                if target not in params_dict:
                    return None
                mlp = self.get_submodule(base) if base else self
                return target, slice(mlp.intermediate_size, 2 * mlp.intermediate_size)
            return None

        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            remap = _remap(name)
            if remap is not None:
                target_name, row_slice = remap
                param = params_dict[target_name]
                with torch.no_grad():
                    param[row_slice].copy_(w)
                # Mark the fused target as loaded.
                loaded.add(target_name)
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)
        return loaded
