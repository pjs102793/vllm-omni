"""Qwen3 Code Predictor -- optimized re-prefill, no KV cache.

Shared by Qwen3-Omni and Qwen3-TTS talker models.

* SDPA attention (F.scaled_dot_product_attention) with native GQA support
* HF-compatible numerics (float32 RMSNorm, float32 RoPE, separate linear layers)
* Per-call embedding buffer to avoid cross-request aliasing
* Pre-allocated position_ids (read-only, safe to persist)
* torch.compile (epilogue_fusion=False) on inner transformer by default
* Optional manual CUDA graph capture per batch-size bucket
* Inline sampling (top-k + top-p) -- no custom op overhead
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Optional flashinfer-fused sampler (top-K + sample in one kernel).
try:
    from flashinfer.sampling import (
        top_k_sampling_from_probs as _fi_topk_sample,
        top_k_top_p_sampling_from_probs as _fi_topk_topp_sample,
    )
    _HAS_FLASHINFER_SAMPLING = True
except ImportError:
    _HAS_FLASHINFER_SAMPLING = False


# ===================================================================
# HF-numerics-compatible layers for code predictor
# ===================================================================
#
# These use plain PyTorch ops (nn.Linear, manual RMSNorm in float32,
# rotate_half RoPE) to produce outputs numerically identical to the
# HuggingFace reference. vLLM's fused kernels (RMSNorm, QKVParallel,
# get_rope) introduce small precision differences that compound across
# the autoregressive steps of the code predictor, causing severe
# audio quality degradation.
#
# See: https://github.com/vllm-project/vllm-omni/issues/2274


class _RMSNorm(nn.Module):
    """RMSNorm matching HuggingFace's implementation exactly.

    Computes variance in float32 to avoid bfloat16 precision loss.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _RotaryEmbedding(nn.Module):
    """RoPE matching HuggingFace's implementation exactly.

    Forces float32 computation for cos/sin, matching HF's torch.autocast(enabled=False).
    """

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (matching HF)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ===================================================================
#  Attention
# ===================================================================


class CodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor.

    Uses ``F.scaled_dot_product_attention`` with HF-compatible RoPE and RMSNorm.
    No KV cache -- the code predictor always re-prefills the full (short)
    sequence each AR step.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.is_gqa = self.num_kv_heads != self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.max_seq = int(config.num_code_groups) + 1

        # Fused QKV projection (single GEMM instead of three).
        # Mathematically identical to separate q/k/v — concatenated weight rows
        # produce concatenated outputs. ~2.6× faster than separate GEMMs at
        # sub-talker decode shapes (bs=4-16, in CUDA graph).
        # HF checkpoint stores q_proj/k_proj/v_proj separately; load_weights
        # remaps them into slices of qkv_proj.
        bias = getattr(config, "attention_bias", False)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if current_omni_platform.is_npu():
            if self.max_seq > 2048:
                raise ValueError(
                    "Qwen3-TTS code predictor NPU fusion attention uses a fixed 2048x2048 "
                    f"causal mask, but max_seq={self.max_seq} exceeds the mask size."
                )
            # Ascend SDPA is_causal migration example uses a fixed 2048x2048
            # compressed causal mask with sparse_mode=2.
            fusion_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.bool),
                diagonal=1,
            )
            self.register_buffer("_fusion_causal_mask", fusion_mask, persistent=False)

    def _forward_npu_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        import torch_npu

        q_f, k_f, v_f = q, k, v
        if self.is_gqa:
            k_f = (
                k[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
                .reshape(bsz, self.num_heads, seq_len, self.head_dim)
            )
            v_f = (
                v[:, :, None, :, :]
                .expand(bsz, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
                .reshape(bsz, self.num_heads, seq_len, self.head_dim)
            )

        mask = self._fusion_causal_mask
        mask = mask.contiguous()
        q_f = q_f.contiguous()
        k_f = k_f.contiguous()
        v_f = v_f.contiguous()
        return torch_npu.npu_fusion_attention(
            q_f,
            k_f,
            v_f,
            self.num_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=mask,
            scale=float(self.scaling),
            keep_prob=1.0,
            # Keep torch_npu's API spelling.
            pre_tockens=2147483647,
            next_tockens=2147483647,
            inner_precise=0,
            prefix=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            # Ascend SDPA is_causal migration example uses sparse_mode=2.
            sparse_mode=2,
            gen_mask_parallel=True,
            # Keep sync=True for the NPU fused attention path.
            sync=True,
        )[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
        hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Fused QKV: single GEMM, then split.
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(hidden_shape_q)).transpose(1, 2)
        k = self.k_norm(k.view(hidden_shape_kv)).transpose(1, 2)
        v = v.view(hidden_shape_kv).transpose(1, 2)

        cos, sin = position_embeddings
        # cos/sin are [batch, seq_len, head_dim], need unsqueeze at dim=1 for heads
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        if not current_omni_platform.is_npu():
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scaling,
                is_causal=True,
                enable_gqa=self.is_gqa,
            )
        else:
            attn_out = self._forward_npu_attention(q, k, v, bsz, seq_len)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out)


# ===================================================================
#  MLP
# ===================================================================


class CodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching HF's implementation."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        # Fused gate+up projection (single GEMM instead of two). HF checkpoint
        # stores gate_proj/up_proj separately; load_weights remaps them into
        # slices of gate_up_proj. ~1.7× faster than separate at sub-talker
        # decode shapes (bs=4-16, in CUDA graph).
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ===================================================================
#  Decoder Layer
# ===================================================================


class CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer (SDPA, no KV cache)."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = CodePredictorMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ===================================================================
#  Base Transformer Model (re-prefill, no KV cache)
# ===================================================================


class CodePredictorBaseModel(nn.Module):
    """Inner transformer for code predictor.

    Signature: ``forward(inputs_embeds, position_ids) -> hidden_states``
    """

    def __init__(
        self,
        config,
        *,
        embedding_dim: int | None = None,
        use_parallel_embedding: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        emb_dim = int(embedding_dim) if embedding_dim is not None else int(config.hidden_size)
        if use_parallel_embedding:
            self.codec_embedding = nn.ModuleList(
                [VocabParallelEmbedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )
        else:
            self.codec_embedding = nn.ModuleList(
                [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )

        self.layers = nn.ModuleList(
            [
                CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Run the transformer body in float32 when the model is in fp16.
        # fp16 lacks the dynamic range for stable attention scores and
        # SiLU-gated MLP intermediates, producing NaN on GPUs without
        # native bf16 support (Turing, Volta).  The RMSNorm and RoPE
        # layers already upcast internally; this extends the same
        # treatment to attention and MLP.
        input_dtype = inputs_embeds.dtype
        use_fp32 = input_dtype == torch.float16
        if use_fp32:
            inputs_embeds = inputs_embeds.float()
        hidden_states = inputs_embeds
        with torch.amp.autocast(inputs_embeds.device.type, enabled=use_fp32, dtype=torch.float32):
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_embeddings)
            hidden_states = self.norm(hidden_states)
        return hidden_states.to(input_dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with QKV / gate_up fusion remapping.

        HF checkpoint stores ``q_proj`` / ``k_proj`` / ``v_proj`` and
        ``gate_proj`` / ``up_proj`` as separate Linear weights. Our fused
        modules store them as concatenated rows of a single weight tensor;
        we route the individual loads into the appropriate row slices.
        """
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        def _remap(name: str) -> tuple[str, slice] | None:
            # Returns (fused_param_name, row_slice) or None if no remap.
            if name.endswith(".q_proj.weight") or name.endswith(".q_proj.bias"):
                base = name.rsplit(".q_proj.", 1)[0]
                suffix = "weight" if name.endswith("weight") else "bias"
                target = f"{base}.qkv_proj.{suffix}"
                param = params_dict.get(target)
                if param is None:
                    return None
                # q comes first (rows 0..q_size). Use the param's q_size from sibling attrs.
                # Infer q_size from the parent attention module.
                attn = self.get_submodule(base) if base else self
                q_size = attn.q_size
                return target, slice(0, q_size)
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

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # Fused remap for QKV / gate_up.
            remap = _remap(name)
            if remap is not None:
                target_name, row_slice = remap
                param = params_dict[target_name]
                # Write directly into the row slice of the fused weight.
                with torch.no_grad():
                    param[row_slice].copy_(loaded_weight)
                # Mark the fused target as loaded (vLLM strict check uses
                # named_parameters, not the HF source names).
                loaded_params.add(target_name)
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===================================================================
#  Wrapper Configuration
# ===================================================================


@dataclasses.dataclass
class CodePredictorWrapperConfig:
    """Controls behavioral differences between model-specific code predictors."""

    use_cuda_graphs: bool = False
    use_parallel_embedding: bool = False
    use_projection: bool = False
    return_proj_buf: bool = False
    sampling_mode: str = "stored"
    use_kv_cache: bool = False


# ===================================================================
#  Code Predictor Wrapper (optimized re-prefill, persistent buffers)
# ===================================================================


class CodePredictorWrapper(nn.Module):
    """Optimized code predictor -- re-prefill approach, no KV cache.

    Each AR step forwards the full growing sequence (len 2 -> num_code_groups+1)
    through the transformer.  The extra O(T^2) FLOPs are negligible for
    short sequences, and this avoids all KV-cache management overhead.

    Optimizations:
      1. Per-call embedding buffer -- avoids cross-request aliasing.
      2. Pre-allocated position_ids -- no torch.arange per step.
      3. Cached module references -- bypass ModuleList indexing.
      4. torch.compile on inner transformer.
      5. Inline sampling (top-k + top-p) -- no custom op overhead.
      6. Optional manual CUDA graph capture per batch-size bucket.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        cp_config,
        wrapper_config: CodePredictorWrapperConfig,
        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = cp_config
        self._wrapper_config = wrapper_config
        self.prefix = prefix

        self._num_groups = int(cp_config.num_code_groups)
        self._cp_hidden = int(cp_config.hidden_size)

        # For Omni backward compat (accessed by the talker)
        self.num_code_groups = self._num_groups

        # Determine embedding dimension
        _talker_hidden = int(talker_hidden_size) if talker_hidden_size is not None else self._cp_hidden

        if wrapper_config.use_kv_cache:
            from vllm_omni.model_executor.models.common.qwen3_code_predictor_kv_graph import (
                CodePredictorBaseModelKVGraph,
            )

            self.model = CodePredictorBaseModelKVGraph(
                cp_config,
                embedding_dim=_talker_hidden,
            )
        else:
            self.model = CodePredictorBaseModel(
                cp_config,
                embedding_dim=_talker_hidden,
                use_parallel_embedding=wrapper_config.use_parallel_embedding,
                prefix=f"{prefix}.model" if prefix else "model",
            )

        self.lm_head = nn.ModuleList(
            [nn.Linear(cp_config.hidden_size, cp_config.vocab_size, bias=False) for _ in range(self._num_groups - 1)]
        )

        # Projection: Identity when hidden sizes match or not needed
        if wrapper_config.use_projection and _talker_hidden != self._cp_hidden:
            self.small_to_mtp_projection = nn.Linear(_talker_hidden, self._cp_hidden, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Sampling defaults for "stored" mode
        self._top_k: int = 50
        self._top_p: float = 0.8

        # Lazily initialised state
        self._proj_buf: torch.Tensor | None = None
        self._model_dtype: torch.dtype | None = None
        self._compiled_model_fwd = None
        self._bucket_sizes: list[int] = []
        self._bucket_pos_ids: dict[int, torch.Tensor] = {}
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None
        self._device_graphs: dict[int, tuple] = {}  # (graph, static_output) per bucket

        # KV-cache + CUDA-graph state (lazily filled by _setup_kv_graphs).
        self._kv_states: dict[int, object] = {}
        self._kv_prefill_in: dict[int, torch.Tensor] = {}
        self._kv_prefill_pos: dict[int, torch.Tensor] = {}
        self._kv_decode_in: dict[int, torch.Tensor] = {}
        self._kv_decode_pos: dict[int, torch.Tensor] = {}
        self._kv_prefill_graph: dict[int, tuple] = {}
        self._kv_decode_graph: dict[int, tuple] = {}
        self._kv_prefill_fwd = None
        self._kv_decode_fwd = None

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def set_sampling_params(self, top_k: int = 50, top_p: float = 0.8) -> None:
        """Configure sampling parameters to maintain consistency with previous implementation."""
        self._top_k = top_k
        self._top_p = top_p
        logger.debug("Sampling parameters updated: top_k=%d, top_p=%.2f", top_k, top_p)

    # ------------------------------------------------------------------
    #  Lazy-init helpers
    # ------------------------------------------------------------------

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype, bsz: int) -> None:
        """Ensure the projection buffer can hold at least *bsz* rows."""
        max_seq = self._num_groups + 1
        if (
            self._proj_buf is not None
            and self._proj_buf.device == device
            and self._proj_buf.dtype == dtype
            and self._proj_buf.shape[0] >= bsz
        ):
            return
        self._proj_buf = torch.zeros(bsz, max_seq, self._cp_hidden, dtype=dtype, device=device)

    def _setup_kv_graphs(self) -> None:
        """KV-cache path: torch.compile prefill + decode forwards, capture one
        prefill graph and one decode graph per batch-size bucket.

        State is held in dicts keyed by bucket size; ``_forward_kv`` selects
        the bucket via ``_padded_bsz`` and replays the matching graphs.
        """
        from vllm.platforms import current_platform

        # Determine bucket sizes the same way the re-prefill path does.
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        device = next(self.model.parameters()).device
        dtype = self._model_dtype
        cp_hidden = self._cp_hidden

        # Compile prefill/decode forwards. Inductor produces one graph per
        # (shape, mode) combo; cuda-graph capture sits on top of the compiled
        # function so we get both fusion + launch elimination.
        if current_omni_platform.supports_torch_inductor():
            prefill_fwd = torch.compile(
                self.model.forward_prefill,
                dynamic=False,
                options={"epilogue_fusion": False},
            )
            decode_fwd = torch.compile(
                self.model.forward_decode,
                dynamic=False,
                options={"epilogue_fusion": False},
            )
            compile_msg = "torch.compile(epilogue_fusion=False)"
        else:
            prefill_fwd = self.model.forward_prefill
            decode_fwd = self.model.forward_decode
            compile_msg = "eager"

        self._kv_prefill_fwd = prefill_fwd
        self._kv_decode_fwd = decode_fwd

        pool = current_platform.get_global_graph_pool()

        for bsz in self._bucket_sizes:
            # Allocate per-bucket KV state + static input/pos buffers.
            state = self.model.allocate_state(bsz, device, dtype)
            self._kv_states[bsz] = state

            pre_in = torch.zeros(bsz, 2, cp_hidden, dtype=dtype, device=device)
            pre_pos = (
                torch.arange(2, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous()
            )
            dec_in = torch.zeros(bsz, 1, cp_hidden, dtype=dtype, device=device)
            dec_pos = torch.zeros(bsz, 1, device=device, dtype=torch.long)

            self._kv_prefill_in[bsz] = pre_in
            self._kv_prefill_pos[bsz] = pre_pos
            self._kv_decode_in[bsz] = dec_in
            self._kv_decode_pos[bsz] = dec_pos

            # Warmup eager + compiled forwards so Inductor compiles before capture.
            state.reset()
            for _ in range(3):
                _ = prefill_fwd(pre_in, pre_pos, state.k_caches, state.v_caches)
            torch.cuda.synchronize()

            # Warmup decode (after a prefill so cache slot 0,1 are populated).
            state.set_decode_step(2)
            dec_pos.fill_(2)
            for _ in range(3):
                _ = decode_fwd(
                    dec_in, dec_pos, state.k_caches, state.v_caches,
                    state.write_idx, state.attn_mask,
                )
            torch.cuda.synchronize()

            # Capture prefill graph (fresh state).
            state.reset()
            g_pre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_pre, pool=pool):
                out_pre = prefill_fwd(pre_in, pre_pos, state.k_caches, state.v_caches)
            self._kv_prefill_graph[bsz] = (g_pre, out_pre)

            # Capture decode graph with cache_len=2 (post-prefill state).
            state.reset()
            state.set_decode_step(2)
            dec_pos.fill_(2)
            g_dec = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_dec, pool=pool):
                out_dec = decode_fwd(
                    dec_in, dec_pos, state.k_caches, state.v_caches,
                    state.write_idx, state.attn_mask,
                )
            self._kv_decode_graph[bsz] = (g_dec, out_dec)

        # Mark "compile done" so forward() doesn't re-enter.
        self._compiled_model_fwd = object()  # sentinel
        logger.info(
            "code_predictor: KV-cache + CUDA graphs captured (%s) for buckets %s",
            compile_msg,
            self._bucket_sizes,
        )

    def _maybe_quantize_subtalker(self) -> None:
        """Optionally quantize sub-talker Linear weights via torchao.

        Env-gated by QWEN3_TTS_SUBTALKER_QUANT. Modes:
          - "int8_w"   : Int8WeightOnlyConfig            (W8A16, bf16 act)
          - "fp8_w"    : Float8WeightOnlyConfig          (W8A16, bf16 act)
          - "fp8_w8a8" : Float8DynamicActivationFloat8WeightConfig (W8A8, uses
                         Ada FP8 tensor cores)

        Applied before CUDA-graph capture so the quantized matmul kernels
        get captured. Memory bandwidth reduction is the primary gain since
        sub-talker decode is BW-bound at typical batch sizes (1-32).
        """
        mode = os.environ.get("QWEN3_TTS_SUBTALKER_QUANT", "").lower()
        if not mode:
            return
        try:
            from torchao.quantization import quantize_
            if mode == "int8_w":
                from torchao.quantization import Int8WeightOnlyConfig
                cfg = Int8WeightOnlyConfig()
            elif mode == "fp8_w":
                from torchao.quantization import Float8WeightOnlyConfig
                cfg = Float8WeightOnlyConfig()
            elif mode == "fp8_w8a8":
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                cfg = Float8DynamicActivationFloat8WeightConfig()
            else:
                logger.warning(
                    "code_predictor: unknown QWEN3_TTS_SUBTALKER_QUANT=%s; skipping", mode
                )
                return
        except ImportError as exc:
            logger.warning(
                "code_predictor: torchao not available, skipping quantization: %s", exc
            )
            return
        # Quantize attention + MLP linears (inside self.model.layers)
        quantize_(self.model, cfg)
        # Quantize lm_head per code group (one used per AR sub-step)
        for lm in self.lm_head:
            quantize_(lm, cfg)
        # Quantize talker->sub-talker projection (used once per call)
        if isinstance(self.small_to_mtp_projection, nn.Linear):
            quantize_(self.small_to_mtp_projection, cfg)
        logger.info(
            "code_predictor: sub-talker Linear weights quantized with %s", mode
        )

    def _setup_compile(self) -> None:
        """Lazily set up torch.compile with optional device graph capture."""
        if self._compiled_model_fwd is not None:
            return

        # Cache model parameter dtype so forward() doesn't need to query it
        # on every call.  Also ensures warmup buffers match model precision
        # even when upstream modules produce a different dtype (#2385).
        self._model_dtype = next(self.model.parameters()).dtype
        # Apply optional weight quantization BEFORE building references and
        # CUDA-graph capture so the quantized matmul kernels are captured.
        self._maybe_quantize_subtalker()
        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)

        if self._wrapper_config.use_kv_cache:
            # bf16 by default. Sampling-mode perceptual quality (DNSMOS) is
            # parity with re-prefill in our measurements; fp32 was only used
            # for strict greedy equivalence experiments. Toggle to fp32 with
            # QWEN3_TTS_KV_CACHE_FP32=1 if a deployment needs deterministic-
            # equivalent output under greedy.
            if os.environ.get("QWEN3_TTS_KV_CACHE_FP32", "0") == "1":
                self.model = self.model.float()
                for i, lm in enumerate(self.lm_head):
                    self.lm_head[i] = lm.float()
                if isinstance(self.small_to_mtp_projection, nn.Linear):
                    self.small_to_mtp_projection = self.small_to_mtp_projection.float()
                self._model_dtype = torch.float32
                logger.info("code_predictor: KV-cache mode forcing fp32 (QWEN3_TTS_KV_CACHE_FP32=1)")
            else:
                logger.info("code_predictor: KV-cache mode using bf16 (default)")
            self._setup_kv_graphs()
            return

        if not current_omni_platform.supports_torch_inductor():
            # NPU or other platforms without Inductor support
            self._compiled_model_fwd = self.model.forward

            if current_omni_platform.is_npu() and self._wrapper_config.use_cuda_graphs:
                # For NPU, use eager + NPU graphs (no torch.compile)
                self._warmup_buckets()
                self._capture_npu_graphs()
                logger.info("code_predictor: eager mode + NPU graphs")
            else:
                logger.warning_once("code_predictor: torch.compile disabled")
            return

        # torch.compile fuses RMSNorm/RoPE in ways that lose float32
        # precision, compounding across AR steps. Use epilogue_fusion=False
        # to disable the problematic fusions while still getting kernel
        # fusion benefits for the linear layers and SDPA.
        self._compiled_model_fwd = torch.compile(
            self.model.forward,
            dynamic=False,
            options={"epilogue_fusion": False},
        )
        self._warmup_buckets()

        if self._wrapper_config.use_cuda_graphs:
            self._capture_cuda_graphs()
            logger.info("code_predictor: torch.compile (no epilogue fusion) + CUDA graphs")
        else:
            logger.info("code_predictor: torch.compile (dynamic=False, no epilogue fusion)")

    def _padded_bsz(self, bsz: int) -> int:
        """Round batch size up to nearest power-of-2 bucket."""
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    def _warmup_buckets(self) -> None:
        """Warmup power-of-2 batch-size buckets to front-load Inductor compilation."""
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        max_seq = self._num_groups + 1
        device = next(self.model.parameters()).device

        # Ensure proj_buf matches model parameter dtype to avoid dtype
        # mismatch during warmup compilation (see #2385).
        self._ensure_buffers(device, self._model_dtype, max(self._bucket_sizes))
        proj_buf = self._proj_buf

        for bsz in self._bucket_sizes:
            pos_ids = torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous()
            self._bucket_pos_ids[bsz] = pos_ids
            for _ in range(3):
                self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
        logger.info("code_predictor: warmup done for buckets %s", self._bucket_sizes)

    def _capture_cuda_graphs(self) -> None:
        """Capture a CUDA graph per bucket using vLLM's global graph pool."""
        from vllm.platforms import current_platform

        pool = current_platform.get_global_graph_pool()
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf

        for bsz in self._bucket_sizes:
            static_input = proj_buf[:bsz, :max_seq, :]
            pos_ids = self._bucket_pos_ids[bsz]

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                static_output = self._compiled_model_fwd(static_input, pos_ids)

            self._device_graphs[bsz] = (g, static_output)

        logger.info("code_predictor: captured CUDA graphs for buckets %s", self._bucket_sizes)

    def _capture_npu_graphs(self) -> None:
        """Capture an NPU graph per bucket using torch_npu's NPUGraph."""
        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf
        pool = torch.npu.graph_pool_handle()

        for bsz in self._bucket_sizes:
            static_input = proj_buf[:bsz, :max_seq, :]
            pos_ids = self._bucket_pos_ids[bsz]

            g = torch.npu.NPUGraph()
            with torch.npu.graph(g, pool=pool):
                static_output = self._compiled_model_fwd(static_input, pos_ids)

            self._device_graphs[bsz] = (g, static_output)

        logger.info("code_predictor: captured NPU graphs for buckets %s", self._bucket_sizes)

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------

    def _sample_code(
        self,
        logits: torch.Tensor,
        *,
        stored_mode: bool,
        s_top_k: int,
        s_top_p: float,
        use_sampling: bool,
        inv_temperature: float,
        top_k: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Sample one residual code from ``logits``; returns ``[B, 1]`` long.

        Two regimes, matching HF numerics:
          * ``stored_mode``: top-k then top-p over raw logits (params captured
            once as ``s_top_k`` / ``s_top_p``).
          * per-call: temperature-scaled (``inv_temperature``) + top-k, or
            argmax when ``use_sampling`` is False.
        Uses the flashinfer fused sampler when available, else a torch fallback.
        """
        if stored_mode:
            if _HAS_FLASHINFER_SAMPLING:
                probs = F.softmax(logits, dim=-1, dtype=torch.float32)
                if s_top_p < 1.0:
                    return _fi_topk_topp_sample(probs, s_top_k, s_top_p, generator=generator).long().unsqueeze(-1)
                return _fi_topk_sample(probs, s_top_k, generator=generator).long().unsqueeze(-1)
            if s_top_k > 0:
                topk_vals, _ = logits.topk(s_top_k, dim=-1)
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
            if s_top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = (cumulative_probs - sorted_probs) >= s_top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)
            return torch.multinomial(probs, num_samples=1, generator=generator)
        if use_sampling:
            scaled = logits * inv_temperature
            if _HAS_FLASHINFER_SAMPLING and top_k > 0:
                probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
                return _fi_topk_sample(probs, top_k, generator=generator).long().unsqueeze(-1)
            if top_k > 0:
                topk_vals, _ = scaled.topk(top_k, dim=-1)
                scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
            probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
            return torch.multinomial(probs, num_samples=1, generator=generator)
        return logits.argmax(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    #  Forward -- re-prefill + inline sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict residual codebooks 1..G-1 autoregressively."""
        if self._wrapper_config.use_kv_cache:
            return self._forward_kv(
                layer0_code, layer0_embed, last_talker_hidden,
                do_sample, temperature, top_k, top_p, generator,
            )

        bsz = int(layer0_code.shape[0])
        num_groups = self._num_groups
        device = layer0_code.device

        # _setup_compile caches _model_dtype on first call; use it for buffers
        # so they always match model weight precision (#2385).
        self._setup_compile()
        dtype = self._model_dtype

        padded_bsz = self._padded_bsz(bsz)
        self._ensure_buffers(device, dtype, padded_bsz)

        proj_buf = self._proj_buf
        max_seq = num_groups + 1
        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        # Zero the padded region of the buffer
        proj_buf[:padded_bsz].zero_()

        # Fill buffer positions 0 (talker hidden) & 1 (layer0 embed)
        proj_buf[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
        proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)

        # Get pre-computed pos_ids for this bucket
        full_pos_ids = self._bucket_pos_ids.get(padded_bsz)
        if full_pos_ids is None:
            full_pos_ids = (
                torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(padded_bsz, -1).contiguous()
            )

        # Use captured device graph if available, otherwise call compiled fn.
        device_graph_entry = self._device_graphs.get(padded_bsz)

        # Prepare sampling parameters
        stored_mode = self._wrapper_config.sampling_mode == "stored"
        if stored_mode:
            s_top_k = self._top_k
            s_top_p = self._top_p
            use_sampling = True
            inv_temperature = 0.0
        else:
            s_top_k = 0
            s_top_p = 1.0
            use_sampling = do_sample and temperature > 0
            inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
            if use_sampling and top_p != 1.0:
                raise NotImplementedError(
                    "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
                )

        # Output codes -- shape depends on return mode
        if self._wrapper_config.return_proj_buf:
            all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz, -1)[:, :1]
        else:
            all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz)

        # Autoregressive loop: predict layers 1..G-1
        for step in range(1, num_groups):
            # Run transformer (device graph replay or compiled forward)
            if device_graph_entry is not None:
                device_graph_entry[0].replay()
                hidden_out = device_graph_entry[1]
            else:
                hidden_out = model_fwd(proj_buf[:padded_bsz, :max_seq, :], full_pos_ids)

            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])

            # Sample next code
            code = self._sample_code(
                logits,
                stored_mode=stored_mode,
                s_top_k=s_top_k,
                s_top_p=s_top_p,
                use_sampling=use_sampling,
                inv_temperature=inv_temperature,
                top_k=top_k,
                generator=generator,
            )

            # Store code
            if self._wrapper_config.return_proj_buf:
                all_codes[:, step] = code
            else:
                all_codes[:, step] = code.reshape(bsz)

            # Embed predicted code -> project -> next buffer position
            if step < num_groups - 1 or self._wrapper_config.return_proj_buf:
                new_embed = codec_embeds[step - 1](code)
                proj_buf[:bsz, step + 1, :] = projection(new_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)

        if self._wrapper_config.return_proj_buf:
            return all_codes, proj_buf[:bsz].clone()
        return all_codes

    # ------------------------------------------------------------------
    #  Forward (KV-cache variant) -- prefill(seq=2) + (Q-2) decode steps
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _forward_kv(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        bsz = int(layer0_code.shape[0])
        num_groups = self._num_groups
        device = layer0_code.device

        self._setup_compile()  # populates _kv_* maps on first call
        dtype = self._model_dtype

        padded_bsz = self._padded_bsz(bsz)
        state = self._kv_states[padded_bsz]
        pre_in = self._kv_prefill_in[padded_bsz]
        dec_in = self._kv_decode_in[padded_bsz]
        dec_pos = self._kv_decode_pos[padded_bsz]
        g_pre, out_pre = self._kv_prefill_graph[padded_bsz]
        g_dec, out_dec = self._kv_decode_graph[padded_bsz]

        projection = self.small_to_mtp_projection
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        # Reset KV state for this call (cache + mask).
        state.reset()

        # Populate prefill input buffer (pre_in is a graph-captured tensor).
        pos0 = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype))
        pos1 = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype))
        pre_in.zero_()
        pre_in[:bsz, 0:1, :] = pos0
        pre_in[:bsz, 1:2, :] = pos1

        # Replay prefill graph -> hidden_out captured in out_pre.
        g_pre.replay()

        # Sampling helpers
        stored_mode = self._wrapper_config.sampling_mode == "stored"
        if stored_mode:
            s_top_k = self._top_k
            s_top_p = self._top_p
            use_sampling = True
            inv_temperature = 0.0
        else:
            s_top_k = 0
            s_top_p = 1.0
            use_sampling = do_sample and temperature > 0
            inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
            if use_sampling and top_p != 1.0:
                raise NotImplementedError(
                    "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
                )
        sample_kwargs = dict(
            stored_mode=stored_mode,
            s_top_k=s_top_k,
            s_top_p=s_top_p,
            use_sampling=use_sampling,
            inv_temperature=inv_temperature,
            top_k=top_k,
            generator=generator,
        )

        all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz)

        # Step 1: emit logits at prefill output position 1 -> sample code 1.
        logits = lm_heads[0](out_pre[:bsz, 1, :])
        code = self._sample_code(logits, **sample_kwargs)
        all_codes[:, 1] = code.reshape(bsz)
        last_code = code

        # Steps 2..G-1: decode replay with host-side write_idx / mask update.
        for step in range(2, num_groups):
            new_embed = codec_embeds[step - 2](last_code)  # [B, 1, emb]
            next_in = projection(new_embed.reshape(bsz, 1, -1))
            dec_in.zero_()
            dec_in[:bsz, :, :] = next_in
            dec_pos.fill_(step)
            state.set_decode_step(step)
            g_dec.replay()

            logits = lm_heads[step - 1](out_dec[:bsz, 0, :])
            code = self._sample_code(logits, **sample_kwargs)
            all_codes[:, step] = code.reshape(bsz)
            last_code = code

        return all_codes

    # ------------------------------------------------------------------
    #  Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights directly (no fused projection remapping needed)."""
        loaded: set[str] = set()
        model_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], w))
            else:
                other_weights.append((name, w))

        loaded_model = self.model.load_weights(model_weights)
        loaded |= {f"model.{n}" for n in loaded_model}

        params = dict(self.named_parameters(remove_duplicate=False))
        for name, w in other_weights:
            param = params.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)
            loaded.add(name)

        return loaded
