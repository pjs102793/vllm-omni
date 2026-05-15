"""Qwen3-TTS Code Predictor -- thin wrapper over CodePredictorWrapper."""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

# Backward-compat alias used by tests
Qwen3TTSTalkerCodePredictorModelVLLM = CodePredictorBaseModel


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(CodePredictorWrapper):
    """Qwen3-TTS code predictor (CUDA graphs, per-call sampling, projection)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        use_kv_cache = os.environ.get("QWEN3_TTS_USE_KV_CACHE", "0") == "1"
        if use_kv_cache and current_omni_platform.is_npu():
            logger.warning(
                "QWEN3_TTS_USE_KV_CACHE=1 is not supported on NPU; falling back to re-prefill path."
            )
            use_kv_cache = False
        if use_kv_cache:
            logger.info(
                "code_predictor: KV-cache PoC enabled via QWEN3_TTS_USE_KV_CACHE=1 "
                "(prefill + decode graphs captured per bucket)."
            )

        super().__init__(
            vllm_config=vllm_config,
            cp_config=config,
            wrapper_config=CodePredictorWrapperConfig(
                use_cuda_graphs=not use_kv_cache,
                use_parallel_embedding=False,
                use_projection=(config.hidden_size != talker_config.hidden_size),
                return_proj_buf=False,
                sampling_mode="per_call",
                use_kv_cache=use_kv_cache,
            ),
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=prefix,
        )
        # Store talker_config for backward compat (accessed by some callers)
        self.talker_config = talker_config
        self._vllm_config = vllm_config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with vllm config context (required for VocabParallelEmbedding)."""
        with set_current_vllm_config(self._vllm_config):
            return super().load_weights(weights)
