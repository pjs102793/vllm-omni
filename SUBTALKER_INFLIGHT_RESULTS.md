# Qwen3-TTS Sub-talker In-flight — Results Report

## Status (이 브랜치 종료 시점)

| Goal 조건 | 달성 여부 | 비고 |
|---|---|---|
| (1) Prefill + decode in-flight 구현 | **부분 ✓** | Token-flat attention kernel + wrapper integration 완료. 진정한 cross-request phase-mix는 미구현 (scheduler integration 필요) |
| (2) 명확한 속도 개선 | **❌ 미달** | KV-on PoC와 동등 (±3%, noise floor). 본 브랜치만으로 wall-time lever 풀리지 않음 |
| (3) 음질 보장 | ✓ | DNSMOS OVRL parity (-2.1%, sampling stochasticity 내) |
| (4) 별도 브랜치 | ✓ | `feat/qwen3-tts-subtalker-inflight` |

## 작업물

### 코드
- `vllm_omni/model_executor/models/common/qwen3_code_predictor_inflight.py` (신규, 283줄)
  - `CodePredictorAttentionInflight`: token-flat input, per-token `slot_indices` / `write_positions` / `attn_mask`, gather-based SDPA
  - `CodePredictorBaseModelInflight`: plain tensor signature (compile/graph 호환)
  - `InflightCacheState`: 공유 K/V buffer + per-bucket graph param tensors
- `vllm_omni/model_executor/models/common/qwen3_code_predictor.py`
  - `CodePredictorWrapperConfig.use_inflight` flag
  - `_setup_inflight_graphs`: bucket별 graph capture (max_num_seqs까지)
  - `_forward_inflight`: drop-in 14-step AR loop, 각 step token-flat forward
- `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code_predictor_vllm.py`
  - `QWEN3_TTS_USE_INFLIGHT=1` env var

### 검증
- **Micro-bench** (`bench_inflight_sanity.py`, random init):
  - Inflight forward latency vs KV-graph decode at N ∈ {1, 4, 8, 16, 32, 64}: **±5% parity**
- **Server bench** (1.7B-CustomVoice, default deploy yaml, repeats=3, bucket-aware warmup):

| | KV-on PoC RTF agg | Inflight RTF agg | Δ |
|---|---:|---:|---:|
| short bs=1 | 0.157 | 0.153 | −3% |
| short bs=4 | 0.064 | 0.067 | +5% |
| short bs=8 | 0.041 | 0.040 | −2% |
| medium bs=1 | 0.139 | 0.141 | +1% |
| medium bs=8 | 0.034 | 0.032 | −6% |
| long bs=1 | 0.136 | 0.138 | +1% |
| long bs=4 | 0.052 | 0.053 | +2% |
| long bs=8 | 0.032 | 0.033 | +3% |

→ 모든 셀에서 ±6% 이내, mean ≈ 0% (within noise).

- **DNSMOS perceptual quality** (sample mode, 8 prompts, default sampling):

| | OVRL | SIG | BAK |
|---|---:|---:|---:|
| KV-off baseline | 3.031 | 3.358 | 3.890 |
| KV-on PoC (full fp32) | 3.049 | 3.437 | 3.817 |
| Inflight (this branch) | **2.968** | 3.315 | 3.834 |

→ Inflight이 KV-on PoC 대비 OVRL -2.1%. Sampling stochasticity (N=8 sample variance ≈ 0.05) 범위 내.

## 왜 wall-time 개선이 없는가 — 근본 원인

설계 문서 (`SUBTALKER_INFLIGHT_DESIGN.md`)에 사전 분석된 그대로:

```
현 architecture에서 dependency chain:
  main_fwd(n+1) input ← inputs_embeds_out
                       ← audio_codes(n) (codec_embedding sum)
                       ← sub_fwd(n) (14 sequential AR steps)
```

이 chain이 strict. 한 request 입장에서 main token rate은 `main_fwd + 14*sub_step`로 변경 불가능.

본 브랜치의 token-flat in-flight kernel은 다음 두 조건이 충족될 때만 wall-time lever를 풀 수 있음:

1. **다른 sub-step phase의 request들이 한 sub_fwd batched call에 모임** — 그러나 한 `talker_mtp` call의 모든 request가 같은 main talker step에서 시작하므로 같은 phase에서 출발. Phase 다양성 = `gpu_model_runner._talker_mtp_forward` 1-step 분리 + per-request state 추적 필요.
2. **Main+sub multi-stream overlap** — main_fwd와 sub_fwd가 GPU에서 동시 진행. vllm-omni stage system 자체 변경 필요.

둘 다 multi-week PR scope. 본 브랜치는 (1)의 prerequisite (kernel + cache 구조)을 마련했지만 scheduler 통합은 미구현.

### 정량적 한계 분석 (Amdahl)

가정: `main_fwd = 2ms`, `sub_step = 0.4ms`, `Q-1 = 14` AR steps.

| 시나리오 | per-token wall | 비교 |
|---|---:|---|
| Current lockstep | `main_fwd + 14×sub_step` = **7.6ms** | baseline |
| Naive 1-step inflight (14 main+sub interleaved) | `14×(main_fwd + sub_step)` = **33.6ms** | 4.4× 손해 (main_fwd 14× 더 자주) |
| Multi-stream main+sub overlap | `max(main_fwd, 14×sub_step) ≈ max(2, 5.6) = 5.6ms` | 1.36× 개선 (이론 상한) |
| Speculative decoding (사용자 평가 risk) | `< 7.6ms` (sub-talker steps 일부 skip) | 음질 위험 |

→ **multi-stream overlap이 본 architecture에서 wall-time 풀 수 있는 유일한 lever** (≤ 1.36×).

## 다음 단계 (이 브랜치 종료 후)

| 우선순위 | 작업 | 예상 효과 | 음질 위험 | 비용 |
|---|---|---|---|---|
| **A** | Stage 1 (C2W) graph capture 활성화 | wall −20~30% (C2W on) | 없음 | 1주 |
| **B** | INT8 weight-only quant for sub-talker | sub-step time −30~50% (memory-bound) | 작음 | 2-3주 |
| **C** | Multi-stream main+sub overlap | wall ≤ 1.36× | 없음 | 4-6주 (vllm-omni stage 변경) |
| **D** | Scheduler-level in-flight (이 브랜치 follow-up) | 본 분석상 negative or marginal | 작음 | 2-3주 (효과 검증 필요) |
| **E** | Sub-talker model 압축 (layer/hidden 축소) | sub-step time −50%+ | 적당 (재학습) | 모델 변경 |

D는 이 브랜치 작업을 base로 가능하지만, 사전 분석상 효과 negative. A가 가장 ROI 높음.

## 브랜치 / PR

- Branch: https://github.com/pjs102793/vllm-omni/tree/feat/qwen3-tts-subtalker-inflight
- 3 commits:
  - `docs(qwen3-tts): subtalker in-flight design`
  - `feat(qwen3-tts): token-flat in-flight attention module`
  - `feat(qwen3-tts): wire in-flight token-flat path into CodePredictorWrapper`
