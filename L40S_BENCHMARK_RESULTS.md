# Qwen3-TTS L40S 최적화 결과

> 환경: L40S (sm_89), bs=64 deploy, C2W skip — 모든 정밀 측정 `repeats=7, warmup=5, bucket-warmup=3`

**핵심 결과** — Fork 시점 코드 대비 throughput **1.21-1.75× 개선**. bs=64 medium에서 **857 → 1500 tok/s**. Sub-talker 6단계 최적화 + main talker FP8 누적.

| bs=1 (저배치) | bs=8 (중배치) | bs=32 | bs=64 (최고) |
|:---:|:---:|:---:|:---:|
| **1.29×** | **1.29×** | **1.52×** | **1.75×** |

## 📑 목차

1. [시작 — 어디가 병목이었나](#시작--어디가-병목이었나)
2. [Sub-talker 개선 (6단계)](#sub-talker-개선-6단계)
3. [Main talker FP8](#main-talker-fp8)
4. [최종 정밀 비교 (Fork → Final)](#최종-정밀-비교-fork--final)
5. [핵심 발견](#핵심-발견)
6. [부록 A — Sub-talker Architecture Sweep](#부록-a--sub-talker-architecture-sweep)
7. [부록 B — 측정 메소돌로지](#부록-b--측정-메소돌로지)

---

## 시작 — 어디가 병목이었나

Fork 시점 (commit `adb2291c`) 의 step time 분해 (bs=8 medium 기준, 약 22ms per main token):

```
[Sub-talker AR 38%] [Main talker 25%] [vLLM core 20%] [preprocess+sample 17%]
```

| 영역 | ms / step | % | 주요 작업 |
|---|---:|---:|---|
| **Sub-talker AR loop** | ~8.5 | 38% | 5L × 14 sub-steps decoder + sample + codec_embed × 13 |
| Main talker forward | ~5.3 | 25% | 28L × 2048H Qwen3 forward (1×) |
| vLLM core overhead | ~4.2 | 20% | engine loop, RPC, _prepare_inputs |
| preprocess + sample | ~2.9 | 17% | per-request loop, output post-process |

**병목은 sub-talker — step time의 38%**. 14 sub-steps × 5 layers × 7 GEMMs = 490 GEMM call per main token 의 누적이 크다. Main talker는 이미 vLLM의 `Qwen3Model` (QKVParallelLinear fused 등) 거치고 있어 추가 leverage 작음.

그래서 **sub-talker를 먼저 적극 최적화**한 뒤, 마지막에 main talker FP8 적용.

---

## Sub-talker 개선 (6단계)

각 단계는 **이전 단계 대비 추가 효과**를 보여줍니다. 모든 측정은 medium bucket bs=8 기준 RTF_agg (lower = faster).

### ① KV cache + CUDA graph `a5e1d5a0`

Sub-talker AR 루프에서 매 sub-step마다 전체 시퀀스를 re-prefill 하던 것을 KV cache로 변경. CUDA graph로 14개 sub-step 캡처.

| bucket | bs | 이전 (KV-off) | 이후 (KV-on) | speedup |
|---|---:|---:|---:|---:|
| short | 8 | 0.039 | 0.040 | 0.98× |
| short | 32 | 0.020 | 0.017 | **1.18×** |
| short | 64 | 0.016 | 0.014 | **1.14×** |
| medium | 32 | 0.017 | 0.015 | **1.13×** |
| medium | 64 | 0.013 | 0.012 | **1.08×** |
| long | 64 | 0.013 | 0.012 | **1.08×** |

FLOPs 15.9× 감소. 하지만 low batch에선 amortize 안 됨 (kernel launch overhead 지배).

### ② Batched preprocess `7332dc87`

매 step의 per-request preprocess Python loop (25.6 ms @ bs=64) 를 batched call로 융합 + inline merge.

| bucket | bs | 이전 | 이후 | speedup |
|---|---:|---:|---:|---:|
| short | 8 | 0.040 | 0.038 | **1.05×** |
| short | 32 | 0.017 | 0.015 | **1.13×** |
| short | 64 | 0.014 | 0.012 | **1.17×** |
| medium | 32 | 0.015 | 0.013 | **1.15×** |
| medium | 64 | 0.012 | 0.009 | **1.33×** |
| long | 64 | 0.012 | 0.009 | **1.33×** |

High batch (bs≥32) 에서 큰 win — preprocess는 bs에 비례 증가하니까. 두 단계 (① + ②) 합치면 bs=64 medium에서 **1.44×** (8% + 33%).

### ③ flashinfer top-K sampling `ae3053c8`

Sub-talker AR 의 매 sub-step sampling (top-K + multinomial) 을 `flashinfer.sampling.top_k_sampling_from_probs` 1-kernel로 교체.

| bucket | bs | 이전 | 이후 | speedup |
|---|---:|---:|---:|---:|
| short | 1 | 0.193 | 0.182 | **1.06×** |
| short | 8 | 0.038 | 0.035 | **1.09×** |
| medium | 1 | 0.183 | 0.171 | **1.07×** |
| medium | 8 | 0.033 | 0.032 | **1.03×** |
| medium | 16 | 0.021 | 0.019 | **1.11×** |
| medium | 64 | 0.009 | 0.009 | *saturated* |

마이크로벤치는 1.79×지만 서버는 5-10% — sampling이 sub-talker time의 약 16% 차지하므로.

### ④ QKV + Gate/Up fusion `e8f088b9`

Sub-talker의 별도 q/k/v Linear 3개 → 단일 `qkv_proj`, gate/up 2개 → 단일 `gate_up_proj`. Plain `nn.Linear` 위에 weight concat 만 적용 (**bit-exact, max diff 0**).

| bucket | bs | 이전 | 이후 | speedup |
|---|---:|---:|---:|---:|
| short | 1 | 0.182 | 0.178 | **1.02×** |
| short | 8 | 0.035 | 0.034 | **1.03×** |
| medium | 2 | 0.103 | 0.097 | **1.06×** |
| medium | 4 | 0.056 | 0.053 | **1.06×** |
| medium | 8 | 0.032 | 0.030 | **1.07×** |
| medium | 64 | 0.009 | 0.009 | *saturated* |

QKV fusion 자체는 마이크로벤치 2.6× 빠르지만 layer total은 그 중 일부 (Linear가 layer time의 ~30%).

### ⑤ SDPA priority fix — cuDNN dispatch `e48390ea`

PyTorch SDPA가 sub-talker decode에서 항상 MATH로 fallback되던 3중 gate 진단:

- **Gate 1**: `enable_gqa=True` + `attn_mask` → 모든 fused backend 거부
- **Gate 2**: PyTorch priority order에서 MATH가 CUDNN 앞
- **Gate 3**: vLLM이 cudnn_sdp 전역 disable (diffusers 호환)

해결: 수동 K/V expand + `sdpa_kernel(list, set_priority=True)`. SDPA backend 단독 시간 (decode @ bs=8): MATH 30.3μs → **CUDNN 11.5μs (-62%)**.

| bucket | bs | 이전 | 이후 | speedup |
|---|---:|---:|---:|---:|
| short | 1 | 0.178 | 0.176 | **1.01×** |
| medium | 1 | 0.166 | 0.164 | **1.01×** |
| medium | 4 | 0.053 | 0.053 | 1.00× |
| medium | 8 | 0.030 | 0.031 | 0.97× |
| medium | 32 | 0.014 | 0.012 | **1.17×** |

SDPA가 sub-talker의 ~10% 뿐이라 step time 영향 1-3% 수준. 다만 진단 자체가 의미 있음 — 다른 모델도 같은 문제 가능.

### ⑥ Sub-talker 누적 효과

위 5단계 모두 적용 후 (fork → SDPA priority fix까지). 다음 단계 FP8 적용 직전 상태:

| bucket | bs | fork RTF | sub-talker만 후 | sub-talker speedup |
|---|---:|---:|---:|---:|
| short | 1 | 0.196 | 0.176 | **1.11×** |
| short | 8 | 0.038 | 0.033 | **1.15×** |
| medium | 1 | 0.184 | 0.164 | **1.12×** |
| medium | 8 | 0.034 | 0.031 | **1.10×** |
| medium | 32 | 0.017 | 0.012 | **1.42×** |
| medium | 64 | 0.014 | 0.009 | **1.56×** |

sub-talker 자체만으로 high batch에서 큰 win (medium bs=64에서 1.56×). 저배치 (bs=1) 에서는 1.11× 정도.

---

## Main talker FP8 `5a660d31`

Sub-talker가 작아진 뒤, 이제 main talker가 상대적으로 큰 비중. vLLM의 `Fp8OnlineLinearMethod` 활성화 — deploy yaml에 `quantization: fp8` 한 줄 추가.

| 구성 | 값 |
|---|---|
| Weight | bf16 → fp8 e4m3fn (1회 online quant) |
| Activation | dynamic per-tensor scale (매 forward) |
| Compute | CutlassFP8ScaledMM (Ada sm_89) |
| Output | bf16 |
| Sub-talker 영향 | 없음 — plain `nn.Linear` 사용해서 vLLM LinearBase 거치지 않음 |

| bucket | bs | 이전 (BF16) | 이후 (FP8) | speedup |
|---|---:|---:|---:|---:|
| short | 1 | 0.176 | 0.152 | **1.16×** |
| short | 4 | 0.058 | 0.051 | **1.14×** |
| short | 8 | 0.033 | 0.029 | **1.14×** |
| medium | 1 | 0.164 | 0.143 | **1.15×** |
| medium | 4 | 0.053 | 0.045 | **1.18×** |
| medium | 8 | 0.031 | 0.028 | **1.11×** |
| medium | 64 | 0.009 | **0.008** | **1.13× (saturation 해제)** |
| long | 64 | 0.009 | **0.008** | **1.13×** |

모든 batch에서 일관 10-18% throughput. 이전에 bs=64 medium/long의 ceiling이던 1333 tok/s가 **1500 tok/s 로 돌파** — main talker forward가 진짜 saturation 원인이었음.

> **주의 — Amdahl 효과**: sub-talker 최적화로 step time이 줄어든 상태에서 FP8를 적용했기 때문에 % speedup이 크게 보임. FP8가 main talker에서 절감하는 **절대 시간 (~2-3 ms)** 는 sub-talker 최적화 여부와 무관하므로, fork state에서 FP8만 적용했다면 더 작은 %로 보였을 것. 그래도 절대 효과는 동일.

---

## 최종 정밀 비교 (Fork → Final)

apples-to-apples — 양쪽 모두 동일한 bench 조건 `--repeats 7 --warmup 5 --bucket-warmup 3`.

| bucket | bs | fork RTF | final RTF | speedup | tok/s 변화 |
|---|---:|---:|---:|---:|---:|
| short | 1 | 0.196 | 0.152 | **1.29×** | 61 → 79 |
| short | 2 | 0.117 | 0.091 | **1.29×** | 103 → 132 |
| short | 4 | 0.068 | 0.051 | **1.33×** | 176 → 235 |
| short | 8 | 0.038 | 0.029 | **1.31×** | 316 → 414 |
| short | 16 | 0.025 | 0.019 | **1.32×** | 480 → 632 |
| short | 32 | 0.019 | 0.013 | **1.46×** | 632 → 923 |
| short | 64 | 0.016 | 0.011 | **1.45×** | 750 → 1091 |
| medium | 1 | 0.184 | 0.143 | **1.29×** | 65 → 84 |
| medium | 2 | 0.107 | 0.081 | **1.32×** | 112 → 148 |
| medium | 4 | 0.062 | 0.045 | **1.38×** | 194 → 267 |
| medium | 8 | 0.034 | 0.028 | **1.21×** | 353 → 429 |
| medium | 16 | 0.022 | 0.017 | **1.29×** | 545 → 706 |
| medium | 32 | 0.017 | 0.011 | **1.55×** | 706 → 1091 |
| medium | 64 | 0.014 | 0.008 | **1.75×** | **857 → 1500** |
| long | 1 | 0.182 | 0.141 | **1.29×** | 66 → 85 |
| long | 2 | 0.102 | 0.082 | **1.24×** | 118 → 146 |
| long | 4 | 0.061 | 0.046 | **1.33×** | 197 → 261 |
| long | 8 | 0.035 | 0.026 | **1.35×** | 343 → 462 |
| long | 16 | 0.023 | 0.017 | **1.35×** | 522 → 706 |
| long | 32 | 0.017 | 0.011 | **1.55×** | 706 → 1091 |
| long | 64 | 0.013 | 0.008 | **1.62×** | 923 → 1500 |

### Stage별 누적 commit

| Commit | 단계 | 기여 영역 |
|---|---|---|
| `a5e1d5a0` | ① KV cache + CUDA graph | high batch (특히 bs=32+) |
| `7332dc87` | ② Batched preprocess | high batch (bs≥32) |
| `ae3053c8` | ③ flashinfer sampling | 모든 bs (5-10%) |
| `e8f088b9` | ④ QKV + Gate/Up fusion | 모든 bs (3-6%, bit-exact) |
| `e48390ea` | ⑤ SDPA priority fix | 작음 (1-3%), 진단 가치 큼 |
| `5a660d31` | ⑥ Main talker FP8 | 모든 bs (10-18%), bs=64 saturation 해제 |

---

## 핵심 발견

### vLLM SDPA가 cuDNN을 못 쓰는 진짜 이유

PyTorch 2.5+가 default로 cuDNN SDPA를 쓰는데 sub-talker에서는 항상 MATH로 fallback. 3중 gate:

1. `enable_gqa=True` + `attn_mask` → fused kernel 모두 거부
2. PyTorch priority order에서 MATH가 CUDNN 앞에 위치
3. vLLM 전역 `enable_cudnn_sdp(False)` (diffusers 호환)

해결: `sdpa_kernel(list, set_priority=True)`로 임시 enable + priority 재배열, context exit 시 글로벌 상태 복원.

### Sub-talker quantization은 안 됨

| 시도 | 결과 |
|---|---|
| torchao Int8WeightOnly | 2-3× 느림 (dequant fallback) |
| torchao Float8WeightOnly | 5% 느림 |
| torchao FP8DynamicActW8A8 | 50-100% 느림 |
| vLLM cutlass_scaled_mm | wrapper overhead 25µs/call로 무용 |
| 순수 CUTLASS FP8 | 빠르지만 PyTorch 통합 비용 큼 |

원인: sub-talker matrix (~1024×3072) 가 Ada FP8 tensor core sweet spot 미달. Main talker (2048H, 14336 MLP) 는 반대로 FP8 효과적.

### 왜 main talker FP8가 가장 ROI 좋은 PR 후보인가

| Commit | 코드 변경 (LOC) | Throughput 효과 | ROI |
|---|---|---|---|
| `5a660d31` Main FP8 | 1줄 (yaml + 주석) | **+10-18% 모든 bs** | 🏆 최고 |
| `7332dc87` Batched preprocess | 327+/37- | **+14-25% high batch** | 중-상 |
| `ae3053c8` flashinfer | 67+/20- | +5-10% | 중 |
| `e8f088b9` Fusion | 254+/29- | +3-6% | 중 |
| `e48390ea` SDPA priority fix | 33+/32- | +1-3% | 낮음 (효과) |
| `a5e1d5a0` KV cache | 619+/8- | +5-10% | 낮음 (양 큼) |

### 남은 leverage (변경 못 한 것들)

| 영역 | 잠재 효과 | 노력 |
|---|---|---|
| Sub-talker AR loop 전체 단일 graph | 3-7% | 큼 |
| Speculative decoding | 잠재 50%+ | 매우 큼 |
| vLLM `_prepare_inputs` upstream patch | 5-10% | 중 |
| H100/H200 upgrade | 2-3× | hardware |

---

## 부록 A — Sub-talker Architecture Sweep

> 실험: sub-talker를 wider+shallower (H=2048) 변형하면 L40S에서 더 빠른가?
> 측정: `QWEN3_TTS_SUBTALKER_LAYERS/HIDDEN/INTERMEDIATE` env로 config 오버라이드, random weights (audio 품질 무의미, profile only).

### bs=64 throughput (main tokens/sec)

| Config | short | medium | long |
|---|---:|---:|---:|
| **5L × 1024 × 3072** (baseline) 🏆 | **857** | **1333** | **1333** |
| 3L × 2048 × 3072 | 545 | 632 | 667 |
| 4L × 2048 × 3072 | 571 | 522 | 600 |
| 5L × 2048 × 3072 | 545 | 545 | 500 |

**결론**: H=2048 변형은 throughput 30-45% 손해. weight 2.6×↑가 BW-bound L40S에 그대로 반영. Layer 수 감소 (5→3) 만으로 H=2048 GEMM 증가분 못 따라잡음 — **wider+shallower 가설 L40S에서 성립 안 함**. 현재 5L × 1024H 가 L40S에서 가장 효율적 architecture.

---

## 부록 B — 측정 메소돌로지

- **측정 도구**: `bench_qwen3_tts.py` — httpx async 클라이언트로 `/v1/audio/speech` POST 스트리밍
- **측정 조건**: 정밀 측정은 모두 `--repeats 7 --warmup 5 --bucket-warmup 3`
- **모드**: C2W skip (audio decode 생략, sub-talker compute 비중 isolate)
- **RTF_agg**: `wall_total / sum(audio_durations)` — length-normalized
- **Throughput**: `12 / RTF_agg` (codec 12Hz → main tokens per second)
- **Hardware**: L40S (sm_89, Ada Lovelace, 46GB VRAM, 864 GB/s memory BW)
- **모델**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, voice `Vivian`
- **Buckets**: short (~12 chars, ~2.4s audio) / medium (~45, ~9.5s) / long (~120, ~30s)
