# Qwen3-TTS Sub-talker In-flight Batching — Design

## Goal

`feat/qwen3-tts-kv-cache-poc` 브랜치에서 KV cache + CUDA graph로 sub-talker self
speedup 1.36-2.5× (micro-bench)를 얻었지만, real server wall time은 1.3× 부근에서
plateau. 분석상 원인은 두 가지:

1. **Sub-talker의 14 AR step이 lock-step**: 한 `talker_mtp` 호출 안에서 모든
   request가 같은 sub-step에 있음. 다른 request의 다른 sub-step과 batched 안 됨.
2. **Sub-talker가 main talker step과 강하게 동기**: `talker_mtp` 14 step이 한
   `model.forward` 안에서 atomic하게 처리. 다음 main talker step은 sub-talker
   14 step이 끝나야 시작.

이 doc은 (1)을 푸는 방향 — **request 간 다른 sub-step phase를 한 sub-talker
forward에 batched** (in-flight). 즉 한 sub-talker call에 prefill(step 1)과
decode(step 2..14)가 섞여서 처리되는 구조.

## 현재 lockstep 구조

```
gpu_model_runner._preprocess (per main talker step):
  for req in active_decode_reqs:           # span_len == 1 인 req들
      preprocess(req)                       # talker_mtp_inputs 수집
  talker_mtp_forward(decode_req_ids):
      audio_codes = code_predictor(...)     # batched, internally Q-1=14 step lock
        for step in range(1, Q):            # ← 모든 req가 같은 step
            graph[bucket].replay()          #   step별 dedicated graph
        return audio_codes                  # [B, Q]
```

특징:
- talker_mtp_forward는 main_fwd 직후 한 번 호출. 그 안에서 14 step atomic.
- 모든 active decode req가 같은 sub-step에 있음 → cache_len, attn_mask 동일 →
  단일 graph (per bucket) replay 가능.

## In-flight 후 구조 (목표)

```
gpu_model_runner._preprocess (per main talker step):
  for req in active_decode_reqs:
      preprocess(req)
      if req has fresh main token:
          subtalker_scheduler.enqueue(req)  # 새 sub-talker 시작 (cache_len=0)
  
  # subtalker_scheduler가 1 step씩 진행
  subtalker_scheduler.step():
      flat_inputs = []                       # token-flat
      slot_indices = []
      cache_lens = []
      for slot in active_slots:
          flat_inputs.append(slot.next_input)
          slot_indices.append(slot.cache_slot_id)
          cache_lens.append(slot.cache_len)
          slot.cache_len += 1
      graph[N=len(slots)].replay()           # 한 forward에 N tokens flat
      
      # done 검출 → audio_codes 완료 → main talker enqueue
      for slot in active_slots:
          if slot.cache_len == Q:
              slot.return_audio_codes()
              subtalker_scheduler.dequeue(slot)
```

특징:
- 한 sub-talker forward에 N reqs × 1 token. cache_len 다양.
- attention: 각 token이 자기 cache slot의 valid prefix attend.
- request의 main token rate은 같음 (14 main+sub interleaved steps per main token).
- 다만 한 시점 active sub-talker reqs = N (다양한 sub-step phase 있음).

## Throughput 효과 분석 (사전 정량)

### 단일 request throughput
- Lock-step (현재): 1 main token per `main_fwd + 14 × sub_step` time.
- In-flight: 1 main token per `14 × (main_fwd + sub_step_each)` time.
- main_fwd가 sub_step_each보다 크면 (실측 main ~2ms, sub_step ~0.4ms),
  **in-flight이 14× main_fwd 더 들어 손해 가능**.
- main_fwd << sub_step이면 비슷.

### N concurrent requests throughput
- Lock-step: `N main tokens per (main_fwd_N + 14 × sub_step_N)` time.
  - 여기서 sub_step_N은 batch N이라 amortize.
- In-flight: `N main tokens per (14 × main_fwd_N + 14 × sub_step_N')` time.
  - sub_step_N'은 N (다양한 phase) batched, batch dim 자체는 N.

→ in-flight의 sub_step batch dim도 N (운영 max_num_seqs)으로 변화 없음.
   GEMM 자체 시간 same. 다양성(phase 다른 cache_len)만 늘어남.

### 결론
**Throughput 측면에서 in-flight 자체가 wall time을 줄이는 lever는 미미**할 가능성
높음. 진짜 lever는:
- (A) main talker와 sub-talker forward를 **다른 CUDA stream에 동시 진행**
- (B) sub-talker batch dim 자체를 **multi-step amortization으로 키우기** (단,
      multi-stream 없이는 sequential이라 의미 작음)
- (C) **Attention의 cache_len 다양성이 GPU utilization에 미치는 영향**: 평균
      cache_len 같지만 multi-shape이라 cudagraph 캡처 측면 손해 가능

→ Goal "속도 개선이 명확하게 있어야"는 본 in-flight 구현만으로 보장 어려움.

다만 다음 잠재 이점이 있을 수 있음:
1. **Long-tail latency 개선**: lock-step에선 batch에 1개 슬로우 req가 있으면 다른
   reqs도 14 step 대기. in-flight은 fast req가 빠르게 main talker 진행 가능 (단,
   본 architecture에서도 main_fwd_n+1이 sub_fwd_n 14 step 완료 wait이라 동일).
2. **CPU overhead 감소**: 14 step lockstep의 Python AR loop이 1 step씩 amortize
   되며 vLLM scheduler에 흡수.
3. **Pipeline parallelism이 가능한 ground work**: 현 architecture에서 multi-stream
   확장 가능한 기반 마련.

## 구현 단계

### Step 1: Token-flat KV cache + attention (이 doc)
새 모듈 `qwen3_code_predictor_inflight.py`:
- `CodePredictorAttentionInflight`: token-flat input, per-token slot_idx + cache_len
- gather-based SDPA: 각 token이 cache[slot_idx]의 [:cache_len+1] attend
- bool attn_mask: `[N, max_seq=17]`, valid prefix marking

### Step 2: Sub-talker scheduler
새 module 또는 `qwen3_tts_talker.py` 확장:
- per-request state: `(cache_slot_id, cache_len, last_code, next_input_embed)`
- enqueue (main talker step end): new slot, cache_len=0
- step (per main talker step): batched forward over active slots
- dequeue (cache_len == Q): audio_codes 완료, main talker 다음 token enqueue

### Step 3: gpu_model_runner 통합
`_talker_mtp_forward` 변경:
- 14 step atomic loop 제거
- 1 step씩 호출, scheduler가 state 관리
- main talker step end → 그 request의 새 sub-talker enqueue

### Step 4: Graph capture
bucket별로 `(N_max, prefill_or_decode)` 두 graph:
- N_max ∈ [1, 2, 4, 8, 16, 32, 64, 128]
- prefill: N tokens, all with cache_len=0~1, slot_idx 다양
- decode: N tokens, cache_len 1~Q-1, slot_idx 다양

### Step 5: 음질 검증
- 동일 prompt + greedy → 동일 token sequence
- DNSMOS perceptual: sampling 모드 stat 분포 비교

## 음질 보장 전략

In-flight architecture에서 numerics drift cascade 더 복잡해질 가능성:
- Token timing (한 main step에 처리되는 sub-talker reqs 조합)이 매번 다를 수 있음
- Cache layout 바뀜 (per-request slot index 다름)
- Attention path 변경 (gather vs slice)

대책:
- 동일 prompt + 동일 sub-talker seed → 동일 sub-talker output 보장 (deterministic)
- 음질 측정: `audio_quality.py`로 KV-cache PoC 결과와 strict 비교

## Acceptance Criteria

1. **속도**: long bs=64 (C2W skip) RTF agg ≤ 0.007 (현재 PoC 0.007). 동등 또는
   개선.
2. **음질**: DNSMOS OVRL ≥ 3.0 (sampling 모드). KV cache PoC와 통계적 동등.
3. **torch.compile 호환**: main talker compile + sub-talker compile + 새 graph
   capture 동시 동작.
4. **별도 브랜치**: `feat/qwen3-tts-subtalker-inflight`.

## 진행 상황 (initial commit)
- [x] Design doc
- [ ] Step 1: token-flat attention module
- [ ] Step 2: per-request scheduler
- [ ] Step 3: gpu_model_runner integration
- [ ] Step 4: graph capture
- [ ] Step 5: bench + quality verification
