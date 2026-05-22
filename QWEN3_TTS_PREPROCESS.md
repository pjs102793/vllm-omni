# Qwen3-TTS Preprocess 처리 방식

Qwen3-TTS의 preprocess는 두 가지 경로로 처리됩니다.

- **비배치 (per-request) 경로** — 기존 경로, prefill / mixed batch에서 사용
- **배치 (all-decode fast path) 경로** — 커밋 `7332dc87`에서 추가된 AR decode 최적화 경로

호출 진입점은 `vllm_omni/worker/gpu_model_runner.py`의 `_preprocess`이며, 조건에 따라 두 경로 중 하나로 분기됩니다.

---

## 1. 비배치 (per-request) 경로 — `preprocess()`

**정의**: `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:541`
`Qwen3TTSTalkerForConditionalGeneration.preprocess()`

요청 한 개씩 호출되며, `span_len`에 따라 두 분기를 갖습니다.

### Prefill (`span_len > 1`) — line 579–650

- **첫 prefill 청크**
  - `_build_prompt_embeds()`로 전체 prompt embedding 생성
  - CPU에 캐시(`embed.prefill`), `tts_pad_embed` / `trailing_text` 등은 GPU에 보관
- **이후 prefill 청크**
  - 저장된 CPU embedding을 `talker_prefill_offset`으로 슬라이스
  - 길이가 모자라면 `tts_pad_embed`로 패딩
- placeholder 정렬을 위해 `input_ids`를 `codec_pad_id`로 덮어쓰고, `codes.audio` 자리표시자 zeros를 채워 반환

### Decode (`span_len == 1`) — line 652–684

- `trailing_text` 큐에서 한 스텝의 텍스트 벡터를 pop, 없으면 `tts_pad_embed`
- postprocess가 남긴 `hidden_states.last`를 `past_hidden`으로 사용
- 새 토큰을 embed → `inputs_embeds_out` 생성
- `mtp_inputs=(past_hidden, text_step)`를 update_dict에 실어 반환 → 러너가 `talker_mtp_*` 버퍼에 복사

### 호출 측 (slow path)

`vllm_omni/worker/gpu_model_runner.py:1389-1419` — `self.input_batch.req_ids`를 순회하면서 요청마다 `self.model.preprocess(...)`를 호출하고, 결과를 한 줄씩 `talker_mtp_*` 버퍼에 복사합니다.

---

## 2. 배치 (all-decode fast path) — `batch_preprocess_decode()`

**정의**: `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py:686`
**커밋**: `7332dc87 feat(qwen3-tts): batched preprocess fast-path + inline merge for AR decode`

### 활성 조건 (`gpu_model_runner.py:1317-1323`)

다음을 모두 만족해야 fast path가 켜집니다.

- `has_talker_mtp == True`
- 배치 내 **모든 요청이 decode** (`num_scheduled_tokens == 1`)
- 환경변수 `QWEN3_TTS_DISABLE_BATCH_PREPROCESS=1`로 끄지 않음
- 모델에 `batch_preprocess_decode` 메서드가 정의돼 있음

### per-row 루프 대비 핵심 차이

1. **배치 embed lookup** — `talker.py:712-713`
   - `embed_input_ids`를 `[N, 1]`에 한 번에 호출 (per-row 호출 N회 → 1회)
2. **`torch.stack` 한 번으로 배치 텐서 구성** — `talker.py:772-773`
   - `last_hidden`, `tts_pad` / `trailing_text` 참조는 이미 GPU에 bf16으로 상주 (`gpu_resident_buffer_keys`)
   - Python loop는 참조만 수집, `torch.stack`이 `[N, H]` 두 텐서를 한 번에 생성
   - per-row `.to()` / `.reshape()` 호출 전부 제거
3. **per-request 캐시** — `talker.py:724-748`
   - `additional_information`의 flatten 결과를 `info_dict["_flat"]`에 저장
   - `codec_streaming` bool도 `info_dict["_cs"]`에 저장 → 매 step 재계산 안 함
4. **Bulk `.copy_()`** — `gpu_model_runner.py:1351-1354`
   - 결과 `{input_ids, embeds, last_hidden, text_step, info_updates}`를 4개의 `talker_mtp_*` GPU 버퍼에 bulk copy (4N → 4)
5. **Inline merge** — `gpu_model_runner.py:1366-1385`
   - `_update_intermediate_buffer`의 함수 호출 / `gpu_keys` 체크 / `_store_value` 간접화를 우회
   - `trailing_text` + `codec_streaming`만 직접 dict 업데이트

### 출력 레이아웃

```python
{
    "input_ids":   Tensor[N],
    "embeds":      Tensor[N, H],
    "last_hidden": Tensor[N, H],
    "text_step":   Tensor[N, H],
    "info_updates": list[dict],  # 길이 N, runner 측에서 buffer에 merge
}
```

---

## 3. 폴백 동작

다음 경우에는 자동으로 per-request `preprocess()` 루프 (`gpu_model_runner.py:1389-1419`) 로 떨어집니다.

- Prefill이 포함된 배치
- prefill + decode가 섞인 배치
- `batch_preprocess_decode` 메서드가 없는 모델

즉, **AR decode 루프(대부분의 step)** 에서만 fast path가 켜집니다.

---

## 4. 성능 (커밋 7332dc87 측정값)

L40S, 1.7B-CustomVoice / bs=64 yaml / C2W skip:

| cell           | KV-on baseline | batched fast-path | Δ       |
|----------------|----------------|-------------------|---------|
| short  bs=64   | RTF 0.014 / 1.80s  | RTF 0.012 / 1.51s   | −16%    |
| medium bs=64   | RTF 0.012 / 6.21s  | RTF 0.009 / 4.80s   | −23%    |
| long   bs=64   | RTF 0.012 / 19.46s | RTF 0.009 / 14.87s  | −24%    |

배치 크기에 비례해 효과가 커집니다 (amortize되는 것이 per-row Python loop이기 때문):

- bs ≤ 4: 노이즈 수준
- bs = 32: ~12–15%
- bs = 64: ~14–25%

---

## 5. 확장 가능성

다른 omni TTS 모델 (`fish_speech`, `qwen3_omni`, `voxcpm2`, `voxtral_tts`, `covo_audio`, `mimo_audio`, `qwen2_5_omni`)도 같은 `has_preprocess=True` 훅을 갖고 있어 자체 `batch_preprocess_decode`만 추가하면 러너의 fast path를 그대로 재사용할 수 있습니다.
