# PagedAttention & KV Cache 관리

> Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, SOSP 2023)

---

## 1. KV Cache가 뭐고 왜 문제인가?

### KV Cache의 정의

Autoregressive decoding에서 매 token마다 attention을 다시 계산하면:
- 이전 token들의 K, V를 매번 다시 projection → 중복 계산
- 해결: **K, V를 한 번 계산하고 메모리에 저장 (cache)** → 재사용

```
Decode step t:
  Q_t = x_t @ W_Q                              ← 새 token만 계산
  K_t, V_t = x_t @ W_K, x_t @ W_V              ← 새 token만 계산
  KV_cache = concat(KV_cache, [K_t, V_t])      ← cache에 append
  O = softmax(Q_t @ KV_cache.K^T) @ KV_cache.V ← 전체 cache 사용
```

### 메모리 크기 계산

```
KV cache 크기 = 2 (K,V) × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

LLaMA-7B 기준 (32 layers, 32 heads, 128 dim, fp16):
- token 1개당: 2 × 32 × 32 × 128 × 2 = **512 KB/token**
- context 2048: **1 GB**
- context 32K: **16 GB**  ← 모델 weight(13GB)보다 큼!

### 기존 시스템의 문제

기존 (FasterTransformer, Orca 등): **연속된 메모리 블록**으로 KV cache 관리

```
Request A (max_len=2048 예약): ████████████████░░░░░░░░░░░░  ← 실제 사용 500, 낭비 1548
Request B (max_len=2048 예약): ████████░░░░░░░░░░░░░░░░░░░░  ← 실제 사용 100, 낭비 1948
Request C (max_len=2048 예약):                                ← 메모리 없어서 못 받음
```

**낭비의 3가지 유형:**
1. **Reserved fragmentation**: 미래 token용으로 미리 예약한 빈 공간
2. **Internal fragmentation**: max_len 예약했는데 짧게 끝남
3. **External fragmentation**: 다양한 크기 할당으로 빈 공간 파편화

→ 실제 측정: **20–40%만 활용**, 60–80% 낭비

---

## 2. PagedAttention의 아이디어

OS의 **virtual memory + paging** 개념을 차용:
- KV cache를 **고정 크기 block(page)** 단위로 관리
- 논리적으로는 연속, 물리적으로는 비연속
- block table이 logical → physical 매핑

### 구조

```
Logical KV blocks (per request):
  [block 0][block 1][block 2][block 3]...

Block table (per request):
  logical 0 → physical 7
  logical 1 → physical 2
  logical 2 → physical 15
  ...

Physical KV cache pool (GPU memory):
  [block 0][block 1][block 2]...[block N]   ← 모든 request가 공유
```

각 block은 고정 크기 (예: 16 tokens)
→ Internal fragmentation 최대 16 tokens로 제한

### PagedAttention kernel

기존 FlashAttention은 연속 메모리 가정 → block table로 indirection 추가:

```
for each query token:
    for each logical block i:
        physical_block_id = block_table[i]
        K_block = kv_cache[physical_block_id].K
        V_block = kv_cache[physical_block_id].V
        ... (FlashAttention 로직과 동일)
```

추가 cost는 작음 (block table lookup), 메모리 절약 효과는 큼.

---

## 3. PagedAttention의 추가 이점

### 3-1. Memory sharing — Parallel sampling

같은 prompt에서 여러 답변 sampling (beam search, parallel sampling):
- prompt 부분의 KV는 **모든 sample이 공유** 가능 (copy-on-write)
- 답변 생성 시 갈라지는 부분만 복사

```
Prompt blocks: [P0][P1][P2]  ← 1번만 저장, 모든 sample이 참조
Sample 1:                [S1_0][S1_1]...
Sample 2:                [S2_0][S2_1]...
Sample 3:                [S3_0][S3_1]...
```

메모리 55% 절약 (논문 측정)

### 3-2. Prefix caching

여러 request가 같은 system prompt를 공유:
- system prompt의 KV block을 캐시
- 같은 prefix 요청은 prefill 단계에서 재사용

vLLM의 `--enable-prefix-caching` 옵션으로 활성화

### 3-3. Preemption / Swapping

GPU 메모리 부족 시:
- 일부 request의 KV blocks를 CPU memory로 swap out
- 또는 recomputation (다시 prefill)
- 새 request 받고, 나중에 swap in

---

## 4. 성능 영향

| 항목 | 기존 시스템 | vLLM (PagedAttention) |
|------|-------------|----------------------|
| KV memory 활용률 | 20–40% | **96%+** |
| Throughput | 1x | **2–4x** |
| Latency (p50) | 1x | 비슷하거나 낮음 |
| 동시 처리 가능 requests | 적음 | 많음 |

특히 **batch size를 크게 가져갈 수 있어** throughput이 크게 증가.

---

## 5. KV Cache 관련 최적화 트렌드

### 5-1. KV Cache Quantization
- KV를 INT8/INT4로 저장 → 메모리 2–4x 절약
- FP8 KV cache (H100): 정확도 손실 거의 없음
- vLLM, TensorRT-LLM 모두 지원

### 5-2. GQA / MQA (모델 구조 차원)
- **MQA (Multi-Query)**: 모든 head가 같은 K, V 공유 → KV cache 1/h 배
- **GQA (Grouped Query)**: 그룹 단위 공유 → MQA의 정확도 손실 보완
- LLaMA-2 70B, LLaMA-3는 GQA 채택

### 5-3. KV Cache Compression / Eviction
- **H2O**: 중요한 token만 cache에 유지 (heavy hitters)
- **StreamingLLM**: sink tokens + recent window
- **SnapKV**: prompt-aware compression

### 5-4. Disaggregated Serving
- Prefill과 decode를 다른 GPU에서 실행
- KV cache를 NVLink/RDMA로 전송
- DistServe, Splitwise 등 — latency/throughput SLA 최적화

---

## 6. 한계 / 주의점

- **Block size tradeoff**: 작으면 lookup overhead ↑, 크면 internal frag ↑ (보통 16)
- **Custom CUDA kernel 필요**: standard PyTorch에서는 직접 구현 어려움
- **Prefix caching cache invalidation**: 동일 prefix 판단 비용
- **Swap I/O 비용**: CPU↔GPU 전송이 느리면 latency spike

---

## 다음 단계

- [[continuous_batching]] — PagedAttention과 함께 작동
- [[quantization_speculative_decoding]] — KV quantization 상세
- vLLM 소스코드: `vllm/attention/backends/`
