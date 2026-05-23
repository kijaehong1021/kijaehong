# Continuous Batching

> Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022)

---

## 1. 왜 batching이 필요한가?

GPU는 SIMD 머신 → batch가 클수록 throughput ↑

LLM serving의 특성:
- Request마다 generation 길이가 **다름** (10 tokens vs 2000 tokens)
- Decode는 한 step에 1 token씩 → 매우 짧은 step이 반복
- 단일 request만 처리하면 GPU 활용률 5–10%

---

## 2. 기존 Static Batching의 문제

### 2-1. Request-level batching

```
Time →
Batch [Req A (10 tok), Req B (1000 tok), Req C (50 tok)]

Step 1:  [A1][B1][C1]   ← 3개 동시 처리
Step 2:  [A2][B2][C2]
...
Step 10: [A10][B10][C10]  ← A 완료, 하지만 자리 차지
Step 11:      [B11][C11]  ← A 자리는 비어있음 (낭비)
...
Step 50:      [B50][C50]  ← C 완료
Step 51:      [B51]       ← B만 남음, GPU 거의 노는 상태
...
Step 1000:    [B1000]     ← 매우 비효율
```

**문제:**
- 가장 긴 request에 batch 전체가 묶임
- 짧은 request 완료 후에도 자리 점유
- 새 request는 batch가 끝날 때까지 대기

### 2-2. Padding 낭비

가변 길이 batch → padding token 추가 → padding에도 연산 수행

---

## 3. Continuous Batching (Iteration-level Scheduling)

Orca의 핵심: **iteration(token step) 단위로 batch 재구성**

```
Time →
Iter 1:  [A][B][C]
Iter 2:  [A][B][C]
...
Iter 10: [A][B][C]      ← A 완료
Iter 11: [B][C][D]      ← A 자리에 새 request D 즉시 투입
Iter 12: [B][C][D]
...
Iter 50: [B][D][E]      ← C 완료, E 투입
...
```

**핵심 변화:**
- 매 iteration마다 새 request 추가 / 완료된 request 제거 가능
- Batch가 항상 가득 차 있음 → GPU 활용률 최대
- 새 request의 대기 시간 ↓

### 3-1. Selective Batching

문제: prefill (n tokens 한번에) + decode (1 token)을 같은 batch에 넣으려면?

해결: **연산 단위로 다르게 처리**
- Attention: request마다 독립적으로 (다른 seq_len)
- Linear (Q/K/V projection, FFN): 모든 token을 하나로 concat해서 batch

```
Batch composition:
  Req A (prefill, 500 tokens)
  Req B (decode, 1 token)
  Req C (decode, 1 token)

Linear layer 입력: [502, d_model]  ← 한번에 처리
Attention: 각 request 따로 (다른 seq 길이)
```

---

## 4. Scheduling Policy

매 iteration에서 결정할 것:
1. 어떤 request를 batch에 포함?
2. 새 request prefill을 언제 끼워넣을까?
3. KV cache 부족 시 어떤 request preempt?

### 4-1. Prefill vs Decode 균형

Prefill은 compute-bound (긴 시퀀스, matmul 큼)
Decode는 memory-bound (KV cache load가 병목)

**같은 batch에 prefill + decode 섞으면**:
- Decode latency 증가 (prefill이 step 시간 지배)
- 하지만 throughput은 좋음

**전략:**
- vLLM: Prefill 우선 (긴 prefill은 chunked)
- TGI: Decode 우선 (낮은 latency 보장)
- SGLang: 적응형

### 4-2. Chunked Prefill

긴 prompt (10K tokens)를 한 번에 prefill하면 → decode latency spike

해결: prefill을 **chunk 단위 (예: 512 tokens)** 로 쪼개서 decode와 섞음

```
Iter N:   decode batch + chunk 1 of long prompt (512 tok)
Iter N+1: decode batch + chunk 2 of long prompt (512 tok)
...
Iter N+20: decode batch + chunk 20 (마지막)
Iter N+21: 새 request의 decode 시작
```

→ Decode latency 보장 + throughput 유지

### 4-3. Preemption 전략

KV cache full일 때:
- **Recompute**: request를 죽이고 나중에 prefill 다시
- **Swap**: KV blocks를 CPU memory로 옮김

vLLM은 짧은 request → recompute, 긴 request → swap

---

## 5. PagedAttention과의 시너지

Continuous batching의 전제: KV cache 메모리를 동적으로 추가/해제 가능해야 함
- 기존 연속 메모리 방식 → 불가능
- **PagedAttention** → 가능 (block 단위 할당)

즉:
- Orca (2022): continuous batching 제안
- vLLM (2023): PagedAttention + continuous batching → 본격 채택
- 이제는 모든 serving engine의 표준

---

## 6. 성능 영향

| 항목 | Static batching | Continuous batching |
|------|-----------------|---------------------|
| Throughput (tok/s) | 1x | **2–10x** |
| TTFT (Time to First Token) | 길어질 수 있음 | 짧음 (즉시 시작) |
| Tail latency | 매우 김 (긴 request 대기) | 안정적 |
| GPU 활용률 | 20–40% | **80%+** |

---

## 7. 실제 serving engine 비교

| Engine | Continuous batching | Chunked prefill | Scheduling |
|--------|---------------------|-----------------|------------|
| vLLM | ✓ | ✓ (V1) | priority + FCFS |
| TGI (HuggingFace) | ✓ | 제한적 | latency 우선 |
| TensorRT-LLM | ✓ (in-flight batching) | ✓ | 정책 설정 가능 |
| SGLang | ✓ | ✓ | 적응형 + RadixAttention |
| DeepSpeed-MII | ✓ (dynamic SplitFuse) | ✓ | throughput 우선 |

---

## 8. SLO/SLA 관점

LLM serving의 핵심 지표:
- **TTFT (Time To First Token)**: prefill 완료까지 — 사용자 대기 체감
- **TPOT (Time Per Output Token)**: token 간 간격 — streaming UX
- **Throughput (req/s, tok/s)**: 시스템 처리량

Trade-off:
- 큰 batch → throughput ↑, TPOT ↑ (각 step 느려짐)
- 작은 batch → throughput ↓, TPOT ↓
- Chunked prefill → TTFT와 TPOT 분리 제어 가능

---

## 다음 단계

- [[paged_attention_kv_cache]] — 메모리 관리 측면
- [[quantization_speculative_decoding]] — 처리량을 더 늘리는 다른 축
- Disaggregated serving (DistServe, Splitwise) — prefill/decode 분리
