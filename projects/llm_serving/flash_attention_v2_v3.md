# FlashAttention-2 & FlashAttention-3

> v2: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
> v3: Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024)

---

## 1. FlashAttention v1의 한계

v1은 메모리 문제는 해결했지만 GPU 활용률이 낮음 (A100 기준 ~25% FLOPs utilization).

원인:
- **Work partitioning이 비효율적**: K, V를 outer loop로 → warp 간 동기화 필요
- **Non-matmul FLOPs 비중이 큼**: softmax 관련 연산이 matmul 대비 느림
- **Causal mask 처리 비효율**: masked 영역도 계산 후 버림

GPU의 Tensor Core matmul은 non-matmul보다 16x 빠름 → non-matmul 줄이는 게 핵심.

---

## 2. FlashAttention-2의 개선

### 2-1. Loop 순서 변경 (Q를 outer로)

```
v1:                          v2:
for j in K_blocks:           for i in Q_blocks:
    for i in Q_blocks:           for j in K_blocks:
        S = Q_i @ K_j^T              S = Q_i @ K_j^T
        update O_i                   update O_i
        (warp 간 통신 필요)          (warp 독립 처리 가능)
```

**효과:**
- 각 warp가 자신의 Q block에 대해 독립적으로 작업
- Inter-warp 동기화/통신 불필요
- Shared memory 사용량 감소

### 2-2. Non-matmul FLOPs 감소

Online softmax 계산에서 매 iteration마다 rescaling하던 것을:
- 매번 나누지 않고 **누적만** 하다가 **마지막에 한 번만** 정규화

```
v1: O_new = (l_old * e^(...) * O_old + e^(S) * V) / l_new   ← 매번 나눔
v2: O_unnorm 누적 후 마지막에 1번 나눔                       ← 나눗셈 1회
```

### 2-3. Sequence 차원 병렬화

v1: batch × num_heads로만 병렬 → 작은 batch에서 SM 활용 못함
v2: batch × num_heads × **seq_len**으로 병렬 → 긴 sequence에서도 SM 가득 활용

### 2-4. 결과

| 항목 | v1 | v2 |
|------|----|----|
| FLOPs utilization (A100) | ~25% | ~50–73% |
| 속도 (긴 seq, fp16) | 1x | **~2x** |
| 표준 attention 대비 | 2–4x | 5–9x |

---

## 3. FlashAttention-3 (H100 전용)

H100의 새 기능을 적극 활용:
- **Tensor Memory Accelerator (TMA)**: 비동기 메모리 전송
- **WGMMA (Warp Group MMA)**: warp 4개가 함께 더 큰 matmul
- **FP8 support**: 처리량 2x

### 3-1. Warp Specialization

기존: 모든 warp가 같은 일을 함 (data parallel)
v3: warp를 역할별로 나눔
- **Producer warps**: TMA로 K, V 비동기 load
- **Consumer warps**: GEMM 계산
- **Softmax warps**: softmax 처리

→ Producer가 다음 tile을 load하는 동안 Consumer가 현재 tile 계산 (pipeline)

### 3-2. GEMM과 Softmax overlap

이전: GEMM 끝 → softmax 시작 → GEMM 시작 (sequential)
v3: 두 개의 GEMM 사이에 softmax를 끼워넣어 **ping-pong** 실행

```
시간 →
GEMM-0 (QK^T) | softmax-0    | GEMM-1 (PV)
              | GEMM-2 (next QK^T) | softmax-1   | ...
```

### 3-3. FP8 지원

- FP8 matmul: H100에서 BF16 대비 2x 처리량
- Outlier 처리: per-block scaling + incoherent processing (Hadamard transform)
- Block quantization으로 정확도 유지

### 3-4. 결과

| 항목 | v2 (FP16) | v3 (FP16) | v3 (FP8) |
|------|-----------|-----------|----------|
| H100 FLOPs utilization | ~35% | **~75%** | ~75% |
| 속도 (vs v2 FP16) | 1x | 1.5–2x | **3x** |
| 정확도 손실 (FP8) | - | - | 2.6x lower error than baseline FP8 |

---

## 4. LLM Serving 관점

### 어느 단계에 영향?

| 단계 | 영향 |
|------|------|
| Prefill (긴 prompt) | **매우 큼** — sequence 길이 n²에 비례 |
| Decode (token 1개) | 작음 — n=1이라 KV cache 접근이 병목 |

→ Long context serving (RAG, agent, 32K+) 에서 prefill 속도가 곧 사용자 latency

### 실제 채택 현황

- vLLM, TGI, SGLang 등 주요 inference engine이 FlashAttention 기본 사용
- FA3는 H100 환경에서 자동 선택
- A100 환경은 FA2까지만 사용 가능 (H100 명령어 필요)

---

## 5. 한계 / 트레이드오프

- **GQA/MQA**: head 수가 적으면 병렬화 이득 제한
- **Variable length batching**: padding 없이 처리하려면 추가 logic (FA varlen API)
- **Custom mask** (sliding window 등): 직접 kernel 수정 필요
- **Backward pass**: forward만큼 최적화 안 됨 (training보다 inference 우선)

---

## 다음 단계

- [[paged_attention_kv_cache]] — KV cache 메모리 관리
- [[continuous_batching]] — request scheduling
- FA3 + FP8 실제 정확도 측정
