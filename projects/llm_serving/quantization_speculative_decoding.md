# Quantization & Speculative Decoding

LLM serving에서 throughput과 latency를 줄이는 두 가지 큰 축:
1. **Quantization** — 모델/KV cache를 저정밀도로 표현 → 메모리·연산 절약
2. **Speculative Decoding** — 작은 모델로 미리 추측해서 큰 모델은 검증만 → step 수 절약

---

## 1. Quantization

### 1-1. 왜 필요한가?

LLM의 병목:
- **Memory bandwidth**: decode 시 weight를 HBM에서 매번 load
- **Memory capacity**: 70B 모델 FP16 = 140GB → 한 GPU에 안 들어감
- **Compute**: 큰 batch에서는 matmul도 병목

Quantization 효과:
| Precision | 메모리 | 처리량 (matmul) | 정확도 |
|-----------|--------|-----------------|--------|
| FP16 / BF16 | 1x | 1x | baseline |
| FP8 (H100) | 0.5x | 2x | ~baseline |
| INT8 | 0.5x | 2x | 약간 손실 |
| INT4 | 0.25x | 4x | 손실 있음 |

### 1-2. Weight-only vs Activation Quantization

**Weight-only (W4A16, W8A16)**:
- Weight만 양자화, activation은 FP16 유지
- 양자화된 weight를 load 후 dequantize → FP16 matmul
- 메모리 대역폭 절감 위주 (decode에 유리)
- 대표: **AWQ, GPTQ**

**Weight + Activation (W8A8)**:
- Activation도 양자화 → INT8 matmul (Tensor Core 활용)
- 처리량 직접 증가 (prefill에 유리)
- 대표: **SmoothQuant, FP8**

### 1-3. 주요 방법들

#### GPTQ (2022)
- 사후 양자화 (Post-Training Quantization)
- Hessian 기반으로 양자화 오차 최소화
- INT4 weight + FP16 activation
- 정확도 손실 적음

#### AWQ (Activation-aware Weight Quantization)
- 핵심 관찰: weight의 **중요도는 activation 크기에 비례**
- 큰 activation에 대응하는 weight channel만 보호 (scaling)
- INT4에서도 정확도 거의 유지
- 현재 가장 널리 쓰이는 weight-only 방법

#### SmoothQuant
- Activation outlier 문제 해결
- 어려운 activation의 크기를 weight로 "이전"
- W8A8 INT8 가능

#### FP8 (H100)
- E4M3 / E5M2 두 가지 포맷
- Per-tensor 또는 per-block scaling
- 정확도 가장 좋음 (INT8보다)
- vLLM/TRT-LLM 모두 지원

### 1-4. KV Cache Quantization

KV cache는 long context에서 weight보다 큼 → 양자화 효과 큼

```
LLaMA-3 70B, context 128K:
  - FP16 KV: ~40GB
  - FP8 KV:  ~20GB
  - INT4 KV: ~10GB
```

**구현:**
- vLLM `--kv-cache-dtype fp8` / `fp8_e5m2`
- TensorRT-LLM의 KV cache INT8
- 정확도 영향 거의 없음 (특히 FP8)

### 1-5. Quantization의 한계

- Activation outlier 처리가 어려운 layer 존재 (예: FFN의 일부)
- INT4에서 long context 성능 저하 가능
- Calibration dataset 의존성
- 양자화/역양자화 overhead (custom kernel 필요)

---

## 2. Speculative Decoding

### 2-1. 핵심 아이디어

Autoregressive decode의 본질적 한계:
- Token N은 token N-1이 정해져야 계산 가능 → **sequential**
- 매 step마다 큰 모델 forward → 비쌈

**아이디어**:
- 작은 **draft model**로 K개 token을 빠르게 생성
- 큰 **target model**로 K개를 한 번에 검증 (parallel)
- 맞으면 채택, 틀리면 거기서부터 다시

```
Step 1: Draft model이 5 tokens 추측: [t1, t2, t3, t4, t5]
Step 2: Target model이 1번의 forward로 [t1..t5] 확률 분포 모두 계산
Step 3: 처음 3개까지 일치 → t1,t2,t3 채택
        t4가 거부 → t4를 target model 분포로 재샘플링
        t5 버림
결과: 1번의 target forward로 4 tokens 생성 (1배 → 4배 속도)
```

### 2-2. 왜 빠른가?

Target model의 forward 1회 비용 ≈ K tokens 한꺼번에 처리 비용
- Decode는 memory-bound → batch K로 처리해도 시간 거의 동일
- KV cache load는 1번만, matmul만 K배

→ Draft 정확도 70%면 평균 ~3 tokens / step → **3x speedup**

### 2-3. 수학적 보증

**중요**: 단순한 휴리스틱이 아니라 **확률 분포가 target과 동일함이 증명됨**
(rejection sampling)

```
p = target model 분포
q = draft model 분포

draft가 t를 제안할 때:
  p(t) >= q(t) → 항상 채택
  p(t) <  q(t) → 확률 p(t)/q(t) 로 채택, 아니면 거부

거부 시: (p - q)+ 분포에서 재샘플링
```

→ 출력 분포가 target만 쓴 것과 정확히 동일 (greedy든 sampling이든)

### 2-4. Draft model 선택

| 방법 | 설명 | 속도 향상 |
|------|------|----------|
| Small LM | 같은 family의 작은 모델 (LLaMA-7B for 70B) | 2–3x |
| n-gram | prompt 기반 통계 기반 | 1.5–2x |
| Self-speculative | 큰 모델의 일부 layer만 사용 | 1.5–2x |
| Medusa | 추가 head 학습 (multi-token) | 2–3x |
| EAGLE | feature-level prediction | **3–4x** |

### 2-5. Variants

#### Medusa
- 기존 LM head 외에 K개의 추가 head 학습
- 각 head가 다음, 다음다음, ... token을 예측
- Tree-based 검증으로 여러 후보 평가
- 모델 fine-tuning 필요

#### EAGLE
- Feature(hidden state) level에서 예측
- 단순 token 예측보다 정확
- Token level speculative보다 일관되게 빠름

#### Lookahead Decoding
- Draft model 없이 self-speculation
- Jacobi iteration 활용
- 단일 모델로 동작 (배포 간단)

### 2-6. Serving 시 고려사항

**Batch size의 영향**:
- Batch가 작을 때 (memory-bound): speculative 효과 큼 (2–4x)
- Batch가 클 때 (compute-bound): 효과 감소 또는 역효과
- → low-traffic 시간에 자동 활성화 같은 정책 가능

**Acceptance rate**:
- Draft와 target 분포 차이가 크면 acceptance ↓
- Domain-specific fine-tuning으로 개선 가능

---

## 3. Combined Strategies

실제 serving에서는 여러 기법을 함께 사용:

```
LLaMA-3-70B serving 예시:
  - Model: AWQ INT4 (메모리 70GB → 18GB)
  - KV cache: FP8 (메모리 절반)
  - Attention: FlashAttention-3
  - Batching: continuous + chunked prefill
  - Speculation: EAGLE draft (small head)
  - Result: 단일 H100에서 70B 모델 서빙 가능, 5–10x throughput
```

---

## 4. Trade-off 요약

| 기법 | 메모리 ↓ | Throughput ↑ | Latency ↓ | 정확도 ↓ | 구현 복잡도 |
|------|----------|--------------|-----------|----------|-------------|
| FP8 weight | ✓ | ✓ | - | 미미 | 중 |
| INT4 (AWQ) | ✓✓ | ✓ | - | 약간 | 중 |
| FP8 KV cache | ✓ | ✓ | - | 거의 없음 | 낮음 |
| Speculative decoding | - | - (batch 작을 때 ↑) | ✓✓ | 없음 | 높음 |
| Medusa/EAGLE | - | - | ✓✓ | 없음 | 높음 (학습 필요) |

---

## 5. 한계 / 향후 방향

**Quantization**:
- 1-bit / 2-bit (BitNet 등) 활발히 연구 중
- Mixed precision (layer마다 다른 정밀도)
- Per-token / per-group scaling 최적화

**Speculative**:
- Multi-model speculation (cascade)
- 입력에 따라 draft 모델 선택
- Long context에서의 효율 (KV cache 공유)

---

## 다음 단계

- [[flash_attention_v2_v3]] FP8 attention과의 연계
- [[paged_attention_kv_cache]] KV quantization 상세
- [[continuous_batching]] speculative와 batch scheduling 상호작용
