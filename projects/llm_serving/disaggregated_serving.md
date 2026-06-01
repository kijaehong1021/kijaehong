# Disaggregated Serving (Prefill-Decode 분리)

> Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving" (OSDI 2024)
> Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024)

---

## 1. 왜 분리가 필요한가?

### Prefill과 Decode의 근본적 차이

| 특성 | Prefill | Decode |
|------|---------|--------|
| 처리 단위 | 전체 prompt (수백~수천 token) | 1 token/step |
| 연산 특성 | **Compute-bound** (큰 matmul) | **Memory-bound** (KV cache load) |
| GPU 활용 | Tensor Core 풀 가동 | 대역폭 병목, ALU 노는 상태 |
| 소요 시간 | 수십ms ~ 수초 (prompt 길이 비례) | 수ms/token (거의 일정) |
| 사용자 체감 | TTFT (Time To First Token) | TPOT (Time Per Output Token) |

### 같은 GPU에서 돌릴 때의 문제

Prefill과 Decode가 **같은 GPU에서 batching** 되면:

```
Iter N:   [decode req A] [decode req B] [prefill req C (1000 tokens)]
           ← decode: 수ms        ← prefill: 수백ms

결과: req A, B의 TPOT이 prefill 때문에 spike
      → streaming 중인 사용자가 뚝뚝 끊기는 경험
```

**근본 원인**: Prefill은 긴 시간을 독점 → decode의 step latency 불안정

Continuous batching의 chunked prefill이 부분 완화하지만:
- Chunk 크기 선택 어려움 (작으면 throughput ↓, 크면 latency spike)
- 결국 두 workload의 resource 요구가 다름 → 같은 GPU에서 완전한 해결 불가

---

## 2. Disaggregated Serving의 핵심 아이디어

**Prefill GPU와 Decode GPU를 물리적으로 분리**:

```
사용자 요청 흐름:

  Request
     │
     ▼
┌─────────────┐    KV cache 전송    ┌─────────────┐
│  Prefill    │ ──────────────────▶ │   Decode    │
│  Instance   │    (NVLink/RDMA)    │  Instance   │
│  (GPU pool) │                     │  (GPU pool) │
└─────────────┘                     └─────────────┘
  • 긴 matmul 최적화                • 많은 request 동시 처리
  • 높은 MFU 유지                   • KV cache 효율 관리
  • TTFT 최적화                     • TPOT 안정화
```

Prefill 완료 후 → KV cache를 Decode GPU로 전송 → Decode 시작

---

## 3. 각 단계의 최적화 방향

### Prefill Instance

목표: **TTFT 최소화**, MFU (Model FLOP Utilization) 최대화

최적 설정:
- 큰 batch size (compute-bound라 batch 커도 시간 큰 차이 없음)
- 더 많은 GPU 메모리를 KV cache 대신 **activation**에 할당
- Tensor Parallelism 적극 활용 (큰 matmul 분산)
- FlashAttention으로 long prompt 처리

### Decode Instance

목표: **TPOT 안정화**, 높은 throughput

최적 설정:
- 많은 request를 동시에 처리 (memory-bound → batch 넓힐수록 효율)
- 큰 KV cache pool (PagedAttention)
- Pipeline Parallelism 유리 (작은 matmul → TP 효율 낮음)
- Speculative Decoding 적용 용이

---

## 4. KV Cache 전송 — 핵심 기술 과제

Prefill → Decode로 KV cache를 이동해야 함:

```
KV cache 크기 (LLaMA-70B, prompt 2K token, fp16):
  2 × 80 layers × 8 kv_heads × 128 dim × 2K tokens × 2 bytes
  = 약 655 MB
```

전송 방식:
| 방식 | 대역폭 | 지연 |
|------|--------|------|
| PCIe (CPU 경유) | ~32 GB/s | 높음 |
| NVLink (같은 노드) | ~600 GB/s | 낮음 |
| RDMA (노드 간) | ~200 GB/s | 중간 |
| InfiniBand | ~400 GB/s | 낮음 |

**655MB @ NVLink 600GB/s → ~1ms** → 허용 가능  
**655MB @ PCIe 32GB/s → ~20ms** → TTFT에 큰 영향

→ **같은 노드 내 NVLink 활용**이 현실적으로 가장 중요.  
→ 노드 간은 RDMA (RoCE / InfiniBand) 필요.

### 전송 최적화

- **Pipeline overlap**: KV cache 전송과 decode 첫 step을 겹침
- **Chunked transfer**: layer별로 나눠서 전송 + 계산 overlap
- **Compression**: KV cache를 INT8/FP8로 줄여서 전송

---

## 5. DistServe의 접근 (OSDI 2024)

### Goodput 최적화

단순 throughput이 아니라 **SLO(Service Level Objective)를 만족하는 처리량** = Goodput

```
Goodput = SLO를 만족한 request 수 / 전체 시간
```

SLO 예: TTFT < 500ms, TPOT < 100ms/token

### Resource 분리 + 독립 스케일링

```
Prefill:Decode 비율을 workload에 맞게 조정

  Short prompt, long output  → Decode GPU 더 많이
  Long prompt, short output  → Prefill GPU 더 많이
  Balanced                   → 1:1
```

실험 결과 (논문):
- 기존 colocated 대비 goodput **3.8x** 향상
- TTFT P99 **10x** 감소
- TPOT P99 안정화

### 스케줄링

- Prefill queue → Decode pool로 request 라우팅
- KV cache 전송 완료 확인 후 decode 시작
- Decode instance의 KV cache pool 모니터링 → 부족 시 prefill 속도 조절

---

## 6. Splitwise의 접근 (ISCA 2024)

하드웨어 이질성 활용:
- **Prefill**: 고성능 GPU (A100/H100) — compute 집약
- **Decode**: 저비용 GPU (A10G 등) — bandwidth만 필요

```
비용 최적화:
  A100 (compute-heavy)    → Prefill 전용
  A10G (bandwidth-heavy)  → Decode 전용

  A100 1개 = A10G 3~4개 비용
  Decode는 A10G로도 충분 → 비용 절감
```

실험 결과:
- 동일 비용 대비 throughput **1.4x**
- Decode latency SLO 만족률 향상

---

## 7. 실제 구현 현황

| 시스템 | 분리 방식 | 상태 |
|--------|-----------|------|
| DistServe | 완전 분리 + RDMA | 연구 오픈소스 |
| Splitwise | 이질적 GPU | MS Azure 적용 |
| vLLM | disaggregated prefill 지원 시작 (v0.4+) | 프로덕션 |
| TensorRT-LLM | disaggregated 지원 | NVIDIA 공식 |
| LMDeploy | 실험적 지원 | — |
| Mooncake (월리엄 팀) | KV cache 중심 분리 | ByteDance |

---

## 8. 한계 / 트레이드오프

**단점:**
- KV cache 전송 overhead — 짧은 prompt에서 오히려 손해
- 시스템 복잡도 증가 (두 pool 관리, 라우팅, 전송 모니터링)
- 네트워크 장애 시 가용성 문제
- 작은 클러스터에서는 효과 제한적

**유리한 조건:**
- 긴 prompt (RAG, agent) + 긴 output → 두 단계 gap이 클 때
- 엄격한 TTFT / TPOT SLO가 모두 있을 때
- NVLink 또는 고속 interconnect가 있을 때
- 수백~수천 GPU 규모의 대형 클러스터

---

## 9. 연관 개념

### KV Cache Migration

Decode 중에 KV cache를 다른 GPU로 이동 (load balancing):
- 한 Decode instance 과부하 → 일부 request를 다른 instance로 live migration
- KV cache를 중간에 옮기면서 generation 계속 → 사용자는 모름

### Mooncake (2024)

KV cache를 **first-class citizen**으로 취급:
- KV cache만 따로 관리하는 분산 스토리지 레이어
- Prefill/Decode GPU 양쪽에서 read/write
- CPU DRAM도 KV cache tier로 활용

---

## 다음 단계

- [ ] Multi-GPU Parallelism (TP/PP/SP) — Disaggregated serving의 기반
- [ ] Mooncake 논문 분석
- [ ] vLLM disaggregated prefill 코드 분석
- [[continuous_batching]] — Disaggregation의 전 단계
- [[paged_attention_kv_cache]] — KV cache 전송의 기반
