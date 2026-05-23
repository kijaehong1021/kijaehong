# FlashAttention

> Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)

---

## 1. 핵심 문제 의식

Standard attention의 병목은 **compute가 아니라 메모리 접근(IO)**이다.

```
Standard Attention 흐름:
  1. S = QK^T         → HBM에 write  [n × n]
  2. P = softmax(S)   → HBM에 write  [n × n]
  3. O = PV           → HBM에 write  [n × d]

n=4096, d=64 기준:
  → Attention matrix: 4096 × 4096 × 2bytes ≈ 32MB (fp16)
  → 이걸 매 layer마다 HBM read/write 반복
```

GPU 메모리 계층:
| 메모리 | 크기 | 대역폭 |
|--------|------|--------|
| HBM (GPU DRAM) | ~40–80GB | ~2 TB/s |
| SRAM (on-chip, L1/shared memory) | ~192KB per SM | ~19 TB/s |

**핵심 통찰:** n×n matrix를 HBM에 쓰지 않고, SRAM에서 tile 단위로 처리하면 IO를 대폭 줄일 수 있다.

---

## 2. 두 가지 핵심 기법

### 2-1. Tiling

전체 Q, K, V를 한 번에 처리하지 않고 블록 단위로 나눠서 처리.

```
Q를 블록으로 분할: Q = [Q_1, Q_2, ..., Q_T]   (각 블록: B_r × d)
K, V를 블록으로 분할: K = [K_1, ..., K_S]     (각 블록: B_c × d)

for each Q_i:
    for each K_j, V_j:
        S_ij = Q_i @ K_j^T           ← SRAM에서 계산
        O_i  = update(O_i, S_ij, V_j) ← SRAM에서 누적
    HBM에 O_i write                   ← 블록 완성 시 한 번만
```

→ HBM에 n×n attention matrix를 **절대 쓰지 않음**

### 2-2. Online Softmax (Recomputation)

Tiling의 문제: softmax는 행 전체를 봐야 정규화 가능 → 블록 처리가 불가능해 보임.

해결: **running max와 normalizer를 블록마다 업데이트**

```
블록 j를 처리할 때:
  m_new = max(m_old, rowmax(S_ij))          ← running max 갱신
  l_new = e^(m_old - m_new) * l_old + rowsum(e^(S_ij - m_new))  ← normalizer 갱신
  O_new = (l_old * e^(m_old - m_new) * O_old + e^(S_ij - m_new) * V_j) / l_new
```

마지막 블록 처리 후 O가 정확한 softmax 결과와 동일함 (수학적으로 exact).

---

## 3. IO 복잡도 비교

| | Standard Attention | FlashAttention |
|---|---|---|
| HBM reads | O(n·d + n²) | O(n²·d / M) |
| HBM writes | O(n²) | O(n·d) |
| SRAM 사용 | 거의 없음 | O(M) — tile fit |
| 메모리 사용 | O(n²) | **O(n)** |

M = SRAM 크기 (per SM)  
d = head dimension  
n >> M^(1/2) 일 때 FlashAttention이 유리 (실제로 항상 해당)

---

## 4. Backward Pass — Recomputation

Standard attention backward:
- Forward에서 저장한 n×n attention matrix P를 다시 읽어야 함 → O(n²) 메모리

FlashAttention backward:
- P를 저장하지 않음 → backward 시 **on-the-fly recomputation**
- Q, K, V (O(n·d))와 output O, softmax normalizer (l, m) 만 저장
- 계산량은 늘지만 메모리는 O(n)으로 유지

→ training에서도 긴 sequence가 가능해짐

---

## 5. LLM Serving에서의 의미

### Prefill 단계 (prompt 처리)
- 전체 prompt sequence를 한 번에 처리
- sequence가 길수록 standard attention은 OOM 위험
- FlashAttention: O(n) 메모리로 긴 context 처리 가능

### Decode 단계 (token 생성)
- 매 step마다 새 token 1개 vs 전체 KV cache
- n=1이라 attention matrix가 작음 → FlashAttention 이득 적음
- 대신 **KV Cache** 관리가 핵심 병목

### 실제 효과
| 항목 | 효과 |
|------|------|
| 메모리 사용 | 표준 대비 5–20x 감소 |
| 속도 | A100 기준 2–4x 빠름 |
| 최대 sequence 길이 | 8K → 64K+ 가능 |

---

## 6. FlashAttention-2에서 달라진 점 (간략)

| 항목 | v1 | v2 |
|------|----|----|
| Work partitioning | K, V를 outer loop | Q를 outer loop |
| Warp 간 통신 | 많음 | 최소화 |
| Non-matmul FLOPs | 많음 | 줄임 |
| GPU 활용률 | ~25% | ~50–70% |

v2의 핵심: Q를 outer loop로 바꿔서 warp 간 동기화 없이 독립적으로 처리 가능.

---

## 7. 한계

- **Causal mask + 짧은 sequence**: 절반이 masked → 실제 compute가 절반인데 tile은 전체 처리
- **GQA (Grouped Query Attention)**: K, V head 수가 Q보다 적은 경우 별도 최적화 필요
- **Custom hardware**: SRAM 크기가 다르면 tile size 재조정 필요

---

## 다음 단계

- [ ] FlashAttention-2 상세 분석 (work partitioning 수식)
- [ ] PagedAttention — KV cache 관리
- [ ] Continuous Batching과의 상호작용
