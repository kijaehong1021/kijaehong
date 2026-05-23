# Attention 메커니즘 기초 이론

## 1. 왜 Attention인가?

RNN/LSTM의 한계:
- 시퀀스를 순차적으로 처리 → GPU 병렬화 불가
- Long-range dependency: 거리가 멀수록 정보 손실
- Fixed-size hidden state에 모든 정보를 압축해야 함

Attention은 이를 해결: **모든 위치 간 직접 연결**, 병렬 계산 가능

---

## 2. Scaled Dot-Product Attention

### 핵심 수식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 구성 요소

| 기호 | 의미 | 크기 |
|------|------|------|
| Q (Query) | 현재 토큰이 "무엇을 찾는가" | [seq_len, d_k] |
| K (Key) | 각 토큰이 "무엇을 제공하는가" | [seq_len, d_k] |
| V (Value) | 실제로 가져올 정보 | [seq_len, d_v] |

### 단계별 계산

```
1. Score 계산:   S = QK^T         → [seq_len, seq_len]
2. 스케일링:     S / sqrt(d_k)    → 수치 안정성 확보
3. Softmax:      A = softmax(S)   → [seq_len, seq_len]  (attention weight)
4. 가중합:       O = AV           → [seq_len, d_v]
```

### sqrt(d_k)로 나누는 이유

d_k가 클수록 내적값의 분산이 커짐:
- 분산이 크면 softmax의 gradient가 극단적으로 작아짐 (vanishing gradient)
- d_k 차원의 벡터 내적의 분산 = d_k → 표준편차 = sqrt(d_k)
- 나눠주면 분산을 1로 정규화

---

## 3. Multi-Head Attention (MHA)

### 수식

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 왜 여러 head?

단일 attention은 하나의 관계만 포착. 여러 head를 사용하면:
- head 1: 문법적 의존 관계
- head 2: 의미적 유사성
- head 3: 위치 정보
- ... 각 head가 다른 subspace에서 다른 패턴 학습

### 파라미터

- d_model = 전체 임베딩 차원 (e.g., 512)
- h = head 수 (e.g., 8)
- d_k = d_v = d_model / h = 64

W_i^Q, W_i^K ∈ R^{d_model × d_k}  
W_i^V ∈ R^{d_model × d_v}  
W^O ∈ R^{hd_v × d_model}

---

## 4. 복잡도 분석 — GPU 최적화의 핵심 문제

### 시간 복잡도

| 연산 | 복잡도 |
|------|--------|
| QK^T 계산 | O(n² · d_k) |
| Softmax | O(n²) |
| AV 계산 | O(n² · d_v) |
| **전체** | **O(n² · d)** |

n = 시퀀스 길이, d = 차원

### 메모리 복잡도 — 진짜 문제

```
Attention matrix S = QK^T 의 크기: [n, n]

n = 1,000   → 1M elements  →    4MB (fp32)
n = 8,000   → 64M elements →  256MB
n = 32,000  → 1B elements  →    4GB  ← GPU VRAM 한계 초과!
n = 100,000 → 10B elements →   40GB  ← 불가능
```

**이것이 FlashAttention이 해결하는 핵심 문제:**  
n×n attention matrix를 HBM에 materialize하지 않고 계산

### 메모리 대역폭 병목

GPU 연산의 병목은 보통 compute가 아니라 **메모리 접근**:
- HBM(High Bandwidth Memory) bandwidth: ~2TB/s
- SRAM(on-chip) bandwidth: ~19TB/s (10x 빠름)
- Attention: QK^T, softmax, AV 매번 HBM read/write → 느림

---

## 5. Self-Attention vs Cross-Attention

### Self-Attention
Q, K, V 모두 같은 시퀀스에서 생성  
→ 시퀀스 내 토큰 간 관계 포착 (Encoder에서 사용)

### Cross-Attention
Q는 한 시퀀스, K/V는 다른 시퀀스  
→ 번역: Q=target 언어 토큰, K/V=source 언어 토큰

### Causal (Masked) Self-Attention
미래 토큰을 보지 못하도록 mask 적용:
```
mask[i][j] = -inf  if j > i
           =    0  otherwise
```
LLM의 autoregressive 생성에 사용 (Decoder)

---

## 6. Transformer 전체 구조에서의 위치

```
Input Embedding
      ↓
[Positional Encoding]
      ↓
┌─────────────────────┐
│  Encoder Block × N  │
│  ┌───────────────┐  │
│  │ Multi-Head    │  │
│  │ Self-Attention│  │
│  └───────────────┘  │
│  Add & LayerNorm     │
│  ┌───────────────┐  │
│  │  Feed Forward │  │
│  └───────────────┘  │
│  Add & LayerNorm     │
└─────────────────────┘
      ↓
    Output
```

---

## 7. GPU 최적화 관점에서의 핵심 포인트

| 문제 | 원인 | 해결 방향 |
|------|------|-----------|
| O(n²) 메모리 | Attention matrix 저장 | Tiling / FlashAttention |
| HBM bandwidth 병목 | 반복적 메모리 접근 | Kernel fusion, SRAM 활용 |
| Softmax의 sequential 특성 | 전체 row를 봐야 함 | Online softmax (numerically stable) |
| Causal mask | 절반이 -inf | 효율적 mask 표현 |

---

## 다음 단계

- [ ] Online Softmax 알고리즘 이해
- [ ] FlashAttention v1 논문 분석
- [ ] Tiling 전략과 SRAM 활용
- [ ] FlashAttention-2 개선점
