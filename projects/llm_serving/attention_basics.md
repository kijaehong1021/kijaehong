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

### 수식적 정식화 — Head를 정확히 무엇이 정의하는가

#### 1. Tensor shape의 흐름

입력 $X \in \mathbb{R}^{n \times d_{\text{model}}}$ (n개 token, 각 d_model 차원)

**Step 1.** Linear projection으로 Q, K, V 생성:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

여기서 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$

→ $Q, K, V \in \mathbb{R}^{n \times d_{\text{model}}}$

**Step 2.** Head별로 분할 (reshape):

$$Q \to [Q_1, Q_2, \ldots, Q_h], \quad Q_i \in \mathbb{R}^{n \times d_k}$$

이때 $d_k = d_{\text{model}} / h$. **수학적으로 이건 W^Q를 block partition한 것과 동등:**

$$W^Q = [W_1^Q \mid W_2^Q \mid \cdots \mid W_h^Q]$$

각 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$. 따라서:

$$Q_i = X W_i^Q$$

즉 **head i는 W^Q의 i번째 column block에 의해 정의되는 subspace** 로의 projection.

#### 2. Per-head 계산

각 head는 독립적인 scaled dot-product attention:

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{n \times d_v}$$

여기서 핵심: **softmax는 head별로 따로 적용**. 이게 비선형성을 만들어내고, head들이 서로 다른 패턴을 학습하는 이유.

#### 3. Concat + Output projection

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

여기서 $W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$. W^O도 block partition 가능:

$$W^O = \begin{bmatrix} W_1^O \\ W_2^O \\ \vdots \\ W_h^O \end{bmatrix}, \quad W_i^O \in \mathbb{R}^{d_v \times d_{\text{model}}}$$

이 형태로 쓰면:

$$\text{MHA}(X) = \sum_{i=1}^{h} \text{head}_i \cdot W_i^O$$

**→ MHA는 h개의 head 출력의 합** (concat+행렬곱 = 부분곱의 합). 각 head는 **출력에 독립적으로 기여**한다는 게 수식적으로 명확해짐.

#### 4. Rank 관점에서의 해석

단일 head의 출력 $\text{head}_i \cdot W_i^O \in \mathbb{R}^{n \times d_{\text{model}}}$

이건 다음 두 곱의 결과:
- $\text{head}_i$: rank ≤ $d_v$
- $W_i^O \in \mathbb{R}^{d_v \times d_{\text{model}}}$: rank ≤ $d_v$

→ 한 head의 기여는 **최대 rank $d_v$짜리 행렬**

전체 MHA 출력 = $h$개의 rank-$d_v$ 행렬의 합 → **최대 rank $h \cdot d_v = d_{\text{model}}$**

**중요한 통찰:**
- 단일 head로 d_model rank를 만들려면 $d_k = d_{\text{model}}$ 필요 → 비용 h배
- Multi-head는 **저랭크 행렬들의 합으로 full-rank 변환** 구성 → 효율적

이건 **low-rank decomposition** (LoRA 등)과 같은 원리:
- $W \approx \sum_i u_i v_i^T$ 처럼 rank가 낮은 항들의 합으로 풍부한 변환 표현

#### 5. Softmax가 없으면? — Multi-head의 본질

만약 attention에서 softmax를 제거하면:

$$\widetilde{\text{head}}_i = \frac{Q_i K_i^T}{\sqrt{d_k}} V_i = \frac{1}{\sqrt{d_k}} (X W_i^Q) (X W_i^K)^T (X W_i^V)$$

이건 $X$에 대한 **선형 함수**. 여러 head를 합쳐도:

$$\sum_i \widetilde{\text{head}}_i \cdot W_i^O = X \cdot (\text{어떤 큰 선형 변환})$$

→ 단일 head로 만들 수 있는 것과 표현력이 같아짐.

**Softmax의 비선형성이 head 다양성의 원천이다.** Head별로 softmax가 다른 위치에 peak를 만들기 때문에, 합쳐도 단일 head로 환원 불가.

#### 6. Head 간 직교성? — 학습된 분리

이론적으로는 $W_i^Q$들이 직교하도록 제약하지 않음 (학습으로 결정).

실제 학습된 모델 분석:
- Head들의 attention pattern은 **서로 다른** 경향 (cosine similarity 낮음)
- Layer 깊이에 따라 다양화 정도 변화 (얕은 layer: 비슷, 깊은 layer: 다양)
- Redundant head는 prune해도 성능 유지 (Voita et al. 2019, Michel et al. 2019)

#### 7. 구체적 dimension 예시 (LLaMA-7B)

| 변수 | 값 |
|------|-----|
| $d_{\text{model}}$ | 4096 |
| $h$ | 32 |
| $d_k = d_v$ | 128 |
| $W^Q$ shape | [4096, 4096] |
| 각 $W_i^Q$ shape | [4096, 128] |
| $Q$ (reshape 전) | [n, 4096] |
| $Q_i$ (head 분할 후) | [n, 128] |
| Attention map per head | [n, n] |
| head_i 출력 | [n, 128] |
| Concat 출력 | [n, 4096] |
| MHA 최종 출력 | [n, 4096] |

#### 8. Forward 계산 그래프 (정확한 수식)

각 token 위치 $t$에서, head $i$의 출력:

$$\text{head}_i[t] = \sum_{s=1}^{n} \alpha_i^{t,s} \cdot V_i[s]$$

attention weight:

$$\alpha_i^{t,s} = \frac{\exp(Q_i[t] \cdot K_i[s] / \sqrt{d_k})}{\sum_{s'=1}^{n} \exp(Q_i[t] \cdot K_i[s'] / \sqrt{d_k})}$$

최종 출력 (token t에서):

$$\text{MHA}[t] = \sum_{i=1}^{h} \text{head}_i[t] \cdot W_i^O$$

→ 한 token의 출력은 (h heads) × (n positions) 만큼의 가중합으로 결정됨.

---

### 왜 여러 head를 쓰는가? — 동기와 직관

#### 동기 1: Softmax의 "단일 모드" 한계

Attention의 핵심 연산은 softmax:

$$A = \text{softmax}(QK^T / \sqrt{d_k})$$

Softmax는 본질적으로 **하나의 확률 분포**를 만든다. 즉, 한 query token은 **하나의 "포커스"** 만 가질 수 있다 — 보통 한두 개의 token에 weight가 몰림.

예: "The animal didn't cross the street because **it** was too tired"
- "it"이 무엇을 가리키는지 알려면 → "animal"에 집중해야 함
- 동시에 "it"의 문법적 역할(주어)을 알려면 → "was"에도 집중해야 함
- 동시에 "tired"의 인과 관계를 보려면 → "because"에도 집중해야 함

**단일 attention은 이 셋을 동시에 못 봄.** Softmax가 가중치를 분산시키면 어느 것도 제대로 포착 못함 (averaging 문제).

→ **해결**: 여러 개의 독립된 softmax를 병렬로 → 각자 다른 곳을 봄

#### 동기 2: Subspace 분리 학습

단일 attention은 d_model 차원 전체에서 하나의 Q, K projection을 만든다:
- "이 token이 무엇을 찾는가"를 **한 가지 방식**으로만 정의

Multi-head는 d_model을 h개의 작은 subspace로 분리하고, **각 subspace마다 독자적인 Q, K, V projection 학습**:
- Head 1의 W_1^Q는 "문법적 관계를 위한 query 변환"을 학습
- Head 2의 W_2^Q는 "의미적 유사성을 위한 query 변환"을 학습
- ...

각 head는 **표현 공간의 다른 측면**을 본다.

#### 동기 3: Ensemble 효과

여러 head의 출력을 concat → W^O로 projection
- 본질적으로 h개의 작은 attention의 **앙상블**
- 한 head가 실수해도 다른 head가 보정 가능
- Gradient도 여러 path로 흘러 안정적

#### 실제로 학습된 head 패턴 (BertViz, Clark et al. 2019)

연구로 관찰된 head 역할:
| Head 유형 | 학습된 패턴 |
|-----------|-------------|
| Positional heads | 직전/직후 token에 집중 (local context) |
| Syntactic heads | 동사 → 주어, 명사 → 수식어 등 의존 관계 |
| Coreference heads | 대명사 → 선행사 (it → animal) |
| Delimiter heads | [CLS], [SEP], punctuation에 집중 |
| Rare token heads | 드문 token (특이 단어)에 집중 |
| "No-op" heads | [SEP] 토큰 등에 모이는 비활성 head |

흥미로운 관찰:
- 일부 head는 **prune 가능** (성능 손실 없이 제거 가능) → 모든 head가 유용하진 않음
- Head 수는 모델 크기에 비례 (LLaMA-7B: 32, LLaMA-70B: 64)

---

### 왜 d_k = d_model / h 인가? — 계산 비용 일정 유지

핵심 관찰: 단일 attention with full d_model vs. h-head with d_k=d_model/h는 **계산량이 거의 동일**

```
단일 attention (d=d_model):
  QK^T: O(n² · d_model)
  AV:   O(n² · d_model)

h-head MHA (d_k=d_model/h, h개 head 합):
  per head: O(n² · d_model/h)
  h heads:  O(n² · d_model)   ← 동일!
```

**즉, 같은 compute budget으로 표현력을 다양화**하는 것이 multi-head의 핵심 트릭.

만약 d_k를 그대로 d_model로 두면:
- 계산량 h배 증가
- 같은 자원이면 단일 head를 더 깊게 쌓는 게 나음

### 파라미터 정리

- d_model = 전체 임베딩 차원 (예: 4096 in LLaMA-7B)
- h = head 수 (예: 32)
- d_k = d_v = d_model / h (예: 128)

W_i^Q, W_i^K ∈ R^{d_model × d_k}  
W_i^V ∈ R^{d_model × d_v}  
W^O ∈ R^{h·d_v × d_model}

실제 구현은 head별로 따로 matmul하지 않고:
- 하나의 큰 W^Q ∈ R^{d_model × d_model}로 한 번에 projection
- 결과를 reshape: [n, d_model] → [n, h, d_k]
- 이렇게 하면 단일 matmul로 모든 head 처리 → GPU 효율적

---

### Head 수 늘리기의 한계

| h가 작을 때 (예: 1–4) | h가 클 때 (예: 64+) |
|----------------------|---------------------|
| 표현 다양성 부족 | d_k 너무 작아 표현력 부족 |
| 위에서 본 averaging 문제 | head별 softmax noise 증가 |
| 단일 패턴만 학습 | 중복된 head 많아짐 |

경험적 sweet spot: d_k = 64~128, h = d_model / d_k

이것이 **MQA / GQA의 동기**:
- 모든 head가 정말 필요한가? K, V는 공유해도 되지 않을까?
- → 다음 단계에서 다룸

---

### MHA → MQA → GQA 진화

KV cache 메모리 문제로 인한 변종:

| 구조 | Q heads | K, V heads | 메모리 | 품질 |
|------|---------|------------|--------|------|
| MHA (원조) | h | h | 1x | baseline |
| MQA (Multi-Query) | h | **1** | 1/h | 떨어짐 |
| GQA (Grouped Query) | h | h/g (예: g=8) | 1/g | ~baseline |

- **MQA**: 모든 query head가 **같은 K, V** 공유 → KV cache 극단적 감소, 품질 손실
- **GQA**: g개 query를 한 그룹으로 묶어 K, V 공유 → 균형
- LLaMA-2/3 70B는 GQA 채택 (h=64, kv_heads=8 → 8개 그룹)

→ LLM serving에서 KV cache 크기는 직접적인 throughput 제약
→ 자세한 내용은 [[paged_attention_kv_cache]]

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
