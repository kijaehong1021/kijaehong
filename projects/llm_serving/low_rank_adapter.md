# Low-Rank Adapter (LoRA) & PEFT

> Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
> Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023)

---

## 1. 왜 Full Fine-tuning이 문제인가?

LLaMA-70B를 full fine-tuning:
- 파라미터 수: 70B
- fp16 기준 gradient + optimizer state: **~420GB** (Adam: weight × 3)
- A100 80GB GPU → **5개 이상** 필요

사전학습된 LLM을 task에 맞게 쓰려면 더 효율적인 방법이 필요 → **PEFT (Parameter-Efficient Fine-Tuning)**.

---

## 2. Rank란 무엇인가?

LoRA를 이해하려면 먼저 행렬의 rank 개념이 필요.

### 직관: 행렬이 담은 실질적 독립 정보의 수

```
A = [1  2  3]     ← 3개 열이 있지만...
    [2  4  6]       2행 = 1행 × 2  (종속)
    [3  6  9]       3행 = 1행 × 3  (종속)

→ 독립적인 행은 1개뿐  →  rank(A) = 1
```

```
B = [1  0  0]
    [0  1  0]       모든 행이 서로 독립
    [0  0  1]

→ rank(B) = 3  (full rank)
```

### 기하학적 의미

rank = **이 행렬이 입력 벡터를 몇 차원 공간으로 보내는가**

```
rank 1 행렬: 어떤 입력이 들어와도 출력은 항상 한 직선 위
rank 2 행렬: 출력이 2차원 평면 위
rank r 행렬: 출력이 r차원 subspace에 놓임
full rank:   출력이 d차원 전체를 채움
```

### rank가 낮은 행렬 = 두 작은 행렬의 곱

rank-r 행렬은 항상 두 개의 작은 행렬 곱으로 분해 가능:

```
M ∈ R^{d × d},  rank(M) = r

M = B · A    where  B ∈ R^{d × r},  A ∈ R^{r × d}

파라미터 수:
  M:   d × d  =  4096 × 4096  =  16,777,216
  B+A: 2 × d × r  =  2 × 4096 × 8  =  65,536   (r=8)
```

이게 SVD(특이값 분해)와 같은 원리:

```
M ≈ U · Σ_r · V^T

  U   ∈ R^{d × r}   ← B에 해당
  Σ_r ∈ R^{r × r}   ← 스케일 (α/r에 흡수)
  V^T ∈ R^{r × d}   ← A에 해당
```

---

## 3. LoRA의 핵심 아이디어

### 관찰: Weight 변화량은 low-rank

사전학습 모델을 fine-tuning할 때 weight 변화 ΔW는 **rank가 낮다** (실험적으로 관찰).

ΔW의 특이값(singular value) 분포를 보면:

```
ΔW ∈ R^{4096 × 4096}   ← 이론상 최대 rank 4096

실제 fine-tuning 후 특이값 분포:
  σ₁ = 1.23  ←  크다 (중요한 방향)
  σ₂ = 0.87
  σ₃ = 0.45
  σ₄ = 0.12
  σ₅ = 0.003  ←  거의 0 (무의미)
  σ₆ = 0.001
  ...
  σ₄₀₉₆ ≈ 0

→ 의미 있는 singular value는 앞의 몇 개뿐
→ ΔW는 실질적으로 low-rank
→ BA로 근사해도 정보 손실이 거의 없음
```

**LoRA의 핵심 가정**:
> "70B 파라미터를 다 바꿀 필요 없이, 실제로 중요한 방향 8~16개만 바꿔도 된다"

```
Full fine-tuning:
  W' = W + ΔW      where ΔW ∈ R^{d × d}   ← d×d 전체 업데이트

LoRA:
  W' = W + ΔW ≈ W + BA   where B ∈ R^{d × r}, A ∈ R^{r × d},  r << d
```

ΔW를 직접 학습하는 대신 **두 개의 작은 행렬 B, A의 곱**으로 근사 (위의 low-rank 분해 원리 그대로).

Plain 표기:

```
  W (d × d, frozen)
  B (d × r, 학습)    r = rank (예: 4, 8, 16)
  A (r × d, 학습)

  forward: output = x · W  +  x · A^T · B^T
                    ↑ frozen    ↑ LoRA 경로 (추가)
```

수식:

$$h = xW + xA^TB^T = x(W + A^TB^T)$$

- 초기화: A는 랜덤 Gaussian, B는 0 → 초기 ΔW = BA = 0 (학습 초기 원래 모델과 동일)
- 스케일링: 실제론 `(α/r) · BA`로 스케일 (α는 하이퍼파라미터)

### 파라미터 절감 계산 (섹션 2 내용 재확인)

```
d = 4096 (LLaMA-7B attention projection)

Full:  ΔW = d × d = 4096 × 4096 = 16,777,216 파라미터
LoRA (r=8):  B + A = d×r + r×d = 2 × 4096 × 8 = 65,536 파라미터

절감: 16M → 65K = 256배 감소
```

---

## 4. 어디에 LoRA를 붙이는가?

Transformer의 어느 weight에 LoRA를 적용할지 선택 가능.

### 원래 논문 (Hu et al.)

Attention의 Q, V projection에만 적용:
```
  W_Q → W_Q + B_Q A_Q
  W_V → W_V + B_V A_V
  W_K, W_O, FFN → 그대로 frozen
```

### 실제 사용

보통 더 넓게 적용:
```
적용 대상: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
           (attention 전체 + FFN까지)
```

모듈별 학습 파라미터 비율 (LLaMA-7B, r=8):
| 모듈 | 원래 파라미터 | LoRA 파라미터 |
|------|-------------|--------------|
| q_proj (4096×4096) | 16.7M | 65K |
| v_proj (4096×4096) | 16.7M | 65K |
| 전체 (q+k+v+o+ffn) | ~300M/layer | ~2M/layer |

---

## 5. 핵심 하이퍼파라미터

### rank (r)

LoRA의 압축 정도 결정:

| r | 파라미터 수 | 표현력 | 일반적 용도 |
|---|------------|--------|-------------|
| 4 | 최소 | 낮음 | 단순 task, 데이터 적을 때 |
| 8 | 작음 | 중간 | 일반적인 instruction tuning |
| 16 | 중간 | 높음 | 복잡한 task |
| 64+ | 많음 | Full에 근접 | Full fine-tuning 대안 |

경험적으로 r=8~16이 대부분의 경우에 충분.

### alpha (α)

스케일링 계수: `(α / r) · BA`

- α = r → 스케일 1.0 (기본)
- α = 2r → 스케일 2.0 (LoRA 기여 강화)
- 보통 α = r 또는 α = 2r로 설정

### dropout

LoRA layer에 dropout 추가 → 과적합 방지:
```
  output = x · W  +  dropout(x · A^T) · B^T
```

---

## 6. Inference 시 — Weight 병합

LoRA의 핵심 장점: 추론 시 **별도 overhead 없음**

```
학습 후:  W_merged = W + (α/r) · BA

이 병합된 W_merged를 원래 W 자리에 대체
→ 원래 모델과 구조 동일, 속도 동일
```

여러 LoRA를 교체하면서 동일 base 모델에 다른 역할 부여 가능:
```
base LLM + LoRA_A (수학 전문가)
base LLM + LoRA_B (코딩 전문가)
base LLM + LoRA_C (의료 전문가)

→ 요청에 따라 LoRA만 교체 (hot-swap)
```

---

## 7. QLoRA — 양자화 + LoRA

> "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

### 아이디어

LoRA도 70B 모델을 2 GPU에 올리려면 base model weight 자체가 너무 큼.
→ **base model을 4-bit로 양자화**해서 메모리 절감 + **LoRA로 학습**

```
메모리 비교 (LLaMA-65B 기준):
  Full fine-tuning:     >780GB
  LoRA (fp16):          ~160GB
  QLoRA (4-bit base):    ~48GB  ← A100 1개로 가능!
```

### NF4 (NormalFloat4)

QLoRA의 핵심 양자화 방식:
- 사전학습 weight는 정규분포를 따름
- 정규분포에 최적화된 4-bit 격자점 배치 → 균등 분할보다 정확
- INT4보다 정보 손실 적음

```
NF4 격자점 (16개):
  -1.0, -0.69, -0.52, -0.39, -0.28, -0.18, -0.09, 0.0,
   0.09,  0.18,  0.28,  0.39,  0.52,  0.69,  0.87, 1.0
  → 정규분포의 quantile에 배치
```

### Double Quantization

NF4 양자화 상수(scale factor) 자체도 다시 양자화:
```
  weight → NF4 (4-bit)             : scale ∈ fp32 → 추가 메모리
  scale  → fp8 (8-bit)             : scale 메모리도 절감
  → 파라미터당 평균 4.127 bit (≈ 4-bit)
```

### Paged Optimizer

GPU OOM 방지:
- Adam optimizer state (fp32 × 2)를 **CPU DRAM에 page**
- GPU 메모리 부족 시 자동으로 CPU로 offload → OOM 대신 느려짐

### QLoRA 전체 구조

```
base model [NF4 4-bit, frozen]
     │
     │  forward 시 BF16으로 dequantize
     ▼
BF16 computation
     │
     ▼
LoRA adapters [BF16, 학습]
     │
gradient → optimizer state [CPU DRAM]
```

학습은 BF16으로, 저장은 NF4로 → 메모리와 정확도의 균형.

---

## 8. LoRA 변형들

### 7-1. LoRA+

A와 B의 learning rate를 다르게 설정:
- A: 낮은 lr (이미 좋은 방향 찾음)
- B: 높은 lr (output side는 빠르게 적응)
→ 동일 step에서 더 좋은 수렴

### 7-2. DoRA (Weight-Decomposed LoRA)

weight를 magnitude(크기)와 direction(방향)으로 분해:
```
  W = m · (V / ||V||)    where m = magnitude, V = direction

  DoRA: magnitude는 학습, direction에 LoRA 적용
```
→ Full fine-tuning에 더 근접한 성능

### 7-3. LoRA-FA (Frozen A)

A matrix를 랜덤 초기화 후 **frozen**, B만 학습:
- 메모리 절반 (A의 gradient, optimizer state 없음)
- 성능 거의 유지

### 7-4. rsLoRA

rank-stabilized LoRA: 스케일링을 `α/r` 대신 `α/√r`로:
- 높은 rank에서 학습 안정화
- rank를 크게 써도 안정적

### 7-5. LongLoRA

긴 context fine-tuning을 위한 LoRA:
- Shifted sparse attention으로 context 확장 (4K → 100K)
- Embedding, normalization layer도 학습 (LoRA와 함께)

---

## 9. VLM에서의 LoRA 적용

`[[vlm_multimodal]]` 와 연결:

```
Vision Encoder [frozen]
        │
    Connector [full fine-tuning]  ← 파라미터 적어서 full로
        │
      LLM [LoRA]  ← 파라미터 많아서 LoRA로

적용 위치:
  - Attention: q, k, v, o projection
  - FFN: gate, up, down projection (선택적)
  - Embedding layer: 보통 제외
```

VLM 특수 고려:
- **Visual instruction tuning**: 다양한 이미지-텍스트 pair → r=16 이상 권장
- **High-res tile**: token이 많아 gradient가 풍부 → r 낮아도 됨
- **LoRA target**: LLM만? Vision encoder도? (보통 LLM만)

---

## 10. 실제 학습 설정 예시

### LLaMA-3-8B + LoRA (일반 instruction tuning)

```python
lora_config = {
    "r": 16,
    "lora_alpha": 32,          # alpha = 2r
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}
```

### LLaMA-3-70B + QLoRA (단일 A100 80GB)

```python
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
}

lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],   # 메모리 제한 시 줄임
    "lora_dropout": 0.1,
}
```

---

## 11. 성능 비교

| 방법 | VRAM (7B) | VRAM (70B) | 성능 (vs full) |
|------|-----------|------------|----------------|
| Full FT | 80GB+ | 500GB+ | 100% |
| LoRA (r=16) | ~20GB | ~160GB | 95~98% |
| QLoRA (r=16) | ~8GB | ~48GB | 93~97% |
| QLoRA (r=8) | ~7GB | ~45GB | 90~95% |

task와 데이터에 따라 LoRA가 full fine-tuning과 **거의 동등**한 경우 많음.

---

## 12. 한계

- **Rank 선택 어려움**: task마다 최적 r이 다름, 실험 필요
- **Multi-task**: 서로 다른 task의 LoRA를 합치면 성능 저하
- **Catastrophic forgetting**: 여전히 발생 (full에 비해 적지만)
- **QLoRA 속도**: dequantize overhead → full보다 학습 느림 (~30%)
- **Attention-only LoRA**: FFN 제외 시 일부 task 성능 제한

---

## 다음 단계

- [ ] IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- [ ] Adapter layers vs LoRA 비교
- [ ] LoRA merging (여러 LoRA를 합치는 방법: TIES, DARE)
- [ ] Multi-task LoRA
- [[vlm_multimodal]] — VLM에서의 LoRA 적용
- [[quantization_speculative_decoding]] — QLoRA의 양자화 기법
