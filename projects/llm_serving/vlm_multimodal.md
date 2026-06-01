# VLM (Vision-Language Model) — 아키텍처부터 Inference까지

> 대표 논문:
> - Radford et al., "CLIP: Learning Transferable Visual Models From Natural Language Supervision" (2021)
> - Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning" (NeurIPS 2022)
> - Liu et al., "Visual Instruction Tuning (LLaVA)" (NeurIPS 2023)
> - Bai et al., "Qwen-VL" (2023), "InternVL" (2024)

---

## 1. VLM이 뭔가? — 기본 구조

LLM은 text token sequence를 입력받음. VLM의 핵심 문제:

> **"이미지를 어떻게 LLM이 이해할 수 있는 token으로 바꿀 것인가?"**

기본 파이프라인:

```
Image
  │
  ▼
┌──────────────┐        ┌─────────────┐        ┌────────────┐
│ Vision       │──────▶ │  Connector  │──────▶ │    LLM     │──▶ Text output
│ Encoder      │        │  (bridge)   │        │            │
└──────────────┘        └─────────────┘        └────────────┘
  이미지를 feature       feature를 LLM이         text token과
  vector로 변환          이해하는 형태로 변환      함께 처리
```

세 컴포넌트를 어떻게 선택하고 연결하는지가 VLM 아키텍처의 핵심.

---

## 2. Vision Encoder — 이미지를 벡터로

### 2-1. ViT (Vision Transformer)

이미지를 patch로 분할 후 Transformer로 처리:

```
Image (H × W × 3)
  │
  ▼ patch 분할 (e.g. 14×14 pixels per patch)
  │
[patch_1][patch_2]...[patch_N]    N = (H/14) × (W/14)
  │
  ▼ Linear projection + Positional embedding
  │
Transformer Encoder (12~24 layers)
  │
  ▼
Visual features: [N, d_vision]    (e.g. [256, 1024])
```

224×224 이미지 + 14×14 patch → N = (224/14)² = 256 patches → 256개 visual token

Plain 표기:
```
  input:  image  ∈ R^{H × W × 3}
  after patch split + proj:  tokens ∈ R^{N × d_patch},   N = (H·W) / (patch_size²)
  after ViT encoder:         visual_features ∈ R^{N × d_vision}
```

### 2-2. 대표 Vision Encoder들

| 모델 | 학습 방식 | d_vision | 특징 |
|------|-----------|----------|------|
| CLIP ViT-L/14 | Image-Text contrastive | 1024 | 가장 널리 쓰임, text-aligned |
| CLIP ViT-L/14@336 | 동일, 고해상도 | 1024 | 336px → 576 patches |
| SigLIP | Sigmoid loss 버전 | 1152 | CLIP보다 안정적 학습 |
| DINOv2 | Self-supervised (이미지만) | 1024 | spatial 정보 더 풍부 |
| InternViT-6B | 6B param ViT | 3200 | InternVL에서 사용 |

### 2-3. CLIP의 역할

CLIP은 image-text pair를 contrastive learning으로 학습:

```
  image embedding ──────┐
                         ├── cosine similarity 최대화 (matching pair)
  text embedding  ──────┘      최소화 (non-matching pair)
```

결과: image feature가 **text semantic과 정렬된 공간**에 놓임  
→ VLM에서 LLM과 연결하기 쉬움 (이미 언어와 관련된 표현)

---

## 3. Connector — Vision과 LLM을 잇는 다리

Vision encoder 출력 (`[N, d_vision]`)을 LLM 입력 형식(`[M, d_model]`)으로 변환.  
방식에 따라 VLM 아키텍처가 크게 달라짐.

### 3-1. Linear Projection (LLaVA style)

가장 단순한 방식: MLP 한두 층으로 바로 projection.

```
visual_features [N, d_vision]
        │
        ▼  Linear(d_vision → d_model) + GELU + Linear(d_model → d_model)
        │
visual_tokens [N, d_model]   ← LLM의 text token과 같은 차원
```

특징:
- 구조 단순, 학습 빠름
- Visual token 수 = patch 수 그대로 유지 (256~576개)
- LLM이 모든 spatial 정보를 직접 attention으로 처리

대표: **LLaVA-1.5**, LLaVA-1.6, Bunny

### 3-2. Q-Former (BLIP-2, InstructBLIP)

Learnable query를 통해 visual feature를 **압축**:

```
visual_features [N, d_vision]
        │
        ▼ Cross-attention (Q-Former Transformer)
        │
[query_1][query_2]...[query_K]   K << N  (예: K=32, N=256)
        │
        ▼ Linear projection
        │
visual_tokens [K, d_model]   ← 압축된 32개 token만 LLM으로
```

K개의 learnable query가 visual feature에서 필요한 정보를 선택적으로 추출.

특징:
- LLM 입력 token 수 대폭 감소 (256 → 32)
- 계산 효율 ↑, 하지만 fine-grained spatial 정보 손실 가능
- Q-Former 자체 학습 필요 → training complexity ↑

대표: **BLIP-2**, InstructBLIP, MiniGPT-4

### 3-3. Cross-Attention (Flamingo style)

Visual token을 LLM의 **중간 layer에 cross-attention으로 주입**:

```
LLM Layer N:
  text hidden state [n_text, d_model]
          │
          ▼ Self-Attention (text 내부)
          │
          ▼ Cross-Attention ← visual_features [N, d_vision]  (새 layer 추가)
          │
          ▼ FFN
          │
  출력 [n_text, d_model]
```

특징:
- LLM weight를 **frozen** 상태로 유지 가능 (cross-attn layer만 학습)
- 이미지가 LLM 깊숙이 영향을 줌 → 강력한 visual grounding
- 추가 cross-attention layer → 메모리 + 계산 증가
- Few-shot learning에 강함

대표: **Flamingo**, OpenFlamingo, Idefics

### 3-4. 방식 비교

| 방식 | Token 수 | 공간 정보 | 학습 복잡도 | 추론 속도 |
|------|----------|-----------|-------------|-----------|
| Linear Proj | N (많음) | 풍부 | 낮음 | 느림 (token 많아서) |
| Q-Former | K (적음) | 중간 | 중간 | 빠름 |
| Cross-Attn | N (많음) | 풍부 | 높음 | 느림 (추가 layer) |

---

## 4. LLM에서의 처리 — Text와 Image의 결합

### 4-1. Token Interleaving

Visual token과 text token을 **concat해서 LLM에 입력**:

```
입력 sequence:
  [SYS] [USER] <image> [v1][v2]...[vN] </image> [질문 text tokens] [ASST]
                        ↑ visual tokens (256개)    ↑ text tokens

LLM은 이 전체 sequence를 자기회귀적으로 처리
visual token에도 causal mask 적용 (또는 bidirectional)
```

### 4-2. Position Encoding 처리

Visual token의 위치 인코딩 방식:
- **연속 번호**: text 앞에 오는 token처럼 0~N-1 번호
- **2D position**: patch의 (row, col) 위치를 인코딩
- **무시**: CLIP feature 자체가 position 정보 포함 → 추가 불필요 (LLaVA 방식)

### 4-3. Attention Mask 설계

```
         [v1..vN]  [text tokens]  [output tokens]
[v1..vN]   ✓          ✗               ✗          ← visual: 서로 attend 가능
[text]     ✓          ✓               ✗          ← text: visual + 앞 text 참조
[output]   ✓          ✓               ✓          ← output: 전부 참조 (causal)
```

일부 모델은 visual token 간 attention을 끊기도 함 (독립 처리).

---

## 5. Training 전략

### 5-1. Two-Stage Training (LLaVA 방식)

**Stage 1: Feature Alignment (Connector만 학습)**
```
Vision Encoder [frozen] → Connector [학습] → LLM [frozen]

데이터: image-caption pair (CC3M 등 600K개)
목표: visual feature와 LLM embedding space를 정렬
```

**Stage 2: Visual Instruction Tuning (Connector + LLM 학습)**
```
Vision Encoder [frozen] → Connector [학습] → LLM [학습]

데이터: visual instruction following 데이터 (158K개, GPT-4로 생성)
목표: 이미지에 대한 대화/추론 능력 학습
```

### 5-2. 학습 데이터

| 종류 | 예시 | 역할 |
|------|------|------|
| Image-Caption | CC3M, LAION | alignment |
| VQA | VQAv2, GQA | 기본 QA |
| Visual Instruction | LLaVA-Instruct, ShareGPT4V | 대화 능력 |
| OCR | TextVQA, DocVQA | 텍스트 인식 |
| Chart/Diagram | ChartQA, AI2D | 구조화 이미지 |
| Grounding | RefCOCO | spatial 이해 |

### 5-3. LoRA / QLoRA for VLM

Full fine-tuning 대신 효율적 학습:

```
LLM의 attention weight에 low-rank adapter 삽입:
  W' = W + BA   where B ∈ R^{d × r}, A ∈ R^{r × d},  r << d

VLM 특수 고려:
  - Vision encoder는 보통 frozen (이미 잘 학습됨)
  - Connector는 full fine-tuning (파라미터 적어서)
  - LLM은 LoRA (파라미터 절약)
```

---

## 6. High-Resolution 처리

초기 VLM은 224×224 → 256 token → 세밀한 텍스트/차트 인식 불가.

### 6-1. 고해상도 ViT

- CLIP ViT-L/14@336: 336px → 576 patches
- InternViT@448: 448px → 1024 patches
- 해상도 ↑ → patch 수 n² 증가 → LLM 입력 token 폭증

### 6-2. Dynamic Resolution / Tile 방식 (LLaVA-1.6, InternVL)

이미지를 여러 tile로 분할해서 각각 처리:

```
원본 이미지 (고해상도)
        │
        ▼ 적절한 grid로 분할 (예: 2×2, 3×2 등)
        │
[tile_1][tile_2][tile_3][tile_4] + [thumbnail (전체 축소)]
        │
각 tile을 ViT로 독립 인코딩
        │
visual tokens 전부 concat → LLM
```

장점:
- 세밀한 정보 유지 (OCR, 차트 읽기)
- 동적 분할로 다양한 aspect ratio 지원

단점:
- Token 수 폭증 (4 tiles × 256 = 1024 + 256 = 1280 tokens)
- LLM context length 소비 심각

### 6-3. Token Compression (AnyRes, TokenPacking)

늘어난 visual token을 줄이는 방법:

**Spatial pooling**: 2×2 patch를 하나로 합침 → 1/4로 감소  
**Token merging (ToMe)**: 유사한 token을 merge  
**FastV**: attention score 낮은 visual token을 중간 layer에서 제거

---

## 7. Inference 최적화

### 7-1. Visual Token의 KV Cache 특성

Text token과 달리 visual token의 특성:
- **고정**: 이미지가 주어지면 visual token 변하지 않음
- **공유 가능**: 같은 이미지에 대한 여러 질문 → visual KV cache 재사용
- **많음**: tile 방식에서 1000+개 → KV cache 메모리 큼

### 7-2. Prefix Caching for Images

같은 이미지를 반복 질의할 때 (챗봇, RAG):

```
1회차: image encode → visual token KV cache 생성 + 저장
2회차: 같은 이미지 → KV cache hit → visual prefill 생략
       text 부분만 처리
```

vLLM의 `--enable-prefix-caching`으로 지원.  
이미지 hash → prefix key → KV cache lookup.

### 7-3. Visual Token Reduction (Inference 시)

긴 이미지 설명 없이 짧은 답변만 필요할 때:

**FastV (2024)**:
```
Shallow layer에서 attention score 계산
→ visual token 중 score 낮은 것들을 deep layer에서 제거
→ LLM 후반부에서 token 수 감소
→ latency 20~40% 감소, 품질 거의 유지
```

**LLaVA-PruMerge**:
```
Q-Former처럼 key visual token만 선택
→ 256 → 64 token으로 압축
→ 4x 빠른 inference
```

### 7-4. Multi-Image / Video Batching

여러 이미지 또는 video frame 처리 시 문제:
- Frame 수 × patch 수 = 엄청난 token 수
- (예: 64 frames × 256 patches = 16,384 visual tokens)

해결:
- **Frame sampling**: 균일 또는 중요도 기반 frame 선택
- **Temporal compression**: 인접 frame간 유사한 token merge
- **Separate KV**: frame별 KV cache 분리 후 cross-frame attention 제한

---

## 8. 주요 VLM 모델 비교

| 모델 | Vision Enc | Connector | LLM | 특징 |
|------|------------|-----------|-----|------|
| LLaVA-1.5 | CLIP ViT-L/336 | MLP | Vicuna-13B | 단순 강력 |
| LLaVA-1.6 | CLIP ViT-L/336 | MLP + tiling | Mistral-7B | 고해상도 |
| InternVL2 | InternViT-6B | MLP | InternLM2 | SOTA 오픈소스 |
| Qwen-VL | ViT | Cross-attn+MLP | Qwen | 다국어, OCR 강함 |
| BLIP-2 | EVA-ViT-g | Q-Former | Flan-T5/OPT | Q-Former 원조 |
| Flamingo | NFNet | Cross-attn | Chinchilla | Few-shot 강함 |
| GPT-4V | 비공개 | 비공개 | GPT-4 | 최강 but closed |
| Claude 3 | 비공개 | 비공개 | Claude 3 | 문서 이해 강함 |
| Gemini | 비공개 | native multimodal | Gemini | 처음부터 multimodal |

---

## 9. Native Multimodal vs. Adapter 방식

지금까지 본 것 = **Adapter 방식**: 기존 LLM + Vision encoder 붙이기

**Native Multimodal** (Gemini, Chameleon 등):
- 처음부터 image + text를 함께 학습
- Image token을 text token과 동등하게 취급
- Tokenizer 단에서 이미지를 discrete token으로 변환 (VQ-VAE 등)
- 더 깊은 통합, 하지만 학습 비용 매우 큼

```
Adapter:  Vision enc → connector → LLM   (이미 학습된 LLM 재활용)
Native:   [image token][text token] → 단일 Transformer   (처음부터 학습)
```

---

## 10. GPU 최적화 관점 요약

| 단계 | 병목 | 해결 |
|------|------|------|
| Image encoding | ViT forward (compute) | 배치 처리, FP16/INT8 |
| Visual token projection | 작은 MLP (빠름) | 큰 문제 없음 |
| LLM prefill (with visual) | 긴 context (n² attention) | FlashAttention, token 압축 |
| LLM decode | KV cache load (memory-bound) | PagedAttention, KV quantization |
| Multi-image/video | 매우 긴 context | Temporal compression, frame sampling |

---

## 다음 단계

- [ ] Training 테크닉 심화 (visual instruction tuning data, DPO for VLM)
- [ ] Video LLM (시간 축 처리)
- [ ] Native multimodal 아키텍처 (Chameleon, Emu3)
- [ ] VLM serving 최적화 (SGLang radix cache + image prefix)
- [[flash_attention]] — visual token의 긴 context 처리
- [[paged_attention_kv_cache]] — visual KV cache 관리
