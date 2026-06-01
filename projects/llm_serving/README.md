# LLM Serving 학습 노트

LLM inference 최적화 / serving 시스템 학습 자료 정리

## 목차

### 기초
- [Attention 기초 이론](attention_basics.md) — Scaled dot-product, MHA, 복잡도 분석

### Attention 최적화
- [FlashAttention](flash_attention.md) — IO-aware exact attention (v1)
- [FlashAttention v2 / v3](flash_attention_v2_v3.md) — Work partitioning, warp specialization, FP8

### Serving System
- [PagedAttention & KV Cache](paged_attention_kv_cache.md) — vLLM 메모리 관리
- [Continuous Batching](continuous_batching.md) — Orca, iteration-level scheduling

### 처리량 / 지연 시간 최적화
- [Quantization & Speculative Decoding](quantization_speculative_decoding.md) — AWQ, FP8, EAGLE/Medusa

### 분산 Serving
- [Disaggregated Serving](disaggregated_serving.md) — Prefill/Decode 분리 (DistServe, Splitwise)

### Multimodal
- [VLM 아키텍처 & Inference](vlm_multimodal.md) — Vision encoder, Connector, Training, Inference 최적화

## 학습 순서 제안

1. `attention_basics` → attention의 수식과 O(n²) 문제 이해
2. `flash_attention` → IO-aware 최적화 개념
3. `flash_attention_v2_v3` → 실전에서 쓰이는 버전
4. `paged_attention_kv_cache` → 메모리 관리의 핵심
5. `continuous_batching` → scheduling 측면
6. `quantization_speculative_decoding` → 추가 최적화 기법
7. `disaggregated_serving` → Prefill/Decode 분리 아키텍처
