# Efficient Graph Data Access for Out-of-Memory GPU Streaming Graph Processing (Grapin)

- **저자**: Qiange Wang, Yongze Yan, Hongshi Tan, Cheng Chen 외 (NUS, Northeastern Univ, ByteDance)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3749646.3749659

---

## 문제 정의

대규모 스트리밍 그래프(수십억 엣지)가 GPU 메모리 용량을 초과할 때, CPU-GPU 협력 처리 방식은 두 가지 문제를 겪는다:

1. **과도한 CPU→GPU 데이터 전송**: 연속적인 그래프 연산에서 같은 데이터를 반복 전송
2. **불규칙한 접근 패턴**: 스트리밍 그래프의 동적 특성으로 인해 전송 패턴이 예측 불가

기존 솔루션들은 이 중복 접근 문제를 효과적으로 해결하지 못함.

---

## 핵심 아이디어

### 1. GPU용 증분 처리 알고리즘 (Incremental Processing)

기존 CPU용 증분 알고리즘은 무거운 데이터 의존성 처리로 인해 GPU에 그대로 적용 불가.
→ 데이터 의존성 처리를 **GPU 친화적인 형태로 변환**하여, 연산 측면에서 중복 그래프 접근을 제거.

### 2. GPU Hot Subgraph 관리 프레임워크

자주 접근되는 동적 서브그래프를 **vertex 단위로 GPU에 캐싱**하는 경량 프레임워크.
→ 이미 GPU에 있는 데이터는 재전송 없이 재활용.

---

## 성능 결과

| 최적화 | 효과 |
|---|---|
| 증분 처리 활성화 | 데이터 전송량 61% 감소 |
| Hot subgraph 재활용 추가 | 남은 전송량 72% 추가 감소 |
| 전체 | 데이터 전송 89% 감소 |
| CPU 솔루션 대비 | 1.8x ~ 96.9x 속도 향상 (평균 17.9x) |

하드웨어: 단일 NVIDIA A5000 GPU
