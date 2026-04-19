# Scaling GPU-Accelerated Databases beyond GPU Memory Size

- **저자**: Yinan Li, Bailu Ding, Ziyun Wei 외 (Microsoft, Cornell Univ, Ohio State Univ)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3749646.3749710

---

## 문제 정의

GPU의 HBM 용량(최대 80GB)은 실제 분석 DB 크기(수백GB~수TB)에 비해 너무 작음. 데이터가 GPU 메모리를 초과하면 PCIe를 통한 데이터 전송이 필요한데, **PCIe 대역폭이 HBM 대역폭에 비해 극도로 낮아** 전체 성능의 병목이 됨.

---

## 핵심 아이디어: Hybrid CPU-GPU Query Processing

CPU와 GPU의 강점을 분리하여 PCIe 전송 병목을 완화.

### 역할 분담

| 역할 | 처리 주체 | 이유 |
|---|---|---|
| **데이터 필터링** | CPU | 필터로 데이터 볼륨을 줄여 PCIe 전송량 최소화 |
| **Join 등 compute-intensive 연산** | GPU | HBM의 높은 대역폭과 대규모 병렬성 활용 |

### 동작 흐름

1. CPU가 전체 데이터에 대해 효율적인 필터링 수행
2. 필터를 통과한 **소량의 데이터만** PCIe를 통해 GPU로 전송
3. GPU에서 join 등 무거운 연산 처리

→ PCIe를 통해 전송되는 데이터 볼륨 자체를 줄이는 것이 핵심.

---

## 성능 결과

- 벤치마크: TPC-H (scale factor 최대 1000 = 1TB)
- 하드웨어: 단일 A100 GPU (80GB HBM)
- GPU 메모리를 크게 초과하는 데이터셋에서도 효과적으로 처리
- 최신 CPU 전용 DB 시스템 대비 성능 및 비용 효율성 모두 우수
