# Powerful GPUs or Fast Interconnects: Analyzing Relational Workloads on Modern GPUs

- **저자**: Marko Kabić, Bowen Wu, Jonas Dann, Gustavo Alonso (ETH Zurich)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3749646.3749698

---

## 문제 정의

GPU 모델(RTX3090, A100, H100, GH200)과 인터커넥트(PCIe 3.0/4.0/5.0, NVLink 4.0)의 다양한 조합이 relational analytics 성능에 미치는 영향이 불명확함.

기존 평가 지표인 **arithmetic intensity**와 **GFlop/s**는 data analytics 워크로드를 설명하기에 부적합.

---

## 핵심 아이디어

### MaxBench: GPU 관계형 워크로드 벤치마킹 프레임워크

TPC-H, H2O-G, ClickBench 워크로드를 대상으로 다양한 GPU-인터커넥트 조합을 체계적으로 분석.

### 새로운 평가 지표 제안

기존 GFlop/s, arithmetic intensity 대신:

| 지표 | 설명 |
|---|---|
| **Characteristic Query Complexity** | 쿼리 자체의 연산 복잡도 특성 |
| **Characteristic GPU Efficiency** | 해당 쿼리에 대한 GPU의 실질적 효율 |

Data analytics 워크로드는 compute-bound보다 **memory-bound**에 가깝기 때문에 기존 지표로 설명이 안 됨.

### 비용 모델 기반 성능 예측

새 지표를 기반으로 GPU 모델과 인터커넥트 조합에 따른 **쿼리 실행 성능을 예측하는 비용 모델** 개발.

→ 미래 하드웨어 트렌드(인터커넥트 대역폭 향상 vs GPU 효율 향상)에 따른 성능 변화를 시뮬레이션 가능.

---

## 주요 인사이트

- GPU 연산 능력과 인터커넥트 대역폭 간의 **trade-off** 관계 분석
- 워크로드 특성에 따라 더 강한 GPU가 유리한 경우 vs 더 빠른 인터커넥트가 유리한 경우가 다름
- NVLink 4.0 같은 고속 인터커넥트는 특정 쿼리에서 GPU 업그레이드보다 효과적
