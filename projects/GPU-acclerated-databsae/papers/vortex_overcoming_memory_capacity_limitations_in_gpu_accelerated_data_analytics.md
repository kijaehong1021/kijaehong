# Vortex: Overcoming Memory Capacity Limitations in GPU-Accelerated Large-Scale Data Analytics

- **저자**: Yichao Yuan, Advait Iyer, Lin Ma, Nishil Talati (University of Michigan)
- **학회**: VLDB 2024
- **링크**: https://doi.org/10.14778/3717755.3717780

---

## 문제 정의

GPU는 높은 연산 처리량을 제공하지만 두 가지 병목이 대용량 데이터 analytics를 방해함:

1. **GPU 메모리 용량 한계**: 현대 GPU는 수십~수백 GB에 불과, CPU 메모리(수 TB)와 격차가 큼
2. **PCIe 대역폭 병목**: CPU→GPU 데이터 전송 속도가 GPU HBM 대역폭보다 훨씬 낮음

특히 멀티-GPU 환경에서 하나의 GPU가 IO-intensive 작업을 할 때 다른 GPU의 PCIe 링크가 유휴 상태로 낭비됨.

---

## 핵심 아이디어

### 1. 최적화된 IO 프리미티브: 멀티-GPU PCIe 집합 활용

**단일 타겟 GPU의 IO 요청을 처리할 때, 시스템의 모든 PCIe 링크를 활용**.

- IO-intensive 작업을 하는 GPU: 다른 GPU들을 경유해서 데이터를 수신
- Compute-bound 작업(AI 학습 등)을 하는 GPU들: 평소 IO 자원을 거의 안 쓰므로 그 PCIe 링크를 데이터 라우팅에 활용

→ 단일 PCIe 링크 대역폭 한계를 다중 PCIe 링크 집합 대역폭으로 극복.

### 2. 분리된 프로그래밍 모델

GPU 커널 개발과 IO 스케줄링을 **분리**하여:
- 개발자가 IO 스케줄링을 신경 쓰지 않고 커널 개발에 집중
- 기존 GPU 코드 재사용 가능

### 3. Late Materialization (Zero-copy Memory Access 기반)

GPU의 zero-copy 메모리 접근을 활용한 late materialization 기법으로 불필요한 데이터 이동 최소화.

---

## 성능 결과

GPU 메모리에 데이터를 캐싱하지 않고도:

| 비교 대상 | 향상 |
|---|---|
| 최신 GPU 기반 시스템 (Proteus) | 평균 5.7x |
| CPU 기반 DuckDB | 2.5x 비용 효율 향상 |
