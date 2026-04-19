# Vortex: Overcoming Memory Capacity Limitations in GPU-Accelerated Large-Scale Data Analytics

- **저자**: Yichao Yuan, Advait Iyer, Lin Ma, Nishil Talati (University of Michigan)
- **학회**: VLDB 2024
- **링크**: https://doi.org/10.14778/3717755.3717780

---

## 문제 정의

GPU는 높은 연산 처리량을 제공하지만 두 가지 병목이 대용량 데이터 analytics를 방해함:

1. **GPU 메모리 용량 한계**: 현대 GPU는 수십~수백 GB에 불과, CPU 메모리(수 TB)와 격차가 큼
2. **PCIe 대역폭 병목**: CPU DRAM 대역폭 ~150GB/s에 비해 GPU 1개의 PCIe 단방향 대역폭은 ~28GB/s에 불과

기존 해결책들의 한계:
- **멀티 GPU**: IO와 compute가 같이 늘어남 → compute 낭비, 비용 증가
- **CPU-GPU 하이브리드**: CPU가 느려서 GPU가 놀게 됨
- **단순 스트리밍**: PCIe 병목 그대로 남음

---

## 핵심 아이디어: 다른 GPU의 PCIe를 빌려쓰자

AI 추론 같은 **compute-bound** 워크로드는 PCIe를 거의 안 씀. 같은 서버에서 analytics와 AI 추론이 같이 돌고 있을 때, analytics GPU가 다른 GPU들의 놀고 있는 PCIe를 활용.

```
4-GPU 시스템: GPU0=analytics, GPU1~3=AI 추론

CPU → GPU1 → GPU0   (GPU1의 PCIe + inter-GPU 링크)
CPU → GPU2 → GPU0
CPU → GPU3 → GPU0
CPU → GPU0           (직접)
```

단일 PCIe ~28GB/s → 4개 합산 ~112GB/s로 CPU DRAM 대역폭에 근접.

---

## 3계층 설계

### 1계층: Exchange Primitive (IO 엔진)

포워딩 시 두 가지 문제 발생:
- **Head-of-line blocking**: 소프트웨어 큐에서 뒤 요청이 앞 요청에 막힘
- **비균등 대역폭**: H2D/D2H가 동시에 쓰이면 서로 간섭 → H2D가 불리

해결 원칙:
1. **즉시 실행**: 큐 대기 없이 즉시 실행 가능한 요청만 제출
2. **Flow control**: H2D/D2H 양방향 트래픽을 같이 관리, D2H가 너무 빠르면 제한

포워딩 GPU는 패킷 단위로 ping-pong 버퍼로 파이프라인:
```
[패킷 받는 중] + [이전 패킷 → GPU0 전달 중]  ← 동시에 실행
```
버퍼 크기 = 패킷 1개만 필요 (포워딩 GPU 메모리 낭비 최소화).

---

### 2계층: IO-Decoupled 프로그래밍 모델

기존 방식: IO 스케줄링과 GPU 커널 코드가 뒤엉킴 → 코드 재사용 불가, 복잡성 증가

Vortex 방식: **ExKernel** = 기존 GPU 커널 + 데이터 분할 방법만 추가

```
ExKernel이 지정하는 것:
- inputs(it):  it번째 청크의 입력 위치
- outputs(it): it번째 청크의 출력 위치
- kernel():    실제 GPU 연산 (기존 on-core 코드 재사용 가능)
```

**Pipelined Executor**가 IO 전담 관리:
```
cycle N:   GPU가 chunk(N-1) 연산
           Exchange가 chunk(N) 로드 + chunk(N-2) 저장  ← 동시에!
```

GPU 연산과 데이터 전송을 겹쳐서(overlap) 실행 → 대기 시간 최소화.

---

### 3계층: 쿼리 연산자 구현

**Sort**: 외부 정렬을 청크 단위 부분 정렬 → 병합 단계로 분해하여 ExKernel로 구현.

**Hash Join (양쪽 테이블이 GPU 메모리 초과)**: 작은 테이블을 파티셔닝 후 해시 테이블 구성, 큰 테이블을 스트리밍하며 프로빙.

**Late Materialization (Zero-copy 기반)**:

Zero-copy = GPU compute unit이 CPU 메모리를 캐시라인(64B) 단위로 **직접** 접근하는 방식 (SDMA 없이).

```
일반 방식: 모든 컬럼을 PCIe로 전송 → 연산
Late Mat.: 키 컬럼만 전송 → 필터링 → 살아남은 행만 zero-copy로 나머지 컬럼 접근
```

불필요한 컬럼 데이터를 PCIe로 전송하지 않아도 됨.

---

## 성능 결과

GPU 메모리에 데이터를 캐싱하지 않고도:

| 비교 대상 | 향상 |
|---|---|
| 최신 GPU 기반 시스템 (Proteus) | 평균 5.7x |
| CPU 기반 DuckDB | 3.4x 성능, 2.5x 비용 효율 향상 |

AI 워크로드 영향: 다른 GPU의 PCIe를 빌려써도 AI 추론 성능 저하 평균 **6.8%** 에 불과.
