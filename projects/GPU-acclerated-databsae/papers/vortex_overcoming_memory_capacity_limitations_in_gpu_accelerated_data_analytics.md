# Vortex: Overcoming Memory Capacity Limitations in GPU-Accelerated Large-Scale Data Analytics

- **저자**: Yichao Yuan, Advait Iyer, Lin Ma, Nishil Talati (University of Michigan)
- **학회**: VLDB 2024
- **링크**: https://doi.org/10.14778/3717755.3717780

---

## 문제 정의

GPU는 높은 연산 처리량을 제공하지만 대용량 analytics에는 두 가지 병목이 있음:

| 문제 | 수치 |
|---|---|
| GPU 메모리 용량 | A100 80GB, H100 80~96GB vs CPU DRAM 수 TB |
| PCIe 단방향 대역폭 | ~28 GB/s vs CPU DRAM ~150 GB/s |

기존 해결책의 한계:
- **멀티 GPU**: 메모리 늘리면 compute도 같이 늘어남 → compute 낭비, 비용 증가. 단일 서버 GPU 수도 한계가 있음
- **CPU-GPU 하이브리드**: CPU가 느려서 GPU가 유휴 상태가 됨
- **단순 스트리밍**: PCIe 대역폭 병목 그대로

**핵심 관찰**: 같은 서버에서 AI 추론 같은 **compute-bound** 워크로드는 PCIe를 거의 사용하지 않음. 이 idle한 PCIe 대역폭을 analytics GPU가 활용할 수 있음.

---

## 핵심 아이디어: 다른 GPU의 PCIe를 빌려쓰자

```
4-GPU 서버: GPU0 = analytics, GPU1~3 = AI 추론 (PCIe 거의 안 씀)

CPU DRAM
  │ (28 GB/s)   │ (28 GB/s)   │ (28 GB/s)   │ (28 GB/s)
  ↓             ↓             ↓             ↓
 GPU0          GPU1          GPU2          GPU3
(analytics)  (forwarding)  (forwarding)  (forwarding)
               │             │             │
               └─────────────┴─────────────┘
                     inter-GPU link로 GPU0에 전달
```

- CPU → GPU0 직접: 28 GB/s
- CPU → GPU1 → GPU0: GPU1의 PCIe + NVLink/PCIe inter-GPU
- CPU → GPU2 → GPU0, CPU → GPU3 → GPU0 동시에

4개 합산 → **~112 GB/s**, CPU DRAM 대역폭(~150 GB/s)에 근접.

---

## 계층 1: Exchange Primitive (IO 엔진)

### 포워딩의 두 가지 문제

**문제 1: Head-of-line blocking**

CUDA 런타임은 요청을 큐에 넣고 순서대로 실행. 앞 요청이 느리면 뒤 요청이 전부 막힘.

```
Queue: [H2D 요청 A (느림)] [H2D 요청 B] [H2D 요청 C]
         ↑ 이게 막히면 B, C도 못 시작
```

**문제 2: 비균등 대역폭 (H2D vs D2H 간섭)**

H2D(CPU→GPU)와 D2H(GPU→CPU) 트래픽이 동시에 쓰이면 GPU 메모리 컨트롤러 용량 초과:
- H2D: 8 GB/s로 떨어짐
- D2H: 20 GB/s로 D2H가 훨씬 빠름
→ D2H가 H2D 대역폭을 잠식. 포워딩 GPU끼리 불균등하게 완료 → 일부 PCIe 링크 유휴

### Exchange 설계 원칙

1. **즉시 실행 (no queuing)**: 큐에 대기 없이 즉시 실행 가능한 요청만 제출 → head-of-line blocking 제거

2. **통합 Flow Control**: H2D와 D2H를 같이 관리.
   - D2H 큐가 H2D 큐보다 빠르게 drain되지 않도록 제한
   - H2D/D2H 태스크 수를 균형 있게 유지 → 대역폭 경쟁 방지

### 포워딩 GPU 내부 구현: Ping-Pong 버퍼

```
포워딩 GPU (예: GPU1):

버퍼 A: [패킷 N 수신 중  ←  CPU]
버퍼 B: [패킷 N-1 → GPU0 전달 중]  ← 동시 실행!

다음 사이클:
버퍼 B: [패킷 N+1 수신 중  ←  CPU]
버퍼 A: [패킷 N → GPU0 전달 중]
```

- 패킷 1개 크기의 버퍼 2개만 필요 → 포워딩 GPU 메모리 낭비 최소
- 수신과 전달을 overlap해서 처리량 극대화

---

## 계층 2: IO-Decoupled 프로그래밍 모델 (ExKernel)

### 기존 방식의 문제

기존 out-of-core GPU 시스템은 IO 스케줄링과 커널 코드가 뒤엉킴:
- 코드 재사용 불가 (특수 목적 커널 재작성 필요)
- IO 관리 로직이 커널 코드와 분리 안 됨

### ExKernel 인터페이스

```
ExKernel = 기존 GPU 커널 + 데이터 분할 명세

개발자가 정의하는 것:
  inputs(it)   → it번째 청크의 입력이 어디 있는지 (DRAM 내 위치)
  outputs(it)  → it번째 청크의 출력이 어디 저장될지
  size()       → 총 청크 수
  kernel(it)   → 실제 GPU 연산 (기존 on-GPU 코드 그대로 재사용 가능!)

Vortex Pipelined Executor가 담당하는 것:
  - Exchange 호출 (IO 스케줄링)
  - GPU 커널 실행
  - 두 작업 overlap
```

### Pipelined Executor 동작

GPU 메모리를 3 영역으로 나눔: `mem_A`, `mem_B`, `tmp`

```
사이클 N:
  GPU 연산: kernel(N-1) 실행 → mem_A 읽고 mem_B에 쓰기
  Exchange:  chunk(N) 로드 → mem_A
             chunk(N-2) 결과 저장 ← mem_B   ← 동시 실행!

사이클 N+1:
  GPU 연산: kernel(N) 실행 → mem_B 읽고 mem_A에 쓰기
  Exchange:  chunk(N+1) 로드 → mem_B
             chunk(N-1) 결과 저장 ← mem_A
```

→ GPU 연산과 데이터 전송이 항상 overlap → PCIe 대역폭을 쉬지 않고 활용.

---

## 계층 3: 쿼리 오퍼레이터 구현

### Sort (외부 정렬)

**SortExKernel**: 입력 배열을 GPU 메모리에 맞는 청크로 나눠 각각 부분 정렬 (radix sort)

**MergeExKernel**: 정렬된 청크들을 single-pass merge

```
[DRAM: 정렬 안 된 큰 배열]
  ↓ SortExKernel (청크별 radix sort)
[DRAM: 부분 정렬된 청크들]
  ↓ MergeExKernel
[DRAM: 전체 정렬된 배열]
```

기존 rocPRIM/CUB 라이브러리 커널 그대로 재사용 → ExKernel 감쌈만 하면 됨.

### Hash Join (양쪽 테이블 모두 GPU 메모리 초과)

```
1. 작은 테이블을 파티셔닝 (RadixPartitionExKernel)
   파티션 하나가 GPU 메모리에 들어오는 크기로

2. 각 파티션으로 해시 테이블 구축 (GPU HBM에 상주)

3. 큰 테이블을 스트리밍하며 각 파티션에 probe
   (ProbeExKernel: 청크 단위로 GPU로 스트리밍)
```

### Late Materialization + Zero-copy

**Late Materialization**: 전체 컬럼을 모두 PCIe로 전송하지 않고, 선택적으로만 전송.

```
일반 방식:
  PCIe로 전체 컬럼 A, B, C 전송 → GPU에서 filter(A) → B, C로 연산

Late Materialization:
  PCIe로 필터 컬럼 A만 전송 → GPU에서 filter(A) → 살아남은 행 인덱스
  → 살아남은 행의 B, C 값만 필요할 때 접근
```

**Zero-copy 메모리 접근**: GPU compute unit이 PCIe를 통해 CPU DRAM을 cache line(64B) 단위로 직접 접근. SDMA 없이, GPU 코어가 필요한 주소를 직접 읽음.

- SDMA 방식(Exchange): 연속된 큰 블록을 배치로 전송 → 효율적이지만 필요 없는 데이터도 옴
- Zero-copy: 필요한 cache line만 on-demand로 가져옴 → selectivity 높을 때 훨씬 효율적

**선택 기준 (threshold)**:

```
threshold θ = LLC_cache_line_size / element_size × num_Exchange_GPUs

AMD MI100 기준 (cache line 128B, element 4B, GPU 4개):
θ = 128 / 4 × 4 = 128

selectivity < 1/128 → zero-copy 사용 (살아남는 행이 1% 미만)
selectivity ≥ 1/128 → Exchange 사용 (배치 전송이 더 효율)
```

→ selectivity가 낮으면 zero-copy로 필요한 것만 cache line 단위로 가져옴.

---

## AI 추론과의 공존

Vortex는 analytics GPU가 다른 AI 추론 GPU의 유휴 PCIe를 빌려쓰는 구조. AI 추론 성능에 미치는 영향:

- AI 추론 GPU 입장에서는 자신의 PCIe를 일부 내어주는 셈
- **평균 6.8% 성능 저하**에 불과 → compute-bound 워크로드는 PCIe 거의 안 쓰기 때문

---

## 성능 결과

**실험 환경**: AMD MI100 4-GPU 서버, Star Schema Benchmark SF=1000 (1TB), 데이터를 GPU 메모리에 캐싱 없이 전부 CPU DRAM에서 처리

| 비교 대상 | Vortex 성능 |
|---|---|
| GPU 기반 Proteus (최신 baseline) | **5.7x** 빠름 |
| CPU 기반 DuckDB | **3.4x** 빠름 |
| DuckDB 대비 비용 효율 | **2.5x** 향상 |

**Exchange primitive 단독 효과**: 단일 GPU 대비 4-GPU 합산 시 대역폭이 거의 선형에 가깝게 증가.

---

## Scaling GPU DB (Li et al.) 와의 비교

두 논문 모두 GPU 메모리 초과 문제를 다루지만 접근이 다름:

| | Scaling GPU DB (VLDB 2025) | Vortex (VLDB 2024) |
|---|---|---|
| 전략 | CPU가 scan, GPU가 join | GPU가 전부, PCIe 병렬화 |
| 전제 | 단일 GPU | 다중 GPU 서버 (AI 추론 공존) |
| PCIe 병목 해결 | 전송량 자체를 줄임 (압축 scan) | 가용 PCIe 대역폭을 늘림 |
| 적합한 환경 | 소규모 GPU 서버 | AI+Analytics 혼합 워크로드 서버 |

---

## 핵심 인사이트

| # | 인사이트 |
|---|---|
| 1 | AI 추론 GPU의 idle PCIe 대역폭을 analytics에 재활용 → 새로운 하드웨어 없이 대역폭 N배 확보 |
| 2 | H2D/D2H 트래픽을 독립적으로 보면 안 됨, 같이 관리해야 균형 유지 |
| 3 | IO 스케줄링과 커널 코드 분리(ExKernel)가 기존 라이브러리(rocPRIM/CUB) 재사용 가능하게 함 |
| 4 | Zero-copy는 selectivity가 극히 낮을 때만 유효 (threshold 공식으로 결정) |
| 5 | compute-bound 워크로드와 memory-bound 워크로드를 같은 서버에서 혼합하면 리소스 효율 극대화 |
