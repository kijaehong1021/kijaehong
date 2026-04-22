# GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP

- **저자**: Nils Boeschen, Tobias Ziegler, Carsten Binnig (TU Darmstadt)
- **학회**: SIGMOD 2024
- **링크**: https://doi.org/10.1145/3698812

---

## 문제 정의

SSD 기반 DB는 비용이 낮지만, 인메모리 시스템 대비 성능 격차가 존재:

| 시스템 | 유효 대역폭 | 비용 |
|---|---|---|
| In-memory DB | ~28 GiB/s (HBM 기준) | DRAM 비쌈 |
| SSD-based DB | ~20 GiB/s (raw SSD 대역폭) | 저렴 |
| SSD + 압축 + CPU 해제 | CPU 병목으로 개선 안 됨 | - |

GPU를 연산 가속기로만 쓰던 기존 방식 (CPU가 SSD → DRAM → GPU로 데이터 전달)에서는 CPU가 병목.

**핵심 관찰**: CPU는 heavy-weight 압축 해제 속도가 느리지만, GPU는 수천 코어로 빠르게 해제 가능. 게다가 NVIDIA GPUDirect Storage(GDS)로 SSD → GPU HBM 직접 전송이 가능해짐.

---

## 핵심 아이디어: GPU-in-Data-Path

GPU를 연산 가속기가 아닌 **I/O 경로 자체에 배치**. SSD에서 GPU로 직접 데이터를 스트리밍하고, GPU가 압축 해제 + 쿼리 실행을 모두 처리.

```
기존 방식:
SSD → [CPU DRAM] → [PCIe] → GPU
              ↑ CPU가 압축 해제 (느림)

GOLAP:
SSD ──────────────────→ GPU HBM   (GDS: GPUDirect Storage)
                         ↓
                      압축 해제 (GPU에서 on-the-fly)
                         ↓
                      쿼리 실행 (scan, filter, join, agg)
```

3단계를 통해 유효 대역폭을 SSD 원시 대역폭(20 GiB/s)에서 **100 GiB/s**까지 끌어올림.

---

## 핵심 기술 1: GPUDirect Storage (GDS) 기반 직접 전송

기존 방식:
```
SSD → CPU pinned memory (bounce buffer) → GPU HBM
```
- 데이터가 CPU 메모리를 거쳐야 해서 CPU-GPU PCIe 대역폭이 병목

GDS 방식:
```
SSD → GPU HBM  (DMA 컨트롤러가 직접 전송, CPU bypass)
```
- CPU는 전송 제어(I/O 요청 발행)만 하고, 데이터 자체는 직접 이동
- CPU 메모리 bounce buffer 불필요

**구현 디테일**: GDS에는 동기(synchronous)와 비동기(asynchronous) 두 방식이 있음.
- 비동기 방식이 I/O 스레드 수를 줄일 수 있을 것 같지만, GDS 드라이버 내부에서 직렬화가 발생 → 최대 대역폭 미달
- **GOLAP은 동기 GDS 사용**: 여러 I/O 스레드가 병렬로 동기 cuFileRead 호출 → 안정적으로 최대 SSD 대역폭 달성

---

## 핵심 기술 2: On-the-fly 압축 해제 (Heavy-weight Compression)

### Heavy-weight vs Light-weight 압축

| 구분 | 알고리즘 예 | 압축률 | GPU 해제 속도 |
|---|---|---|---|
| Light-weight | FOR, bit-packing, RLE | 낮음 | 매우 빠름 |
| Heavy-weight | ZStandard, LZ4, Snappy | **높음** | GPU에선 빠름, CPU에선 느림 |

GOLAP이 heavy-weight를 선택한 이유:
- 더 높은 압축률 → SSD에서 읽어야 할 바이트 수 감소 → 실효 대역폭 증가
- CPU는 이 압축 해제에서 병목이지만, GPU는 수천 코어로 빠르게 처리 가능 (nvCOMP 라이브러리 활용)

### 압축 해제와 쿼리 실행 파이프라인 중첩

```
I/O 스레드들: [청크1 로드] [청크2 로드] [청크3 로드] ...
GPU 스트림:         [청크1 압축해제+scan]   [청크2 압축해제+scan] ...
```

- 고정 크기 청크 단위로 컬럼 데이터 저장 (튜플 수는 설정 가능, 같은 컬럼 내 모든 청크 동일 크기)
- 같은 크기이므로 fixed-size 버퍼 재사용 → GPU 메모리 할당 오버헤드 없음
- 압축 해제 결과를 downstream 오퍼레이터(filter, join 등)에 즉시 파이프라이닝

### 적응형 압축 파라미터 선택

압축 알고리즘과 청크 크기 조합에 따라 성능이 다름 (큰 청크 → 높은 압축률, 작은 청크 → 높은 pruning 효과). GOLAP은 **샘플링 기반 adaptive compression 전략**으로 컬럼별 최적 파라미터 결정.

---

## 핵심 기술 3: GPU-accelerated Opportunistic Pruning

SSD에서 데이터를 읽기 전에 **불필요한 청크를 미리 제거**해서 실제 I/O 양을 줄임.

### 동작 방식

```
1. 각 청크에 대해 메타데이터(min/max, bloom filter, histogram) 사전 저장

2. 쿼리 실행 전, GPU가 청크 메타데이터를 병렬로 평가:
   - 각 단순 predicate에 대해 청크별 pass/prune 비트맵 생성
   - 여러 predicate의 비트맵을 bitwise AND/OR로 결합
   → 최종 pruning 비트맵 생성

3. 비트맵을 CPU로 전송
   I/O 스레드가 pruning 비트맵 기준으로 skip할 청크 건너뜀
   → 필요한 청크만 SSD에서 로드
```

### Pruning 메타데이터 종류

| 기법 | 설명 | 특성 |
|---|---|---|
| **MinMax** | 청크 내 최솟값/최댓값 저장 | 간단, 정렬/균등 분포에 효과적 |
| **Bloom Filter** | 청크에 특정 값 존재 여부 확률적 판단 | point predicate에 유효 |
| **Histogram** | 256개 bin으로 값 분포 기록 | 가장 높은 pruning 비율 달성, 스토리지 오버헤드 있음 |

**Opportunistic**: pruning 자체가 GPU에서 병렬로 처리되어 오버헤드가 거의 없음. pruning 효과가 없는 경우에도 성능 손실 없이 그냥 전체 데이터 로드.

### 데이터 클러스터링 + 소팅으로 pruning 효율 향상

fact 테이블을 dimension 필터 속성 기준으로 정렬/클러스터링하면, 같은 범위의 값들이 같은 청크에 모여 MinMax pruning 효과 극대화.

---

## 핵심 기술 4: GPU-CPU Co-execution (GPU 메모리 초과 시)

쿼리의 downstream 오퍼레이터(join, aggregation)가 GPU 메모리를 초과하는 경우:

```
GPU가 처리 가능한 범위:
  SSD → GPU: 압축 해제 + scan + filter → 중간 결과

GPU 메모리 초과 부분:
  중간 결과 → PCIe → CPU: join, aggregation 처리

최종 결과 → CPU에서 반환
```

- GPU의 강점(scan + 압축해제)은 유지
- CPU의 강점(대용량 메모리)으로 downstream 처리
- **최악의 경우**: downstream 전체를 CPU로 보내야 하면, 압축된 데이터가 PCIe를 타야 함 → SSD-based CPU 시스템 수준으로 성능 저하 가능. 하지만 실제 OLAP 워크로드는 selective해서 대부분 GPU에서 처리 가능.

---

## 유효 대역폭 증폭 효과

각 기술이 유효 대역폭에 기여하는 방식:

```
SSD raw 대역폭:        ~20 GiB/s
  ↓
+ Heavy-weight 압축:   압축률 ~2.8x → 실효 ~56 GiB/s
  ↓
+ Opportunistic Pruning: 불필요 청크 skip → 최대 ~100 GiB/s
  ↓
GOLAP 달성 유효 대역폭: 최대 100 GiB/s
```

→ 인메모리 시스템의 대역폭 한계(~28 GiB/s)를 **초과**.

---

## 성능 결과

**실험 환경**: SSB(Star Schema Benchmark), TPC-H, 실제 택시 데이터셋

| 비교 | 결과 |
|---|---|
| SSD-based DBMS (압축 없음) | GOLAP이 대폭 앞섬 |
| In-memory DBMS (CPU) | GOLAP이 동등하거나 초과 |
| 유효 대역폭 최대 | **100 GiB/s** (SSD 20 GiB/s에서 5x 증폭) |
| 압축률 | ~2.8x (GOLAP, In-memory 동일 설정) |

**비용 효율성**: 인메모리 시스템은 데이터 크기에 비례해 DRAM 비용이 증가. GOLAP은 GPU 고정 비용 + 저렴한 SSD만 추가하면 되어 대용량 데이터에서 훨씬 경제적.

---

## 핵심 인사이트

| # | 인사이트 |
|---|---|
| 1 | GPU를 "연산 가속기"가 아닌 "I/O 경로의 일부"로 설계하면 SSD 대역폭 한계를 초과 가능 |
| 2 | Heavy-weight 압축이 CPU에선 병목이지만 GPU에선 오히려 실효 대역폭 향상 도구가 됨 |
| 3 | GPUDirect Storage로 CPU memory bypass → 전송 지연 제거 |
| 4 | GPU parallel pruning은 오버헤드가 거의 없어서 pruning 효과가 없어도 손실 없음 (opportunistic) |
| 5 | 청크 크기는 압축률과 pruning 효과 사이 trade-off → 압축 우선으로 최적화 권장 |
| 6 | DRAM 비용이 plateau된 현 시점에서 SSD+GPU 조합이 비용 효율적 대안 |
