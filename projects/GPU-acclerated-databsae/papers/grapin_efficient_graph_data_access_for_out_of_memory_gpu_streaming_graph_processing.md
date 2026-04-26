# GRAPIN: Efficient Graph Data Access for Out-of-Memory GPU Streaming Graph Processing

- **저자**: Qiange Wang, Yongze Yan, Hongshi Tan, Cheng Chen 외 (NUS, Northeastern Univ, ByteDance)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3749646.3749659

---

## 문제 정의

### 스트리밍 그래프 처리란?

정적 그래프 처리는 한 번만 계산. **스트리밍 그래프 처리**는 그래프에 엣지/노드가 계속 추가·삭제되는 동안 알고리즘 결과(BFS, SSSP, PageRank 등)를 **지속적으로 최신 상태로 유지**.

예: ByteDance의 추천 시스템에서 사용자-콘텐츠 그래프가 실시간으로 변하면서 랭킹 결과를 계속 업데이트해야 함.

### GPU 메모리 초과 문제

대규모 그래프(수십억 엣지, ~55GB)는 GPU 메모리(A5000 = 24GB)에 다 들어가지 않음. CPU DRAM에 그래프를 두고 GPU가 on-demand로 접근해야 함.

이 때 두 종류의 **중복 접근(redundant access)**이 발생:

| 중복 유형 | 원인 |
|---|---|
| **Intra-batch redundancy** | 한 배치 내에서 이미 수렴한 노드도 다시 계산 → 불필요한 그래프 접근 |
| **Inter-batch redundancy** | 배치마다 반복적으로 같은 엣지 데이터를 CPU→GPU 전송 |

```
SSSP on 3.07B 엣지 그래프 (100K 엣지 변경 배치):
  전체 엣지 접근량: 157.12B
  실제 변경된 엣지 비중: 16.4M (0.01%)
  중복 접근: 137B+ (대부분이 낭비)
```

---

## 핵심 기술 1: DM 기반 증분 계산 (Incremental Computation)

### Dependency Memoization (DM) 알고리즘

기존 naive 방식: 그래프가 바뀔 때마다 **처음부터 재계산**.

DM 알고리즘: 각 노드가 자신의 결과에 영향을 준 **부모 노드(dependency)를 기억**. 변경이 생기면 영향받는 노드만 재계산.

```
SSSP 예:
  Result[v] = 최단 경로 거리
  Parent[v] = 해당 경로에서 v의 직전 노드

엣지 (u→w) 삭제 시:
  1. Parent[w] == u인 노드들 → 경로 무효화 (invalid 전파)
  2. 무효화된 노드들만 재계산
  → 영향받지 않는 노드들은 건드리지 않음 → intra-batch 중복 제거
```

### GPU에서 DM이 어려운 이유

DM은 **Result와 Parent를 원자적으로 동시에 업데이트**해야 함. 예를 들어 더 짧은 경로를 찾았으면 Result와 Parent를 동시에 바꿔야 일관성 유지.

문제: GPU의 native atomic 연산(CAS)은 **단일 값(4바이트 또는 8바이트)**만 지원. Result + Parent = 2개 필드를 동시에 바꾸는 원자 연산 없음.

기존 해결책: exclusive lock → 수천 스레드가 동시에 lock 경합 → GPU 대규모 병렬성 낭비.

### GRAPIN의 해결책: Coupled Update (분리된 CAS)

**핵심 관찰**: 모노토닉 알고리즘(SSSP의 min, BFS의 level 등)에서 Result는 항상 최적값으로 수렴. Parent는 최종 최적 Result에 대응하는 값만 중요하고, 중간 과정의 Parent는 나중에 덮어써도 됨.

```
알고리즘:

Step 1: Result 원자적 업데이트 (atomicMin 등)
  old_path = Result[dst]
  new_path = Result[src] + weight
  atomicMin(&Result[dst], new_path)

Step 2: Result 업데이트에 성공한 스레드들이 CAS로 Parent 경쟁
  old_p = Parent[dst]
  repeat:
    old_p = atomicCAS(&Parent[dst], old_p, src)
  until (new_path != Result[dst])  // 더 좋은 값이 나왔으면 종료
     || (src == Parent[dst])       // 내가 이미 Parent면 종료
```

→ 2개의 독립적인 CAS 연산으로 분리. 무거운 lock 불필요. GPU의 대규모 병렬성과 완벽히 호환.

**정확성 보장**: Result가 먼저 최적값으로 수렴한 후, Parent도 그 최적 Result에 해당하는 값으로 결국 수렴함이 증명됨 (모노토닉 속성 덕분).

---

## 핵심 기술 2: Hot Subgraph 관리 프레임워크

Intra-batch 중복은 증분 계산으로 제거. **Inter-batch 중복**(배치마다 같은 엣지 반복 전송)은 캐싱으로 해결.

### 핫니스 스코어 추적

각 노드가 최근 배치에서 얼마나 자주 접근됐는지 **슬라이딩 윈도우**로 추적:

```
hotness(v) = Σ AFQ(v, batch_i)  (최근 W개 배치에 대한 합)
AFQ(v, b) = 배치 b에서 v의 접근 빈도
```

**메모리 효율적 구현**: AFQ가 보통 수십~수백 범위 → 1바이트면 충분.
- 4바이트 정수 1개로 최근 4개 배치의 AFQ를 각 1바이트씩 저장
- 배치 완료 후 우측 1바이트 shift → 가장 오래된 배치 제거, 새 배치 기록
- 오버헤드가 매우 작음 (노드당 4바이트 추가)

### 후보 노드 선택

```
1. 모든 노드를 hotness 기준 내림차순 정렬 (GPU Thrust BlockRadixSort)
2. 상위 노드부터 누적 이웃 크기를 합산 (BlockScan)
3. GPU 캐시 용량을 초과하지 않는 선에서 상위 K개 선택 (이진 탐색)
```

→ 자주 접근되는 subgraph를 GPU HBM에 캐싱. 캐시 hit → CPU 전송 없음.

### Snapshot-Oriented 캐시 교체 + Chunk 기반 메모리 관리

그래프가 업데이트될 때 캐시도 갱신해야 함. CSR 구조(Compressed Sparse Row)에서 노드별 이웃 크기가 달라서 직접 교체하면 메모리 재편 비용이 큼.

**해결책: Chunk 기반 CSR 관리**

```
GPU 메모리를 동일 크기 청크(32MB)로 분할

캐시 교체 시:
  - 제거할 노드의 데이터가 있는 청크만 무효화/압축
  - 새 노드 데이터를 빈 청크에 로드
  - 영향받지 않는 청크는 그대로 유지
```

전체 CSR 재편 없이 영향받는 청크만 수정 → 교체 오버헤드 최소화.

---

## 핵심 기술 3: GPU 최적화 그래프 자료구조 (PMA-CSR)

동적 그래프는 삽입/삭제가 잦아서 일반 CSR(정적 배열)은 비효율. 기존 GPMA(GPU Packed Memory Array)는 CSR 대비 48% 성능.

### GRAPIN의 PMA-CSR

모든 노드의 이웃을 **단일 PMA 배열**에 연속 저장. 노드별로 `(start, end)` 인덱스로 이웃 범위 접근:

```
삽입: 이웃 리스트 끝에 바로 append (공간 있으면)
      공간 부족 시: 재귀적으로 인접 공간 확보 (adjustment)
삭제: 삭제 플래그만 표시 (lazy deletion)
```

**캐시라인 정렬 최적화**: 각 노드의 이웃 리스트 시작 위치를 **128바이트 배수**로 정렬. GPU warp가 PCIe를 통해 캐시라인(128B) 단위로 접근할 때 정렬이 맞아야 단일 요청으로 처리 가능.

**Zero-copy 접근**: 이웃 데이터를 pinned memory(cudaMallocHost)에 저장. GPU compute unit이 PCIe를 통해 직접 캐시라인 단위로 접근. SDMA 없이 demand 방식으로 필요한 것만 가져옴.

---

## 시스템 구조

```
CPU DRAM: [전체 그래프 (PMA-CSR, pinned memory)]
                    ↕ zero-copy (캐시 miss 시)
GPU HBM:  [Hot Subgraph Cache (CSR, chunk 관리)]
                    ↓
          [DM 기반 증분 계산 엔진 (vertex-centric)]
                    ↓
          [결과 배열 (Result, Parent)]
```

배치 처리 흐름:
1. 그래프 변경 배치 수신 → Hot subgraph 캐시 갱신 (snapshot-oriented 교체)
2. 증분 계산 시작: 변경에 영향받는 노드만 활성화
3. GPU 커널에서 vertex-centric 처리:
   - 캐시 hit → GPU HBM에서 이웃 접근
   - 캐시 miss → zero-copy로 CPU DRAM에서 직접 접근
4. Coupled CAS로 Result + Parent 업데이트
5. 수렴까지 반복

---

## 성능 결과

**실험 환경**: NVIDIA A5000 단일 GPU, 그래프 최대 3.07B 엣지 (55GB), BFS/SSSP/CC/PageRank

### 데이터 전송 감소

| 최적화 | 효과 |
|---|---|
| 증분 계산 단독 (Grapin-ZC) | 데이터 전송 **61%** 감소 |
| Hot subgraph 캐싱 추가 (Grapin) | 남은 전송량 **72%** 추가 감소 |
| 합계 | **89%** 전송 감소 |

### 속도 비교

| 비교 대상 | Grapin 성능 |
|---|---|
| CPU 기반 RisGraph | **1.8x ~ 96.9x** (평균 16.2x) |
| CPU 기반 Ingress | **7.4x ~ 70.2x** (평균 19.4x) |
| GPU 기반 SHGraph | SHGraph는 큰 그래프에서 OOM → Grapin만 성공 |

- PageRank: 부동소수점 연산이 많아 증분 계산 이득이 상대적으로 적음 (그래도 평균 33x vs BFS)
- 그래프 크기가 커질수록 Grapin의 이점이 더 커짐 (억 단위 8x → 수십억 단위 14~46x)

---

## 핵심 인사이트

| # | 인사이트 |
|---|---|
| 1 | 스트리밍 그래프에서 중복 접근이 전체 접근량의 90%+를 차지 → 이를 제거하는 게 핵심 |
| 2 | GPU의 native CAS로 DM 알고리즘 구현 가능 (Result/Parent를 독립 CAS로 분리) |
| 3 | 모노토닉 수렴 속성을 이용하면 원자성 보장 없이도 정확성 유지 가능 |
| 4 | 핫니스 추적은 노드당 4바이트면 충분, GPU에서 병렬 처리 가능 |
| 5 | Zero-copy와 캐싱을 조합: 자주 쓰이는 것은 캐싱, 드물게 쓰이는 것은 zero-copy |
| 6 | Chunk 기반 메모리 관리로 동적 그래프 업데이트 시 CSR 전체 재편 비용 회피 |
