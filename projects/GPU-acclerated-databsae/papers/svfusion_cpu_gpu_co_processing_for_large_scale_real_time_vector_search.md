# SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search

- **저자**: Yuchen Peng, Dingyu Yang, Zhongle Xie, Ji Sun, Lidan Shou, Ke Chen, Gang Chen (Zhejiang Univ, Huawei)
- **학회**: VLDB 2026
- **링크**: https://doi.org/10.14778/3796195.3796216

---

## 문제 정의

ANNS(Approximate Nearest Neighbor Search)는 추천, 검색, RAG 등의 핵심 연산. 실시간 스트리밍 환경에서는 검색과 동적 업데이트(삽입/삭제)가 동시에 요구됨.

| 방식 | 장점 | 한계 |
|---|---|---|
| CPU 기반 (HNSW, FreshDiskANN) | 동적 업데이트 지원, 대용량 가능 | 처리량이 낮음 |
| GPU 기반 (GGNN, PilotANN) | 대규모 병렬 처리, 높은 처리량 | 동적 업데이트 어려움, GPU 메모리 한계 |

→ **높은 처리량 + 동적 업데이트 + 대용량**을 동시에 만족하는 솔루션이 없음.

---

## 배경: HNSW 그래프 기반 ANNS

HNSW(Hierarchical Navigable Small World)는 그래프 기반 ANNS의 대표 알고리즘. 벡터들을 노드로, 유사한 벡터들을 엣지로 연결하는 proximity graph를 구성.

**검색 과정**:
1. 랜덤 진입점에서 시작
2. 현재 노드의 이웃들 중 쿼리 벡터에 더 가까운 노드로 이동
3. 더 이상 가까워지지 않을 때 종료 → 상위 K개 반환

**GPU 가속의 이점**: 여러 쿼리를 동시에 처리, 이웃 거리 계산을 대규모 병렬화.

**동적 업데이트의 어려움**: 삽입 시 그래프 구조(엣지) 수정 필요 → GPU에서 동시 접근 제어가 복잡. 삭제 시 연결성 복구 필요.

---

## SVFusion 아키텍처

GPU-CPU-Disk 3-tier 협력 프레임워크.

```
┌─────────────────────────────────────────────────────┐
│                   Query / Update                    │
│                        ↓                            │
│  ┌──────────────────────────────────────────────┐   │
│  │             GPU HBM (hot 벡터 캐시)           │   │
│  │  · 1 TB/s 대역폭                             │   │
│  │  · hot 벡터 + 해당 subgraph 저장             │   │
│  │  · cache hit → GPU에서 거리 계산              │   │
│  └──────────────────────────────────────────────┘   │
│                    cache miss ↓                     │
│  ┌──────────────────────────────────────────────┐   │
│  │             CPU DRAM (전체 벡터)              │   │
│  │  · 수백 GB, 100 GB/s                         │   │
│  │  · cache miss → CPU에서 거리 계산             │   │
│  │  · 또는 GPU로 on-demand 전송 후 GPU 계산      │   │
│  └──────────────────────────────────────────────┘   │
│                        ↓                            │
│  ┌──────────────────────────────────────────────┐   │
│  │                Disk (확장 계층)               │   │
│  │  · 억~수십억 단위 대용량 확장                  │   │
│  │  · 비동기 prefetch                           │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 핵심 기술 1: GPU-CPU Co-processing 검색

### 검색 워크플로

```
입력: 쿼리 벡터 q, 그래프 인덱스 G
출력: 최근접 이웃 K개

1. 랜덤 진입점으로 candidate pool 초기화 (GPU 병렬 처리)
2. 반복:
   a. candidate pool에서 가장 가까운 노드 선택
   b. 해당 노드의 이웃들 조회 (FetchNeighbors)
   c. 각 이웃 벡터에 대해:
      - cache 상태 확인 (GPU HBM에 있는가?)
      - GPU 캐시 hit → GPU HBM에서 직접 거리 계산
      - GPU 캐시 miss → WVP 알고리즘 호출 (아래 설명)
   d. GPU 계산 결과 + CPU 계산 결과 합쳐서 candidate 업데이트
3. candidate 변화 없을 때까지 반복
```

**핵심**: 이웃들을 GPU 캐시 여부에 따라 두 그룹으로 나눠 GPU/CPU 병렬 계산.

```
이웃 벡터 집합 N
  → cache hit 집합  G_cached  → GPU에서 ParallelComputeDist
  → cache miss 집합 G_cpu     → CPU에서 ParallelComputeDist
두 결과 합쳐서 candidate 업데이트
```

---

## 핵심 기술 2: Workload-aware 벡터 캐싱 (WVP)

### 기존 캐싱과의 차이

일반적인 LRU/LFU는 **캐시 히트율 최대화**가 목표. 하지만 GPU-CPU co-processing에서는 캐시 미스 시 GPU 전송 비용 vs. CPU 계산 비용을 함께 고려해야 함.

- 벡터를 GPU로 올리면 전송 비용 발생 (PCIe 전송)
- 그냥 CPU에서 계산하면 전송 비용 없지만 느림
- → 올리는 게 이득인 경우에만 GPU 캐시에 올려야 함

### WVP 알고리즘 (Workload-aware Vector Placement)

캐시 미스 발생 시 두 단계로 결정:

**Phase 1: Selective Prefetch**

미래 접근 빈도 예측값 `f̂(v)`를 추정. 임계값 `λ`와 비교:

```
if f̂(v) < λ:
    → GPU 전송 안 하고, CPU에서 in-place 계산
       (빈도가 낮아서 올려봤자 손해)
else:
    → Phase 2로 진행 (GPU 캐시에 올릴 후보)
```

**Phase 2: Predictive Eviction**

GPU 메모리가 가득 찼을 때, 어떤 벡터를 내쫓을지 결정:

```
clock-sweep 기반 + 예측 빈도 활용:
  reference bit = 0인 벡터들 중에서
  f̂(v)가 가장 낮은 벡터를 evict
  → 앞으로 덜 쓰일 벡터를 내쫓음 (LRU보다 thrashing 방지)
```

**적응적 임계값 조정**: 스트리밍 중 cache miss rate와 평균 예측 빈도를 모니터링해서 `λ`를 동적 조정.

---

## 핵심 기술 3: 실시간 삽입/삭제 처리

### 삽입 (Insertion)

```
1. 새 벡터 v에 대해 GPU 기반 ANNS로 후보 이웃 탐색
   (on-demand 전송 포함)
2. CPU에서 heuristic ordering으로 최종 이웃 선택
   (거리 + 다양성 고려)
3. 역방향 엣지 삽입으로 양방향 연결성 유지
   (v의 이웃들이 v를 이웃으로 추가)
```

### 삭제 (Deletion)

직접 삭제하면 그래프 연결성이 깨질 수 있음. 두 단계로 처리:

1. **Lazy deletion**: 즉시 구조 변경 없이 bitset에 삭제 마킹만
   - 검색 시 삭제된 노드는 결과에서 제외
   - 즉각적이고 비용 낮음

2. **Asynchronous repair**: 백그라운드에서 연결성 복구
   - 삭제된 노드로 연결된 엣지 탐지
   - Topology-aware local repair: 삭제된 노드의 이웃들끼리 새 엣지 추가
   - 주기적 global consolidation: 전체 그래프 구조 정리

---

## 핵심 기술 4: 동시성 제어

쿼리(검색)와 업데이트(삽입/삭제)가 동시에 실행될 때 일관성 보장.

### Fine-grained Locking

```
검색: 이웃 리스트 순회 시 read lock
      → 여러 검색이 동시에 가능

삽입: 후보 탐색 시 read lock
      영향받는 이웃 리스트에만 write lock으로 업그레이드
      → 전체 그래프 lock이 아니라 필요한 부분만 잠금

삭제: 삭제 bitset 업데이트 시 exclusive lock
```

### Cross-tier 동기화

GPU HBM과 CPU DRAM에 동일 벡터의 복사본이 있을 때 일관성 유지:

```
삭제 발생:
  → GPU/CPU 양쪽 deletion bitset을 atomic하게 동시 업데이트
  → 검색 중 삭제된 벡터 만나면 건너뜀

그래프 구조 변경 (삽입, repair):
  → CPU DRAM에 먼저 커밋 (version 번호 증가)
  → GPU 캐시의 해당 항목은 구버전 유지
  → 비동기 배치로 GPU에 전파
  → 검색 중 version mismatch 감지 시 CPU로 fallback
```

### Multi-version 메커니즘

백그라운드 consolidation(대규모 그래프 정리) 중에도 검색 차단 없이 실행:

```
consolidation 시작 시:
  현재 그래프 메타데이터의 read-only 스냅샷 생성 (특정 version)
  
백그라운드 작업은 스냅샷 위에서 실행
  → 원본 그래프에 영향 없음

삽입/삭제는 active version에서 계속 진행
  → consolidation과 운영 업데이트 충돌 없음

consolidation 완료 후 결과 머지
```

---

## 실시간 조율: CUDA Multi-stream

```
CUDA Stream 구성:
  Search Stream 1  ─┐
  Search Stream 2  ─┼→ 여러 배치 쿼리 병렬 처리
  Search Stream N  ─┘
  Update Stream    ──→ 삽입/삭제 비동기 처리
```

- 배치 크기 조절: 처리량 중심이면 큰 배치, 지연시간 중심이면 소배치 즉시 실행
- Adaptive resource management: 워크로드 변동에 따라 각 stream의 GPU 메모리/compute 할당 동적 조정
- GPU 메모리를 stream별 독립 세그먼트로 분할 → lightweight spinlock으로 충돌 방지

---

## 성능 결과

**실험 환경**: 다양한 스트리밍 워크로드 (SlidingWindow, ExpirationTime, Cluster, MST 등)

| 비교 대상 | 검색 처리량 | 삽입 처리량 |
|---|---|---|
| FreshDiskANN | **9.5x** 향상 | **71.8x** 향상 |
| 전체 평균 | **20.9x** 향상 | - |
| 지연시간 | **1.3x ~ 50.7x** 감소 | - |

- recall@10: **91~96%** 유지 (FreshDiskANN보다 0.4~3.4% 높음)
- HNSW는 삭제 repair 전략이 없어 그래프 fragmentation 누적 → recall 가장 낮음

---

## 핵심 인사이트

| # | 인사이트 |
|---|---|
| 1 | GPU 메모리를 "전체 데이터 저장소"가 아닌 "hot 벡터 캐시"로 사용하면 메모리 한계 극복 가능 |
| 2 | 캐시 미스 시 GPU 전송 vs. CPU 계산 비용을 비교해서 결정해야 함 (단순 히트율 최대화가 목표가 아님) |
| 3 | Lazy deletion + 비동기 repair로 삭제 비용을 분산시켜 지연시간 스파이크 방지 |
| 4 | Fine-grained locking + version 기반 동기화로 GPU/CPU 간 일관성 유지하면서 동시성 확보 |
| 5 | CUDA multi-stream으로 검색과 업데이트를 물리적으로 분리 → 서로 간섭 최소화 |
