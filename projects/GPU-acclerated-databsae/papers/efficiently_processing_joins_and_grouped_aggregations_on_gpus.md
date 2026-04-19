# Efficiently Processing Joins and Grouped Aggregations on GPUs

- **저자**: Bowen Wu, Dimitrios Koutsoukos, Gustavo Alonso (ETH Zurich)
- **학회**: SIGMOD 2025
- **링크**: https://dl.acm.org/doi/10.1145/3709689 / https://arxiv.org/abs/2312.00720

---

## 문제 정의

기존 GPU 기반 DB 연산자들의 3가지 문제:

1. **Random access 오버헤드** — 거의 모든 연산자에서 런타임의 최대 75%를 차지
2. **Group-by 미탐구** — Join과 함께 많이 쓰이는데 GPU 가속 연구가 부족
3. **평가 편향** — 기존 연구들이 제한적이고 비대표적인 워크로드로만 평가

---

## 핵심 기여

| 기여 | 내용 | 성능 향상 |
|---|---|---|
| **GFTR** | Random access를 줄이는 새 기법 | 최대 2.3x |
| **Hash-based Group-by 최적화** | 기존 해시 구현 개선 | 19.4x |
| **Sort-based Group-by 최적화** | 정렬 기반 구현 개선 | 1.7x |
| **Partition-based Group-by** | 높은 group cardinality에 적합한 새 알고리즘 | — |

- **비용 모델**: 최적화 효과를 사전에 예측 가능
- **실용 휴리스틱**: 쿼리 옵티마이저가 워크로드에 맞는 구현을 자동 선택하도록 가이드

---

## 문제 원인과 해결 방법

### 1. Random Access 문제 → GFTR

**왜 느린가?**

Sort-Merge Join / Partitioned Hash Join은 공통적으로 3단계를 거친다:
1. **Transform**: 키를 정렬하거나 분할해서 tuple ID를 재배열
2. **Match Finding**: 정렬/분할된 키에서 일치하는 쌍 탐색
3. **Materialization**: 원본 데이터에서 payload를 수집해서 출력

문제는 3단계에서 발생한다. Transform 단계에서 tuple ID가 무작위로 섞이기 때문에, 원본 payload를 읽을 때 `payload[permuted_tid[i]]` 형태의 불규칙한 메모리 접근이 발생하고 캐시 라인을 낭비한다.

**GFTR의 해결책**

| | 기존 GFUR | GFTR |
|---|---|---|
| Transform 대상 | 키만 변환 | 키 + payload 함께 변환 |
| Materialization | `payload[permuted_tid[i]]` (불규칙) | `transformed_payload[i]` (순차) |

payload를 키와 동일한 정렬/분할 연산으로 함께 변환해두면, 이후 접근이 순차적으로 바뀐다.

- **비용 모델**: match ratio > 0.39이면 GFTR이 GFUR보다 빠름
- **Partitioned Hash Join 주의점**: 기존 bucket-chain 방식은 비결정성(동일 키에 다른 순서) + fragmentation 문제가 있어서 radix-sort 기반 분할로 교체해야 GFTR 적용 가능

---

### 2. Hash-based Group-by 문제 → 최적화된 2단계 처리

#### 기존 구현 (cuDF) 의 문제

cuDF는 전역 해시 테이블에 `(key, tuple_id)` 쌍을 저장하고, 집계값은 별도 배열에 유지한다. 이때 두 가지 문제가 발생한다:

1. **그룹 수 적을 때**: 수백만 스레드가 동시에 소수의 집계 배열 슬롯에 atomicAdd → 심각한 경합(contention) → 직렬화
2. **그룹 수 많을 때**: 해시 테이블 삽입/조회 + 집계 배열 접근 모두 불규칙한 global memory 접근 → 대역폭 낭비

#### 최적화 알고리즘 (Algorithm 1)

**Stage 1 — Group ID 할당**

```
compute_group_id(R.key, HashTable, groupId, groupIdLoc, numGroups)
```

- 각 스레드 i가 자신의 키를 해시 테이블에 삽입 시도
- 처음 보는 키 → `atomicAdd(numGroups, 1)`로 그룹 번호 g 발급
- `groupIdLoc[i]` = 해시 테이블 내 해당 키의 위치 P 저장
- 이후 `groupId[P]`를 조회하면 실제 그룹 번호 g를 알 수 있음

이중 인디렉션이 필요한 이유: 삽입 시점엔 아직 그룹 번호가 확정되지 않을 수 있어서, 해시 테이블 위치를 먼저 기록해두고 나중에 역참조한다.

**Stage 2 — 적응형 집계**

```
compute_agg(R, F, groupId, groupIdLoc, A, numGroups)
```

그룹 수 g를 이미 알고 있으므로 shared memory 수용 가능 여부를 사전 판단:

```
그룹당 필요 공간 = 집계 함수 수 × 8바이트
예) 함수 4개 → 그룹당 32바이트 → SM당 최대 ~1,500개 그룹
```

- **shared memory 충분한 경우** (그룹 수 < 임계값):
  1. 스레드 i → `groupIdLoc[i]`로 해시 테이블 위치 P 조회
  2. `groupId[P]`로 실제 그룹 번호 g 획득
  3. shared memory의 `G[g]`에 atomicAdd/atomicCAS로 로컬 집계
  4. `__syncthreads()` 후 스레드 블록의 로컬 결과를 global memory로 flush

- **shared memory 부족한 경우** (그룹 수 ≥ 임계값):
  - global memory 배열 A에 직접 원자 연산 (폴백)

#### 집계 함수별 원자 연산

| 함수 | 연산 |
|---|---|
| SUM, COUNT | `atomicAdd` |
| MAX, MIN | `atomicCAS` (Compare-And-Swap) |

#### 성능 특성

| 그룹 수 | 전략 | 이유 |
|---|---|---|
| 소수 (< ~1K) | shared memory | 경합 최소화 |
| 중간 (1K ~ 10K) | shared memory | 최적 균형 |
| 많음 (> 2^14) | global memory 폴백 또는 partition-based로 전환 | shared memory 용량 초과 |

전역 메모리 atomic 횟수를 스레드 블록 수만큼으로 줄이는 것이 핵심 → **최대 19.4x 향상**

---

### 3. 높은 Group Cardinality 문제 → Partition-based Group-by

**왜 필요한가?**

그룹 수가 많으면 hash-based도 random access 문제가 되살아난다. Partitioned Hash Join의 아이디어를 group-by에 적용.

**동작 방식**

1. **Transform**: 키 + payload를 radix-partition으로 함께 분할
2. **Group**: 각 partition을 thread block에 할당
3. **Aggregation**: 파티션 크기에 따라 전략 선택
   - 그룹 1개: block-wide reduction
   - 소수 그룹: shared memory에서 aggregation
   - 많은 그룹: global memory에서 aggregation

파티션 단위로 그룹 수가 줄어들어 atomic 경합이 감소.

#### 상세 동작

**Step 1 — Radix Partition (Transform)**

group key와 모든 payload를 **같은 파티셔닝 연산으로 함께** 분할한다 (GFTR 적용).
- key의 특정 비트 범위(radix bits)를 기준으로 분할
- 결과 배열은 메모리상 연속 배열로 파티션들이 순서대로 저장됨

**Step 2 — 파티션을 Thread Block에 할당**

각 파티션을 thread block 하나에 배정. Thread block 내부에서는 **shared memory에 해시 테이블**을 올려서 처리.

바로 집계하지 않고 **모든 key를 먼저 삽입한 뒤** 집계한다:
- 파티션 내 그룹 수와 각 그룹의 output offset을 미리 알아야 dense array 구성 가능
- 그룹 수를 알아야 최적 집계 전략 선택 가능

**Step 3 — 파티션 내 그룹 수에 따라 집계 전략 선택**

| 케이스 | 전략 |
|---|---|
| 그룹 1개 | block-wide reduction (가장 최적화) |
| 그룹 소수 | shared memory에 슬롯 할당 → 집계 후 global로 flush |
| 그룹 너무 많음 | global memory에 직접 집계 (폴백) |

#### 파티션이 예상보다 클 때 (Skew 처리)

| 원인 | 해결책 |
|---|---|
| radix bits 선택이 나빠서 skew | group key를 **해싱한 값 기준**으로 재파티셔닝. 어차피 grouping 단계에서 해싱 필요하므로 오버헤드 거의 없음 |
| 동일 key가 너무 많음 (그룹 수 자체가 적음) | 해당 파티션에만 **sort-based 또는 hash-based group-by로 전환** |
| 파티션 내 그룹 수가 너무 많아 shared memory 부족 | **파티션 수를 충분히 늘려서** 각 파티션이 최대 M개 tuple 이하가 되도록 보장 (M = shared memory가 수용 가능한 최대 그룹 수) |

---

### 4. Sort-based Group-by 정렬 비용 → Dictionary Encoding

**왜 느린가?**

payload 열이 여러 개일 때 각 열마다 정렬을 반복하면 비용이 크다.

**해결책**

1. 첫 번째 열 정렬로 그룹 수 g를 파악
2. 나머지 열은 원본 키 대신 0~(g-1)로 매핑한 encoded key로 정렬
3. encoded key는 log₂(g) 비트만 필요 → 정렬 패스 수 감소

payload 열이 2개 이상이면 대부분의 경우 이득.

---

## 알고리즘 선택 가이드 (휴리스틱)

| 조건 | 선택 |
|---|---|
| Match ratio < 0.39 | GFUR (partitioning overhead가 이득보다 큼) |
| Match ratio ≥ 0.39 | GFTR |
| 그룹 수 < 2^14 | Optimized hash-based group-by |
| 그룹 수 ≥ 2^14 | Partition-based group-by |

---

## 실험 결과 요약

| 기법 | 성능 향상 |
|---|---|
| GFTR (wide join) | 1.6 ~ 2.3x |
| Optimized hash-based group-by | 19.4x |
| Partition-based group-by | 2.88x |
| Dictionary encoding (sort-based) | 2.63x |

---

## 결론

> TODO: 정리 필요
