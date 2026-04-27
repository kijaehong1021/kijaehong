# GPU Hash Table 테크닉 정리

> 참고 논문/구현: BGHT (Awad et al., 2021), SlabHash (Ashkiani et al.), DyCuckoo (ICDE 2021), Hive (2025), WarpSpeed (2025), cuCollections (NVIDIA)

---

## 왜 GPU Hash Table이 어려운가

CPU hash table을 그대로 GPU에 올리면 성능이 안 나옴. GPU 특성에 맞는 설계가 필요:

| CPU 가정 | GPU 현실 |
|---|---|
| 스레드 하나가 한 키를 처리 | 32개 스레드(warp)가 항상 같은 명령어 실행 |
| 자유로운 분기(if/else) | 분기하면 다른 스레드가 기다려야 함 (warp divergence) |
| 순차 접근이 빠름 | warp 내 32개 접근이 연속해야 빠름 (coalesced access) |
| atomic 자주 써도 괜찮음 | atomic 경쟁이 많으면 throughput 급락 |

---

## 1. Linear Probing (선형 탐사)

충돌 시 다음 빈 슬롯을 순서대로 탐색.

```
삽입: hash(k) 위치부터 빈 슬롯 찾을 때까지 순차 탐색
조회: hash(k)부터 키가 나오거나 빈 슬롯 만날 때까지 순차 탐색
```

**GPU에서 나쁜 이유**:
- **클러스터링**: 충돌이 생기면 탐사 길이가 연쇄적으로 늘어남
- **Warp divergence**: 스레드마다 탐사 횟수가 달라 warp 내 일부 스레드가 기다림
- **랜덤 접근**: 클러스터 크기가 가변적 → 비연속 메모리 접근 → 캐시 오염

---

## 2. Cuckoo Hashing (뻐꾸기 해싱)

**현재 GPU에서 가장 검증된 방식**. 각 키에 해시 함수 2~3개를 배정, 충돌 시 기존 항목을 다른 위치로 쫓아냄.

```
키 k에 위치 h1(k), h2(k), h3(k) 세 개 배정

삽입:
  h1(k) 비어있으면 저장 → 완료
  아니면 h1(k)에 있던 항목을 쫓아내고 k 저장
  쫓겨난 항목은 자신의 다른 위치(h2, h3)로 재배치
  → 연쇄 쫓아내기, 사이클 감지 시 rehash

조회:
  h1(k), h2(k), h3(k) 세 위치만 확인 → 항상 O(1)
```

**장점**: 조회가 항상 상수 시간, 높은 부하율(0.99) 달성 가능

**GPU 친화적인 이유**:
- 조회 시 확인할 위치가 고정(2~3개) → warp divergence 없음
- 위치들을 병렬로 동시 조회 가능

### Bucketed Cuckoo (BCHT)

슬롯 하나 대신 **버킷(8~32개 슬롯 묶음)** 단위로 cuckoo 적용:

```
버킷 크기 = 8 (A100 캐시라인에 맞춤)
각 키는 버킷 B1, B2 두 곳 중 하나에 저장

삽입: B1 버킷 내 빈 슬롯 찾기 → 없으면 B2 → 없으면 쫓아내기
조회: B1, B2 버킷 전체를 SIMD로 한번에 스캔
```

- **0.99 load factor에서 평균 1.43회 탐사**
- 버킷 단위로 coalesced 접근 → GPU 메모리 효율 극대화
- 현재 static GPU hash table 중 최고 성능

---

## 3. Power-of-Two-Choice (두 후보 중 선택)

삽입 시 두 버킷을 보고 **더 비어있는 쪽** 선택 → 자연스러운 부하 균형.

```
삽입:
  버킷 B1 = h1(k), 버킷 B2 = h2(k)
  occupancy(B1) < occupancy(B2) → B1에 삽입
  아니면 B2에 삽입

조회:
  B1, B2 두 버킷만 확인
```

- Cuckoo보다 삽입이 단순 (쫓아내기 없음)
- 부하가 고르게 분산 → 클러스터링 없음
- Iceberg hashing의 기반 아이디어

---

## 4. Iceberg Hashing

두 계층 버킷으로 구성:

```
┌──────────────────────────────────────┐
│  Primary Table (큰 버킷, 빠름)        │
│  대부분의 키가 여기에 저장됨            │
├──────────────────────────────────────┤
│  Secondary Table (작은 오버플로우)     │
│  Primary에서 넘치는 키 처리            │
└──────────────────────────────────────┘
```

삽입 시 Power-of-Two-Choice로 Primary 버킷 선택, 넘치면 Secondary로.

- 조회: Primary 두 버킷 + Secondary 한 버킷 = 최대 3번 접근
- 높은 부하율에서도 stable한 성능
- WarpSpeed 라이브러리에서 insert 속도 최고 수준

---

## 5. Fingerprint 기반 메타데이터 최적화

버킷 전체를 읽기 전에 **fingerprint(해시의 일부 비트)로 미리 필터링**:

```
저장 시:
  key = 0xABCD1234
  fingerprint = 상위 8비트 = 0xAB
  버킷에 (fingerprint=0xAB, full_key=0xABCD1234) 함께 저장

조회 시:
  1. fingerprint만 비교 (캐시라인 1번, 매우 빠름)
  2. fingerprint 매칭되는 슬롯만 full key 비교
  → 대부분의 경우 full key 비교 생략
```

**효과**: 캐시라인 접근 횟수 대폭 감소 → throughput 20~30% 향상

---

## 6. Warp-Cooperative 설계 (GPU 핵심)

GPU warp(32 스레드)가 **하나의 해시 오퍼레이션을 협력해서 처리**.

### WABC (Warp-Aggregated Bitmask Claim)

```
warp 내 32개 스레드가 각자 다른 키를 삽입하려 함
→ 같은 버킷을 노리는 스레드들을 bitmask로 집계
→ 대표 스레드 하나가 atomic 연산 1번으로 모두 처리
→ 32개 atomic → 1개 atomic으로 경쟁 감소
```

### WCME (Warp-Cooperative Match-and-Elect)

```
조회 시:
  warp 내 모든 스레드가 버킷 내 다른 슬롯 담당
  동시에 비교 → ballot() 명령어로 매칭 결과 집계
  → lock-free, 완전 병렬
```

**핵심**: 64-bit 정렬된 버킷 슬롯 → 단일 CAS로 원자적 key-value 업데이트 가능.

---

## 7. Dynamic Resizing 전략

정적 해시 테이블은 꽉 차면 전체를 새로 만들어야 함 (비용 큼). 동적 테이블의 세 가지 접근:

### Linked Slab (SlabHash)

```
버킷 = 슬랩(slab) 연결 리스트
가득 차면 새 슬랩을 뒤에 추가

[슬랩1] → [슬랩2] → [슬랩3] → ...
```

- **장점**: 증분적 확장, 전체 rehash 없음
- **단점**: pointer-chasing → 비연속 메모리 접근, 삭제 마커로 인한 메모리 낭비

### Subtable 방식 (DyCuckoo)

```
해시 테이블 = 여러 개의 서브테이블
[서브테이블 0] [서브테이블 1] [서브테이블 2] ...

가득 차면 새 서브테이블 추가
cuckoo 쫓아내기가 서브테이블 간에도 가능
```

- **장점**: 전체 rehash 없음, fill factor 보장
- **단점**: 조회 시 여러 서브테이블 확인 필요

### Linear Hashing (Hive, 2025)

```
warp 단위로 K개 버킷 배치씩 증분 확장
부하율 95%에 도달했을 때만 확장
```

- **가장 최신** (2025)
- 95% load factor 유지하면서 1.5~2× 높은 throughput
- RTX 4090 기준: 삽입 3.5B/s, 조회 4B/s

---

## 주요 구현체 비교

| 구현체 | 방식 | 동적? | 삽입 | 조회 | 비고 |
|---|---|---|---|---|---|
| **BGHT** | Bucketed Cuckoo | ❌ | 최고 수준 | 최고 수준 | 학술 벤치마크용 |
| **cuCollections** | Linear Probing | ❌ | 87.5 GB/s | 134.6 GB/s | NVIDIA 공식 (H100) |
| **SlabHash** | Linked Slab | ✅ | 512M/s | 937M/s | 최초 dynamic GPU HT |
| **DyCuckoo** | Subtable Cuckoo | ✅ | 중간 | 중간 | rehash 없는 cuckoo |
| **WarpCore** | Double Hashing | ❌ | 2.84× CUDPP | 2.84× CUDPP | HiPC 2020 best paper |
| **Hive** | Linear Hashing | ✅ | 3.5B/s | 4B/s | 2025 최신, 최고 성능 |
| **WarpSpeed** | 다양 (8종) | 일부 | IcebergHT 최고 | DoubleHT 최고 | 라이브러리, 비교용 |

---

## DB 엔진에서의 활용

### 해시 조인 (Hash Join)

```
Build phase: 작은 테이블의 join key → GPU hash table 구축
Probe phase: 큰 테이블을 스캔하며 hash table 조회

→ hash table 성능이 join 전체 성능을 좌우
```

- 빌드 테이블이 GPU 메모리에 들어올 때: 정적 hash table (BCHT 등)
- 들어오지 않을 때: partitioning → 파티션별로 처리 (Triton Join, Scaling GPU DB)

### 그룹 집계 (Group-By Aggregation)

```
GROUP BY key → hash table에 key별로 aggregate 값 누적

SUM, COUNT, AVG 등 → atomic 업데이트
→ atomic 경쟁 최소화가 핵심 (WABC 같은 기법 활용)
```

### 중복 제거 (Deduplication)

```
hash table에 키 존재 여부만 기록
→ 이미 있으면 skip, 없으면 insert
→ 높은 부하율에서도 안정적인 cuckoo 방식이 유리
```

---

## 선택 기준 요약

```
정적 데이터 + 최고 성능이 필요     → BGHT (Bucketed Cuckoo)
NVIDIA 공식 지원 필요              → cuCollections
동적 삽입/삭제가 많음             → Hive 또는 DyCuckoo
DB 조인 (부하율 중요)              → Bucketed Cuckoo (0.99 load factor)
메모리 제한 + 높은 부하율          → Iceberg Hashing
동시 삽입+조회 (streaming)         → Hive (warp-synchronous 동시성)
```
