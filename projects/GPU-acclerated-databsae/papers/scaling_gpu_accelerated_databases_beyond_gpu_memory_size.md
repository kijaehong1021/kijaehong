# Scaling GPU-Accelerated Databases beyond GPU Memory Size

- **저자**: Yinan Li, Bailu Ding, Ziyun Wei 외 (Microsoft, Cornell Univ, Ohio State Univ)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3749646.3749710

---

## 문제 정의

GPU HBM 용량(A100 기준 80GB)은 실제 분석 DB(수백GB~수TB)에 비해 너무 작음. 데이터가 GPU 메모리를 초과하면 PCIe를 통해 전송해야 하는데, **PCIe 대역폭(24GB/s)이 HBM(2TB/s) 대비 80배 느림** → 전송이 병목.

| 구성 요소 | 대역폭 |
|---|---|
| A100 HBM | 2 TB/s |
| CPU 메모리 | 80 GB/s |
| PCIe 4.0 | 24 GB/s |

---

## 세 가지 핵심 관찰

**Observation 1: Scan은 PCIe를 타야 하면 CPU가 더 빠름**

- CPU scan: SIMD 최적화로 80GB/s (CPU 메모리 대역폭 근접)
- GPU-cold scan: PCIe 24GB/s가 병목 → CPU보다 느림
- GPU-hot scan (HBM에 있을 때만): GPU가 빠름

**Observation 2: Join은 GPU가 항상 더 빠름 (PCIe를 타도)**

- CPU join: 랜덤 메모리 접근 패턴으로 인해 PCIe보다도 느림
- GPU join: PCIe 전송 비용을 포함해도 CPU보다 빠름

**Observation 3: Join이 처리하는 행 수는 Scan보다 훨씬 적음**

- Bitvector filtering 없이도 대부분 쿼리에서 join 입력 < scan 입력
- Bitvector filtering 활성화 시 절반의 쿼리에서 join이 scan의 **1% 미만**만 처리

---

## 해결책: CPU-GPU 역할 분담

```
CPU: 전체 데이터 스캔 → 필터링 → 압축된 채로 PCIe 전송
GPU: 소량의 필터된 데이터로 Join / Aggregate 처리
CPU: 최종 결과 수신 (결과는 작아서 overhead 없음)
```

→ PCIe를 통해 전송되는 데이터 볼륨 자체를 줄이는 것이 핵심.

---

## 핵심 기술 1: Compressed Scan (Predicate Filtering)

### 기존 scan과의 차이

| 방식 | 동작 | 출력 |
|---|---|---|
| CPU-only scan | 필터 → 압축 해제 → 컴팩션 | 비압축 |
| GPU-only scan | GPU에서 필터 + 압축 해제 + 컴팩션 | 비압축 |
| **이 논문** | CPU에서 필터 → **압축 그대로 컴팩션** | **압축 유지** |

### 핵심: 압축 해제 없이 직접 컴팩션

selection bitmap(어떤 행이 선택됐는지)을 이용해 **압축된 값 배열에서 선택된 값만 추출**, 압축 포맷 그대로 출력.

- 재압축 없음 → CPU 오버헤드 최소
- GPU는 압축된 데이터를 받아 자체적으로 압축 해제 (기존에도 하던 일)
- CPU의 **BMI (Bit Manipulation Instructions)** 활용

### 인코딩별 컴팩션 구현

#### Bit-packing: PEXT + PDEP

4-bit 값 8개가 64bit 워드에 빽빽하게 들어있는 상황. 값들이 byte 경계에 안 맞아 일반 명령어로 추출 불가.

**해결: x86 BMI 명령어**
- `PEXT` (Parallel Bit Extract): mask 위치의 비트들을 뽑아서 하위 비트로 모음
- `PDEP` (Parallel Bit Deposit): 하위 비트들을 mask 위치에 흩뿌림

```
Step 1: selection bitmap (1비트)을 값 bit-width만큼 확장
        bit b → bbbb  (PDEP 두 번 + 빼기 트릭으로 구현)
        예: selection 0b01000110 → extended 0x0F00F0F0

Step 2: PEXT(bit-packed values, extended_bitmap)
        → 선택된 값들만 하위 비트로 추출, 압축 포맷 그대로
```

64bit 워드 하나에서 최대 16개(4-bit 기준) 값을 **명령어 5개**로 처리. Intel/AMD 모두 지원.

#### RLE: POPCNT

`(값, 반복횟수)` run에서, 해당 range의 selection bitmap에서 1의 개수를 세면 끝.

```
run: (A, 10) + selection bitmap에서 해당 10개 중 1의 수 = POPCNT
새 run: (A, POPCNT 결과)
```

#### Dictionary Encoding

- 인덱스 컬럼: bit-packing / RLE 방식으로 컴팩션
- 딕셔너리: 선택된 행에서 참조 안 되는 entry를 빈 값으로 교체 (인덱스 컬럼 수정 없이)

**예시**:
```
원본 컬럼: ["apple", "banana", "apple", "cherry", "banana", "apple"]

Dictionary Encoding:
  딕셔너리:     { 0: "apple", 1: "banana", 2: "cherry" }
  인덱스 컬럼:  [0, 1, 0, 2, 1, 0]  (2-bit packed)

selection bitmap: [1, 0, 1, 0, 1, 0]  (행 0, 2, 4 선택)

인덱스 컴팩션 (PEXT):
  원본: [0, 1, 0, 2, 1, 0]
  출력: [0,    0,    1   ]  → bit-packed

딕셔너리 컴팩션:
  선택된 행이 참조하는 인덱스: {0, 1}
  참조 안 되는 인덱스: {2}

  기존: { 0: "apple", 1: "banana", 2: "cherry" }
  변환: { 0: "apple", 1: "banana", 2: ""       }  ← 2만 비움
```

인덱스 컬럼 값을 건드리지 않고 딕셔너리 entry만 비워서 전송량 감소.
GPU는 동일한 인덱스로 딕셔너리 참조 (2번 참조하는 행이 없으니 빈 entry 무해).

#### 혼합 인코딩 처리

실제 컬럼은 RLE 구간 + bit-packing 구간이 섞여있음. run 단위로 각각 처리 후 빈 run 제거 + 인접 동일 인코딩 run 병합.

> bit-packing 컴팩션의 핵심 아이디어(PEXT/PDEP 활용)는 prior work [39]에서 가져온 것. 이 논문은 해당 기법을 CPU-only가 아닌 hybrid 실행 컨텍스트(압축 출력이 필요한 상황)에 적용한 것이 기여.

---

## 핵심 기술 2: Bitvector Filtering

큰 테이블에 직접 필터 조건이 없는 경우를 처리. 예: lineitem에 필터 없고, part에만 `p_size < 10` 조건.

### 동작 방식

```
1. 작은 테이블(part)에 predicate 적용
   → 매칭된 join key들로 bitvector 구성

2. 큰 테이블(lineitem) 스캔 시
   → bitvector로 l_partkey 사전 필터링
   → join에서 매칭 안 될 행들을 PCIe 전송 전에 제거

3. PCIe로 전송되는 데이터 = 필터된 압축 데이터만
```

### 선택적 적용

bitvector 구축/적용 비용이 절약량보다 클 수 있음 (selectivity가 낮을 때). 비용-효익을 추정해서 **이득이 있을 때만 bitvector filtering 적용**.

---

## 핵심 기술 3: Streaming + Partitioning (GPU 메모리 초과 처리)

**Streaming**: 프로브 테이블을 청크 단위로 PCIe 전송, 해시테이블만 GPU 메모리에 상주.

```
GPU 메모리: [해시테이블 (build side 전체)]
PCIe 스트림: [probe chunk 1] → [probe chunk 2] → ...
```

**Partitioning**: 필터된 후에도 테이블이 크면 range 파티션으로 나눠 하나씩 GPU 처리.

```
파티션 1: p_partkey BETWEEN 1 AND 500000   → GPU 처리
파티션 2: p_partkey BETWEEN 500001 AND 1000000 → GPU 처리
```

- 각 파티션 쌍이 GPU 메모리에 맞도록 파티션 수 조절
- 스캔을 파티션마다 반복하지만, CPU scan이 빠르기 때문에 overhead 작음

---

## 한계: N:1 Join에 최적화된 설계

이 논문의 기술들은 암묵적으로 **star schema (dimension ↔ fact)** 구조를 가정:

- **Bitvector filtering**: 작은 build side key로 큰 probe side 사전 필터링. dimension이 작고 fact가 클 때 효과적.
- **Streaming**: build side 해시테이블이 GPU 메모리에 들어갈 만해야 작동.

**N:N join (양쪽 다 큰 경우)**:
- Bitvector selectivity 낮아 효과 없음
- Build side도 GPU 메모리 초과 → partitioning 필요
- 파티션 수만큼 스캔 반복 → 비용 증가

논문 평가도 TPC-H (star/snowflake schema) 에만 집중. 실제 OLAP 워크로드 대부분이 N:1 패턴이라 실용적이지만, N:N에 대한 효과는 별도로 검증되지 않음.

---

## 성능 결과

- 벤치마크: TPC-H scale factor 1000 (1TB), 단일 A100 (80GB)
- TPC-H 22개 쿼리 전부 처리 성공 (기존 GPU DB들은 대부분 실패)

| 비교 대상 | 성능 |
|---|---|
| CPU-only SQL Server (1TB) | **3.5x** 빠름 |
| 비슷한 가격 VM의 SQL Server | **3.4x** speedup |
| 고가 VM의 SQL Server | 1.4x 빠르면서 **2.9x 저렴** |
