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

**Bit-packing**: 값들이 byte 경계에 안 맞게 패킹됨. 선택된 값의 비트들을 추출해서 재조립. BMI의 `PEXT` 명령어로 byte 단위 병렬 추출.

**RLE (Run-Length Encoding)**: `(값, 반복 횟수)` 형태. 선택된 행 범위와 run 범위의 교집합을 계산해 새 run 생성. 압축률 거의 유지.

**Dictionary Encoding**: 딕셔너리는 그대로 두고, 코드 배열만 bit-packing 방식으로 컴팩션.

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
