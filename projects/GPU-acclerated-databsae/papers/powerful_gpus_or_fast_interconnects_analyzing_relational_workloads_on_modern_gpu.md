# Powerful GPUs or Fast Interconnects: Analyzing Relational Workloads on Modern GPUs

- **저자**: Marko Kabić, Bowen Wu, Jonas Dann, Gustavo Alonso (ETH Zurich, Systems Group)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3749646.3749698

---

## 문제 정의

GPU DB 성능에서 "더 강한 GPU"와 "더 빠른 인터커넥트" 중 무엇이 더 중요한가? 다양한 GPU-인터커넥트 조합이 analytics 성능에 미치는 영향이 정량적으로 분석되지 않았음.

기존 평가 지표인 **arithmetic intensity (ops/byte)** 와 **GFlop/s** 는 data analytics 워크로드를 설명하기에 부적합. 대부분의 analytics 쿼리가 memory-bound라 arithmetic intensity가 너무 낮게 나옴.

---

## 실험 환경

4가지 하드웨어 조합에서 TPC-H, H2O-G, ClickBench 총 54개 쿼리 실행:

| 구성 | GPU | 인터커넥트 | IC 대역폭 |
|---|---|---|---|
| C1 | A100 (40GB) | PCIe 3.0 | 16 GB/s |
| C2 | H100 (80GB) | PCIe 5.0 | 64 GB/s |
| C3 | H200 (96GB) | **NVLink 4.0** | **450 GB/s** |
| C4 | RTX3090 (24GB) | PCIe 4.0 | 32 GB/s |

---

## Contribution 1: MaxBench 프레임워크

TPC-H, H2O-G, ClickBench 워크로드를 heterogeneous 하드웨어에서 벤치마킹/프로파일링/모델링하는 프레임워크.

**3계층 구조**:
- **Input Layer**: 벤치마크 선택, 데이터셋 생성, CSV/Parquet 지원
- **Profiling Configuration Layer**:
  - 데이터 위치: CPU / GPU, pinned / non-pinned 메모리 선택
  - Copy Policy: per-dataset (전체 DB 한 번에 전송) vs per-query (쿼리마다 필요한 컬럼만 전송)
  - 프로파일링 수준: 전체 시간 / 데이터 전송 분해 / 오퍼레이터별 분해
- **Analysis & Modeling Layer**: SQL/DataFrame API로 결과 분석, NVIDIA Nsight 연동

---

## Contribution 2: 새로운 특성화 지표

기존 GFlop/s, arithmetic intensity 대신 두 가지 새 지표 제안:

```
실행 시간 = query_complexity / device_efficiency + data_transfer_cost
```

**Characteristic Query Complexity (α_q)**: 쿼리 고유의 연산 복잡도. GPU 종류와 무관.

**Characteristic Device Efficiency (β_d)**: GPU 고유의 효율. 쿼리와 무관.

### 수학적 근거

쿼리 q를 GPU d에서 실행하는 오퍼레이터 시간 T_op는 데이터 크기 n에 대해 선형:
```
T_op(n) = slope_{q,d} × n + intercept_{q,d}
```

slope 행렬 S를 SVD 분해하면 rank-1에 가까울 경우:
```
S_{q,d} ≈ α_q / β_d
```

→ 한 번만 실측하면 다른 GPU에서의 성능을 예측 가능.

**왜 유용한가**: A100에서 측정한 α_q를 이용해 H200에서 얼마나 걸릴지 예측 가능. 실제로 대부분의 쿼리에서 **오차 10% 이내** 예측 달성.

---

## Contribution 3: 인터커넥트 vs GPU 성능 분석

### 핵심 발견: PCIe 환경에서는 데이터 전송이 지배적

TPC-H 기준 PCIe 구성(C1, C2, C4)에서 쿼리별 런타임 분해:
- C1(PCIe 3.0): 데이터 전송이 **67~98%** 차지 (Q13: 67%, Q12/Q14/Q15/Q18/Q20: 98%+)
- C2(PCIe 5.0): 데이터 전송이 **33~97%** 차지

→ GPU 아무리 좋아도 PCIe가 병목이면 GPU 연산 시간이 전체의 2~30%에 불과.

### NVLink로 전환 시 패러다임 변화

C3(NVLink 4.0)에서는 대부분의 쿼리가 **GPU-bound**로 전환:
- 22개 TPC-H 쿼리 중 단 6개만 데이터 전송 > 50%
- 나머지 16개 쿼리에서 데이터 전송이 평균 **24%** 수준

### "강한 GPU vs 빠른 인터커넥트" 정량 비교

PCIe 3.0에서 RTX3090 → H200으로 GPU 업그레이드: **최대 1.99x** 향상 (H2O-G 기준)

NVLink 4.0에서 RTX3090 → H200으로 GPU 업그레이드: **최대 4.82x** 향상

→ 인터커넥트가 빠를수록 GPU 성능이 비로소 의미 있어짐. **인터커넥트가 선결 조건**.

---

## Contribution 4: 비용 모델로 미래 트렌드 예측

**인터커넥트 대역폭을 1000 GB/s까지 늘리면?**

대역폭 > 137 GB/s 이상이면 모든 벤치마크에서 GPU-bound로 전환됨.
- PCIe 7.0(이론값 242 GB/s, 실효 ~96 GB/s)은 여전히 일부 워크로드에서 interconnect-bound 가능성 있음
- NVLink 환경은 이미 GPU-bound 영역

**GPU 효율을 H200 대비 2배로 늘리면?**

NVLink 4.0 환경에서 TPC-H는 interconnect-bound로 다시 역전됨.
H2O-G, ClickBench는 여전히 GPU-bound 유지.

→ 워크로드마다 병목이 다름. 실측 없이 cost model로 예측 가능.

---

## 주요 인사이트 정리

| # | 인사이트 |
|---|---|
| 1 | PCIe 환경에서는 대부분의 쿼리가 데이터 전송이 지배적 |
| 2 | NVLink로 전환 시 대부분의 쿼리가 interconnect-bound → GPU-bound로 전환 |
| 3 | 빠른 인터커넥트가 있을 때 비로소 GPU 효율이 성능에 큰 영향을 미침 |
| 4 | Join이 가장 많이 연구된 오퍼레이터지만, 문자열 비교/regex를 포함한 Filter도 주요 병목 |
| 5 | GPU에서 문자열/regex 필터는 비정렬 메모리 접근 + 분기로 성능 저하 |
| 6 | CPU-side 필터링(selective transfer)은 느린 인터커넥트에서만 유효. 빠른 인터커넥트에서는 오히려 2x 성능 저하 |
| 7 | PCIe에서 peak 대역폭은 pinned(page-locked) 메모리에서만 달성. non-pinned는 40% 미만 |
| 8 | NVLink는 non-pinned 메모리도 peak 대역폭 달성 (GH200의 cache-coherent 접근 덕분) |
| 9 | GPU 실행 시간은 입력 데이터 크기에 대해 선형 스케일 (memory-bandwidth bound) |
| 13 | 인터커넥트 대역폭 > 137 GB/s이면 모든 워크로드에서 GPU-bound |
| 15 | PCIe 7.0도 일부 워크로드에서 여전히 interconnect-bound 가능. NVLink는 GPU-bound 유지 |
