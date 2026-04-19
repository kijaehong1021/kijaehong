# GPU Acceleration of SQL Analytics on Compressed Data

- **저자**: Zezhou Huang, Krystian Sakowski, Hans Lehnert, Wei Cui, Carlo Curino, Matteo Interlandi, Marius Dumitru, Rathijit Sen (Microsoft)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3778092.3778095

---

## 문제 정의

GPU는 SQL analytics 가속에 적합하지만, GPU HBM(High Bandwidth Memory)이 CPU 메인 메모리보다 훨씬 작아 대용량 데이터셋을 처리하지 못하는 근본적 한계가 있음.

기존 해결책(멀티-GPU, 배치 처리, CPU-GPU 하이브리드)은 모두 낮은 대역폭이나 I/O 오버헤드, 높은 비용 문제를 가짐.

**핵심 관찰**: 압축으로 데이터 크기를 줄이면 GPU 메모리에 더 많은 데이터를 올릴 수 있지만, 압축 해제 비용이 너무 크다. → **압축 해제 없이 압축 데이터 위에서 직접 연산하자.**

---

## 시스템 구조: TQP (Tensor Query Processor)

이 논문은 **TQP** 위에서 구현됨. TQP는 SQL 쿼리를 PyTorch 텐서 연산으로 변환하여 GPU에서 실행하는 프레임워크. PyTorch를 사용하기 때문에 CPU/GPU/APU 등 다양한 하드웨어에 이식 가능.

---

## 핵심 아이디어: 인코딩 체계 (Encoding Schemes)

### 기본 인코딩

| 인코딩 | 텐서 표현 | 적합한 데이터 |
|---|---|---|
| **Plain** | 값 배열 1개 (위치=인덱스) | 일반 데이터 |
| **RLE** | `val[]`, `start[]`, `end[]` 3개 텐서 | 연속 반복값 |
| **Index** | `val[]`, `pos[]` 2개 텐서 | 희소(sparse) 데이터 |

**RLE는 연속된 반복에만 적용됨.** 연속이 끊기면 별개의 런으로 분리:

```
원본: [A, A, B, A, A, A]

val   = [A,  B,  A]
start = [0,  2,  3]
end   = [1,  2,  5]
```

→ A가 두 군데 나와도 연속이 끊겼으니 런 2개로 분리됨.

RLE 효과 비교:
```
[A, B, A, B, A, B]  → 런 길이 전부 1 → 압축 효과 없음
[A, A, A, A, A, A]  → 런 1개          → 압축 극대화
```

→ **정렬된 컬럼** (날짜, 지역코드 등 같은 값이 뭉쳐있는 경우)에 효과적.

---

**Index 예시**: 값이 드문드문 있는 희소 데이터에 적합:

```
원본: [0, 0, 0, 0, 5, 0, 0, 0, 3, 0]  (10행 중 값이 2개뿐)

val = [5, 3]
pos = [4, 8]
```

→ Plain이면 10개 저장, Index면 2개만 저장.

---

**RLE vs Index 비교**:

| | RLE | Index |
|---|---|---|
| 잘 맞는 데이터 | 정렬된 컬럼 (값이 뭉쳐있음) | 희소 컬럼 (값이 드문드문) |
| `[A,A,A,B,B,B]` | ✅ 효과적 | ❌ 별로 |
| `[0,0,5,0,0,3]` | ❌ 별로 | ✅ 효과적 |

### 복합 인코딩 (Composite Encoding)

Plain과 Index를 조합하여 더 강력한 압축:

- **Plain + Index**: 대부분은 좁은 비트폭(예: int8)으로 저장, 아웃라이어만 Index로 별도 저장
  - 예: 10억 개 값 중 일부 아웃라이어가 int64 필요 → 나머지는 int8 저장으로 ~7 GiB 절약
- **RLE + Index**: 연속 구간은 RLE, 불연속 구간은 Index로 처리

---

## 핵심 프리미티브 구현

### 전략: for 루프 없이 PyTorch 병렬 연산만 사용

GPU 병렬성을 최대화하기 위해 명시적 for 루프나 조건문 없이 PyTorch 벡터 연산만으로 구현.

주요 PyTorch 연산들:
- `bucketize(input, boundaries)`: 이진 탐색으로 각 값이 어느 버킷에 속하는지 반환
- `repeat_interleave(input, repeats)`: 각 원소를 지정 횟수만큼 반복
- `scatter(src, index, reduce)`: 같은 인덱스에 값을 누적 (SUM, MIN, MAX 등)
- `unique(input)`: 중복 제거 + 원래 원소→유니크 인덱스 매핑 반환

### range_intersect: 두 RLE 컬럼 교집합

두 RLE 컬럼의 구간 교집합을 구함. 멀티-컬럼 필터, AND 연산의 핵심.

```
입력: c1 = {start=[2], end=[7]},  c2 = {start=[1,4,6], end=[3,5,8]}
```

```python
bins  = bucketize(c1.start, c2.end,   right=False)  # [0]   - c1 시작이 c2 끝 중 어디?
bine  = bucketize(c1.end,   c2.start, right=True)   # [3]   - c1 끝이 c2 시작 중 어디?
cnt   = bine - bins                                  # [3]   - 겹치는 c2 구간 수
idx1  = repeat_interleave(arange(len(cnt)), cnt)     # [0,0,0]
idx2  = range_arange(bins, cnt)                      # [0,1,2]
s = max(c1.start[idx1], c2.start[idx2])             # [2,4,6]
e = min(c1.end[idx1],   c2.end[idx2])               # [3,5,7]
```

결과: `start=[2,4,6]`, `end=[3,5,7]` (3개 교집합 구간)

**핵심**: bucketize로 겹치는 구간 수를 먼저 파악 → repeat_interleave로 인덱스 생성 → max/min으로 실제 교집합 계산. 전부 벡터 연산.

### idx_in_rle: Index 위치가 RLE 구간 안에 속하는지

```python
bin  = bucketize(c_idx.pos, c_rle.start, right=True) - 1
mask = (bin >= 0) & (c_idx.pos <= c_rle.end[bin])
return c_idx.pos[mask]
```

**성능**: 100M 원소에서 CPU 대비 **21~46배 빠름** (교차점: 10K~100K 원소)

---

## 연산자 구현

### 논리 연산 (AND, OR, NOT)

**AND가 왜 필요한가?** SQL WHERE 절의 복합 조건이 AND로 변환됨:

```sql
SELECT * FROM orders WHERE region = 'Seoul' AND status = 'paid'
```

1. `region = 'Seoul'` → MaskColumn (True/False 배열)
2. `status = 'paid'`  → MaskColumn
3. 두 MaskColumn AND → 둘 다 True인 행만 선택

**압축 데이터에서 AND가 어려운 이유**: 두 RLE 컬럼의 런 경계가 달라 단순 값 비교 불가.

```
region (RLE): [Seoul: 0~4,  Busan: 5~9]
status (RLE): [paid: 0~2,   unpaid: 3~6,  paid: 7~9]
```

→ "Seoul AND paid" = 0~2행. 구간 교집합(`range_intersect`)으로 계산해야 함.

**RLE mask는 True 구간만 저장**: False는 묵시적으로 나머지 전부. 따라서 AND = True 구간끼리의 교집합만 구하면 됨. True인 구간 수가 적을수록 연산량이 줄어드는 이유.

**GPU에서 range_intersect가 빠른 이유**: `bucketize`(이진 탐색)를 c1의 모든 런에 대해 동시에 실행.

```
c1 런 100만 개, c2 런 100만 개일 때:

CPU: c1 런 하나씩 순서대로 이진 탐색 → 100만 번 순차 실행
GPU: 스레드 100만 개가 동시에 이진 탐색 → 한꺼번에 실행
```

`bucketize(c1.start, c2.end)` 한 줄이 텐서 연산이라 for 루프 없이 GPU 병렬 처리 가능. 실측: 100M 원소에서 CPU 대비 **21~46배** 빠름.

입력 인코딩 조합에 따라 다른 구현 선택:

| 입력 조합 | AND 구현 |
|---|---|
| RLE AND RLE | `range_intersect` |
| RLE AND Plain | selectivity에 따라 RLE→Index 또는 RLE→Plain 변환 후 AND |
| RLE AND Index | `idx_in_rle` 또는 `rle_contain_idx` |
| Index AND Index | `idx_in_idx` (bucketize 기반) |
| Plain AND Plain | PyTorch `&` 연산자 |

**Index AND Index 상세 예시** (`idx_in_idx`):

```
mask1.pos = [0, 2, 4, 7]
mask2.pos = [2, 4, 5, 7, 9]
AND 결과  = [2, 4, 7]  (둘 다 True인 위치)
```

```python
bin  = bucketize([0,2,4,7], [2,4,5,7,9], right=True) - 1
     = [0,1,2,4] - 1
     = [-1, 0, 1, 3]

# bin이 가리키는 mask2 값과 실제로 같은지 확인
# bin=-1 → False (범위 밖)
# mask2[0]=2 == mask1[1]=2 → True
# mask2[1]=4 == mask1[2]=4 → True
# mask2[3]=7 == mask1[3]=7 → True

mask = [False, True, True, True]
결과 = mask1.pos[mask] = [2, 4, 7]
```

for 루프 없이 bucketize 한 번으로 모든 위치를 GPU에서 병렬 탐색.

**RLE AND Plain의 변환 방향**: Plain→RLE 변환이 AND 연산은 빠르지만 변환 비용(4.29ms)이 RLE→Plain(1.02ms)보다 4배 이상 커서, RLE→Plain 방향이 전체적으로 4~10배 빠름.

### 산술/비교 연산: Alignment 필요

RLE 컬럼끼리 더하기 등은 **런 경계가 맞지 않아** 직접 연산 불가.

예: `c1 = [(A, 0~9), (B, 10~19), (C, 20~39)]` + `c2 = [(X, 0~14), (Y, 15~39)]`

→ **Alignment**: `range_intersect`로 두 컬럼의 런 경계를 통일시킨 뒤 값 텐서끼리 연산
→ 결과: `[(A+X, 0~9), (B+X, 10~14), (B+Y, 15~19), (C+Y, 20~39)]`

### Group-by 집계

**2단계 처리**:

**1단계 - Grouping** (`torch.unique`): 각 행을 그룹 ID에 매핑하는 inverse index 생성

```sql
SELECT region, SUM(amount) FROM orders GROUP BY region
```

```
region 컬럼:   [Seoul, Seoul, Busan, Seoul, Busan]

unique값:      [Busan, Seoul]   ← 정렬된 그룹 목록
inverse index: [1, 1, 0, 1, 0] ← 각 행의 그룹 ID
```

**2단계 - Aggregation** (`torch.scatter`): inverse index 기준으로 그룹별 집계

```
amount:        [10,  20,  30,  40,  50]
inverse index: [1,   1,   0,   1,   0 ]

그룹 0 (Busan): 30 + 50 = 80
그룹 1 (Seoul): 10 + 20 + 40 = 70
```

---

**RLE 컬럼이 있을 때: Alignment 필요**

런 경계가 다른 두 RLE 컬럼은 바로 집계 불가. 먼저 `range_intersect`로 경계를 통일:

```
region (RLE): [Seoul: 0~2, Busan: 3~4]
amount (RLE): [10: 0~1,    20: 2~4   ]
```

Alignment 후:
```
구간   런길이  region  amount
0~1     2     Seoul    10
2~2     1     Seoul    20
3~4     2     Busan    20
```

경계가 통일된 후 집계 (압축 해제 없이 런 길이 직접 활용):

| 함수 | 계산 방법 |
|---|---|
| COUNT | 런 길이 합산 |
| SUM | `val * 런길이` 합산 → Seoul: 10×2 + 20×1 = 40 |
| MIN/MAX | val 텐서만 스캔 (런 길이 무시) |
| AVG/STD | SUM과 COUNT의 후처리로 계산 |

### Join 연산

기존 TQP의 hash join을 압축 컬럼으로 확장:

1. **Get Join Index**: 매칭되는 행 쌍의 인덱스 텐서 생성
   - RLE 컬럼이 포함되면 1:N 또는 N:N 조인 → 런 길이만큼 인덱스 복제
2. **Apply Join Index**: Join Index를 각 컬럼에 적용하여 결과 생성

---

## 성능 결과

- CPU 전용 상용 analytics 시스템 대비 **수십 배(order of magnitude) 속도 향상**
- GPU 메모리에 비압축 상태로 올릴 수 없는 실제 프로덕션 데이터셋에서 검증
- 하드웨어: Azure NC24ads A100 v4 VM (A100 GPU, AMD EPYC 7V13 CPU)
