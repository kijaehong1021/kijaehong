# Tile-based Lightweight Integer Compression in GPU

- **저자**: Anil Shanbhag, Bobbi Yogatama, Xiangyao Yu, Samuel Madden (MIT, UW-Madison)
- **학회**: SIGMOD 2022
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3514221.3526132

---

## 문제 정의

GPU에서 정수 컬럼 압축/해제 시 **Cascading Compression** 문제가 발생.

기존 DB 시스템은 여러 압축 기법을 레이어처럼 쌓음:
```
Delta → FOR → Bit-packing → (압축된 데이터)
```

각 레이어가 별도 커널로 실행되며 **매번 global memory에 읽고 씀**:

```
[Global Mem] → Kernel 1(Delta) → [Global Mem]
                                        ↓
[Global Mem] → Kernel 2(FOR)   → [Global Mem]
                                        ↓
[Global Mem] → Kernel 3(BitPack) → [Global Mem]  (최종 압축 결과)
```

HBM 대역폭(2 TB/s)을 가지고도, 이 중간 쓰기/읽기들이 전체 처리량을 낮춤.

---

## 배경: 경량 압축 기법들

### FOR (Frame of Reference)

배열 전체에서 **기준값(reference)을 하나 정하고**, 모든 값에서 기준값을 뺀 차이만 저장.

```
원본:    [1000, 1003, 1005, 1002, 1007]
reference = 1000 (보통 최솟값)
저장값:  [   0,    3,    5,    2,    7]
```

값 범위가 [1000, 1007]이면 원래 10 bits 필요하지만, reference를 빼면 [0, 7] → **3 bits**로 줄어듦. 값들이 좁은 범위에 몰려있을수록 효과적.

### Bit-packing

정수를 **필요한 최소 비트 수로 빽빽하게** 저장. 32-bit 경계에 맞추지 않고, n bits짜리 값들을 연속으로 이어붙임.

```
값: [3, 1, 2, 3, 0, 2]  → 최댓값 3 → 2 bits면 충분
bit stream: 11 01 10 11 00 10  (12 bits)
```

32-bit int 배열로 저장하면 6×32=192 bits지만, bit-packing하면 **12 bits**. 단, 특정 원소를 읽으려면 비트 단위 연산 필요.

### Delta Encoding

**연속된 값들의 차이**만 저장. 정렬된 데이터에서 값 자체보다 차이가 훨씬 작을 때 유효.

```
원본:   [1000, 1003, 1007, 1009, 1015]
deltas: [1000,    3,    4,    2,    6]  (첫 값은 그대로)
```

정렬된 컬럼(날짜, PK 등)에서 deltas의 범위가 매우 좁아 bit-packing 효율이 극대화됨.

### 조합해서 쓰는 이유

세 기법을 조합하면 시너지:
```
정렬 데이터 → Delta → 차이값들이 작아짐
              → FOR  → reference 빼서 더 작아짐
              → Bit-packing → 최소 비트로 저장
```

하지만 기존에는 각 단계를 별도 커널로 돌려서 **Cascading 문제** 발생.

---

## 핵심 아이디어: Tile-based Compression Model

**Thread Block = Tile = 압축 단위**. 128개(또는 512개)의 정수를 하나의 thread block이 담당.

```
[Global Mem] → 한 번 로드 → [Shared Memory]
                               ↓ Delta (SMEM 내)
                               ↓ FOR   (SMEM 내)
                               ↓ Bit-packing (SMEM 내)
                            → [Global Mem]  (최종 압축 결과)
```

→ Global memory 접근 **1 pass**로 모든 압축 단계 완료. Decompression도 동일하게 1 pass.

Shared memory 대역폭이 Global memory보다 ~10x 빠르기 때문에, 중간 단계를 SMEM에서 처리하는 것이 핵심.

---

## GPU-FOR: Bit-packing + Frame of Reference

### 데이터 포맷

```
┌─────────────────────────────────────────────────────┐
│           Block (128 integers)                      │
│  ┌──────────┬──────────┬──────────┬──────────┐     │
│  │ Mini-blk │ Mini-blk │ Mini-blk │ Mini-blk │     │
│  │ (32 int) │ (32 int) │ (32 int) │ (32 int) │     │
│  │  b1 bits │  b2 bits │  b3 bits │  b4 bits │     │
│  └──────────┴──────────┴──────────┴──────────┘     │
│                                                      │
│  Header: [reference_val] [b1|b2|b3|b4 bitwidths]   │
└─────────────────────────────────────────────────────┘
```

- **Block**: 128개 정수, 1개의 thread block이 담당
- **Mini-block**: 32개 정수, mini-block마다 독립적인 bitwidth
- **Reference**: block 전체의 최솟값. 모든 값에서 빼서 delta 저장
- **Bitwidth**: mini-block 내 최댓값을 표현하는 데 필요한 비트 수
- 32개 × any bitwidth → 항상 32-bit 워드 경계에 맞춰 떨어짐 → SMEM 접근 효율적

### 구체적인 인코딩 예시

값 128개가 있고, 처음 64개가 [99, 102], 나머지 64개가 [99, 114]인 경우:

```
Block:
  reference = 99  (block minimum)

  Mini-block 1 (32개): values [99~102]
    - values - 99 ∈ [0, 3]
    - 최댓값 3 → 2 bits 필요

  Mini-block 2 (32개): values [99~102]  
    - 동일 → 2 bits

  Mini-block 3 (32개): values [99~114]
    - values - 99 ∈ [0, 15]
    - 최댓값 15 → 4 bits 필요

  Mini-block 4 (32개): values [99~114]
    - 동일 → 4 bits

Header 저장:
  [reference = 99] [bitwidths = 2, 2, 4, 4]

Bit-packed data:
  Mini-block 1: 32개 × 2 bits = 64 bits = 2 × 32-bit 워드
  Mini-block 2: 32개 × 2 bits = 64 bits = 2 × 32-bit 워드
  Mini-block 3: 32개 × 4 bits = 128 bits = 4 × 32-bit 워드
  Mini-block 4: 32개 × 4 bits = 128 bits = 4 × 32-bit 워드
```

압축률:
- 원본: 128 × 32 bits = 4096 bits
- 압축: 헤더(~96 bits) + 64+64+128+128 bits ≈ 480 bits
- **~8.5x 압축**

비교: bit-packing만 쓰면 block 전체 최댓값 기준 → 130-0=130이면 8 bits 필요, 128×8=1024 bits. GPU-FOR은 reference로 뺀 덕분에 더 작은 bitwidth 사용 가능.

### Decompression (Unpacking)

각 thread가 자신의 원소 1개를 담당:

```
1. block_start 배열에서 내 블록의 데이터 시작 위치 읽기
2. bitwidth_word에서 내 mini-block의 bitwidth 읽기
3. mini-block offset 계산: 앞 mini-block들의 크기 합
4. bit offset 계산: (내 위치 % 32) × bitwidth
5. 8-byte 로드 (값이 32-bit 워드 경계를 넘칠 수 있으므로)
6. 비트 추출 + shift
7. reference 값 더하기 → 원래 값 복원
```

---

## GPU-DFOR: Delta + FOR + Bit-packing

정렬/준정렬 데이터 대상. 연속된 값의 차이(delta)가 작을 때 효과적.

### 데이터 포맷

```
│ first_value │ [K blocks] │ [K blocks] │ ...
│ (절댓값)     │  (deltas)  │  (deltas)  │
```

- K개 블록씩 묶어 독립적으로 delta 인코딩
- 각 묶음의 첫 번째 원래 값을 별도로 저장
- 각 블록 내 delta들을 GPU-FOR 포맷으로 bit-packing

### 구체적인 인코딩 예시

정렬된 데이터: `[100, 103, 107, 109, 115, 120, 125, 128]`

```
GPU-FOR 방식으로 인코딩:
  reference = 100
  values - 100: [0, 3, 7, 9, 15, 20, 25, 28]
  최댓값 28 → 5 bits 필요
  8개 × 5 bits = 40 bits 사용

GPU-DFOR 방식으로 인코딩:
  first_value = 100 (별도 저장)
  deltas: [3, 4, 2, 6, 5, 5, 3]  (연속 차이)
  reference = 2 (delta 최솟값)
  deltas - 2: [1, 2, 0, 4, 3, 3, 1]
  최댓값 4 → 3 bits 필요
  7개 × 3 bits = 21 bits + first_value(32 bits)
  ≈ 53 bits로 비슷하지만, 실제 컬럼 규모에서 차이 발생
```

500M 정수, 정렬된 데이터 기준:
- GPU-FOR: 평균 **~24 bits/integer**
- GPU-DFOR: 평균 **~4 bits/integer** (연속 값 차이가 매우 작아서)

### Decompression에서 Prefix Sum 사용

Deltas를 복원할 때 prefix sum(cumulative sum)이 필요:

```
deltas: [3, 4, 2, 6, 5, 5, 3]
first_value: 100

prefix_sum(deltas):   [3, 7, 9, 15, 20, 25, 28]
결과:  100 + [3, 7, 9, 15, 20, 25, 28] = [103, 107, 109, 115, 120, 125, 128]
```

Prefix sum이 순차적 연산처럼 보이지만, **병렬 prefix sum 알고리즘**(Blelloch et al.)으로 O(log n) 병렬 처리:

```
SMEM 내에서:
  bit unpacking → deltas 배열
  block-wide prefix sum (shared memory 내에서 완결)
  reference 값 더하기
  → 원래 값 복원
→ Global memory에 1번만 쓰기
```

---

## GPU-RFOR: RLE + FOR + Bit-packing

반복 값이 많은 데이터 대상 (cardinality가 낮은 컬럼).

### 데이터 포맷

```
Block (512 integers):
  ┌─────────────────────────┬──────────────────────────┐
  │ values array (FOR+BP)   │ run_lengths array (FOR+BP)│
  └─────────────────────────┴──────────────────────────┘
  Header: [metadata: runs count, values count]
```

- Block size = 512 integers (GPU-FOR의 128보다 큼, 더 많은 run 포함)
- values 배열과 run_length 배열을 **각각 독립적으로** FOR+bit-packing
- Decompression 시 값 배열과 run_length 배열을 각각 unpack → scatter → prefix sum

### 구체적인 인코딩 예시

데이터: `[5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3]`

```
RLE 변환:
  values:      [5, 7, 3]
  run_lengths: [6, 4, 8]

values 배열 FOR+bit-packing:
  reference = 3
  values - 3: [2, 4, 0]
  최댓값 4 → 3 bits
  3개 × 3 bits = 9 bits

run_lengths 배열 FOR+bit-packing:
  reference = 4
  run_lengths - 4: [2, 0, 4]
  최댓값 4 → 3 bits
  3개 × 3 bits = 9 bits

원본: 18 × 32 bits = 576 bits
압축: ~18 bits + metadata ≈ 매우 높은 압축률
```

### Decompression: Scatter + Inclusive Prefix Sum

```
Step 1: values 배열 unpack:      [5, 7, 3]
Step 2: run_lengths 배열 unpack: [6, 4, 8]
Step 3: run_lengths의 prefix sum: [6, 10, 18]  (각 run의 끝 인덱스)
Step 4: scatter: 인덱스 범위에 값 채우기
  [0,6) → 5, [6,10) → 7, [10,18) → 3
  결과: [5,5,5,5,5,5, 7,7,7,7, 3,3,3,3,3,3,3,3]
```

GPU-RFOR는 GPU-FOR/GPU-DFOR보다 **2배 더 많은 shared memory**를 사용 (values + run_lengths 두 배열 모두 SMEM에 올려야 함).

---

## Cascading vs. Tile-based 비교

`Delta + FOR + Bit-packing` 파이프라인을 예시로:

### Cascading (기존 방식)

```
Decompression:

[Global Mem: bit-packed data]
        ↓ (커널 1 실행: bit-unpack)
[Global Mem: FOR-encoded data]    ← 중간 결과 쓰기
        ↓ (커널 2 실행: FOR decode)
[Global Mem: delta-encoded data]  ← 중간 결과 쓰기
        ↓ (커널 3 실행: delta decode)
[Global Mem: decoded integers]    ← 최종 결과

Global memory 접근: 읽기 3회 + 쓰기 3회 = 6회
```

### Tile-based (이 논문)

```
Decompression:

[Global Mem: bit-packed data]
        ↓ (1번 로드)
[Shared Memory: bit-packed data]
        ↓ bit-unpack in SMEM
[Shared Memory: FOR-encoded data]
        ↓ FOR decode in SMEM
[Shared Memory: delta-encoded data]
        ↓ delta decode (prefix sum) in SMEM
[Shared Memory: decoded integers]
        ↓ (1번 쓰기)
[Global Memory: decoded integers]

Global memory 접근: 읽기 1회 + 쓰기 1회 = 2회
```

→ Global memory 접근 횟수를 **6x → 2x**로 줄임. HBM이 병목인 GPU에서 큰 차이.

---

## Crystal 프레임워크 통합

Crystal은 GPU OLAP 엔진 (오픈소스). 압축 루틴을 **device function**으로 캡슐화해서, 한 줄 변경으로 비압축 → 압축 컬럼 전환 가능:

```cuda
// 기존 비압축 코드
int val = col[idx];

// 압축 컬럼으로 교체 (단 한 줄 변경)
int val = GPU_FOR_unpack(col_compressed, idx);
```

Device function이므로 쿼리 커널 내부에서 inline 실행 → 별도 decompression pass 불필요. 쿼리 실행과 동시에 압축 해제.

---

## 압축 스킴 선택 기준

| 스킴 | 적합한 데이터 |
|---|---|
| GPU-FOR | uniform 분포, 값 범위가 좁은 컬럼 |
| GPU-DFOR | 정렬/준정렬 컬럼 (날짜, 타임스탬프, PK 등) |
| GPU-RFOR | low cardinality 컬럼 (카테고리, 국가 코드 등) |
| GPU-* (하이브리드) | 컬럼별로 최적 스킴 자동 선택 |

---

## 성능 결과

- 비교 대상: COMP (최고 압축률 스킴), 기존 bit-packing 구현들
- SSB(Star Schema Benchmark) 기준:

| 지표 | 결과 |
|---|---|
| 압축률 | COMP와 **유사** (1~2% 차이) |
| 압축 속도 | 기존 대비 **2.2x** 빠름 |
| 쿼리 실행 시간 | 기존 대비 **2.6x** 빠름 |

- 250M 정수 압축 시간: GPU-FOR ≈ GPU-DFOR ≈ **0.1초 미만** (A100 기준)
- 인터커넥트 병목이 있는 환경에서: 압축된 데이터 전송 → PCIe 부하 감소
