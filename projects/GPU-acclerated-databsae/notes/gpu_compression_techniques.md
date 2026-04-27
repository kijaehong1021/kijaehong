# GPU DB 압축 테크닉 정리

> 참고: Tile-based Integer Compression (SIGMOD 2024), GOLAP (SIGMOD 2024), Scaling GPU DB (VLDB 2025)

---

## 왜 GPU DB에서 압축이 중요한가

GPU의 연산 처리량(수십 TFLOPS)은 메모리 대역폭(HBM ~3 TB/s, PCIe ~28 GB/s)보다 훨씬 빠름. 대부분의 DB 워크로드는 **메모리/IO 바운드** → 압축으로 실제 전송 바이트 수를 줄이는 것이 성능의 핵심.

```
압축을 하는 이유:
  1. SSD → GPU: SSD 대역폭(~20 GB/s)이 병목 → 압축률만큼 실효 대역폭 증가
  2. CPU DRAM → GPU: PCIe 대역폭(~28 GB/s)이 병목 → 동일한 효과
  3. GPU HBM 내: 데이터가 작을수록 캐시 효율 증가, 더 많은 데이터 처리 가능

유효 대역폭 = 원시 대역폭 × 압축률
예: SSD 20 GB/s, 압축률 2.8× → 유효 56 GB/s (GOLAP 실측치)
```

**압축 선택 기준**:
- **어디서 압축 해제하는가**: CPU? GPU?
- **압축률 vs 해제 속도**: 높은 압축률 vs 빠른 해제
- **데이터 특성**: 정렬 여부, 값 범위, 반복 패턴

---

## 1. 경량 압축 (Light-weight Compression)

압축률은 낮지만 해제 속도가 매우 빠름. GPU에서 query 실행 중 on-the-fly 해제 가능.

### 1-1. FOR (Frame of Reference)

컬럼 내 값들이 좁은 범위에 몰려 있을 때 유효. **기준값(reference)을 빼서 값의 범위를 줄이고 bit-packing 적용**.

```
원본: [1000, 1002, 1005, 1003, 1001]  (32비트 필요)

reference = min = 1000
델타:       [0, 2, 5, 3, 1]           (3비트면 충분)

저장: reference(1000) + 3비트 packed 배열
압축률: 32/3 ≈ 10.7×
```

**해제**:
```
원본값 = reference + packed_delta
→ 단순 덧셈 하나 → GPU에서 매우 빠름
```

**한계**: 이상치(outlier)가 하나라도 있으면 압축률 급락.
```
[1000, 1002, 99999]  →  range = 98999  →  17비트 필요  →  압축 효과 없음
```

→ 해결책: 블록/타일 단위로 나눠 각 블록마다 별도 reference 사용 (tile-based FOR)

---

### 1-2. Bit-packing

값의 최대 비트 수만큼만 저장. 바이트 경계 무시하고 비트 단위로 빽빽하게 채움.

```
값 범위 [0, 7] → 3비트면 충분

원본 (32비트 × 4개 = 128비트):
  00000000 00000000 00000000 00000101  (5)
  00000000 00000000 00000000 00000011  (3)
  00000000 00000000 00000000 00000110  (6)
  00000000 00000000 00000000 00000010  (2)

Bit-packed (3비트 × 4개 = 12비트):
  101 011 110 010
  = 1010 1111 0010  (2 바이트로 압축)

압축률: 128 / 12 ≈ 10.7×
```

**GPU 해제**: 비트 시프트 + 마스킹으로 간단히 추출
```
value = (packed_array >> (i * bit_width)) & ((1 << bit_width) - 1)
```

**PEXT/PDEP (x86 BMI2 명령어)**: CPU에서 비트 추출/삽입을 하드웨어 명령어 하나로 처리. GPU에는 직접 대응 명령어 없지만 유사한 bit manipulation 최적화 가능.
```
PEXT(src, mask): src에서 mask=1인 비트만 추출해서 연속 배열로 만듦
PDEP(src, mask): src의 비트를 mask=1인 위치에 분산 삽입

활용: bit-packed 값 해제를 단일 명령어로 처리
```

---

### 1-3. Delta Encoding

**연속된 값의 차이(delta)** 를 저장. 정렬된 컬럼(타임스탬프, 순번 등)에 특히 효과적.

```
원본:  [100, 103, 107, 108, 115]
델타:  [100,   3,   4,   1,   7]  (첫 값은 그대로)

정렬된 컬럼이면 델타가 항상 작음 → bit-packing과 결합시 큰 효과
```

**FOR + Delta (DFOR)**:
```
블록 내 값을 먼저 delta → 그 delta 값들에 FOR 적용
→ 정렬 컬럼에서 압축률 극대화
```

**GPU 해제**: prefix sum 필요 (앞 값을 알아야 다음 값 복원 가능)
```
원본[i] = delta[0] + delta[1] + ... + delta[i]
→ GPU parallel prefix sum (scan)으로 병렬 처리
→ A100 기준 수백 GB/s 처리량
```

---

### 1-4. RLE (Run-Length Encoding)

**연속으로 반복되는 값**을 (값, 반복 횟수) 쌍으로 저장.

```
원본:  [A, A, A, A, B, B, C, C, C]  (9개)
RLE:   [(A, 4), (B, 2), (C, 3)]     (6개 값으로 표현)
```

**DB에서의 활용**:
- 정렬된 컬럼 (같은 값이 연속) → 높은 압축률
- GROUP BY 결과, 파티션 키 등

**GPU 해제 패턴**:
```
각 (값, count) 쌍을 warp가 담당
→ count만큼 값을 output 배열에 병렬 기록
→ scatter 방식으로 병렬성 확보

문제: count가 다 달라서 warp divergence 발생 가능
해결: 큰 run은 여러 warp로 분할, 작은 run들은 묶어서 처리
```

**RLE + Compaction (Scaling GPU DB)**:
```
압축 해제 없이 RLE 상태로 연산 가능한 경우도 있음
예: SUM(column) → 각 run을 (값 × count)로 계산 후 합산
→ 데이터 폭발 없이 집계 연산 수행
```

---

### 1-5. Dictionary Encoding

값의 종류가 적을 때 (low cardinality), **사전(dictionary)** 에 고유 값을 저장하고 컬럼에는 인덱스만 저장.

```
원본:  ["Seoul", "Busan", "Seoul", "Incheon", "Seoul"]  (문자열, 가변 길이)

Dictionary:  {0: "Seoul", 1: "Busan", 2: "Incheon"}
컬럼:        [0, 1, 0, 2, 0]  (2비트면 충분, 3개 항목)

압축률: 문자열 수십 바이트 → 2비트
```

**GPU에서의 Dictionary 연산**:
```
filter (column = "Seoul"):
  1. Dictionary에서 "Seoul" → index 0 찾기 (O(1))
  2. 컬럼에서 index == 0인 위치 찾기 (bit-parallel)
  → 전체 문자열 비교 없이 정수 비교만으로 처리

join (두 테이블의 dictionary 컬럼):
  두 dictionary를 merge → 통합 dictionary 생성
  각 컬럼의 인덱스를 새 인덱스로 remapping
  → 실제 데이터(문자열 등) 없이 인덱스만으로 join
```

**Compaction (Scaling GPU DB)**:
```
삭제/필터 후 Dictionary에 쓸모없는 항목이 남을 수 있음
→ GPU가 살아있는 인덱스만 스캔해서 새 Dictionary 재구성
→ Dictionary 크기 축소 → 이후 연산 더 빠름
```

---

## 2. 중량 압축 (Heavy-weight Compression)

압축률이 높지만 해제에 복잡한 연산 필요. **CPU에서는 병목, GPU에서는 빠름**.

| 알고리즘 | 압축률 | GPU 해제 속도 | 특징 |
|---|---|---|---|
| **LZ4** | 중간 (~2×) | 매우 빠름 | 딕셔너리 기반, 빠른 해제 우선 |
| **Snappy** | 중간 (~2×) | 빠름 | Google 개발, 균형형 |
| **ZStandard (ZStd)** | 높음 (~3×) | 빠름 | 현재 가장 많이 쓰임, 압축 레벨 조절 가능 |
| **DEFLATE/gzip** | 높음 | 느림 | GPU에서 비효율 |

**LZ 계열 압축 원리**:
```
반복되는 패턴을 (오프셋, 길이) 참조로 대체

원본: "abcde abcde abcde"
압축: "abcde " + (back_ref: offset=6, len=11)
     → 앞에 나온 "abcde abcde"를 참조
```

**GPU 해제의 어려움**:
```
LZ 해제는 본질적으로 순차적:
  출력[i]은 출력[i - offset]에 의존
  → 이전 출력이 없으면 현재 출력 계산 불가
  → 직접 병렬화가 어려움

해결책: 독립 청크 단위 병렬 해제
  청크마다 별도로 압축 → 각 청크는 이전 청크와 독립
  → warp/block 단위로 청크 병렬 해제
  → 청크 크기가 작을수록 병렬성↑, 압축률↓ (트레이드오프)
```

### nvCOMP (NVIDIA 공식 라이브러리)

NVIDIA가 제공하는 GPU 압축/해제 라이브러리. LZ4, Snappy, ZStd, DEFLATE, Cascaded, Bitcomp 등 지원.

```
// ZStd 해제 예시 (nvCOMP API)
nvcompBatchedZstdDecompressAsync(
  compressed_data_ptrs,    // 각 청크의 압축 데이터 포인터 배열
  compressed_sizes,        // 각 청크 크기
  output_ptrs,             // 출력 포인터 배열
  output_sizes,            // 출력 버퍼 크기
  num_chunks,              // 청크 수 (= 병렬 스레드 수)
  stream                   // CUDA 스트림
);
```

**GOLAP에서의 활용**:
```
SSD → GPU HBM (GDS 직접 전송, CPU bypass)
         ↓
GPU에서 nvCOMP로 ZStd 해제 (on-the-fly)
         ↓
해제된 데이터로 바로 scan/filter/join
```
CPU가 해제 병목이 되는 걸 GPU로 옮김 → 유효 대역폭 2.8× 향상.

---

## 3. GPU 압축 해제 패턴

### 3-1. Cascading (구식)

여러 압축 기법을 순차적으로 적용할 때 각 단계마다 global memory를 거침.

```
[압축 데이터 in GPU global mem]
         ↓ kernel 1: FOR 해제
[중간 결과 in GPU global mem]
         ↓ kernel 2: delta 복원
[중간 결과 in GPU global mem]
         ↓ kernel 3: scan
[최종 결과 in GPU global mem]
```

**문제**: 각 단계마다 HBM read + write → 대역폭 낭비. 커널 실행 오버헤드.

---

### 3-2. Tile-based (최신, SIGMOD 2024)

**128개 정수 = 1 tile**. 타일을 shared memory에 올려 놓고, 모든 해제 단계를 shared memory 안에서 완료.

```
[압축 데이터 in GPU global mem]
         ↓ (1번만 읽음)
[Shared memory: 128개 정수 타일]
  → FOR 해제 (shared mem 내)
  → delta prefix sum (shared mem 내)
  → scan/filter (shared mem 내)
         ↓ (1번만 씀)
[결과 in GPU global mem]
```

**핵심 구조**:
```
타일 = 4 mini-block × 32 정수
mini-block 단위로 독립적으로 처리 → warp 하나가 mini-block 하나 담당

FOR (GPU-FOR):
  각 mini-block마다 reference 값 저장
  warp 내 32 스레드가 32개 값 동시 해제

Delta (GPU-DFOR):
  mini-block 내 prefix sum → warp-level scan으로 병렬 처리
  mini-block 간 경계값 전파 → 별도 처리

Scatter (GPU-RFOR):
  원래 위치로 흩어진 값들을 제자리로 모음
  PEXT/PDEP 유사 비트 조작으로 처리
```

**shared memory 절약**:
```
mini-block 4개를 동시에 올리면 shared mem 과부하
→ 2개씩 ping-pong 방식:
  mini-block 0,1 해제 → 결과 처리
  mini-block 2,3 해제 → 결과 처리
→ shared mem 사용량 절반
```

---

### 3-3. On-the-fly 파이프라이닝 (GOLAP)

압축 해제와 쿼리 실행을 겹쳐서 처리.

```
I/O 스레드:  [청크1 로드] ───── [청크2 로드] ───── [청크3 로드]
GPU 스트림:          [청크1 해제+scan]   [청크2 해제+scan]   ...
```

**고정 크기 청크** 가 핵심:
```
같은 컬럼 내 모든 청크는 동일한 튜플 수 포함
→ 고정 크기 출력 버퍼 재사용 가능
→ GPU 메모리 할당/해제 오버헤드 없음
→ 다운스트림 오퍼레이터에 즉시 파이프라이닝
```

---

## 4. Opportunistic Pruning (GOLAP)

압축과 결합해서 **읽지 않아도 되는 청크를 미리 스킵**.

```
각 청크에 메타데이터 사전 저장:
  MinMax: 청크 내 최솟값, 최댓값
  Bloom Filter: 특정 값 존재 여부 (확률적)
  Histogram: 256 bin으로 값 분포 기록

쿼리 실행 전:
  GPU가 메타데이터(매우 작음)를 병렬 평가
  → 각 predicate에 대해 청크별 pass/prune 비트맵 생성
  → 비트맵 AND/OR 결합 → 최종 skip 목록

I/O 스레드가 skip 목록 기준으로 SSD 읽기 건너뜀
→ 실제 읽는 데이터 양 감소 → 유효 대역폭 추가 향상
```

**압축 + Pruning 결합 효과 (GOLAP 실측)**:
```
SSD raw 대역폭:          ~20 GiB/s
+ Heavy-weight 압축:      압축률 2.8× → 유효 ~56 GiB/s
+ Opportunistic Pruning:  불필요 청크 skip → 최대 ~100 GiB/s

인메모리 시스템 HBM 대역폭(~28 GiB/s)을 초과
```

---

## 5. 압축 방식 비교 및 선택 기준

### 압축률 vs 해제 속도

```
높은 압축률                              빠른 해제
←────────────────────────────────────────→
ZStd > LZ4/Snappy > FOR/Delta > RLE > Dictionary (저 카디널리티)
```

### 언제 무엇을 쓸까

| 상황 | 권장 압축 방식 | 이유 |
|---|---|---|
| 정렬된 정수 컬럼 | Delta + Bit-packing (DFOR) | 델타가 작아 bit 수 최소화 |
| 좁은 범위 정수 | FOR + Bit-packing | reference로 범위 축소 |
| 낮은 카디널리티 문자열 | Dictionary Encoding | 문자열→정수 인덱스 |
| 연속 반복 값 | RLE | 반복 패턴 제거 |
| SSD → GPU IO 병목 | ZStd (nvCOMP) | 높은 압축률로 IO 감소 |
| Query 중 on-the-fly 해제 | Tile-based FOR/Delta | shared mem 내 단일 패스 |
| Selectivity 낮은 쿼리 | Pruning 메타데이터 | 청크 자체를 skip |

### GPU vs CPU 압축 해제 선택

```
CPU가 압축 해제하는 경우:
  → Light-weight (FOR, delta, bit-packing) → CPU도 충분히 빠름
  → 단, 데이터를 GPU로 보내기 전에 해제하면 PCIe 전송량 증가

GPU가 압축 해제하는 경우 (권장):
  → Heavy-weight (ZStd, LZ4) → CPU 병목 → GPU의 병렬 코어 활용
  → PCIe/SSD로 압축 상태 전송 → GPU에서 해제
  → 전송량 감소 + CPU 병목 해소
```

---

## 6. Crystal (GPU DB 라이브러리) 에서의 활용

Crystal은 GPU 컬럼 DB 연산 라이브러리. Tile-based compression과 연산을 통합:

```
BlockScan + BlockDecompress를 단일 커널로 결합:
  shared memory에 타일 로드 → 해제 → scan → 결과 저장
  → global memory 왕복 없이 단일 패스

Crystal 커널 구조:
  template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
  __global__ void compressed_scan_kernel(...) {
    __shared__ int tile[BLOCK_SIZE * ITEMS_PER_THREAD];

    // 1. Load compressed tile
    load_tile(tile, compressed_data);

    // 2. Decompress in shared memory
    for_decompress(tile, reference);   // FOR 해제
    delta_scan(tile);                  // prefix sum

    // 3. Apply predicate & output
    scan_and_filter(tile, predicate, output);
  }
```

---

## 핵심 인사이트 요약

| # | 인사이트 |
|---|---|
| 1 | GPU DB에서 압축의 목적은 저장 공간 절약이 아니라 **유효 대역폭 증가** |
| 2 | Light-weight 압축은 GPU에서 해제 비용이 거의 없어 항상 이득 |
| 3 | Heavy-weight 압축은 CPU에선 병목이지만 GPU에선 오히려 대역폭 증폭 도구 |
| 4 | Tile-based 해제로 global memory 왕복 제거 → shared memory 단일 패스 |
| 5 | 압축 + Pruning 결합이 시너지: 압축으로 전송량↓, Pruning으로 읽는 청크↓ |
| 6 | Dictionary Encoding은 문자열 비교를 정수 비교로 바꿔 GPU 연산 효율 극대화 |
| 7 | 청크 단위 독립 압축이 GPU 병렬 해제의 핵심 (청크 간 의존성 제거) |
