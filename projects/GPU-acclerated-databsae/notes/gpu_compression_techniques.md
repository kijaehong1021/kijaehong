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

| 알고리즘 | 압축률 | GPU 해제 속도 | CPU 해제 속도 | 특징 |
|---|---|---|---|---|
| **LZ4** | 중간 (~2×) | 매우 빠름 | 빠름 | 해제 속도 우선 설계 |
| **Snappy** | 중간 (~2×) | 빠름 | 빠름 | Google 개발, 균형형 |
| **ZStandard (ZStd)** | 높음 (~3×) | 빠름 | 중간 | 압축 레벨 조절 가능 (1~22) |
| **DEFLATE/gzip** | 높음 (~3×) | 느림 | 느림 | GPU에서 비효율, 레거시 |
| **Brotli** | 매우 높음 | 매우 느림 | 매우 느림 | 웹용, DB에선 미사용 |

---

### 2-1. LZ 계열 압축의 원리

LZ4, Snappy, ZStd 모두 **LZ77** 알고리즘을 기반으로 함. 핵심 아이디어는 이미 나온 패턴을 다시 쓰지 않고 "앞에 있던 거 참조해"라는 포인터로 대체하는 것.

#### 압축 과정 (step-by-step)

```
원본: "abcde_abcde_abcde"  (17바이트)

슬라이딩 윈도우 방식으로 왼쪽부터 스캔:

위치 0: 'a' → 윈도우에 없음 → literal
위치 1: 'b' → 없음 → literal
...
위치 5: '_' → 없음 → literal
        현재까지 "abcde_" = 6바이트 literal 누적

위치 6: 'a' → 윈도우에서 "abcde" 패턴 탐색!
              pos=0에서 시작하는 "abcde"와 일치 → 얼마나 길게 이어지나?
              pos=0: a=a, b=b, c=c, d=d, e=e, _=_ → 6글자 일치!
              match (offset=6, length=6) 기록

위치 12: 'a' → 윈도우에서 탐색
               pos=6에서 "abcde_" 다시 일치 → match (offset=6, length=5)
               (마지막 글자 없으면 5)

압축 결과: [literal "abcde_"] [match offset=6, len=6] [match offset=6, len=5]
원본 17바이트 → 압축 결과 몇 바이트 (표현 방식에 따라 다름)
```

#### LZ4 토큰 포맷과 실제 바이트 예시

LZ4의 압축 스트림은 **시퀀스(sequence)** 의 연속. 각 시퀀스 = literal 묶음 + match 1개.

```
각 시퀀스 바이트 배열:
  [token: 1B] [literal_len_ext: 0~NB] [literals: NB] [offset: 2B] [match_len_ext: 0~NB]

token (1바이트):
  ┌────────────────┬────────────────┐
  │  상위 4비트     │  하위 4비트     │
  │  literal 길이   │  match 길이-4  │
  └────────────────┴────────────────┘

  상위/하위가 15(=0xF)이면 다음 바이트에 추가 길이 저장 (255씩 더함, 0이 아닌 값으로 종료)
```

**실제 예시: "hello world hello world" 압축**

```
원본 (22바이트):
  h e l l o   w o r l d   h e l l o   w o r l d
  0 1 2 3 4 5 6 7 8 9 A B C D E F 10 11 12 13 14 15

--- 시퀀스 1 ---
위치 0~11: "hello world " → 12바이트 literal (윈도우에 없음)
위치 12~21: "hello world" → 12바이트 앞(offset=12)에서 10글자 일치 → match

token:
  literal 길이 = 12 → 상위 4비트 = 0xC (12이지만 15 미만이므로 확장 없음)
  match 길이   = 10 → 하위 4비트 = 10-4 = 6

  token = 0xC6

압축 바이트 스트림:
  [0xC6]                  ← token
  [h,e,l,l,o, ,w,o,r,l,d, ] ← 12바이트 literal
  [0x0C, 0x00]            ← offset=12 (little-endian 2바이트)
  (match_len_ext 없음, 하위 4비트 6으로 충분)

총 압축 크기: 1 + 12 + 2 = 15바이트 (원본 22 → 68% 크기)
```

#### 해제 과정 (step-by-step)

```
압축 스트림: [0xC6] [h,e,l,l,o, ,w,o,r,l,d, ] [0x0C, 0x00]
출력 버퍼:   (비어있음)

Step 1: token = 0xC6 읽기
  literal_len = 상위 4비트 = 0xC = 12
  match_len   = 하위 4비트 + 4 = 6 + 4 = 10

Step 2: literal 12바이트 읽어서 출력 버퍼에 그대로 복사
  출력: "hello world "  (위치 0~11)

Step 3: offset 읽기 = 0x0C 0x00 = 12 (little-endian)
  match_pos = 현재 출력 위치(12) - offset(12) = 0

Step 4: 출력[0..9] (10바이트)를 현재 위치(12)에 복사
  출력[12] = 출력[0] = 'h'
  출력[13] = 출력[1] = 'e'
  ...
  출력[21] = 출력[9] = 'd'

최종 출력: "hello world hello world"  ✓
```

#### Overlapping copy (자기 참조) 예시

offset < match_len 일 때 발생. memcpy로 처리 불가 — 반드시 1바이트씩 순서대로.

```
현재 출력 버퍼: "ab"  (위치 0~1)
match: offset=1, length=6

  out[2] = out[2-1] = out[1] = 'b'  → 출력: "abb"
  out[3] = out[3-1] = out[2] = 'b'  → 출력: "abbb"  ← 방금 쓴 값 재사용!
  out[4] = out[4-1] = out[3] = 'b'  → 출력: "abbbb"
  out[5] = out[5-1] = out[4] = 'b'  → 출력: "abbbbb"
  out[6] = out[6-1] = out[5] = 'b'  → 출력: "abbbbbb"
  out[7] = out[7-1] = out[6] = 'b'  → 출력: "abbbbbbb"

결과: 2바이트 → 8바이트로 확장 (RLE처럼 동작)

이 때문에:
  memcpy(out+2, out+1, 6) → 구현에 따라 다름, 보장 안 됨
  for (i=0; i<6; i++) out[2+i] = out[2+i-1]  → 항상 정확
```

#### LZ4 압축 시 해시 테이블 활용

압축기가 윈도우에서 패턴을 탐색할 때 매번 전체 스캔하면 O(N²). 실제로는 **해시 테이블**로 O(1) 탐색.

```
압축 중 유지하는 해시 테이블:
  key   = 현재 4바이트의 해시
  value = 해당 4바이트가 마지막으로 나온 위치

위치 pos에서:
  h = hash(input[pos..pos+3])
  candidate = table[h]           ← 같은 해시의 이전 위치
  table[h] = pos                 ← 현재 위치로 업데이트

  if input[candidate..] == input[pos..] (최소 4바이트):
    → match 발견! 얼마나 길게 이어지는지 확인
  else:
    → 해시 충돌 (false positive) → literal로 처리
```

이 덕분에 LZ4 압축 속도가 매우 빠름 (~500 MB/s 이상). 압축률보다 속도를 우선한 설계.

---

### 2-2. 엔트로피 코딩: Huffman

LZ 계열 압축이 **반복 패턴**을 제거한다면, 엔트로피 코딩은 **자주 나오는 심볼에 짧은 비트 코드를 배정**해서 추가로 줄이는 기법. ZStd는 LZ 결과물에 Huffman 또는 FSE를 한 번 더 적용해서 압축률을 높임.

#### 엔트로피와 이론적 한계

엔트로피(H)는 데이터를 표현하는 데 필요한 **최소 평균 비트 수**. 심볼 s가 확률 p(s)로 등장할 때:

```
H = -Σ p(s) × log2(p(s))

예: 심볼 {a, b, c, d}, 각 확률 1/4
  H = -(4 × 1/4 × log2(1/4)) = 2비트
  → 고정 2비트 코드로 표현 가능, 이미 최적

예: 심볼 {a, b, c, d}, 확률 1/2, 1/4, 1/8, 1/8
  H = -(1/2×log2(1/2) + 1/4×log2(1/4) + 1/8×log2(1/8) + 1/8×log2(1/8))
    = -(1/2×(-1) + 1/4×(-2) + 1/8×(-3) + 1/8×(-3))
    = 0.5 + 0.5 + 0.375 + 0.375 = 1.75비트
  → 평균 1.75비트면 이론적으로 표현 가능

직관: 자주 나오는 심볼(높은 p)은 log2(p) 절댓값이 작음 → 짧은 코드로 표현하면 최적
```

#### Huffman 트리 생성 (압축)

**최소 힙(min-heap)** 을 이용한 bottom-up 트리 구성. 빈도가 낮은 심볼부터 합쳐나감.

```
입력: "aabbbbcccccccc"  (a×2, b×4, c×8)

Step 1: 각 심볼을 빈도와 함께 리프 노드로 만들고 힙에 삽입
  힙: [a:2] [b:4] [c:8]

Step 2: 가장 작은 두 노드를 꺼내 합침
  꺼냄: [a:2], [b:4]
  합침: 내부 노드 [ab:6] 생성 (왼쪽=a, 오른쪽=b)
  힙:   [ab:6] [c:8]

Step 3: 다시 가장 작은 두 노드를 꺼내 합침
  꺼냄: [ab:6], [c:8]
  합침: 루트 노드 [abc:14] 생성 (왼쪽=ab, 오른쪽=c)

최종 트리:
        [14]
       /    \
    [6]      c(8)
   /   \
  a(2)  b(4)

코드 배정 (왼쪽=0, 오른쪽=1):
  a → 00  (2비트)
  b → 01  (2비트)
  c → 1   (1비트)
```

**엔트로피와 비교**:
```
실제 평균 비트 수:
  a: 2/14 × 2비트 = 0.286
  b: 4/14 × 2비트 = 0.571
  c: 8/14 × 1비트 = 0.571
  합계 = 1.428비트/심볼

이론적 엔트로피:
  H = -(2/14×log2(2/14) + 4/14×log2(4/14) + 8/14×log2(8/14))
    ≈ 1.399비트/심볼

Huffman = 1.428, 엔트로피 = 1.399 → 매우 근접 (약 2% 차이)
```

#### Huffman 압축/해제 예시

```
원본: "aabbbbcccccccc"  (14바이트 = 112비트)
코드: a→00, b→01, c→1

압축:
  a  a  b  b  b  b  c  c  c  c  c  c  c  c
  00 00 01 01 01 01 1  1  1  1  1  1  1  1
  = 00000101010111111111  = 20비트 = 2.5바이트

압축률: 112비트 → 20비트 = 5.6× (Huffman 테이블 오버헤드 제외)
```

**해제 (트리 순회)**:
```
압축 비트스트림: 00000101010111111111

Huffman 트리를 루트부터 순회:
  읽음 '0' → 왼쪽으로 → [6] 노드
  읽음 '0' → 왼쪽으로 → a 리프 → 'a' 출력, 루트로 복귀

  읽음 '0' → 왼쪽으로 → [6] 노드
  읽음 '0' → 왼쪽으로 → a 리프 → 'a' 출력, 루트로 복귀

  읽음 '0' → 왼쪽으로 → [6] 노드
  읽음 '1' → 오른쪽으로 → b 리프 → 'b' 출력, 루트로 복귀

  ...

  읽음 '1' → 오른쪽으로 → c 리프 → 'c' 출력, 루트로 복귀

결과: "aabbbbcccccccc"  ✓
```

**해제의 핵심 특성: prefix-free 코드**

Huffman 코드는 어떤 코드도 다른 코드의 접두어가 아님. 이 덕분에 비트스트림을 앞에서부터 읽기만 해도 심볼 경계를 자동으로 파악 가능. (구분자 불필요)

```
a=00, b=01, c=1 이면:
  비트스트림 "001..." 읽을 때
  '0' → 아직 코드 안 끝남 (0으로 시작하는 코드: 00, 01)
  '0' → "00" = a 확정! 다음 심볼 시작

  만약 a=0, b=01 이었다면 (prefix-free 아님):
  '0' 읽었을 때 a인지 b의 시작인지 알 수 없음 → 불가
```

#### 테이블 기반 고속 해제

트리 순회는 비트 하나씩 처리 → 느림. 실제 구현에서는 **룩업 테이블**로 한번에 여러 비트 처리.

```
테이블 크기 = 2^k (k = 미리 읽을 비트 수, 보통 8~15)

테이블 구성:
  비트스트림 앞 k비트 → 테이블 인덱스
  테이블[인덱스] = (심볼, 실제 소비한 비트 수)

예 (k=4, a=00, b=01, c=1):
  0000 → (a, 2)  ← 앞 2비트로 a 결정, 나머지 2비트는 다음 심볼
  0001 → (a, 2)
  0100 → (b, 2)
  1000 → (c, 1)  ← 앞 1비트로 c 결정, 나머지 3비트는 다음 심볼
  1001 → (c, 1)
  ...

해제:
  bits = read_k_bits()
  (symbol, consumed) = table[bits]
  output(symbol)
  rewind(k - consumed)  ← 안 쓴 비트 돌려놓기

→ 비트 하나씩 처리 대신 k비트를 한 번에 처리 → 수배 빠름
```

#### Huffman의 한계

```
심볼 확률이 2의 거듭제곱(1/2, 1/4, 1/8...)이 아닐 때 낭비 발생:

예: 심볼 {a}, 확률 1/3
  이상적 코드 길이: log2(3) = 1.585비트
  Huffman 배정: 2비트 → 0.415비트 낭비 (26% 초과)

해결: FSE/ANS (다음 섹션)
  → 비정수 비트 배정으로 엔트로피에 더 근접
  → ZStd가 literal에 Huffman 대신 FSE를 쓰는 이유
```

---

### 2-3. ZStandard (ZStd) 상세

ZStd = LZ77 + **엔트로피 코딩** 결합. LZ4보다 압축률이 높은 이유가 여기 있음.

#### LZ4 vs ZStd: 같은 입력으로 비교

```
원본: "aaabbbaaaccc"  (12바이트)
심볼 빈도: a=6, b=3, c=3

--- LZ4 처리 ---
LZ 분석:
  "aaabbb" → literal 6바이트 (처음 등장)
  "aaa"    → offset=6, length=3 (앞의 "aaa" 재사용)
  "ccc"    → literal 3바이트 (처음 등장)

저장:
  [token] [literal "aaabbb"] [offset=6] [token] [literal "ccc"]
  literal은 그대로 ASCII 저장 → 'a'=0x61, 'b'=0x62, 'c'=0x63

--- ZStd 처리 ---
LZ 분석: LZ4와 동일한 시퀀스 탐색

Huffman 추가 압축 (literal에 적용):
  빈도 분석: a(6회), b(3회), c(3회)
  Huffman 코드 생성:
    a → 0      (1비트, 가장 빈번)
    b → 10     (2비트)
    c → 11     (2비트)

  literal "aaabbb":
    LZ4: 6바이트 (48비트)
    ZStd: 000 + 101010 = 0001 0101 0 → 9비트 ≈ 2바이트

  literal "ccc":
    LZ4: 3바이트 (24비트)
    ZStd: 111111 = 6비트 → 1바이트

ZStd 총 크기: Huffman 테이블 + 압축 시퀀스 → LZ4보다 작음
(단, 짧은 입력에서는 Huffman 테이블 오버헤드가 클 수 있음 → 실제론 64KB+ 입력에서 효과 큼)
```

#### FSE (Finite State Entropy)

ANS(Asymmetric Numeral Systems) 기반 엔트로피 코딩. ZStd에서 sequences 인코딩에 사용. 이론적 엔트로피 한계에 거의 도달하면서 테이블 룩업 한 번으로 디코딩.

**핵심 아이디어**: Huffman은 심볼 하나에 정수 비트를 배정. FSE는 **상태(state)** 를 통해 여러 심볼을 함께 인코딩 → 심볼 하나당 비정수 비트 효과.

```
직관:
  Huffman: "이 심볼은 항상 2비트"
  FSE:     "이 심볼은 평균 1.58비트"
           → 어떤 상태에서는 1비트, 어떤 상태에서는 2비트 소비
           → 평균 내면 1.58비트
```

#### ANS (Asymmetric Numeral Systems) 원리

ANS는 전체 스트림을 하나의 큰 정수 `x`로 인코딩. 심볼을 추가할 때마다 x를 갱신하며, x의 크기가 곧 정보량.

```
핵심 직관:
  x = 100이고 심볼 a(확률 1/2)를 인코딩:
    새 x ≈ 100 × (1 / 0.5) = 200  → x가 약 2배 증가 → 1비트 정보 추가

  x = 100이고 심볼 c(확률 1/4)를 인코딩:
    새 x ≈ 100 × (1 / 0.25) = 400 → x가 약 4배 증가 → 2비트 정보 추가

  → 희귀한 심볼일수록 x가 더 많이 커짐 = 더 많은 비트 소비
  → 자주 나오는 심볼은 x를 조금만 키움 = 적은 비트 소비

실제로는 x가 무한히 커지지 않도록 renormalization:
  x가 너무 커지면 → 하위 비트 몇 개를 비트스트림에 출력하고 x를 줄임
  x가 너무 작아지면 → 비트스트림에서 비트를 읽어서 x를 늘림
  → x를 항상 [L, 2L) 범위에 유지
```

#### FSE 테이블 구성

FSE는 ANS를 **테이블 룩업**으로 구현. 상태 0~(L-1) 중 각 상태에서 어떤 심볼을 디코딩하는지, 몇 비트를 읽는지 사전에 계산해서 저장.

```
예: 심볼 {A(빈도 4), B(빈도 2), C(빈도 2)}, 테이블 크기 L=8

각 심볼에 테이블 슬롯 배정 (빈도에 비례):
  A → 슬롯 4개 (확률 4/8 = 1/2)
  B → 슬롯 2개 (확률 2/8 = 1/4)
  C → 슬롯 2개 (확률 2/8 = 1/4)

spread 알고리즘으로 슬롯을 균등하게 분산 배치:
  (같은 심볼이 뭉치면 안 좋음 → 테이블 전체에 고르게)

state:    0    1    2    3    4    5    6    7
symbol:   A    B    A    C    A    B    A    C

num_bits 결정:
  A는 슬롯 4개 = L/2 → 디코딩 시 1비트 읽음 (다음 상태 2가지 중 선택)
  B, C는 슬롯 2개 = L/4 → 디코딩 시 2비트 읽음 (다음 상태 4가지 중 선택)

최종 테이블:
  state │ symbol │ num_bits │ new_state_base
  ──────┼────────┼──────────┼───────────────
    0   │   A    │    1     │      0
    1   │   B    │    2     │      4
    2   │   A    │    1     │      2
    3   │   C    │    2     │      6
    4   │   A    │    1     │      0
    5   │   B    │    2     │      4
    6   │   A    │    1     │      2
    7   │   C    │    2     │      6
```

**비정수 비트가 가능한 이유**:

```
만약 A 확률이 딱 3/8 (슬롯 3개)이라면:
  A 슬롯 3개이지만 L=8이므로 어떤 슬롯은 1비트, 어떤 슬롯은 2비트 배정
  예: 슬롯 0,2,4 → num_bits = 1 (A를 만날 때 절반은 1비트)
       슬롯 6    → num_bits = 2 (A를 만날 때 절반은 2비트)
       (정확히 A 슬롯 3개를 어디에 배치하는지에 따라 다름)

  → 평균 비트 수: (2×1 + 1×2) / 3 = 1.33비트
  → 이론값: log2(8/3) ≈ 1.415비트
  → 테이블 크기를 늘릴수록 이론값에 더 근접 (L=256이면 오차 거의 0)
```

#### FSE 디코딩 step-by-step

디코딩은 **비트스트림을 거꾸로** 읽으면서 처리. (인코딩이 거꾸로 진행됐으므로)

```
비트스트림 (뒤에서부터 읽음): ...01 1 10 1...
초기 state = 4  (인코더가 마지막에 기록한 상태)

Step 1: state=4
  symbol    = table[4].symbol    = A  → 'A' 출력
  num_bits  = table[4].num_bits  = 1
  bits      = read_bits(1)       = 1   (비트스트림 끝에서 1비트 읽음)
  new_state = table[4].new_state_base + bits = 0 + 1 = 1

Step 2: state=1
  symbol    = B  → 'B' 출력
  num_bits  = 2
  bits      = read_bits(2)  = 10 (= 2)
  new_state = 4 + 2 = 6

Step 3: state=6
  symbol    = A  → 'A' 출력
  num_bits  = 1
  bits      = read_bits(1)  = 0
  new_state = 2 + 0 = 2

Step 4: state=2
  symbol    = A  → 'A' 출력
  ...

디코딩: 매 스텝 = 테이블 룩업 1번 + 비트 읽기 1번 → 매우 빠름
```

#### Huffman vs FSE 비교

```
                Huffman              FSE
────────────────────────────────────────────────
코드 방식       가변 비트 코드        상태 기계 + 테이블
심볼당 비트     정수                  비정수 가능
이론 한계 근접  약간 못 미침          거의 도달 (< 0.001비트/심볼)
디코딩          트리 순회             테이블 룩업 1회
디코딩 속도     중간                  매우 빠름
병렬성          낮음 (순차)           낮음 (상태 의존, 순차)
ZStd 활용       literal (짧을 때)     sequences (offset/length)
```

**ZStd에서 세 FSE 상태 기계 interleaved 동작**:

```
sequences 디코딩 시 세 개의 독립 FSM이 동시에 동작:
  FSM_LL: literal_length 디코딩
  FSM_OF: offset 디코딩
  FSM_ML: match_length 디코딩

한 시퀀스 디코딩 순서:
  1. FSM_LL 테이블 룩업 → literal_length 심볼 + 읽을 비트 수
  2. FSM_OF 테이블 룩업 → offset 심볼 + 읽을 비트 수
  3. FSM_ML 테이블 룩업 → match_length 심볼 + 읽을 비트 수
  4. 세 FSM 각각 비트 읽어서 다음 상태로 전환

→ 세 룩업을 파이프라인으로 실행 가능 (서로 독립)
→ 한 FSM이 테이블 읽는 동안 다른 FSM이 비트 읽기 수행
→ CPU 명령어 수준 파이프라이닝 극대화 → ZStd 해제 속도의 핵심
```

#### ZStd 블록 구조

ZStd는 **프레임 → 블록** 계층 구조. 블록 단위로 독립 압축 가능.

```
ZStd 프레임:
  [Magic: 4B] [Frame Header] [Block 0] [Block 1] ... [Checksum: 4B]

각 블록:
  [Block Header: 3B]
    - 하위 1비트: Last_Block 여부
    - 다음 2비트: Block_Type (Raw/RLE/Compressed/Reserved)
    - 나머지: Block_Size

Block_Type = Compressed 일 때 내부 구조:
  [Literals Section]   ← literal 데이터 (Huffman/FSE/Raw 중 선택)
  [Sequences Section]  ← LZ match 시퀀스들 (offset, match_len, literal_len을 FSE 코딩)

Sequences 해제 순서:
  1. FSE 테이블 복원 (블록 헤더에 저장된 정보로)
  2. 각 시퀀스: (literal_len, offset, match_len) 세 값을 FSE로 동시 디코딩
     → 세 개의 독립 FSE 상태 기계가 interleaved로 동작 (파이프라인 효율)
  3. literal 복사 → match 복사 반복
```

**ZStd 압축 레벨**:
```
레벨 1  (fastest): 압축률 ~2×, CPU 압축 ~500 MB/s
레벨 3  (default): 압축률 ~3×, CPU 압축 ~200 MB/s
레벨 19 (best):    압축률 ~5×, CPU 압축 ~10 MB/s

해제 속도는 레벨과 무관하게 항상 빠름 (~1.5 GB/s on CPU, ~10 GB/s on GPU)
→ 압축은 오프라인에서 한 번, 해제는 쿼리마다 온라인 → 레벨 높여도 이득
```

#### LZ4 vs ZStd 실전 선택

```
LZ4 선택:
  - 해제가 초저지연(μs 단위) 필요
  - 압축률보다 CPU/GPU 해제 처리량이 중요
  - 스트리밍 분석처럼 해제가 핫패스

ZStd 선택:
  - SSD/네트워크 IO가 병목 → 압축률로 IO 감소가 더 이득
  - 압축은 오프라인 (배치 ETL 등)
  - DB 컬럼 스토리지 (GOLAP, Parquet 등)에서 주로 선택
```

---

### 2-3. GPU 해제의 핵심 난제: 순차 의존성

LZ 계열 해제는 본질적으로 순차적. 이게 GPU 병렬화의 최대 장벽.

```
해제 의존성 체인:

출력[0..5]   = literal "abcdef"
출력[6..9]   = 출력[0..3] 복사  ← 출력[0..5]가 있어야 계산 가능
출력[10..13] = 출력[4..7] 복사  ← 출력[6..9]가 있어야 계산 가능
...

→ 스레드 B가 스레드 A의 출력에 의존
→ A가 끝나야 B 시작 가능 → 순차 실행
```

**Overlapping copy 문제** (더 까다로운 케이스):
```
offset < length인 경우:
  현재 출력 위치 = 100
  offset = 3, length = 8

  출력[100] = 출력[97]
  출력[101] = 출력[98]
  출력[102] = 출력[99]
  출력[103] = 출력[100]  ← 방금 쓴 값을 다시 읽음!
  출력[104] = 출력[101]  ← 방금 쓴 값을 다시 읽음!
  ...

→ memcpy로 한번에 처리 불가, 바이트 하나씩 순서대로 복사해야 함
→ SIMD/vectorized copy 불가 → GPU 병렬화 더욱 어려움
```

---

### 2-4. GPU 병렬 해제 전략

#### 전략 1: 청크 단위 병렬 해제 (가장 일반적)

압축 시 데이터를 **독립된 청크**로 나눠 각각 압축. 청크 간 back-reference 금지.

```
압축 시:
  [청크0: 독립 압축] [청크1: 독립 압축] [청크2: 독립 압축] ...
  청크 경계를 넘는 back-reference 없음

해제 시:
  GPU thread block 0 → 청크0 해제
  GPU thread block 1 → 청크1 해제  (동시 실행!)
  GPU thread block 2 → 청크2 해제
  ...

병렬성 = 청크 수 = 수천 ~ 수만 개 → GPU 활용률 극대화
```

**트레이드오프**:
```
청크 크기 크면: 압축률↑ (윈도우 크면 더 긴 패턴 탐색 가능), 병렬성↓
청크 크기 작으면: 병렬성↑, 압축률↓ (짧은 윈도우 = 찾을 수 있는 반복 패턴 한정)

일반적으로 64KB ~ 1MB 사이에서 결정
nvCOMP 기본값: 64KB
```

#### 전략 2: 청크 내 warp 협력 해제

청크 하나를 단일 스레드가 아닌 **warp(32 스레드)가 협력**해서 처리.

```
청크를 sub-sequence로 분할:
  [seq0] [seq1] [seq2] ... (각 seq = literal + match 1개씩)

2단계 처리:
  1단계 (분석 패스): 각 스레드가 자신의 seq를 스캔 → 출력 위치 계산
     thread 0: seq0 → 출력 위치 0~15
     thread 1: seq1 → 출력 위치 16~28 (seq0 결과 필요 → prefix sum으로 계산)
     thread 2: seq2 → 출력 위치 29~45
     → warp-level prefix sum으로 모든 스레드의 출력 위치 동시 결정

  2단계 (기록 패스): 각 스레드가 자신의 출력 범위에 기록
     → back-reference 의존성이 없는 literal은 병렬 기록
     → match는 의존성 해결 후 기록
```

#### 전략 3: 두 패스 해제 (nvCOMP ZStd 방식)

```
패스 1 (분석, 완전 병렬):
  각 스레드 블록이 압축 스트림에서 토큰만 파싱
  → 출력 오프셋, literal 위치, match 위치 메타데이터 생성
  → 실제 데이터 복사 안 함

패스 2 (기록):
  메타데이터 기반으로 실제 출력 기록
  back-reference 의존성이 해결된 경우 병렬 기록
  → 긴 match는 여러 스레드가 나눠서 기록
```

---

### 2-5. nvCOMP (NVIDIA 공식 라이브러리)

NVIDIA가 제공하는 GPU 압축/해제 라이브러리. 배치(batch) API로 수천 개 청크를 한 번에 병렬 처리.

```
// ZStd 해제 예시 (nvCOMP Batched API)
nvcompBatchedZstdDecompressAsync(
  compressed_data_ptrs,    // 각 청크의 압축 데이터 포인터 배열 (device memory)
  compressed_sizes,        // 각 청크의 압축된 크기 배열
  output_ptrs,             // 각 청크의 출력 버퍼 포인터 배열
  output_sizes,            // 각 출력 버퍼 크기 배열
  actual_decompressed_sizes, // 실제 해제된 크기 (output)
  num_chunks,              // 총 청크 수
  scratch,                 // 임시 작업 메모리
  scratch_size,
  statuses,                // 각 청크의 성공/실패 상태
  stream                   // CUDA 스트림 (비동기 실행)
);
```

**지원 알고리즘 및 특성**:
```
LZ4:      해제 속도 최우선. 컬럼 스캔처럼 해제가 핫패스일 때
Snappy:   LZ4와 유사, Google 생태계 호환성
ZStd:     압축률과 해제 속도 균형. DB 스토리지에 주로 선택
Deflate:  zlib/gzip 호환 필요할 때 (해제 느림)
Cascaded: nvCOMP 자체 경량 압축 (RLE + delta + bit-packing 조합), 정수 컬럼 특화
Bitcomp:  nvCOMP 자체 비트 압축, 매우 빠른 해제
```

**Cascaded (nvCOMP 자체 형식)**:
```
정수 컬럼에 특화된 경량 압축을 GPU 친화적으로 설계:
  1단계: delta encoding
  2단계: RLE
  3단계: bit-packing
→ 각 단계가 GPU shared memory 내에서 처리 (tile-based와 유사한 접근)
→ 압축률은 낮지만 해제가 극도로 빠름 (~100 GB/s 이상)
→ 실시간 스트리밍 워크로드에 적합
```

**GOLAP에서의 활용**:
```
SSD → GPU HBM (GDS 직접 전송, CPU bypass)
         ↓
nvCOMP ZStd 배치 해제 (수천 청크 동시)
         ↓
해제된 컬럼 데이터로 바로 scan/filter/join
```

CPU가 해제 병목이 되는 걸 GPU로 옮김 → 유효 대역폭 2.8× 향상.

**GPU vs CPU 해제 속도 비교 (참고치)**:
```
ZStd 해제:
  CPU (단일 코어): ~1.5 GB/s
  CPU (32코어):   ~40 GB/s
  GPU (A100):     ~200 GB/s  (nvCOMP, 충분한 청크 수)

LZ4 해제:
  CPU (단일 코어): ~4 GB/s
  CPU (32코어):   ~100 GB/s
  GPU (A100):     ~350 GB/s  (nvCOMP)

→ GPU가 CPU 단일 코어 대비 100×, 멀티코어 대비 5~7× 빠름
→ PCIe 대역폭(28 GB/s)보다 GPU 해제 속도가 훨씬 빠름
  → 해제 자체가 병목이 되는 경우는 없음, IO가 병목
```

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
