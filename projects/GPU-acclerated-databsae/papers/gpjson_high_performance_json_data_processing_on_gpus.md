# GpJSON: High-performance JSON Data Processing on GPUs

- **저자**: Sacheendra Talluri, Guido Walter Di Donato, Luca Danelutti 외 (VU Amsterdam, Politecnico di Milano, Oracle Labs)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3746405.3746439

---

## 문제 정의

JSON은 **순차적으로 읽어야 하는 포맷**이라 파싱이 느림. 파싱만으로 전체 처리 시간의 최대 80%를 차지하는 경우도 있음.

GPU로 가속하고 싶은데, JSON의 **불규칙한 구조** 때문에 단순 포팅이 어려움:
- 중첩 구조 (depth가 가변)
- 문자열 내부의 특수문자 처리 (escape)
- 줄마다 길이가 다름

기존 CPU 접근(SIMD, 멀티스레딩)은 병렬성이 한계가 있고, GPU DB들은 JSON 원본 포맷을 지원하지 않음.

---

## 핵심 아이디어: Structural Index + In-situ Query

**Structural Index**: JSON을 파싱(역직렬화) 대신 **인덱스로 변환**. 실제 값은 원본 그대로 두고, 어디에 어떤 구조가 있는지만 비트맵으로 기록.

**In-situ Query**: 인덱스를 이용해 원본 데이터에서 필요한 값만 직접 추출. 역직렬화 없이 JSONPath 쿼리 실행.

---

## Structural Index 구축 과정 (5단계 GPU 커널)

```
원본 JSON → [Newline] → [Escape] → [Quote] → [String] → [Leveled Bitmap]
```

각 단계가 독립적인 GPU 커널로 실행되고, 인덱스를 메모리에서 직접 주고받음 (zero-copy).

| 단계 | 인덱스 | 역할 |
|---|---|---|
| 1 | **Newline Index** | LD-JSON에서 각 JSON 객체의 시작 위치 파악 |
| 2 | **Escape Index** | `\"` 같은 이스케이프 문자 위치 기록 |
| 3 | **Quote Index** | 따옴표 위치 기록 |
| 4 | **String Index** | 각 문자가 문자열 내부인지 외부인지 비트맵으로 기록 |
| 5 | **Leveled Bitmap Index** | 최종 인덱스. depth별 구조 문자(`{`, `}`, `:`) 위치 기록 |

String Index가 있어야 문자열 내부의 `{`, `}` 같은 구조 문자를 무시할 수 있음.

### 1. Newline Index

줄 수를 모르니까 메모리를 미리 할당할 수 없음 → **2단계**로 해결:

```
1단계 커널: 각 스레드가 담당 구간에서 줄바꿈 수만 셈
            counts = [1, 6, 3, 1]  (스레드 4개 기준)

prefix sum으로 오프셋 계산:
            offsets = [0, 1, 7, 10]  (각 스레드가 쓸 시작 위치)
            총 크기 = 11 → 이 크기로 메모리 할당

2단계 커널: 각 스레드가 offsets[i]부터 자기 줄바꿈 위치 기록
```

### 2. Escape Index

`\"` 안의 `"` 는 문자열 끝이 아님 → String Index 전에 이스케이프 처리 필요.

**Escape Carry Index**: 스레드 경계에서 이스케이프가 넘어가는지 체크.
```
스레드 A 마지막 문자가 `\` → 스레드 B 첫 문자가 이스케이프됨
carry = True/False로 다음 스레드에 전달
```
carry 정보가 있으면 각 스레드가 독립적으로 병렬 계산 가능.

### 3. Quote Index

Escape Index를 참고해 이스케이프된 `"` 는 제외하고, 진짜 따옴표 위치만 비트맵에 기록:

```
입력:        { "foo" : "bar" }
Quote Index: 0 1 0 0 0 1 0 1 0 0 0 1 0
```

### 4. String Index

**Prefix-XOR Sum**으로 병렬 계산. 따옴표를 만날 때마다 문자열 안/밖 상태가 토글됨:

```
Quote Index: 0 1 0 0 0 1 0 1 0 0 0 1 0
Prefix-XOR:  0 1 1 1 1 0 0 1 1 1 1 0 0
             ←문자열 내부→   ←내부→
```

결과적으로 1인 구간 = 문자열 내부. GPU parallel scan으로 구현 (CPU의 `pclmulqdq` 명령어 없이 직접 구현).

### 5. Leveled Bitmap Index

String Index로 문자열 내부 구조 문자를 무시하고, 진짜 구조 문자만 depth별로 기록:

```json
{"foo": {"bar": "baz"}}
```
```
L0: { 위치 0,  : 위치 6,  } 위치 22  (루트 레벨)
L1: { 위치 8,  : 위치 13, } 위치 21  (한 단계 안)
```

쿼리 실행 시 JSONPath의 depth에 맞는 레벨만 스캔해서 값 위치를 바로 찾음.

---

## GPU에서의 어려움: String Index 병렬 구축

문자열 내부 판단은 **앞 문자의 컨텍스트**가 필요해서 병렬화가 어려움.

예: `\"foo\"bar"` → 어느 따옴표가 진짜 끝인지 순차적으로 봐야 앎.

**해결책: Prefix-XOR Sum으로 병렬 계산**

```
Quote 위치:  0 1 0 0 1 0 0 0
Prefix-XOR:  0 1 1 1 0 0 0 0  ← 1인 구간이 문자열 내부
```

각 비트의 값 = 그 이전 모든 Quote 비트의 XOR → GPU parallel scan 연산으로 처리 가능.

CPU의 `pclmulqdq` 전용 명령어 없이 GPU 커널로 직접 구현.

---

## 파티셔닝 + 파이프라이닝 (GPU 메모리 한계 극복)

파일을 **동일 크기 파티션**으로 나눠 여러 CUDA 스트림으로 파이프라인 처리:

```
파티션 1: [로드] → [인덱스 구축] → [쿼리 실행]
파티션 2:          [로드]         → [인덱스 구축] → [쿼리 실행]
파티션 3:                          [로드]         → ...
```

### Pre-allocation의 핵심

파티션 크기가 **고정**이라 대부분의 인덱스 크기를 미리 계산 가능:

| 인덱스 | 크기 | 할당 방식 |
|---|---|---|
| Escape Index | 파티션 크기의 1/8 (1비트/바이트) | **미리 할당** |
| Quote Index | 파티션 크기의 1/8 | **미리 할당** |
| String Index | 파티션 크기의 1/8 | **미리 할당** |
| Leveled Bitmap | 파티션 크기 × 쿼리 최대 depth / 8 | **미리 할당** |
| Newline Index | 줄 수에 따라 가변 | 동적 할당 (2단계 커널) |

Newline Index만 동적 할당이고 나머지는 전부 미리 할당 → GPU 커널 내부에서 동적 할당 거의 없음.

### 쿼리 depth 최적화

Leveled Bitmap은 JSONPath 쿼리를 정적 분석해서 **필요한 depth까지만** 저장. 쿼리보다 깊은 레벨은 아예 기록하지 않아 메모리 절약.

---

## 쿼리 엔진: JSONPath → 바이트코드

JSONPath 쿼리를 **커스텀 바이트코드로 컴파일** → GPU에서 바이트코드 인터프리터로 실행. **GPU에서 바이트코드 인터프리터를 돌린 첫 사례**.

### 컴파일 과정

JSONPath 표현식을 Lexer → AST → 바이트코드 순으로 변환:

```
$.users[?(@.lang == "en")]

→ 바이트코드:
MOVE_TO_KEY "users"   // Leveled Bitmap에서 "users" 키 탐색
MOVE_DOWN             // L0 → L1로 depth 이동
MOVE_TO_KEY "lang"    // L1에서 "lang" 키 탐색
EXPRESSION_STRING_EQUALS "en"  // 값 비교
STORE_RESULT          // 매칭된 줄 저장
```

### 바이트코드 명령어 종류

| 분류 | 명령어 | 설명 |
|---|---|---|
| Index 탐색 | `MOVE_UP/DOWN` | depth 이동 |
| | `MOVE_TO_KEY` | 특정 키 탐색 |
| | `MOVE_TO_INDEX` | 배열 인덱스 이동 |
| 제어 흐름 | `END` | 현재 줄 실행 종료 |
| | `STORE_RESULT` | 결과 저장 |
| 표현식 | `EXPRESSION_STRING_EQUALS` | 문자열 비교 |

### 실행 방식

각 GPU 스레드가 여러 JSON 줄을 담당. 줄마다 바이트코드 인터프리터를 실행:

```
1. Newline Index로 줄 시작 위치 확인
2. MOVE_TO_KEY "users" → Leveled Bitmap L0 스캔
3. MOVE_DOWN → L1로 이동
4. MOVE_TO_KEY "lang" → L1 스캔
5. EXPRESSION_STRING_EQUALS "en" → 원본 JSON에서 값 직접 비교
6. 매칭되면 STORE_RESULT, 아니면 END
```

매칭 실패 시 즉시 종료 → 불필요한 처리 없음.

### 현재 한계

- 재귀 탐색(`$..author`) 미지원
- 동적 배열 슬라이싱(`$.user[2:]`) 미지원
- 입력 JSON이 well-formed이어야 함

---

## 성능 결과

| 비교 대상 | 향상 |
|---|---|
| CPU SIMD/멀티스레딩 (simdjson 등) | **2.9x** 이상 |
| NVIDIA RAPIDS cuDF | **6~8x** |

단일 NVIDIA A100에서 측정. 역직렬화 + 쿼리 end-to-end 기준.
