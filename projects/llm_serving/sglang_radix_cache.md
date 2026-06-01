# SGLang & RadixAttention

> Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (SOSP 2025)

---

## 1. SGLang이 뭔가?

**SGLang (Structured Generation Language)**: LLM을 프로그래밍하기 위한 DSL + 이를 효율적으로 실행하는 serving runtime.

두 가지로 구성:
- **Frontend**: Python-like 언어로 LLM 프로그램 작성 (multi-turn, branching, parallelism 등)
- **Runtime**: 이 프로그램을 효율적으로 실행 — 핵심이 **RadixAttention**

---

## 2. 왜 RadixAttention이 필요한가?

### 기존 KV cache prefix caching의 한계

vLLM의 prefix caching:
- 동일한 prefix를 가진 요청은 KV cache 재사용 가능
- 하지만 **정확히 같은 prefix일 때만** hit

현실의 LLM 프로그램은 훨씬 복잡한 prefix 공유 패턴을 가짐:

```
패턴 1 — Few-shot prompting:
  [system prompt][example 1][example 2][example 3][query A]
  [system prompt][example 1][example 2][example 3][query B]
  → 공통 prefix: [system prompt][example 1][example 2][example 3]

패턴 2 — Tree-of-Thought:
  [system][문제]
  ├── [풀이 방향 1][다음 단계 1a]
  ├── [풀이 방향 1][다음 단계 1b]   ← [system][문제][풀이 방향 1] 공유
  └── [풀이 방향 2][다음 단계 2a]

패턴 3 — Multi-turn chat:
  [system][turn1][turn2][turn3][turn4]
  [system][turn1][turn2][turn3][turn5]   ← turn4까지 공유

패턴 4 — RAG:
  [system prompt][retrieved doc 1][retrieved doc 2][query]
  → doc들이 여러 query에서 반복 등장
```

→ 단순한 "같은 prefix" 매칭으로는 이런 패턴을 모두 포착 못함.

---

## 3. Radix Tree — 자료구조 이해

### Trie (Prefix Tree) 먼저

문자열 집합을 prefix 공유로 압축하는 트리:

```
저장된 문자열: "cat", "car", "card", "dog"

Trie:
  root
  ├── c
  │   └── a
  │       ├── t  (end: "cat")
  │       └── r  (end: "car")
  │           └── d  (end: "card")
  └── d
      └── o
          └── g  (end: "dog")
```

공통 prefix는 한 번만 저장.

### Radix Tree (Compressed Trie)

연속된 단일 자식 노드를 하나로 합침:

```
Radix Tree (위와 동일):
  root
  ├── "ca"
  │   ├── "t"   (end: "cat")
  │   └── "r"   (end: "car")
  │       └── "d"  (end: "card")
  └── "dog"   (end: "dog")
```

→ 노드 수 감소, 공통 prefix 효율적 공유

### Token sequence에 적용

SGLang은 token sequence를 radix tree로 관리:
- 각 노드 = token sequence segment + 해당 KV cache 블록
- 공통 prefix를 가진 요청들이 동일 노드(= 동일 KV cache)를 공유

---

## 4. RadixAttention의 동작

### 핵심 아이디어

> KV cache를 radix tree로 관리하여 **임의의 공통 prefix**를 자동으로 감지하고 재사용.

### 자료구조

```python
# 개념적 표현
class RadixTreeNode:
    key: list[int]         # token ids (이 노드의 구간)
    kv_cache: KVBlock      # 이 구간에 해당하는 KV cache
    children: dict[int, RadixTreeNode]  # 다음 token → 자식 노드
    last_access_time: float  # LRU eviction용
    ref_count: int           # 현재 이 노드를 쓰고 있는 request 수
```

### Request 처리 흐름

**Case 1: Cache miss (처음 보는 prefix)**

```
입력: [sys][A][B][C][D]

Radix tree: 비어있음

처리:
  1. 전체 prefill 실행 → KV cache 생성
  2. tree에 삽입:
     root → [sys][A][B][C][D] (노드 1개)
  3. decode 시작
```

**Case 2: Partial cache hit**

```
입력: [sys][A][B][C][X][Y]   ← [sys][A][B][C]는 이미 있음

Radix tree:
  root → [sys][A][B][C][D]

처리:
  1. longest prefix match: [sys][A][B][C] hit → KV cache 재사용
  2. 나머지 [X][Y]만 prefill → KV cache append
  3. tree 업데이트:
     root → [sys][A][B][C]    ← 분기점 (split)
                ├── [D]
                └── [X][Y]
  4. decode 시작
```

Split 과정 (핵심):

```
Before:
  root → [sys][A][B][C][D]    (노드 하나)

[sys][A][B][C][X][Y] 요청 들어옴
→ [sys][A][B][C]까지 match, [D] vs [X]에서 분기

After:
  root → [sys][A][B][C]       ← 새 분기 노드 (KV cache 공유)
              ├── [D]          ← 기존 suffix
              └── [X][Y]      ← 새 suffix
```

**Case 3: Full cache hit**

```
입력: [sys][A][B][C][D]   ← 이미 tree에 있음

처리:
  1. 전체 KV cache hit → prefill 완전 skip
  2. decode만 실행
  → TTFT ≈ 0 (prefill 비용 없음)
```

---

## 5. LRU Eviction — 메모리 관리

GPU KV cache는 유한 → 오래된 노드 제거 필요.

### 조건

- `ref_count == 0`: 현재 사용 중인 request가 없는 노드만 evict 가능
- LRU (Least Recently Used): `last_access_time`이 가장 오래된 노드부터 제거

### Eviction 규칙

```
Evict 불가: ref_count > 0  (누군가 쓰는 중)
Evict 가능: ref_count == 0 and 메모리 부족

부모 노드보다 자식 노드를 먼저 evict
→ 공통 prefix는 최대한 유지
```

예시:

```
tree:
  root → [sys][A][B]         ref=0, last=T1
              ├── [C][D]     ref=0, last=T2
              └── [E][F]     ref=2, last=T3  ← 사용 중

메모리 부족 → evict 후보: [C][D] (ref=0, 더 오래됨)
→ [sys][A][B]는 유지 ([E][F]가 이 prefix 필요)
→ [C][D]만 제거
```

---

## 6. 구체적 성능 예시

### Few-shot 시나리오

```
System prompt: 500 tokens
Few-shot examples: 1000 tokens
Query: 10 tokens
Output: 100 tokens

총 입력: 1510 tokens

RadixAttention 없이:
  매 request마다 1510 token prefill → 느림

RadixAttention 있이:
  첫 request: 1510 token prefill + tree에 저장
  이후 request: 10 token prefill (query만) + KV cache hit
  → prefill 비용 99% 절감
```

### Multi-turn chat

```
Turn 1: [sys][Q1][A1]                     → prefill 3 segments
Turn 2: [sys][Q1][A1][Q2][A2]             → 앞 3개 hit, [Q2][A2]만 prefill
Turn 3: [sys][Q1][A1][Q2][A2][Q3][A3]    → 앞 5개 hit, [Q3][A3]만 prefill
...
Turn N: 이전 N-1 turn KV cache 전부 hit   → O(1) prefill
```

### 논문 reported 성능 (vs vLLM)

| 시나리오 | vLLM | SGLang |
|----------|------|--------|
| Few-shot (same examples) | 1x | **5.7x** throughput |
| Multi-turn chat | 1x | **3.3x** |
| Tree-of-Thought | 1x | **4.4x** |
| Single request (no sharing) | 1x | ~1x (비슷) |

공유 패턴이 없을 때는 이득 없음. 공유가 많을수록 이득 큼.

---

## 7. VLM에서의 RadixAttention

`[[vlm_multimodal]]`과의 연결:

### Image prefix 재사용

같은 이미지에 여러 질문:
```
[image KV cache (256 tokens)] + [질문 1]
[image KV cache (256 tokens)] + [질문 2]
[image KV cache (256 tokens)] + [질문 3]
```

RadixAttention:
- 첫 번째 요청: image encode + prefill → KV cache 저장
- 이후 요청: image KV cache hit → **image prefill 완전 skip**
- 이미지 인코딩 자체도 캐시 가능 (vision encoder 출력)

RAG + VLM 시나리오:
```
[system][retrieved image 1][retrieved image 2][query A]
[system][retrieved image 1][retrieved image 2][query B]
→ image KV cache 전부 공유
```

### Tile 처리와 결합

고해상도 tile (4개 tile × 256 token = 1024 token):
- 같은 이미지의 tile들은 hash 기반으로 식별
- tile 단위 KV cache prefix sharing → 동일 이미지 반복 질의에 강력

---

## 8. 구현 세부사항

### Token-level hashing

prefix 매칭은 token sequence의 **hash**로 빠르게:

```
node_key = hash(parent_hash + token_ids)
→ O(1) 조회 (vs 전체 sequence 비교)
```

충돌 시 실제 token 비교 fallback.

### Chunked prefix

긴 prefix는 chunk 단위(예: 256 token)로 노드 분할:
- 너무 긴 노드는 eviction 단위가 커져서 비효율
- chunk 단위로 자르면 partial eviction 가능

### Thread-safe 동작

여러 request가 동시에 tree 접근:
- 읽기: ref_count 증가 → concurrent read OK
- 쓰기(insert/split): lock 필요하지만 짧은 critical section

---

## 9. 다른 시스템과의 비교

| 기능 | vLLM prefix cache | SGLang RadixAttention |
|------|-------------------|----------------------|
| 동일 prefix 공유 | ✓ | ✓ |
| 부분 prefix 공유 | ✗ | ✓ (자동 split) |
| Tree 구조 공유 | ✗ | ✓ |
| LRU eviction | ✓ (block 단위) | ✓ (node 단위, 세밀) |
| Multi-modal prefix | 제한적 | ✓ |
| API 노출 | 자동 | 자동 |

TensorRT-LLM, LMDeploy도 유사한 prefix caching 지원하지만 radix tree 기반은 SGLang이 원조.

---

## 10. 한계

- **메모리 overhead**: radix tree 자체 메모리 + ref counting
- **Hash 충돌**: 드물지만 발생 시 KV cache miss로 처리
- **Cold start**: 처음 요청들은 cache miss → 워밍업 필요
- **Dynamic content**: 매번 달라지는 prefix (timestamp, random seed 등) → 공유 불가
- **Eviction 비용**: LRU 유지를 위한 lock contention (고부하 시)

---

## 다음 단계

- [ ] SGLang의 다른 최적화: continuous batching, overlap scheduling
- [ ] Speculative Decoding + RadixAttention 결합
- [ ] Disaggregated prefill에서의 KV cache 전송과의 차이
- [[paged_attention_kv_cache]] — RadixAttention의 기반이 되는 block 관리
- [[vlm_multimodal]] — image prefix caching 응용
- [[continuous_batching]] — SGLang의 스케줄링 기반
