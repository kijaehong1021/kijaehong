# GPH: An Efficient and Effective Perfect Hashing Scheme for GPU Architectures

- **저자**: Jiaping Cao, Le Xu, Man Lung Yiu, Jianbin Qin, Bo Tang (Hong Kong PolyU, Shenzhen Univ, SUSTech)
- **학회**: SIGMOD 2025
- **링크**: https://doi.org/10.1145/3725406

---

## 문제 정의

GPU 해시 테이블은 충돌 해결 방식에 따라 성능이 크게 달라짐. 기존 방식들의 한계:

| 방식 | 프로브 수 | 문제 |
|---|---|---|
| Separate Chaining | 무제한 | 링크드 리스트 → 느림 |
| Open Addressing | 무제한 | 충돌 시 여러 슬롯 탐색 |
| Cuckoo Hashing | O(1) | 1~2번 프로브, 불일치 발생 |
| **Perfect Hashing** | **정확히 1번** | 최고 성능, 기존엔 동적 업데이트 미지원 |

GPU 기반 perfect hashing으로 동적 업데이트(insert)까지 지원하는 구현이 없었음.

---

## 성능 분석 모델

GPH 설계에 앞서 기존 해시 테이블 성능을 정량적으로 분석하는 모델 제안:

```
Time ∝ (EI × CPI) / AO

AO  = Achieved Occupancy      : 활성 warp 비율 (높을수록 좋음)
EI  = Executed Instructions   : 총 실행 명령어 수 (적을수록 좋음)
CPI = Cycles Per Instruction  : 메모리 레이턴시 지표 (낮을수록 좋음)
```

기존 해시 테이블 측정 결과:

| | AO | EI(억) | CPI | 실행 시간 |
|---|---|---|---|---|
| WarpCore (Open Addr.) | 70.89% | 93.6 | 22.58 | 45ms |
| DyCuckoo (Cuckoo) | 86.87% | 186.9 | 24.78 | 88ms |
| CUDPP (Cuckoo) | 92.37% | 7.3 | 445.38 | 55ms |
| **GPH (Perfect)** | **98.24%** | **64.3** | **23.16** | **23ms** |

### 핵심 인사이트 3가지

1. **AO가 낮은 이유**: 프로브 수가 들쭉날쭉하면 먼저 끝난 warp이 다른 warp을 기다림 (tail effect). → 정확히 1번 프로브하면 AO 최대화.
2. **EI 결정 요인**: cooperative group 크기. 그룹이 클수록 warp당 처리하는 lookup 수가 줄어 EI 증가.
3. **CPI 결정 요인**: 메모리 코얼레싱. 연속 메모리 접근 시 CPI 감소. CUDPP는 랜덤 접근이라 CPI가 445로 폭발.

---

## GPH 데이터 구조

**2계층 구조**:
- **Cell Index** (shared memory): `(Mapping Φ, Offset Δ)` 쌍 저장 → 빠른 접근
- **Hash Table Buckets** (global memory): 실제 key-value 저장

### lookup(k) 과정 (global memory 접근 단 1회)

```
1. Hc(k) → Cell id i              (shared memory 접근, 빠름)
2. Hv(k, Φi) → Virtual Bucket id j
3. Hb(i, j, Φi, Δi) → Bucket id  (global memory 딱 1번!)
4. Bucket 확인 → 결과 반환
```

Cell Index가 shared memory에 있어 빠르고, global memory는 정확히 1번만 접근 → AO 98.24% 달성.

---

## Bucket Requester: EI × CPI 최소화

버킷 접근이 가장 비싼 연산 → **Vectorization + ILP**로 최적화.

### Vectorization

```
기존 (DyCuckoo): 16스레드 그룹 → 슬롯 1개씩 (8바이트)
GPH:              4스레드 그룹  → 벡터 1개씩 (16바이트 = 슬롯 2개)
```

CUDA의 `ulonglong2` 벡터 타입으로 슬롯 2개를 한 번에 읽음. 그룹 크기가 4배 줄어드니 같은 warp에서 4배 더 많은 lookup 처리 → EI 감소.

### ILP (Instruction-Level Parallelism)

스레드당 여러 벡터를 루프 언롤링으로 동시에 요청 → 파이프라인 실행으로 메모리 레이턴시 숨김.

---

## Insert 커널 (동적 업데이트 지원)

기존 GPU perfect hashing은 정적 데이터만 지원. GPH는 **동적 삽입을 최초로 지원**.

**2단계 삽입**:

1. **Saturation Fill Phase**: lock-free로 빈 슬롯에 직접 삽입 (overflow 없을 때)
2. **Refinement Phase**: overflow 발생 시 Virtual Bucket 내에서 Extended Move로 재배치

Virtual Bucket 덕분에 overflow 시 **전체 재해시 없이** 같은 셀 내에서만 키-값 재배치 가능.

---

## 성능 결과

- 8500 MOPS (Million Operations Per Second) 이상 달성
- 기존 최고 성능 GPU 해시 테이블 대비 **1.70x ~ 2.78x** lookup 처리량 향상
- synthetic 및 real-world 워크로드 모두에서 검증
