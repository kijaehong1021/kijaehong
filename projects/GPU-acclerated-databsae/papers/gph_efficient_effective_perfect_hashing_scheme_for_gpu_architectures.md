# GPH: An Efficient and Effective Perfect Hashing Scheme for GPU Architectures

- **저자**: Jiaping Cao, Le Xu, Man Lung Yiu, Jianbin Qin, Bo Tang (Hong Kong PolyU, Shenzhen Univ, SUSTech)
- **학회**: SIGMOD 2025
- **링크**: https://doi.org/10.1145/3725406

---

## 문제 정의

GPU 기반 해시 테이블은 높은 메모리 대역폭과 대규모 병렬성 덕분에 빠른 lookup을 제공하지만:

1. **기존 GPU 해시 테이블 분석 부재**: 성능을 균일하게 평가하는 벤치마크/모델이 없음
2. **Lookup 성능 한계**: 기존 구현들은 bucket probe 수가 많아 random memory access 증가
3. **동적 업데이트 미지원**: 대부분의 GPU 해시 테이블이 static (insert 불가)

---

## 핵심 아이디어

### 1. 마이크로벤치마크 + 성능 분석 모델

GPU 해시 테이블들의 lookup 성능을 균일하게 평가하는 벤치마크와 general 성능 분석 모델 개발.

### 2. Perfect Hashing (GPH 핵심)

**모든 lookup에서 정확히 1번의 bucket probe만 수행**하도록 보장.
- 기존 해시 테이블: collision 시 다수의 probe 필요 → random memory access 증가
- GPH: perfect hashing으로 항상 1 probe → 예측 가능하고 효율적인 메모리 접근

### 3. Global Memory 접근 최적화

- **Vectorization**: 여러 데이터를 묶어 한 번에 메모리 요청
- **Instruction-level Parallelism (ILP)**: 메모리 레이턴시를 숨기기 위해 독립적인 명령어들을 중첩 실행

### 4. 동적 업데이트 지원 (Insert Kernel)

GPU에서 insert 연산을 처리하는 전용 커널 도입 → static 해시 테이블 한계 극복

---

## 성능 결과

- Lookup 처리량: **8,500 MOPS 이상** (합성 및 실제 워크로드 모두)
- 모든 평가된 GPU 기반 해시 테이블 중 최고 성능
