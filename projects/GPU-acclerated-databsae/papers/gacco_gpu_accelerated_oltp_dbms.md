# GaccO: A GPU-accelerated OLTP DBMS

- **저자**: Bang Liu 외 (HKUST)
- **학회**: SIGMOD 2022
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3514221.3517876

---

## 문제 정의

OLTP(온라인 트랜잭션 처리)는 짧은 트랜잭션과 높은 동시성이 특징. 기존 GPU DB는 OLAP(분석)에 집중했고, OLTP에서 GPU 활용은 동시성 제어(concurrency control)의 어려움 때문에 미개척.

---

## 핵심 아이디어

GPU에서 OLTP 트랜잭션을 처리하는 완전한 DBMS:

- **GPU 친화적 동시성 제어**: 낙관적 동시성 제어(OCC) 기반, GPU의 대규모 병렬성에 맞게 설계
- **인덱스 구조**: GPU에 최적화된 해시 인덱스 및 B-tree 변형
- **트랜잭션 배치 처리**: 다수 트랜잭션을 배치로 묶어 GPU 병렬성 최대 활용

---

## 성능 결과

- (미정리 - PDF 없음)
