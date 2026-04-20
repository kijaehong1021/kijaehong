# Everest: GPU-Accelerated System For Mining Temporal Motifs

- **저자**: Yichao Yuan 외 (University of Michigan)
- **학회**: VLDB 2024
- **링크**: https://www.vldb.org/pvldb/vol17/p162-yuan.pdf

---

## 문제 정의

Temporal motif mining(시간 순서가 있는 그래프 패턴 탐색)은 소셜 네트워크, 금융 거래 등에서 중요하지만 계산 비용이 매우 높음. 엣지 수십억 개 규모의 그래프에서 CPU로는 처리 불가.

---

## 핵심 아이디어

GPU 병렬성으로 temporal motif counting을 가속:

- **시간 제약 조인**: 시간 윈도우 내에 발생한 엣지들의 패턴을 GPU join으로 처리
- **메모리 효율적 그래프 표현**: GPU 메모리에 맞게 그래프를 압축 표현
- **병렬 카운팅**: 독립적인 motif 후보들을 GPU 스레드에 분배

---

## 성능 결과

- (미정리 - PDF 없음)
