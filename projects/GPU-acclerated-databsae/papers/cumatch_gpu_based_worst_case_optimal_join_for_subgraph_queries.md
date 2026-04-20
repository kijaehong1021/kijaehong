# cuMatch: A GPU-based Memory-Efficient Worst-Case Optimal Join for Subgraph Queries

- **저자**: 미정리
- **학회**: SIGMOD 2025
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3725398

---

## 문제 정의

복잡한 패턴의 subgraph 쿼리는 일반 binary join으로 처리하면 중간 결과가 폭발적으로 커짐. Worst-Case Optimal Join(WCOJ)은 이론적으로 최적이지만 GPU 구현이 어려움.

---

## 핵심 아이디어

GPU에서 WCOJ 알고리즘을 메모리 효율적으로 구현:

- **WCOJ**: 여러 릴레이션을 동시에 교차시켜 중간 결과 크기를 최소화
- GPU의 병렬성으로 교차 연산을 가속
- 메모리 효율적 설계로 GPU HBM 제약 극복

---

## 성능 결과

- (미정리 - PDF 없음)
