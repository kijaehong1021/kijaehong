# Graph Processing on GPUs: A Survey

- **저자**: Xuanhua Shi 외
- **학회**: ACM Computing Surveys
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3128571

---

## 개요

GPU를 활용한 그래프 처리 기법들을 체계적으로 정리한 서베이 논문.

---

## 주요 내용

### 그래프 표현
- CSR(Compressed Sparse Row), CSC, Edge List 등 GPU에 적합한 그래프 저장 방식
- 메모리 접근 패턴과 코얼레싱 관점에서의 비교

### 그래프 알고리즘의 GPU 구현
- BFS, SSSP, PageRank, 삼각형 카운팅 등 핵심 알고리즘
- 워크로드 불균형(load imbalance) 문제: 허브 노드(degree 높은 노드)가 병렬성을 깸

### 주요 최적화 기법
- **warp-level 병렬화**: 엣지를 warp 단위로 분배
- **동적 워크로드 밸런싱**: 허브 노드 처리를 별도로 분리
- **GPU-CPU 협력**: 메모리 초과 시 CPU와 분담

---

## 성능 결과

- (서베이 논문 - 개별 성능 수치보다 기술 비교 위주)
