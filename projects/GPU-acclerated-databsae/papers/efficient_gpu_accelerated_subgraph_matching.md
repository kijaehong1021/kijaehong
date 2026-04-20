# Efficient GPU-Accelerated Subgraph Matching

- **저자**: Xibo Sun 외
- **학회**: SIGMOD 2023
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3589326

---

## 문제 정의

Subgraph matching(패턴 그래프가 대상 그래프의 부분 그래프인지 찾기)은 NP-hard 문제로 탐색 공간이 폭발적으로 증가. CPU 기반 접근은 대규모 그래프에서 너무 느림.

---

## 핵심 아이디어

GPU의 대규모 병렬성으로 subgraph matching 탐색 공간을 병렬 탐색:

- **워크로드 밸런싱**: GPU 스레드 간 불균등한 탐색 공간을 동적으로 분배
- **메모리 효율화**: 중간 결과(partial matching)를 GPU 메모리 내에서 효율적으로 관리
- **조기 종료**: 유효하지 않은 탐색 경로를 GPU에서 빠르게 가지치기

---

## 성능 결과

- (미정리 - PDF 없음)
