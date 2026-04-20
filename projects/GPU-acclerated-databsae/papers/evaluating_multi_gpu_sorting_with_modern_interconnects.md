# Evaluating Multi-GPU Sorting with Modern Interconnects

- **저자**: Justus Henneberg, Felix Schuhknecht (Johannes Gutenberg Univ.)
- **학회**: SIGMOD 2022
- **링크**: http://dl.acm.org/doi/pdf/10.1145/3514221.3517842

---

## 문제 정의

멀티-GPU 환경에서 정렬(sort)은 GPU 간 데이터 교환이 필수. PCIe vs NVLink 등 다양한 인터커넥트 특성이 정렬 성능에 미치는 영향이 불명확.

---

## 핵심 아이디어

멀티-GPU 정렬 알고리즘들을 다양한 인터커넥트(PCIe 3/4, NVLink 2/3) 조합에서 체계적으로 평가:

- **Merge-based**: 각 GPU에서 부분 정렬 후 병합
- **Partition-based**: radix partition으로 데이터를 GPU에 분배 후 각 GPU에서 로컬 정렬
- 인터커넥트 대역폭이 정렬 성능의 핵심 병목임을 분석

---

## 성능 결과

- (미정리 - PDF 없음)
