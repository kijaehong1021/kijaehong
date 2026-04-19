# GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP

- **저자**: Nils Boeschen, Tobias Ziegler, Carsten Binnig (TU Darmstadt)
- **학회**: SIGMOD 2024
- **링크**: https://doi.org/10.1145/3698812

---

## 문제 정의

SSD 기반 DB는 비용 효율적이지만 인메모리 시스템 대비 3.5배 낮은 대역폭으로 성능 격차가 존재. SSD에서 데이터를 읽어 CPU 메모리에 올린 뒤 처리하는 기존 방식은 이 병목을 해결하지 못함.

---

## 핵심 아이디어: GPU-in-Data-Path

GPU를 연산 가속기가 아닌 **I/O 경로 자체에 배치**하는 아키텍처.

### 동작 방식

1. **SSD → GPU 직접 스트리밍**: 압축된 heavy-weight 블록을 SSD에서 GPU로 직접 전송
2. **On-the-fly 압축 해제**: 테이블 스캔 중 GPU에서 실시간으로 압축 해제
3. **GPU 최적화 Pruning**: 불필요한 데이터 블록을 GPU에서 조기에 제거하여 체감 읽기 대역폭 추가 향상

### 왜 효과적인가?

- CPU를 거치지 않아 CPU-GPU 데이터 전송 병목 제거
- 압축 상태로 전송 → 실효 대역폭 대폭 증가
- GPU의 높은 연산 병렬성으로 압축 해제 비용 최소화

---

## 성능 결과

- 유효 대역폭 최대 **100 GiB/s** 달성
- 기존 인메모리 시스템의 대역폭 한계를 초과
- SSD 기반 시스템과 인메모리 시스템 간 성능 격차 해소
