# Distributed GPU Joins on Fast RDMA-capable Networks

- **저자**: Khaled Hamda 외
- **학회**: SIGMOD 2023
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3588709

---

## 문제 정의

단일 서버의 GPU 메모리로 감당할 수 없는 대규모 join을 위해 여러 서버의 GPU를 활용해야 함. 서버 간 통신 비용(네트워크 대역폭)이 병목.

---

## 핵심 아이디어

**RDMA(Remote Direct Memory Access)** 를 활용해 서버 간 GPU 메모리를 직접 접근:

- CPU를 우회해 GPU → GPU로 데이터 직접 전송 (낮은 레이턴시, 높은 대역폭)
- 분산 hash join: 파티셔닝 → 셔플 → 로컬 join 단계를 RDMA로 최적화
- 네트워크 전송과 GPU 연산을 파이프라인으로 겹쳐 실행

---

## 성능 결과

- (미정리 - PDF 없음)
