# Tile-based Lightweight Integer Compression in GPU

- **저자**: Evangelia Sitaridi 외
- **학회**: SIGMOD 2022
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3514221.3526132

---

## 문제 정의

GPU에서 정수 컬럼 압축/해제는 메모리 용량 절감과 전송 비용 감소에 유리하지만, GPU의 SIMT 실행 모델에 맞는 압축 알고리즘 설계가 필요.

---

## 핵심 아이디어

**Tile 단위**로 데이터를 나눠 GPU warp/thread block에 매핑:

- FOR (Frame of Reference), Delta, BitPacking 등 경량 압축을 GPU에 최적화
- warp 내 스레드들이 tile의 서로 다른 값을 병렬로 압축/해제
- 메모리 코얼레싱을 유지하면서 압축 처리량 최대화

---

## 성능 결과

- (미정리 - PDF 없음)
