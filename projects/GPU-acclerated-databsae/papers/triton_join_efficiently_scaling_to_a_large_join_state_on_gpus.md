# Triton Join: Efficiently Scaling to a Large Join State on GPUs with Fast Interconnects

- **저자**: Justus Henneberg, Felix Schuhknecht (Johannes Gutenberg Univ.)
- **학회**: SIGMOD 2022
- **링크**: https://dl.acm.org/doi/pdf/10.1145/3514221.3517911

---

## 문제 정의

Hash join에서 build side(작은 테이블)가 GPU 메모리를 초과하면 성능이 급락. NVLink 같은 고속 인터커넥트가 있는 멀티-GPU 환경에서 이를 활용하는 방법이 불명확.

---

## 핵심 아이디어

NVLink를 통한 GPU 간 고속 통신을 활용해 **join state(해시 테이블)를 여러 GPU에 분산**:

- Build side를 여러 GPU에 나눠 해시 테이블 구성
- Probe side는 각 GPU에서 **모든 GPU의 해시 테이블을 NVLink로 접근**
- PCIe보다 훨씬 높은 NVLink 대역폭으로 cross-GPU 해시 테이블 접근 가능

---

## 성능 결과

- (미정리 - PDF 없음)
