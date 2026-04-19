# GPU Acceleration of SQL Analytics on Compressed Data

- **저자**: Zezhou Huang, Krystian Sakowski, Hans Lehnert, Wei Cui, Carlo Curino, Matteo Interlandi, Marius Dumitru, Rathijit Sen (Microsoft)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3778092.3778095

---

## 문제 정의

GPU는 SQL analytics 가속에 적합하지만, GPU HBM(High Bandwidth Memory)이 CPU 메인 메모리보다 훨씬 작아 대용량 데이터셋을 처리하지 못하는 근본적 한계가 있음.

기존 해결책(멀티-GPU, 배치 처리, CPU-GPU 하이브리드)은 모두 낮은 대역폭이나 I/O 오버헤드, 높은 비용 문제를 가짐.

**핵심 관찰**: 압축으로 데이터 크기를 줄이면 GPU 메모리에 더 많은 데이터를 올릴 수 있지만, 압축 해제 비용이 너무 크다. → **압축 해제 없이 압축 데이터 위에서 직접 연산하자.**

---

## 핵심 아이디어

### 압축 데이터에 직접 동작하는 연산 프리미티브

다음 압축 방식들을 지원하는 새로운 GPU 연산 메서드:

| 압축 방식 | 특징 |
|---|---|
| **RLE (Run-Length Encoding)** | 연속 동일 값을 (값, 횟수) 쌍으로 저장 |
| **Index Encoding** | 값을 인덱스로 대체 |
| **Bit-width Reduction** | 필요한 비트 수만 사용 |
| **Dictionary Encoding** | 고유값 사전으로 대체 |

### 주요 기술적 기여

1. **Multi-column RLE 연산**: 압축 해제 없이 여러 RLE 열을 동시에 처리
2. **Heterogeneous column encoding 처리**: 열마다 다른 압축 방식이 혼재해도 처리 가능
3. **PyTorch tensor 연산 활용**: 다양한 디바이스(CPU, GPU)에 이식 가능한 구현

---

## 성능 결과

- CPU 전용 상용 analytics 시스템 대비 **수십 배(order of magnitude) 속도 향상**
- GPU 메모리에 비압축 상태로 올릴 수 없는 실제 프로덕션 데이터셋에서 검증
