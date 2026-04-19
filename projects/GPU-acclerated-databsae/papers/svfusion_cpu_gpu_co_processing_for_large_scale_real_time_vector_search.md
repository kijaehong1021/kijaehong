# SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search

- **저자**: Yuchen Peng, Dingyu Yang, Zhongle Xie, Ji Sun, Lidan Shou, Ke Chen, Gang Chen (Zhejiang Univ, Huawei)
- **학회**: VLDB 2026
- **링크**: https://doi.org/10.14778/3796195.3796216

---

## 문제 정의

ANNS(Approximate Nearest Neighbor Search)는 정보 검색, 추천 시스템 등의 핵심 연산인데, 대규모 실시간 벡터 검색에서 기존 방법들이 각각의 한계를 가짐:

| 방식 | 문제 |
|---|---|
| CPU 기반 | 동적 업데이트 지원하지만 낮은 처리량 |
| GPU 기반 | 높은 성능이지만 동적 업데이트 어렵고 GPU 메모리 한계 |

→ 높은 처리량 + 동적 업데이트 + 대용량 데이터를 동시에 만족하는 솔루션이 없음.

---

## 핵심 아이디어: SVFusion

GPU-CPU-디스크 3-tier 협력 프레임워크.

### 1. 계층적 벡터 인덱스 (CPU-GPU Co-processing)

- GPU: 핫 데이터(자주 조회되는 벡터) 처리
- CPU: 나머지 데이터 처리 및 인덱스 관리
- 두 처리 결과를 통합하여 최종 검색 결과 생성

### 2. Workload-aware 벡터 캐싱

제한된 GPU 메모리를 최대한 활용하기 위해 **워크로드 패턴을 분석하여 GPU에 캐싱할 벡터를 동적으로 결정**.

### 3. CUDA Multi-stream 최적화

쿼리 처리와 데이터 업데이트를 CUDA 멀티스트림으로 동시 실행하여 간섭 최소화.

### 4. 적응형 리소스 관리 + 동시성 제어

- 쿼리와 업데이트가 interleave될 때 데이터 일관성 보장
- CPU/GPU 리소스를 워크로드에 맞게 동적으로 조정

---

## 성능 결과 (대규모 스트리밍 워크로드 기준)

| 지표 | 향상 |
|---|---|
| 처리량 (throughput) | 평균 20.9x 향상 |
| 지연 시간 (latency) | 1.3x ~ 50.7x 감소 |

높은 recall 유지하면서 달성.
