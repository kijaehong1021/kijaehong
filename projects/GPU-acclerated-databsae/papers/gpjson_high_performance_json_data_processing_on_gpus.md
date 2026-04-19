# GpJSON: High-performance JSON Data Processing on GPUs

- **저자**: Sacheendra Talluri, Guido Walter Di Donato, Luca Danelutti 외 (VU Amsterdam, PoliMi, Oracle Labs)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3746405.3746439

---

## 문제 정의

JSON은 가장 널리 쓰이는 데이터 포맷이지만 매우 비효율적인 형식:
- **역직렬화(파싱) 비용**: 텍스트 기반 구조 파싱이 CPU에서 주요 병목
- **쿼리 비용**: JSON 데이터를 in-situ로 쿼리하는 고성능 엔진 부재
- 기존 병렬 JSON 파서들도 데이터 볼륨 증가에 따른 성능 한계 존재

---

## 핵심 아이디어

### 1. GPU 기반 병렬 구조 인덱스 구축

JSON 파싱을 GPU에서 **병렬 구조 인덱스(structural index) 구축**으로 구현.
- 텍스트를 순차적으로 파싱하는 대신, GPU의 대규모 병렬성을 이용해 JSON 구조를 동시에 분석
- 인덱스 구축 후 데이터에 직접 접근 가능

### 2. GPU용 경량 쿼리 엔진

구축된 구조 인덱스를 활용해 **JSON 데이터를 in-situ로 쿼리**하는 GPU 쿼리 엔진 설계.
- 역직렬화 없이 원본 JSON에서 직접 쿼리 처리
- GPU 메모리 접근 패턴을 최적화한 경량 설계

### 3. 고수준 언어 바인딩

JavaScript, Python 등에서 사용 가능하도록 GraalVM 런타임 바인딩 제공.

---

## 성능 결과

하드웨어: NVIDIA Ampere A100 단일 GPU

| 비교 대상 | 향상 (end-to-end) |
|---|---|
| 최신 병렬 JSON 파서/쿼리 엔진 | 최소 2.9x |
| NVIDIA RAPIDS | 6~8x |
