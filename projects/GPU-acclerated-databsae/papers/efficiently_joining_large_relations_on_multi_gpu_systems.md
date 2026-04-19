# Efficiently Joining Large Relations on Multi-GPU Systems

- **저자**: Tobias Maltenberger, Ilin Tolovski, Tilmann Rabl (Hasso Plattner Institute)
- **학회**: VLDB 2025
- **링크**: https://doi.org/10.14778/3749646.3749720

---

## 문제 정의

대용량 데이터의 relational join을 처리할 때 기존 멀티-GPU 알고리즘의 두 가지 한계:

1. **P2P 인터커넥트 미활용**: GPU 간 고속 NVLink/NVSwitch를 활용하지 못하고 PCIe 경유
2. **Out-of-core 데이터 처리 불가**: GPU 메모리를 초과하는 대용량 데이터를 네이티브하게 처리하지 못함

---

## 핵심 아이디어: Heterogeneous Multi-GPU Sort-Merge Join

세 단계로 구성된 하이브리드 알고리즘:

### Phase 1 — Multi-GPU Sort (P2P 활용)
- Merge-based 또는 Radix partitioning 기반 정렬
- GPU 간 **P2P 인터커넥트(NVLink/NVSwitch)를 직접 활용**하여 데이터 이동

### Phase 2 — CPU-based Multiway Merge
- 정렬된 파티션들을 CPU에서 병렬 멀티웨이 머지
- GPU→CPU 데이터 이동과 CPU 연산을 오버랩

### Phase 3 — Hybrid Join
- CPU merge path partitioning + binary search 기반 멀티-GPU join 전략 결합
- Copy와 compute 연산을 오버랩하여 전송 비용 감소 (전체 런타임의 22%만 차지)

---

## 성능 결과

| 비교 대상 | 성능 향상 |
|---|---|
| 병렬 CPU sort-merge join | 최대 15.2x |
| 병렬 CPU radix-hash join | 최대 5.5x |
| non-P2P 멀티-GPU sort-merge | 8.7x |
| non-P2P 멀티-GPU hybrid-radix | 2.5x |
| 입력 pre-sorted 시 hybrid-radix 대비 | 최대 14.4x |

- GPU 수에 따라 선형적으로 스케일
- 데이터 skew에 강인 (최대 12% 더 빠름)
- 플랫폼: NVLink 및 NVSwitch 기반 P2P 인터커넥트
