# GPU-accelerated Database

## CPU architectures

### GPU Architecture and Memory Hierarchy
Let’s begin with the GPU architecture. It consists of a massive number of GPU cores, L2 cache, and Global Memory (GMEM). The physical computing units are organized into a hierarchy: GPCs, SMs, Processing Blocks, and individual GPU Cores. This hardware structure corresponds one-to-one with the logical hierarchy of Thread Block Clusters, Thread Blocks, Warps, and Threads.
To support these units, the GPU provides a hierarchical memory system including registers, shared memory, and distributed shared memory. Specifically, threads in a warp can exchange data using warp-level primitives, while threads within a thread block share data via shared memory. Furthermore, threads across different blocks in a cluster can now share data through distributed shared memory.

### Volta Architecture
In the Volta architecture, several new concepts were introduced. First, Volta introduces independent thread scheduling and cooperative groups with a per-thread program counter and stack. These concepts enable finer-grained concurrent execution. For example, when threads in a warp wait for data loading, other threads in the same warp can execute other instructions. In previous architectures, these threads were required to wait for the data loading.
Second, NVLink and Unified Memory were introduced. NVLink serves as a high-bandwidth interconnect between GPUs and the CPU, overcoming the throughput bottlenecks of PCIe. Unified Memory allows for a single unified virtual address space for CPU and GPU memory. With Unified Memory, programmers no longer need to manually manage data sharing between CPU and GPU memory systems, although manual management remains an option for high performance.

### Ampere and Hopper Architectures
In the Ampere architecture, NVIDIA introduced a new asynchronous copy instruction that loads data directly from global memory into SM shared memory.
In the Hopper architecture, thread block clusters and distributed shared memory were introduced. These new computing units and the efficient data-sharing method enable applications to accelerate by avoiding memory accesses to global memory.
The Tensor Memory Accelerator (TMA) is another feature. TMA improves data fetch efficiency by transferring large blocks of data and multi-dimensional tensors between global and shared memory. By handling address generation and data movement in hardware, TMA reduces addressing overhead and frees threads to execute other independent work. Additionally, DPX instructions allow for the efficient processing of core operations, such as addition and finding the maximum, in dynamic programming implementations.

#### Blackwell Architecture
Finally, in the Blackwell architecture, the decompression engine is a key feature. This engine enables a reduction in the cost of data movement between the CPU and GPUs.

## GPU architecture white papers
- Ampere: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- Hopper: https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf
- Blackwell: https://resources.nvidia.com/en-us-blackwell-architecture


## Papers

> **공부 완료 체크 방법**: 논문 내용을 Claude와 충분히 토론하고 이해했으면 `- [ ]`를 `- [x]`로 변경

### VLDB

| 완료 | 논문 | 정리 노트 |
|:---:|---|:---:|
| [ ] | [GpJSON: High-performance JSON Data Processing on GPUs (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p3216-bonetta.pdf) | [📄](papers/gpjson_high_performance_json_data_processing_on_gpus.md) |
| [ ] | [GRAPIN: Efficient Graph Data Access for Out-of-Memory GPU Streaming Graph Processing (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p3854-wang.pdf) | [📄](papers/grapin_efficient_graph_data_access_for_out_of_memory_gpu_streaming_graph_processing.md) |
| [ ] | [Powerful GPUs or Fast Interconnects: Analyzing Relational Workloads on Modern GPU (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p4350-kabic.pdf) | [📄](papers/powerful_gpus_or_fast_interconnects_analyzing_relational_workloads_on_modern_gpu.md) |
| [ ] | [Scaling GPU-Accelerated Databases beyond GPU Memory Size (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p4518-li.pdf) | [📄](papers/scaling_gpu_accelerated_databases_beyond_gpu_memory_size.md) |
| [ ] | [Efficiently Joining Large Relations on Multi-GPU Systems (VLDB 2025)](https://www.vldb.org/pvldb/vol18/p4653-maltenberger.pdf) | [📄](papers/efficiently_joining_large_relations_on_multi_gpu_systems.md) |
| [x] | [GPU Acceleration of SQL Analytics on Compressed Data (VLDB 2025)](https://doi.org/10.14778/3778092.3778095) | [📄](papers/gpu_acceleration_of_sql_analytics_on_compressed_data.md) |
| [ ] | [Vortex: Overcoming Memory Capacity Limitations in GPU-Accelerated Data Analytics (VLDB 2024)](https://doi.org/10.14778/3717755.3717780) | [📄](papers/vortex_overcoming_memory_capacity_limitations_in_gpu_accelerated_data_analytics.md) |
| [ ] | [SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search (VLDB 2026)](https://doi.org/10.14778/3796195.3796216) | [📄](papers/svfusion_cpu_gpu_co_processing_for_large_scale_real_time_vector_search.md) |
| [ ] | [Everest: GPU-Accelerated System For Mining Temporal Motifs (VLDB 2024)](https://www.vldb.org/pvldb/vol17/p162-yuan.pdf) | - |
| [ ] | [GPU Database Systems Characterization and Optimization (VLDB 2024)](https://www.vldb.org/pvldb/vol17/p441-cao.pdf) | - |
| [ ] | [RTIndeX: Exploiting Hardware-Accelerated GPU Raytracing for Database Indexing](https://arxiv.org/pdf/2303.01139) | - |

### SIGMOD

| 완료 | 논문 | 정리 노트 |
|:---:|---|:---:|
| [ ] | [Efficiently Processing Joins and Grouped Aggregations on GPUs (SIGMOD 2025)](https://dl.acm.org/doi/pdf/10.1145/3709689) | [📄](papers/efficiently_processing_joins_and_grouped_aggregations_on_gpus.md) |
| [ ] | [GPH: An Efficient and Effective Perfect Hashing Scheme for GPU Architectures (SIGMOD 2025)](https://dl.acm.org/doi/pdf/10.1145/3725406) | [📄](papers/gph_efficient_effective_perfect_hashing_scheme_for_gpu_architectures.md) |
| [ ] | [GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP (SIGMOD 2024)](https://doi.org/10.1145/3698812) | [📄](papers/golap_gpu_in_data_path_architecture_for_high_speed_olap.md) |
| [ ] | [cuMatch: GPU-based worst-case optimal join for subgraph queries (SIGMOD 2025)](https://dl.acm.org/doi/pdf/10.1145/3725398) | - |
| [ ] | [GPU-Accelerated Graph Cleaning with a Single Machine (SIGMOD 2025)](https://dl.acm.org/doi/pdf/10.1145/3725303) | - |
| [ ] | [Efficient GPU-Accelerated Subgraph Matching (SIGMOD 2023)](https://dl.acm.org/doi/pdf/10.1145/3589326) | - |
| [ ] | [Distributed GPU Joins on Fast RDMA-capable Networks (SIGMOD 2023)](https://dl.acm.org/doi/pdf/10.1145/3588709) | - |
| [ ] | [Triton Join: Efficiently Scaling to a Large Join State on GPUs with Fast Interconnects (SIGMOD 2022)](https://dl.acm.org/doi/pdf/10.1145/3514221.3517911) | - |
| [ ] | [Evaluating Multi-GPU Sorting with Modern Interconnects (SIGMOD 2022)](http://dl.acm.org/doi/pdf/10.1145/3514221.3517842) | - |
| [ ] | [GaccO - A GPU-accelerated OLTP DBMS (SIGMOD 2022)](https://dl.acm.org/doi/pdf/10.1145/3514221.3517876) | - |
| [ ] | [Tile-based Lightweight Integer Compression in GPU (SIGMOD 2022)](https://dl.acm.org/doi/pdf/10.1145/3514221.3526132) | - |
| [ ] | [Graph Processing on GPUs: A Survey](https://dl.acm.org/doi/pdf/10.1145/3128571) | - |


### Scan

### Join

### Group-by
