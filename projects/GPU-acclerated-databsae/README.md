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

### VLDB
- [GpJSON: High-performance JSON Data Processing on GPUs (VLDB)](https://www.vldb.org/pvldb/vol18/p3216-bonetta.pdf)
- [Efficient Graph Data Access for Out-of-Memory GPU Streaming Graph Processing](https://www.vldb.org/pvldb/vol18/p3854-wang.pdf)
- [Powerful GPUs or Fast Interconnects: Analyzing Relational Workloads on Modern GPU](https://www.vldb.org/pvldb/vol18/p4350-kabic.pdf)
- [Scaling GPU-Accelerated Databases beyond GPU Memory Size](https://www.vldb.org/pvldb/vol18/p4518-li.pdf)
- [Efficiently Joining Large Relations on Multi-GPU Systems](https://www.vldb.org/pvldb/vol18/p4653-maltenberger.pdf)
- [Everest: GPU-Accelerated System For Mining Temporal Motifs](https://www.vldb.org/pvldb/vol17/p162-yuan.pdf)
- [GPU Database Systems Characterization and Optimization](https://www.vldb.org/pvldb/vol17/p441-cao.pdf)
- [RTIndeX: Exploiting Hardware-Accelerated GPU Raytracing for Database Indexing](https://arxiv.org/pdf/2303.01139)

### SIGMOD
- [cuMatch: A GPU-based memory-efficient worst-case optimal join processing method for subgraph queries with complex patterns](https://dl.acm.org/doi/pdf/10.1145/3725398)
- [GPU-Accelerated Graph Cleaning with a Single Machine](https://dl.acm.org/doi/pdf/10.1145/3725303)
- [GPH: An Efficient and Effective Perfect Hashing Scheme for GPU Architectures](https://dl.acm.org/doi/pdf/10.1145/3725406)
- [Efficient GPU-Accelerated Subgraph Matching](https://dl.acm.org/doi/pdf/10.1145/3589326)
- [Distributed GPU Joins on Fast RDMA-capable Networks](https://dl.acm.org/doi/pdf/10.1145/3588709)
- [Triton Join: Efficiently Scaling to a Large Join State on GPUs with Fast Interconnects](https://dl.acm.org/doi/pdf/10.1145/3514221.3517911)
- [Evaluating Multi-GPU Sorting with Modern Interconnects](http://dl.acm.org/doi/pdf/10.1145/3514221.3517842)
- [GaccO - A GPU-accelerated OLTP DBMS](https://dl.acm.org/doi/pdf/10.1145/3514221.3517876)
- [Tile-based Lightweight Integer Compression in GPU](https://dl.acm.org/doi/pdf/10.1145/3514221.3526132)
- [Efficiently Processing Joins and Grouped Aggregations on GPUs](https://dl.acm.org/doi/pdf/10.1145/3709689)
- [Graph Processing on GPUs: A Survey](https://dl.acm.org/doi/pdf/10.1145/3128571)

### arxiv
- [GPU Acceleration of SQL Analytics on Compressed Data](https://arxiv.org/pdf/2506.10092)


### Scan

### Join

### Group-by 
