# GPU-accelerated Database


## GPU architecture white papers
- Ampere: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- Hopper: https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf
- Blackwell: https://resources.nvidia.com/en-us-blackwell-architecture


### Ampere
- Asynchronous Copy
- Asynchronous Barrier
- Task Graph Acceleration
- Multi-instance GPU

### Hopper
- DPX instructions for dynamic programming algorithms
- Thread block cluster and distributed shared memory
- Tensor Memory Accelerator (TMA, shared meomry <-> global memory)

### Blackwell
- Decompression Engine
- Reliability, Availability, and Serviceability (RAS) Engine

## Papers

### Scan

### Join

### Group-by 


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