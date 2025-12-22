# 👋 Kijae Hong (홍기재)

📧 [E-mail](kijaehong1021@gmail.com) | 🐙 [Github](https://github.com/kijaehong) | 💼 [LinkedIn](https://www.linkedin.com/in/ki-jae-hong-5643b730a/) | 📚 [Google Scholar](https://scholar.google.com/citations?user=QHGq7GIAAAAJ&hl=ko) | 📖 [DBLP](https://dblp.org/pid/266/5820.html)

---

## Introduction

I am a Researcher at Ceres Technologies and a Ph.D. graduate in Computer Science from POSTECH, specializing in the convergence of Database systems and Artificial Intelligence. My core expertise lies in GPU acceleration, Graph Data Analysis, and LLM serving optimization.

Currently, I focus on building a high-frequency-trading system, building efficient RAG systems, and optimizing LLM inference (vLLM, caching, scheduling). With a strong academic foundation demonstrated by publications in top-tier conferences like VLDB and SIGMOD, and a portfolio of patents in query processing and AI, I am dedicated to developing scalable, high-speed architectures for next-generation data processing.

---

## 🔬 Ongoing Projects

### [Maximizing LLM Caching](https://github.com/kijaehong1021/LLMCachingBoost)

With the rapid advancement of LLMs, there are emerging attempts to expand the scope of data analysis in databases by leveraging LLMs. I'm conducting research on optimizing LLM caching for this purpose.

### [Optimizing Semantic Operators in a Query Plan](https://github.com/kijaehong1021/SemanticOperatorOptimizer)

To broaden the analytical capabilities of databases, recent research is incorporating LLMs into database operators to 1) check specific conditions or 2) extract new attributes from records. However, since LLM inference is a highly costly operation, finding ways to minimize this overhead is essential, and this is a topic I am currently exploring with great interest.

---

## 💼 Previous Projects

<!--
### T-R3X
TBD

### RAG-based Customer Support System for 현대홈쇼핑
TBD

### Knowledge Management System
TBD -->

### SLM-based Operator of RAG

- Developed Small Language Model (SLM)-based operators to execute core RAG system functions with low cost and latency
- Deployed vLLM-based inference services on Kubernetes and implemented auto-scaling based on workload fluctuations
- Project supported by the Ministry of SMEs and Startups
- **Tech Stack**: vLLM, Kubernetes, LLM Fine-tuning

### High Frequency Trading

- Developed a system for collecting tens of thousands of market events per second and performing real-time price prediction and automated trading
- Designed a distributed and parallel processing system to handle large-scale traffic with stability
- Successfully sold to and operated by multiple clients
- Project supported by the Ministry of SMEs and Startups
- **Tech Stack**: Distributed/Parallel Processing System, Real-time Data Processing, C++, WebSocket, CPython

### [GPU-accelerated Relational Query Execution Engine](https://www.vldb.org/pvldb/vol18/p426-han.pdf)

- Conducted research on accelerating relational data analysis queries using GPUs
- Resolved load imbalance issues (inter-warp and intra-warp thread divergence) during parallel processing
- Achieved query processing performance approximately 379 times faster than competing technologies
- Published in VLDB 2025
- **Tech Stack**: GPU, CUDA, Parallel Processing Optimization

### [QaaD (query-as-a-data)](https://dl.acm.org/doi/abs/10.1145/3589279)

- Developed a system leveraging Apache Spark to efficiently process massive volumes of small queries
- Proposed a reverse strategy: merging numerous small queries into a single large-scale query for batch processing (opposite of traditional Spark approach)
- Deeply involved in implementing the core logic
- Gained valuable experience in designing large-scale distributed processing systems
- **Tech Stack**: Apache Spark, Distributed Processing System

### Product Search Engine

- Developed a distributed crawling framework that periodically collects product information from global e-commerce platforms and normalizes it into user-desired formats
- Addressed challenges such as anti-crawling mechanisms and designed large-scale crawling architectures
- Acquired background knowledge in Entity Matching technology to link data from disparate sources
- **Tech Stack**: Distributed Crawling Framework, Entity Matching

### [iturbograph](https://dl.acm.org/doi/abs/10.1145/3448016.3457243)

- Participated in developing a distributed and parallel processing system that supports incremental updates for the results of massive graph analysis queries (e.g., PageRank, SCC, WCC, SSSP)
- Conducted experiments and analyzed competing systems
- Published in SIGMOD 2021
- **Tech Stack**: Rust, MPI, Distributed/Parallel Processing

### [G-CARE](https://www.researchgate.net/profile/Sourav-S-Bhowmick/publication/341750604_G-CARE_A_Framework_for_Performance_Benchmarking_of_Cardinality_Estimation_Techniques_for_Subgraph_Matching/links/5ee45f61a6fdcc73be780998/G-CARE-A-Framework-for-Performance-Benchmarking-of-Cardinality-Estimation-Techniques-for-Subgraph-Matching.pdf)

- Participated in a project proposing a benchmark for cardinality estimation techniques in subgraph matching and analyzing SOTA methodologies
- Extended the query optimizer of RDF-3X to support SOTA methods as plug-in modules
- Gained deep insights into query optimization and cardinality estimation
- Published in SIGMOD 2020
- **Tech Stack**: RDF-3X, Query Optimizer, Cardinality Estimation

<!-- ### SIMD-based B+-tree
TBD -->

---

## 🎓 Education

- **Ph.D.** in Computer Science and Engineering, POSTECH, South Korea, (2018.2-2025.8)  
  Advisor: Wook-Shin Han
- **B.S.** in Industrial and Management Engineering and Computer Science and Engineering, POSTECH, South Korea, (2011.3-2018.2)

---

## 💼 Employment

- **Researcher**, Ceres Technologies, Republic of Korea, (2025.1 - Present)
  - High frequency trading
  - LLM serving (vLLM, Kubernetes, etc.), LLM fine-tuning
  - LLM inference optimization (caching, scheduling)
  - RAG system
- **Intern**, Exem, Republic of Korea, (2016. 6 - 2016. 8)

---

## 📚 Publications

### 2025

**The Effect of Scheduling and Preemption on the Efficiency of LLM Inference Serving**  
_Kyoungmin Kim, Jiacheng Li, **Kijae Hong**, and Anastasia Ailamaki_  
arXiv:2411.07447, 2025 (Under Review at VLDB 2026)

**Large Language Models for Semantic Join: A Comprehensive Survey**  
**_Kijae Hong_** _and Yeonsu Park_  
_IEEE Access_, 2025

**Design and Evaluation of a GPU-Accelerated SQL Query Engine for Large-Scale Analytics**  
**_Kijae Hong_**  
Ph.D. Dissertation, POSTECH, 2025

**[Themis: A GPU-accelerated Relational Query Execution Engine](https://www.vldb.org/pvldb/vol18/p426-han.pdf)**  
**_Kijae Hong_**, _Kyoungmin Kim, Young-Koo Lee, Yang-Sae Moon, Sourav S Bhowmick, and Wook-Shin Han_  
_Very Large Data Bases Conference (VLDB)_, 2025 ⭐

### 2022

**Survey on the GPU-Based Graph Analytics Methods**  
**_Kijae Hong_**, _Jinho Ko, Taesung Lee, and Wook-Shin Han_  
_Korea Computer Congress (KCC)_, 2022

**Performance Analysis of Property Graph Queries on Graph and Relational Databases**  
_Jinho Ko, Taesung Lee, **Kijae Hong**, and Wook-Shin Han_  
_Korea Computer Congress (KCC)_, 2022

### 2021

**[iTurboGraph: Scaling and Automating Incremental Graph Analytics](https://dl.acm.org/doi/abs/10.1145/3448016.3457243)**  
_Seongyun Ko, Taesung Lee, **Kijae Hong**, Wonseok Lee, In Seo, Jiwon Seo, and Wook-Shin Han_  
_ACM International Conference on Management of Data (SIGMOD)_, 2021 ⭐

**A Study on Property Graph Partitioning for Graph Analytic Query Processing in Distributed Environment**  
_Jinho Ko, **Kijae Hong**, Taesung Lee, Jeong-Hoon Lee, and Wook-Shin Han_  
_Korea Computer Congress (KCC)_, 2021

**Graph Analytics Query Acceleration Using Filtering Techniques in Page-Based Dynamic Graph Storage**  
_Taesung Lee, **Kijae Hong**, Jeong-Hoon Lee, and Wook-Shin Han_  
_Korea Computer Congress (KCC)_, 2021

### 2020

**[G-CARE: A Framework for Performance Benchmarking of Cardinality Estimation Techniques for Subgraph Matching](https://www.researchgate.net/profile/Sourav-S-Bhowmick/publication/341750604_G-CARE_A_Framework_for_Performance_Benchmarking_of_Cardinality_Estimation_Techniques_for_Subgraph_Matching/links/5ee45f61a6fdcc73be780998/G-CARE-A-Framework-for-Performance-Benchmarking-of-Cardinality-Estimation-Techniques-for-Subgraph-Matching.pdf)**  
_Yeonsu Park, Seongyun Ko, Sourav S. Bhowmick, Kyoungmin Kim, **Kijae Hong**, and Wook-Shin Han_  
_ACM International Conference on Management of Data (SIGMOD)_, 2020 ⭐

### 2018

**A Survey on Top-down and Bottom-up Inductive Logic Programming Methods**  
_In Seo, Jeong-Hoon Lee, Kyoungmin Kim, **Kijae Hong**, Byunghoon So, and Wook-Shin Han_  
_Korea Computer Congress (KCC)_, 2018

---

### 🔖 Patents

**GPU-Based Query Processing Acceleration Method and Computing System**  
_Wook-Shin Han, **Kijae Hong**, Taesung Lee, and Kyoungmin Kim_  
U.S. Patent Application No. 19/144,304 (Filed: Jul. 2025)

**METHOD FOR GENERATING CONTENTS BASED ON TEMPLATE**  
**_Kijae Hong_**  
Korean Patent Registration No. 10-2025-0068845 (Filed: May. 2025)

**METHOD FOR OPTIMIZING INFERENCE OF LANGUAGE MODEL**  
**_Kijae Hong_**  
Korean Patent Registration No. 10-2025-0066840 (Filed: May. 2025)

**Method for Summarizing Document Using Large Language Model**  
_Inhyeok Na, **Kijae Hong**, and Jaehyun Lim_  
Korean Patent Registration No. 10-2025-0084553 (Published: Jun. 2025)

**Method for Retrieving Document Related to Natural Language Query**  
_Inhyeok Na, **Kijae Hong**, Jaehyun Lim, and Haechan Lee_  
Korean Patent Registration No. 10-2815043-0000 (Registered: May. 2025)

**Method and Computing System for Acceleration of Processing Queries Based on GPU**  
\*Wook-Shin Han, Taesung Lee, Kyungmin Kim, and **Kijae Hong\***  
Korean Patent Registration No. 10-2649076-0000 (Registered: Mar. 2024)

**Distributed Processing System and Method for Processing Data**  
\*Wook-Shin Han, Yeonsu Park, and **Kijae Hong\***  
Korean Patent Registration No. 10-2024-0026045 (Published: Feb. 2024)

**A Method for Mapping a Natural Language Sentence to an SQL Query**  
\*Wook-Shin Han, Hyunji Kim, Jungho Jo, Yukyung Lee, and **Kijae Hong\***  
Korean Patent Registration No. 10-2149701-0000 (Registered: Aug. 2020)

---

## 🎤 Academic Talks

- Vector-Based Search for Semantic Join, Kangwon National University, (Nov. 2025)
- Retrieval-Augmented Generation and Vector Databases, Chung-ang University, (Nov. 2024)

---

## 👨‍🏫 Teaching Experience

- Big Data Course, Samsung, (2018-2019) - TA
- Database Course, POSTECH, (2018-2019) - TA

---

## 🛠️ Skills

- **Programming languages**: C++, Python, Java, JavaScript, CUDA
- **Development frameworks**: Flutter, React, Django
- **Infra**: Kubernetes, Firebase, AWS
- **DB+AI**: vLLM, LLMCache, etc.

---

## 🔗 Other Links

- Wook-Shin Han, Professor, POSTECH - wshan@dblab.postech.ac.kr
- Yang-Sae Moon, Professor, Kangwon National University - ysmoon@kangwon.ac.kr
- Sourav S. Bhowmick, Associate Professor, Nanyang Technological University - assourav@ntu.edu.sg
- Young-Koo Lee, Professor, Kyung Hee University - yklee@khu.ac.kr
