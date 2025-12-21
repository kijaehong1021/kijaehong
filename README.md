# 👋 Kijae Hong (홍기재)

## Introduction

I am a Researcher at Ceres Technologies and a Ph.D. graduate in Computer Science from POSTECH, specializing in the convergence of Database systems and Artificial Intelligence. My core expertise lies in GPU acceleration, Graph Data Analysis, and LLM serving optimization.

포항공대(POSTECH) 컴퓨터공학과 박사 출신으로, 현재 세레스테크놀로지스(Ceres Technologies)에서 **데이터베이스와 AI 기술의 결합(DB+AI)** 을 연구하고 있습니다. 주력 분야는 GPU 가속 기반의 데이터 처리, 그래프 데이터 분석, 그리고 LLM 서빙 및 추론 최적화입니다.

Currently, I focus on building a high-frequency-trading system, building efficient RAG systems, and optimizing LLM inference (vLLM, caching, scheduling). With a strong academic foundation demonstrated by publications in top-tier conferences like VLDB and SIGMOD, and a portfolio of patents in query processing and AI, I am dedicated to developing scalable, high-speed architectures for next-generation data processing.


현재는 초고빈도 매매(High-Frequency Trading) 시스템 개발, vLLM 및 쿠버네티스(Kubernetes)를 활용한 고성능 RAG 시스템 구축과 LLM 미세조정(Fine-tuning) 업무를 수행하고 있습니다. VLDB, SIGMOD 등 세계 최고 수준의 학회에 다수의 논문을 게재하고 관련 특허를 보유하는 등, 이론적 깊이와 실무적 구현 능력을 바탕으로 대규모 데이터 처리 시스템의 혁신을 추구합니다.

---
## 📧 Contact

- Email: kijaehong1021@gmail.com
- GitHub: [@kijaehong](https://github.com/kijaehong)
- LinkedIn: [@Ki-jae Hong](https://www.linkedin.com/in/ki-jae-hong-5643b730a/)
- Location: Seoul, South Korea
- Others: [Google Scholar](https://scholar.google.com/citations?user=QHGq7GIAAAAJ&hl=ko), [DBLP](https://dblp.org/pid/266/5820.html)

---
## 🔬 Ongoing Projects

### [Maximizing LLM Caching](https://github.com/kijaehong1021/LLMCachingBoost)

With the rapid advancement of LLMs, there are emerging attempts to expand the scope of data analysis in databases by leveraging LLMs. I'm conducting research on optimizing LLM caching for this purpose.

LLM의 엄청난 발전 속도에 따라, LLM을 활용하여 데이터베이스의 데이터 분석 가능 범위를 더 넓히려는 시도들이 생겨나고 있습니다. 그리고 저는 이러한 환경에서, LLM 캐싱 테크닉들의 효과를 극대화하기 위해 어떤 방법이 있을지 탐구해보고 있습니다.

### [Optimizing Semantic Operators in a Query Plan](https://github.com/kijaehong1021/SemanticOptimizerOptimizer)

To broaden the analytical capabilities of databases, recent research is incorporating LLMs into database operators to 1) check specific conditions or 2) extract new attributes from records. However, since LLM inference is a highly costly operation, finding ways to minimize this overhead is essential, and this is a topic I am currently exploring with great interest.

LLM을 활용해서 데이터베이스의 분석 가능 범위를 넓히기 위해, 최근 연구들은 database의 연산자들이 LLM inference를 통해 record에 대한 1) condition을 체크하거나, 2)새로운 attribute를 추출할 수 있도록 확장하고 있습니다. 하지만, LLM inference는 매우 비싼 연산이기 때문에 이를 최대한 줄이는 방안이 필요하며, 이는 제가 요즘 흥미롭게 탐구하고 있는 주제입니다.

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

Amid the surging demand for RAG systems, I developed an SLM-based operator designed to execute core system functions with low latency and cost efficiency, a project supported by the Ministry of SMEs and Startups. I established a robust operational infrastructure by deploying vLLM-based inference services on Kubernetes and implementing auto-scaling tailored to workload fluctuations. Through this process, I acquired in-depth expertise in LLM fine-tuning techniques as well as serving infrastructure utilizing vLLM and Kubernetes.

RAG 시스템의 수요가 급증함에 따라, 시스템의 핵심 기능을 저비용·저지연으로 수행할 수 있는 Small Language Model(SLM) 기반의 operator들을 중소벤처기업부 지원하에 개발했습니다. 또한, vLLM을 활용한 추론 서비스를 쿠버네티스 환경에 구축하고 워크로드에 따른 오토 스케일링을 구현하는 등 안정적인 운영 인프라를 마련했습니다. 이 과정을 통해 LLM 파인튜닝 기술은 물론, vLLM 및 쿠버네티스를 활용한 서빙 인프라 구축에 대한 심도 있는 역량을 확보했습니다. 


### High Frequency Trading

At Ceres Technologies, I developed a distributed and parallel processing system capable of collecting tens of thousands of market events per second to perform real-time price prediction and automated trading. This project, supported by the Ministry of SMEs and Startups, was engineered to handle high-volume traffic with stability and has been successfully deployed and operated for multiple client companies.

세레스 테크놀로지스에서 초당 수만 건에 달하는 시장 이벤트를 수집하고, 실시간 가격 예측 및 매매를 자동으로 수행하는 분산/병렬 처리 시스템을 중소기업벤처부 지원하에 개발했습니다. 대규모 트래픽을 안정적으로 처리하도록 설계된 이 시스템은 다수의 고객사에 성공적으로 도입되어 운용되고 있습니다.


### [GPU-accelerated Relational Query Execution Engine](https://www.vldb.org/pvldb/vol18/p426-han.pdf)

During my doctoral studies, I conducted research on accelerating relational data analysis queries using GPUs, a distinct achievement that led to a publication in VLDB, a top-tier database conference. This research focused on resolving load imbalance issues—specifically inter-warp and intra-warp thread divergence—during parallel processing, ultimately achieving a query processing performance approximately 379 times faster than competing technologies. Through this research, I gained a profound understanding of GPU architecture and mastered various optimization techniques within the CUDA environment.

박사 과정 중 GPU를 기반으로 관계형 데이터 분석 질의를 가속화하는 연구를 수행하였으며, 그 성과를 인정받아 데이터베이스 분야 최고 권위 학회인 VLDB에 논문을 게재하였습니다. 본 연구는 병렬 처리 과정에서 발생하는 워프 간, 그리고 워프 내 스레드 간의 부하 불균형(Load Imbalance) 문제를 해결하는 데 집중하였으며, 이를 통해 경쟁 기술 대비 약 379배 빠른 질의 처리 성능을 달성하는 데 성공했습니다. 이 과정을 통해 GPU 아키텍처에 대한 깊은 이해를 갖추게 되었으며, CUDA 환경에서의 다양한 최적화 기법을 체득했습니다.


### [QaaD (query-as-a-data)](https://dl.acm.org/doi/abs/10.1145/3589279)

I participated in the development of a system leveraging Apache Spark to efficiently process massive volumes of small queries. Diverging from the traditional Spark approach of splitting a single query into multiple sub-queries, I proposed and implemented a reverse strategy that merges numerous small queries into a single large-scale query for batch processing. Although I concluded my involvement prior to the paper publication to focus on GPU acceleration research, I was deeply involved in implementing the core logic and gained valuable experience in designing large-scale distributed processing systems.

Apache Spark를 활용하여 대량의 소규모 질의(Small Queries)를 효율적으로 처리하는 시스템 개발에 참여했습니다. 기존 Spark가 하나의 질의를 여러 하위 작업(Sub-query)으로 나누어 병렬 처리하는 것과 달리, 본 프로젝트에서는 역발상으로 수많은 소규모 질의를 하나의 거대 질의로 병합하여 Spark에서 일괄 처리하는 기법을 제안했습니다. 본 프로젝트의 핵심 로직 구현에 깊이 관여했으나, 이후 GPU 가속 연구에 집중하기 위해 논문 집필 단계 이전에 프로젝트를 마무리하여 저자 목록에는 포함되지 않았습니다. 하지만 대규모 분산 처리 시스템을 설계하는 귀중한 경험을 쌓았습니다.

### Product Search Engine

I developed a distributed crawling framework that periodically collects product information from global e-commerce platforms and normalizes it into user-desired formats. Through this project, I gained practical experience in addressing challenges such as anti-crawling mechanisms and designing large-scale crawling architectures, while also acquiring background knowledge in Entity Matching technology to link data from disparate sources. I possess the technical insight that integrating modern LLM technologies into this workflow could have significantly enhanced data transformation and matching efficiency.

글로벌 전자상거래 업체들의 상품 정보를 주기적으로 수집하고, 이를 사용자가 원하는 데이터 포맷으로 정규화하여 저장하는 분산 크롤링 프레임워크를 개발했습니다. 이 과정에서 크롤링 방지(Anti-crawling) 대응, 대규모 크롤링 아키텍처 설계 등 실무적인 이슈들을 해결하며 풍부한 경험을 쌓았습니다. 또한, 서로 다른 출처의 데이터를 연결하는 Entity Matching 기술에 대한 배경 지식도 습득했습니다. 최근의 발전된 LLM 기술을 당시 프로젝트에 접목했다면 데이터 변환 및 매칭 효율을 획기적으로 높일 수 있었을 것이라는 기술적 인사이트를 가지고 있습니다.

### [iturbograph](https://dl.acm.org/doi/abs/10.1145/3448016.3457243)

I participated in the development of a distributed and parallel processing system that supports incremental updates for the results of massive graph analysis queries (e.g., PageRank, SCC, WCC, SSSP). My primary role involved conducting experiments and analyzing competing systems, which allowed me to build extensive experience in distributed and parallel computing environments utilizing technologies such as Rust and MPI.

초거대 그래프 분석 질의(i.e., PageRank, SCC, WCC, SSSP) 결과에 대한 점진적으로 업데이트를 지원하는 분산/병렬 처리 시스템 개발에 참여하였습니다. 경쟁 시스템에 대한 실험 및 분석을 맡았으며 이를 통해 분산/병렬 환경(e.g., Rust, MPI)에 대한 경험을 쌓을 수 있었습니다.

### [G-CARE](https://www.researchgate.net/profile/Sourav-S-Bhowmick/publication/341750604_G-CARE_A_Framework_for_Performance_Benchmarking_of_Cardinality_Estimation_Techniques_for_Subgraph_Matching/links/5ee45f61a6fdcc73be780998/G-CARE-A-Framework-for-Performance-Benchmarking-of-Cardinality-Estimation-Techniques-for-Subgraph-Matching.pdf)

"I participated in a project proposing a benchmark for cardinality estimation techniques in subgraph matching and analyzing state-of-the-art (SOTA) methodologies. My main contribution was extending the query optimizer of RDF-3X, an RDF database, to support SOTA methods as plug-in modules. This experience provided me with deep insights into the fields of query optimization and cardinality estimation.

그래프 분석 질의 중 하나인 서브그래프 매칭 결과의 수를 예측하는 방법론들을 위한 벤치마크를 제안하고 당시 SOTA 방법들을 분석한 프로젝트에 참가하였습니다. 주요 업무는 RDF 데이터베이스 중 하나인 RDF-3X의 query optimizer를 확장하여 SOTA 방법들을 플러그인 형태로 사용할 수 있게 하는 것이었으며, 이를 통해 query optimizer와 cardinality estmation 분야에 대한 인사이트를 쌓을 수 있었습니다.

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
*Kyoungmin Kim, Jiacheng Li, **Kijae Hong**, and Anastasia Ailamaki*  
arXiv:2411.07447, 2025 (Under Review at VLDB 2026)

**Large Language Models for Semantic Join: A Comprehensive Survey**  
***Kijae Hong*** *and Yeonsu Park*  
*IEEE Access*, 2025

**Design and Evaluation of a GPU-Accelerated SQL Query Engine for Large-Scale Analytics**  
***Kijae Hong***  
Ph.D. Dissertation, POSTECH, 2025

**[Themis: A GPU-accelerated Relational Query Execution Engine](https://www.vldb.org/pvldb/vol18/p426-han.pdf)**  
***Kijae Hong***, *Kyoungmin Kim, Young-Koo Lee, Yang-Sae Moon, Sourav S Bhowmick, and Wook-Shin Han*  
*Very Large Data Bases Conference (VLDB)*, 2025 ⭐

### 2022

**Survey on the GPU-Based Graph Analytics Methods**  
***Kijae Hong***, *Jinho Ko, Taesung Lee, and Wook-Shin Han*  
*Korea Computer Congress (KCC)*, 2022

**Performance Analysis of Property Graph Queries on Graph and Relational Databases**  
*Jinho Ko, Taesung Lee, **Kijae Hong**, and Wook-Shin Han*  
*Korea Computer Congress (KCC)*, 2022

### 2021

**[iTurboGraph: Scaling and Automating Incremental Graph Analytics](https://dl.acm.org/doi/abs/10.1145/3448016.3457243)**  
*Seongyun Ko, Taesung Lee, **Kijae Hong**, Wonseok Lee, In Seo, Jiwon Seo, and Wook-Shin Han*  
*ACM International Conference on Management of Data (SIGMOD)*, 2021 ⭐

**A Study on Property Graph Partitioning for Graph Analytic Query Processing in Distributed Environment**  
*Jinho Ko, **Kijae Hong**, Taesung Lee, Jeong-Hoon Lee, and Wook-Shin Han*  
*Korea Computer Congress (KCC)*, 2021

**Graph Analytics Query Acceleration Using Filtering Techniques in Page-Based Dynamic Graph Storage**  
*Taesung Lee, **Kijae Hong**, Jeong-Hoon Lee, and Wook-Shin Han*  
*Korea Computer Congress (KCC)*, 2021

### 2020

**[G-CARE: A Framework for Performance Benchmarking of Cardinality Estimation Techniques for Subgraph Matching](https://www.researchgate.net/profile/Sourav-S-Bhowmick/publication/341750604_G-CARE_A_Framework_for_Performance_Benchmarking_of_Cardinality_Estimation_Techniques_for_Subgraph_Matching/links/5ee45f61a6fdcc73be780998/G-CARE-A-Framework-for-Performance-Benchmarking-of-Cardinality-Estimation-Techniques-for-Subgraph-Matching.pdf)**  
*Yeonsu Park, Seongyun Ko, Sourav S. Bhowmick, Kyoungmin Kim, **Kijae Hong**, and Wook-Shin Han*  
*ACM International Conference on Management of Data (SIGMOD)*, 2020 ⭐

### 2018

**A Survey on Top-down and Bottom-up Inductive Logic Programming Methods**  
*In Seo, Jeong-Hoon Lee, Kyoungmin Kim, **Kijae Hong**, Byunghoon So, and Wook-Shin Han*  
*Korea Computer Congress (KCC)*, 2018

---

### 🔖 Patents

**GPU-Based Query Processing Acceleration Method and Computing System**  
*Wook-Shin Han, **Kijae Hong**, Taesung Lee, and Kyoungmin Kim*  
U.S. Patent Application No. 19/144,304 (Filed: Jul. 2025)

**METHOD FOR GENERATING CONTENTS BASED ON TEMPLATE**  
***Kijae Hong***  
Korean Patent Registration No. 10-2025-0068845 (Filed: May. 2025)

**METHOD FOR OPTIMIZING INFERENCE OF LANGUAGE MODEL**  
***Kijae Hong***  
Korean Patent Registration No. 10-2025-0066840 (Filed: May. 2025)

**Method for Summarizing Document Using Large Language Model**  
*Inhyeok Na, **Kijae Hong**, and Jaehyun Lim*  
Korean Patent Registration No. 10-2025-0084553 (Published: Jun. 2025)

**Method for Retrieving Document Related to Natural Language Query**  
*Inhyeok Na, **Kijae Hong**, Jaehyun Lim, and Haechan Lee*  
Korean Patent Registration No. 10-2815043-0000 (Registered: May. 2025)

**Method and Computing System for Acceleration of Processing Queries Based on GPU**  
*Wook-Shin Han, Taesung Lee, Kyungmin Kim, and **Kijae Hong***  
Korean Patent Registration No. 10-2649076-0000 (Registered: Mar. 2024)

**Distributed Processing System and Method for Processing Data**  
*Wook-Shin Han, Yeonsu Park, and **Kijae Hong***  
Korean Patent Registration No. 10-2024-0026045 (Published: Feb. 2024)

**A Method for Mapping a Natural Language Sentence to an SQL Query**  
*Wook-Shin Han, Hyunji Kim, Jungho Jo, Yukyung Lee, and **Kijae Hong***  
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