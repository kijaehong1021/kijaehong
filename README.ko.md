# 👋 Kijae Hong (홍기재)

📧 [E-mail](kijaehong1021@gmail.com) | 🐙 [Github](https://github.com/kijaehong) | 💼 [LinkedIn](https://www.linkedin.com/in/ki-jae-hong-5643b730a/) | 📚 [Google Scholar](https://scholar.google.com/citations?user=QHGq7GIAAAAJ&hl=ko) | 📖 [DBLP](https://dblp.org/pid/266/5820.html)

---

## Introduction

포항공대(POSTECH) 컴퓨터공학과 박사 출신으로, 주력 분야는 GPU 가속 기반의 데이터 처리, 그래프 데이터 분석, 그리고 LLM 서빙 및 추론 최적화입니다.
이전에 초고빈도 매매(High-Frequency Trading) 시스템 개발, vLLM 및 쿠버네티스(Kubernetes)를 활용한 고성능 RAG 시스템 구축과 LLM 미세조정(Fine-tuning) 업무를 수행하였고, 최근에는 LLM inference 스케줄링, KV 캐싱 등에 대한 연구를 진행하고 있습니다. VLDB, SIGMOD 등 세계 최고 수준의 학회에 다수의 논문을 게재하고 관련 특허를 보유하는 등, 이론적 깊이와 실무적 구현 능력을 바탕으로 대규모 데이터 처리 시스템의 혁신을 추구합니다. 

---

## 🔬 Ongoing Projects

### [Maximizing LLM Caching](https://github.com/kijaehong1021/LLMCachingBoost)

- LLM의 KV 캐싱 테크닉의 효과 극대화 방법 탐구
- **기술 스택**: LLM, LMCache, SGLang, LMCache, Database

### [Optimizing Semantic Operators in a Query Plan](https://github.com/kijaehong1021/SemanticOperatorOptimizer)

- Database 연산자들이 LLM inference를 통해 record에 대한 condition 체크 및 새로운 attribute 추출 가능하도록 기능 확장 연구
- 비용이 높은 LLM inference의 수를 최대한 줄이는 최적화 방안 탐구
- **기술 스택**: LLM, Query Optimization, Database Operators

---

## 💼 Previous Projects

<!--
### T-R3X
TBD

### RAG-based Customer Support System for 현대홈쇼핑
TBD

### Knowledge Management System
TBD -->
### LLM Inference Scheduling
- LLM Inference의 스케줄링 방법들을 평가하고 성능 개선 방향 제시
- vLLM의 스케줄러를 수정하여 사용자가 입력한대로 inference를 수행 가능하도록 구현
- **기술 스택**: vLLM, sarathi

### SLM-based Operator of RAG

- RAG 시스템의 핵심 기능을 저비용·저지연으로 수행할 수 있는 Small Language Model(SLM) 기반의 operator 개발
- vLLM을 활용한 추론 서비스를 쿠버네티스 환경에 구축 및 워크로드에 따른 오토 스케일링 구현
- 중소벤처기업부 지원 프로젝트
- **기술 스택**: vLLM, Kubernetes, LLM Fine-tuning

### High Frequency Trading

- 초당 수만 건에 달하는 시장 이벤트 수집 및 실시간 가격 예측 및 매매 자동화 시스템 개발
- 대규모 트래픽을 안정적으로 처리하도록 설계된 분산/병렬 처리 시스템
- 다수의 고객에 성공적으로 판매되어 운용 중
- 중소기업벤처부 지원 프로젝트
- **기술 스택**: 분산/병렬 처리 시스템, 실시간 데이터 처리, C++, 웹소켓, CPython

### [GPU-accelerated Relational Query Execution Engine](https://www.vldb.org/pvldb/vol18/p426-han.pdf)

- GPU 기반 관계형 데이터 분석 질의 가속화 연구 수행
- 병렬 처리 과정에서 발생하는 워프 간, 워프 내 스레드 간의 부하 불균형(Load Imbalance) 문제 해결
- 경쟁 기술 대비 약 379배 빠른 질의 처리 성능 달성
- VLDB 2025 논문 게재
- **기술 스택**: GPU, CUDA, 병렬 처리 최적화

### [QaaD (query-as-a-data)](https://dl.acm.org/doi/abs/10.1145/3589279)

- Apache Spark를 활용한 대량의 소규모 질의(Small Queries) 효율적 처리 시스템 개발
- 기존 Spark의 역발상: 수많은 소규모 질의를 하나의 거대 질의로 병합하여 일괄 처리하는 기법 제안
- 핵심 로직 구현에 깊이 관여
- 대규모 분산 처리 시스템 설계 경험 확보
- **기술 스택**: Apache Spark, 분산 처리 시스템

### Product Search Engine

- 글로벌 전자상거래 업체들의 상품 정보 주기적 수집 및 데이터 포맷 정규화 분산 크롤링 프레임워크 개발
- 크롤링 방지(Anti-crawling) 대응 및 대규모 크롤링 아키텍처 설계
- 서로 다른 출처의 데이터를 연결하는 Entity Matching 기술 습득
- **기술 스택**: 분산 크롤링 프레임워크, Entity Matching

### [iturbograph](https://dl.acm.org/doi/abs/10.1145/3448016.3457243)

- 초거대 그래프 분석 질의(i.e., PageRank, SCC, WCC, SSSP) 결과에 대한 점진적 업데이트를 지원하는 분산/병렬 처리 시스템 개발
- 경쟁 시스템에 대한 실험 및 분석 수행
- SIGMOD 2021 논문 게재
- **기술 스택**: Rust, MPI, 분산/병렬 처리

### [G-CARE](https://www.researchgate.net/profile/Sourav-S-Bhowmick/publication/341750604_G-CARE_A_Framework_for_Performance_Benchmarking_of_Cardinality_Estimation_Techniques_for_Subgraph_Matching/links/5ee45f61a6fdcc73be780998/G-CARE-A-Framework-for-Performance-Benchmarking-of-Cardinality-Estimation-Techniques-for-Subgraph-Matching.pdf)

- 서브그래프 매칭 결과의 수를 예측하는 방법론들을 위한 벤치마크 제안 및 SOTA 방법 분석
- RDF-3X의 query optimizer를 확장하여 SOTA 방법들을 플러그인 형태로 사용 가능하도록 구현
- Query optimizer와 cardinality estimation 분야에 대한 인사이트 확보
- SIGMOD 2020 논문 게재
- **기술 스택**: RDF-3X, Query Optimizer, Cardinality Estimation

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
