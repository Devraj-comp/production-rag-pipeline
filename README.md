# Production-Grade RAG Pipeline

A high-performance, privacy-first Retrieval-Augmented Generation (RAG) pipeline implementing advanced AI engineering patterns for maximum precision and minimal latency.

## 🚀 Technical Architecture

This system moves beyond basic RAG by implementing a multi-stage retrieval and optimization process:

### 1. Two-Stage Retrieval (Recall $\rightarrow$ Precision)
- **Stage 1: Dense Retrieval:** Uses `BAAI/bge-small-en-v1.5` embeddings and **Qdrant** to quickly retrieve the top 20 candidate chunks.
- **Stage 2: Cross-Encoder Reranking:** Uses a `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) to re-score candidates, ensuring the most relevant context is promoted to the top 3, mitigating the "Lost in the Middle" phenomenon.

### 2. Performance Optimization
- **Semantic Caching:** Implements a **Redis-based semantic cache**. Instead of exact string matches, it uses vector similarity to detect semantically identical queries, returning cached responses in milliseconds without calling the LLM.

### 3. Local-First & Privacy-Centric
- **LLM:** Powered by **Llama 3** running locally via **Ollama**.
- **Infrastructure:** Fully containerized using **Docker Compose** (FastAPI, Qdrant, Redis).

### 4. Quantitative Evaluation
- **RAGAS Framework:** Implements the "RAG Triad" (Faithfulness, Answer Relevance, Context Precision) to quantitatively measure and prove system performance using a local judge LLM.

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **API:** FastAPI
- **Vector DB:** Qdrant
- **Cache:** Redis
- **LLM:** Ollama (Llama 3)
- **Embeddings:** HuggingFace (BGE-Small)
- **Reranker:** Sentence-Transformers (Cross-Encoder)
- **Evaluation:** RAGAS

## 🏁 Getting Started

### Prerequisites
- Docker & Docker Compose
- Ollama installed and running (`ollama run llama3`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Devraj-comp/production-rag-pipeline.git
   cd production-rag-pipeline
   ```
2. Start the infrastructure:
   ```bash
   docker-compose up -d --build
   ```
3. Ingest a document:
   ```bash
   curl -X POST "http://localhost:8000/api/ingest?file_path=/home/devraj/test.txt"
   ```
4. Query the system:
   ```bash
   curl "http://localhost:8000/api/query?q=What+are+the+Core+Collaboration+Days+at+Nexus+Nova?"
   ```
