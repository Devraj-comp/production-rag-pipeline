from typing import List
import os
import json
import numpy as np
import redis
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from app.core.config import settings

class RAGService:
    def __init__(self):
        # Initialize Qdrant
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)

        # Local Embeddings (BGE-Small is fast and professional)
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # Local LLM via Ollama - Use host.docker.internal to reach Mac host from container
        self.llm = OllamaLLM(
            model="llama3",
            base_url="http://host.docker.internal:11434"
        )

        # Redis Client for Semantic Caching
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.cache_threshold = 0.9

        # Reranker for two-stage retrieval
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.collection_name = "document_knowledge"
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensures the Qdrant collection exists with correct vector dimensions."""
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            # BGE-Small embeddings size is 384
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def _calculate_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _rerank(self, query: str, candidates: List) -> List:
        """Rerank candidates using a Cross-Encoder."""
        if not candidates:
            return []

        # Prepare pairs for the reranker: [(query, doc1), (query, doc2), ...]
        pairs = [[query, cand.payload["text"]] for cand in candidates]

        # Predict relevance scores
        scores = self.reranker.predict(pairs)

        # Attach scores to candidates and sort descending
        scored_candidates = [
            (score, cand) for score, cand in zip(scores, candidates)
        ]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Return only the candidates (strip scores)
        return [cand for score, cand in scored_candidates]

    def chunk_text(self, text: str) -> List[str]:
        """Recursive Character Splitting with overlap."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)

    async def ingest_document(self, file_path: str):
        """Full pipeline: Load -> Chunk -> Embed -> Store"""
        print(f"DEBUG: Starting ingestion for {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = self.chunk_text(text)
            print(f"DEBUG: Created {len(chunks)} chunks")

            points = []
            for i, chunk in enumerate(chunks):
                vector = self.embeddings.embed_query(chunk)
                points.append({
                    "id": i,
                    "vector": vector,
                    "payload": {
                        "text": chunk,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "chunk_index": i
                        }
                    }
                })

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"DEBUG: Successfully upserted {len(points)} points to Qdrant")
            return {"status": "success", "chunks_created": len(chunks)}
        except Exception as e:
            print(f"DEBUG ERROR during ingestion: {str(e)}")
            raise e

    async def query(self, user_query: str):
        """Retrieval Pipeline: Query -> Embed -> Cache Check -> Retrieve -> Rerank -> Augment -> Generate -> Cache Store"""
        print(f"DEBUG: Processing query: {user_query}")
        try:
            # 1. Embed the user query
            query_vector = self.embeddings.embed_query(user_query)
            print("DEBUG: Query embedding generated")

            # 2. Semantic Cache Lookup
            print("DEBUG: Checking semantic cache...")
            cached_queries = self.redis_client.keys("cache:query:*")
            for key in cached_queries:
                cached_data = json.loads(self.redis_client.get(key))
                similarity = self._calculate_similarity(query_vector, cached_data["vector"])
                if similarity >= self.cache_threshold:
                    print(f"DEBUG: Cache hit! Similarity: {similarity:.4f}")
                    return {
                        "answer": cached_data["answer"],
                        "sources": cached_data["sources"],
                        "cached": True
                    }

            print("DEBUG: Cache miss. Proceeding to retrieval...")

            # 3. Retrieve candidates from Qdrant (Coarse Retrieval)
            # Increase limit to 20 to ensure recall
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=20
            ).points
            print(f"DEBUG: Retrieved {len(search_result)} coarse candidates")

            # 4. Rerank candidates (Fine-grained Retrieval)
            print("DEBUG: Reranking candidates with Cross-Encoder...")
            reranked_result = self._rerank(user_query, search_result)

            # Take top 3 for final context
            top_candidates = reranked_result[:3]
            print(f"DEBUG: Reranked to top {len(top_candidates)} candidates")

            # 5. Combine retrieved chunks into a single context string
            context = "\n\n".join([res.payload["text"] for res in top_candidates])

            # 6. Construct the Augmented Prompt
            prompt = f"""
            You are a professional AI Assistant. Use the provided context to answer the user's question.
            If the answer is not contained within the context, honestly state that you do not know.
            Do not use external knowledge to make up an answer.

            Context:
            {context}

            Question: {user_query}

            Answer:
            """

            # 7. Generate answer using Local Llama 3
            print("DEBUG: Calling Ollama LLM...")
            response = self.llm.invoke(prompt)
            print("DEBUG: LLM response received")

            sources = list(set([res.payload["metadata"]["source"] for res in top_candidates]))

            # 8. Store result in Semantic Cache
            cache_key = f"cache:query:{hash(user_query)}"
            cache_payload = {
                "vector": query_vector,
                "answer": response,
                "sources": sources
            }
            self.redis_client.set(cache_key, json.dumps(cache_payload))
            print("DEBUG: Result stored in semantic cache")

            return {
                "answer": response,
                "sources": sources,
                "cached": False,
                "context": context
            }
        except Exception as e:
            print(f"DEBUG ERROR during query: {str(e)}")
            raise e

rag_service = RAGService()
