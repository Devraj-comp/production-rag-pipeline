import asyncio
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from app.services.rag_service import rag_service

# Custom Wrapper for RAGAS to use local Ollama
from ragas.llms import BaseRAGASLLM
from ragas.embeddings import BaseRAGASEmbeddings

class OllamaRAGASLLM(BaseRAGASLLM):
    def __init__(self, model_name="llama3"):
        super().__init__()
        self.llm = OllamaLLM(model=model_name, base_url="http://host.docker.internal:11434")

    async def generate_text(self, prompt: str) -> str:
        return await asyncio.to_thread(self.llm.invoke, prompt)

class LocalRAGASEmbeddings(BaseRAGASEmbeddings):
    def __init__(self):
        super().__init__()
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    def embed_query(self, text: str) -> list:
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list) -> list:
        return self.embeddings.embed_documents(texts)

async def run_evaluation():
    print("Starting RAGAS Evaluation...")

    # Load Golden Dataset
    with open("eval/golden_dataset.json", "r") as f:
        golden_data = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in golden_data:
        print(f"Evaluating: {item['question']}")
        # Get response from the actual pipeline
        result = await rag_service.query(item['question'])

        questions.append(item['question'])
        answers.append(result['answer'])
        # RAGAS expects contexts as a list of strings
        contexts.append([result['context']])
        ground_truths.append(item['ground_truth'])

    # Create RAGAS Dataset
    data = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Initialize local judge LLM and embeddings
    judge_llm = OllamaRAGASLLM()
    judge_embeddings = LocalRAGASEmbeddings()

    # Run Evaluation
    print("Calculating metrics (this may take a few minutes)...")
    result = evaluate(
        dataset=data,
        metrics=[faithfulness, answer_relevance, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    # Save Results
    df = result.to_pandas()
    df.to_csv("eval/evaluation_report.csv", index=False)
    df.to_json("eval/evaluation_report.json", orient="records")

    print("\nEvaluation Complete!")
    print(df[['question', 'faithfulness', 'answer_relevance', 'context_precision']])

if __name__ == "__main__":
    asyncio.run(run_evaluation())
