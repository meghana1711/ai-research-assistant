from ai_research_assistant.generation import generate_answer
from ai_research_assistant.qasper_hybrid_retrieval import (
    retrieve_qasper_hybrid_reranked,
)


def answer_qasper(
    question: str,
    paper_id: str,
    store,
    context_k: int = 5,
):
    # Retrieve 10 candidates and rerank them
    retrieved = retrieve_qasper_hybrid_reranked(
        question=question,
        paper_id=paper_id,
        store=store,
        k=10,
    )

    # Only send the best 5 to Ollama
    context_documents = retrieved[:context_k]

    answer = generate_answer(
        question=question,
        documents=context_documents,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": context_documents,
    }