from langchain_core.documents import Document

from ai_research_assistant.reranker import rerank
from ai_research_assistant.vectorstore import load_vector_store


def retrieve(
    question: str,
    k: int = 5,
    fetch_k: int = 10,
) -> list[Document]:

    vector_store = load_vector_store()

    candidates = vector_store.similarity_search(
        query=question,
        k=fetch_k,
    )

    return rerank(
        question=question,
        documents=candidates,
        k=k,
    )