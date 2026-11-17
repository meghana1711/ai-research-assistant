from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


def rerank(
    question: str,
    documents: list[Document],
    k: int = 10,
) -> list[Document]:
    """
    Rerank retrieved documents according to
    their relevance to the user's question.
    """

    if not documents:
        return []

    pairs = [
        [question, document.page_content]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        document
        for document, _ in ranked[:k]
    ]