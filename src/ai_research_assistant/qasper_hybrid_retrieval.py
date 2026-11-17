from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from ai_research_assistant.reranker import rerank


def retrieve_qasper_hybrid(
    question: str,
    paper_id: str,
    store,
    k: int = 10,
):
    paper_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.paper_id",
                match=MatchValue(
                    value=paper_id
                ),
            )
        ]
    )

    return store.similarity_search(
        query=question,
        k=k,
        filter=paper_filter,
    )


def retrieve_qasper_hybrid_reranked(
    question: str,
    paper_id: str,
    store,
    k: int = 10,
):
    # First retrieve using hybrid search
    candidates = retrieve_qasper_hybrid(
        question=question,
        paper_id=paper_id,
        store=store,
        k=k,
    )

    # Then reorder those same candidates
    return rerank(
        question=question,
        documents=candidates,
    )