from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)
from ai_research_assistant.reranker import rerank

def retrieve_qasper(
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

def retrieve_qasper_reranked(
    question: str,
    paper_id: str,
    store,
    k: int = 10,
):
    # Same 10 dense candidates as the baseline
    candidates = retrieve_qasper(
        question=question,
        paper_id=paper_id,
        store=store,
        k=k,
    )

    return rerank(
        question=question,
        documents=candidates,
    )