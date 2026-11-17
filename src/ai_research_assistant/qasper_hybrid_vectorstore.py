from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)

from ai_research_assistant.config import EMBEDDING_MODEL


HYBRID_PATH = "data/qdrant_qasper_validation_hybrid"
HYBRID_COLLECTION = "qasper_validation_hybrid"


dense_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)


def create_hybrid_store(documents):
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        path=HYBRID_PATH,
        collection_name=HYBRID_COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True,
    )


def load_hybrid_store():
    return QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        path=HYBRID_PATH,
        collection_name=HYBRID_COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
    )