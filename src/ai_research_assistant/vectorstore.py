from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from ai_research_assistant.config import (
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={
        "normalize_embeddings": True,
    },
)


def create_vector_store(
    documents: list[Document],
) -> QdrantVectorStore:
    """
    Embed LangChain Documents and persist them
    in a local Qdrant collection.

    For now, rebuilding the index recreates the collection.
    """

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=QDRANT_COLLECTION,
        retrieval_mode=RetrievalMode.DENSE,
        force_recreate=True,
    )

    return vector_store


def load_vector_store() -> QdrantVectorStore:
    """
    Connect to an already indexed local Qdrant collection.
    """

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=QDRANT_COLLECTION,
    )