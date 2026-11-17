from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from ai_research_assistant.config import EMBEDDING_MODEL


QASPER_PATH = "data/qdrant_qasper_validation_dense"
QASPER_COLLECTION = "qasper_validation_dense"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)


def create_qasper_store(documents):
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=QASPER_PATH,
        collection_name=QASPER_COLLECTION,
        retrieval_mode=RetrievalMode.DENSE,
        force_recreate=True,
    )


def load_qasper_store():
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=QASPER_PATH,
        collection_name=QASPER_COLLECTION,
    )