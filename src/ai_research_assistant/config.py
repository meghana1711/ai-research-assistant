import os
from dotenv import load_dotenv


load_dotenv()


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_MAX_TOKENS = int(
    os.getenv("CHUNK_MAX_TOKENS", "256")
)

OLLAMA_MODEL = os.getenv(
    "HF_MODEL",
    "qwen3:4b",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "data/qdrant",
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "ai_research",
)