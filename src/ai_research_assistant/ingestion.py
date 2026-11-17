from pathlib import Path
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from transformers import AutoTokenizer

from ai_research_assistant.config import (
    CHUNK_MAX_TOKENS,
    EMBEDDING_MODEL,
)


# Use the same tokenizer family that our embedding model uses.
hf_tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=CHUNK_MAX_TOKENS,
)


chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)


def load_document(
    file_path: str | Path,
) -> list[Document]:
    """
    Parse and chunk one document using Docling.

    Returns LangChain Document objects.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    loader = DoclingLoader(
        file_path=[str(path)],
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker,
    )

    return loader.load()