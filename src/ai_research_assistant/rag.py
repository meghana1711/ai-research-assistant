from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ai_research_assistant.config import OLLAMA_MODEL
from ai_research_assistant.retrieval import retrieve


SYSTEM_PROMPT = """
You are an AI Engineering Research Assistant.

Answer the question using only the provided research context.

Rules:
- Do not use outside knowledge.
- If the context does not contain enough information, say so.
- Do not invent information.
- Cite supporting passages using [1], [2], etc.
- Give a clear technical explanation.
- Prefer concise answers unless more detail is required.
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Research context:

{context}

Question:
{question}
""",
        ),
    ]
)


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


chain = prompt | llm | StrOutputParser()


def format_context(documents) -> str:
    """
    Convert retrieved LangChain Documents
    into numbered context blocks for the LLM.
    """

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        context_parts.append(
            f"""
[{index}]
{document.page_content}
"""
        )

    return "\n".join(context_parts)


def extract_pages(document) -> list[int]:
    """
    Extract page numbers from Docling metadata.
    """

    pages = set()

    dl_meta = document.metadata.get(
        "dl_meta",
        {},
    )

    for item in dl_meta.get(
        "doc_items",
        [],
    ):
        for provenance in item.get(
            "prov",
            [],
        ):
            page_no = provenance.get(
                "page_no"
            )

            if page_no is not None:
                pages.add(page_no)

    return sorted(pages)


def answer_question(
    question: str,
    k: int = 5,
) -> dict:
    """
    Complete RAG pipeline:

    question
        ↓
    dense retrieval
        ↓
    reranking
        ↓
    prompt construction
        ↓
    Ollama generation
    """

    documents = retrieve(
        question=question,
        k=k,
    )

    context = format_context(
        documents
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        dl_meta = document.metadata.get(
            "dl_meta",
            {},
        )

        origin = dl_meta.get(
            "origin",
            {},
        )

        sources.append(
            {
                "id": index,
                "file_name": origin.get(
                    "filename",
                    document.metadata.get(
                        "source",
                        "Unknown",
                    ),
                ),
                "pages": extract_pages(
                    document
                ),
                "headings": dl_meta.get(
                    "headings",
                    [],
                ),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }