from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ai_research_assistant.config import OLLAMA_MODEL, OLLAMA_BASE_URL


llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI research assistant.

Answer the question using ONLY the supplied research-paper context.

Rules:
- Give only the direct answer.
- Be as short as possible while remaining correct.
- Do not repeat the question.
- Do not add explanations unless they are required to answer correctly.
- If the answer is a name, dataset, method, number, list, or phrase, return only that.
- For yes/no questions, begin with exactly "Yes" or "No".
- If the answer truly cannot be found in the supplied context, output exactly:
  Unanswerable
- Do not use outside knowledge.
- Do not guess.
- Add supporting citation(s) like [1] at the end.
""",
        ),
        (
            "human",
            """
CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
""",
        ),
    ]
)


chain = prompt | llm | StrOutputParser()


def format_context(documents):
    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        context_parts.append(
            f"""
[{index}]
Paper: {metadata.get("title")}
Section: {metadata.get("section")}
Paragraph: {metadata.get("paragraph_index")}

{document.page_content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    question: str,
    documents,
) -> str:

    context = format_context(documents)

    return chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )