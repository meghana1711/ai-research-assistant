from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ai_research_assistant.config import OLLAMA_MODEL
from ai_research_assistant.ingestion import load_document


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


map_prompt = ChatPromptTemplate.from_template(
    """
You are summarizing part of an AI/ML research paper.

Summarize the provided section faithfully.

Focus on:
- the purpose of this section
- important concepts or methods
- datasets or experimental setup
- important numerical results
- conclusions
- limitations, if mentioned

Do not introduce information that is not present.

SECTION:

{content}
"""
)


reduce_prompt = ChatPromptTemplate.from_template(
    """
You are an AI research assistant.

Using ONLY the intermediate summaries below,
write a comprehensive summary of the complete research paper.

Organize the result under these headings:

1. Paper Overview
2. Problem Being Addressed
3. Main Contributions
4. Proposed Method / Architecture
5. Training and Experimental Setup
6. Datasets and Tasks
7. Key Results
8. Limitations
9. Final Takeaway

Preserve important numerical results when available.

Do not invent missing information.
If a detail is not available in the summaries, do not guess.

INTERMEDIATE SUMMARIES:

{summaries}
"""
)


map_chain = (
    map_prompt
    | llm
    | StrOutputParser()
)

reduce_chain = (
    reduce_prompt
    | llm
    | StrOutputParser()
)

def group_documents(
    documents,
    group_size: int = 4,
):
    """
    Group neighboring chunks together so that we do not
    call the LLM once for every tiny chunk.
    """

    groups = []

    for i in range(
        0,
        len(documents),
        group_size,
    ):
        group = documents[i:i + group_size]

        combined_text = "\n\n".join(
            document.page_content
            for document in group
        )

        groups.append(combined_text)

    return groups

def summarize_document(
    file_path: str,
    group_size: int = 4,
) -> str:
    """
    Generate a complete summary of a research paper
    using a map-reduce workflow.
    """

    documents = load_document(file_path)

    print(
        f"Loaded {len(documents)} chunks."
    )

    groups = group_documents(
        documents,
        group_size=group_size,
    )

    print(
        f"Created {len(groups)} summary groups."
    )

    intermediate_summaries = []

    for index, content in enumerate(
        groups,
        start=1,
    ):
        print(
            f"Summarizing group "
            f"{index}/{len(groups)}..."
        )

        summary = map_chain.invoke(
            {
                "content": content,
            }
        )

        intermediate_summaries.append(
            summary
        )

    combined_summaries = "\n\n".join(
        f"SECTION SUMMARY {i}\n{summary}"
        for i, summary in enumerate(
            intermediate_summaries,
            start=1,
        )
    )

    final_summary = reduce_chain.invoke(
        {
            "summaries": combined_summaries,
        }
    )

    return final_summary
