from ai_research_assistant.qasper_ingestion import (
    load_qasper,
    paper_to_documents,
)
from ai_research_assistant.qasper_hybrid_vectorstore import (
    create_hybrid_store,
)


papers = load_qasper(
    split="validation",
    limit=None,
)

documents = []

for paper in papers:
    documents.extend(
        paper_to_documents(paper)
    )

print(f"Indexing {len(documents)} paragraphs...")

create_hybrid_store(documents)

print("Hybrid indexing complete.")