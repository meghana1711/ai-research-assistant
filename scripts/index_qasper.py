from ai_research_assistant.qasper_ingestion import (
    load_qasper,
    paper_to_documents,
)
from ai_research_assistant.qasper_vectorstore import (
    create_qasper_store,
)


def main():

    papers = load_qasper(
        split="validation",
        limit=None,
    )

    documents = []

    for paper in papers:
        documents.extend(
            paper_to_documents(paper)
        )

    print(f"Validation papers: {len(papers)}")
    print(f"Paragraphs: {len(documents)}")
    print("PAPERS BEING INDEXED:", len(papers))
    print("FIRST PAPER:", papers[0]["id"])

    create_qasper_store(documents)

    print("Dense validation index complete.")


if __name__ == "__main__":
    main()