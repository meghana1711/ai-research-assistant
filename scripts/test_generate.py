from ai_research_assistant.qasper_ingestion import load_qasper
from ai_research_assistant.qasper_hybrid_vectorstore import (
    load_hybrid_store,
)
from ai_research_assistant.qasper_rag import answer_qasper


def main():

    papers = load_qasper(
        split="validation",
        limit=1,
    )

    paper = papers[0]

    question = paper["qas"]["question"][0]

    store = load_hybrid_store()

    result = answer_qasper(
        question=question,
        paper_id=paper["id"],
        store=store,
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(result["question"])

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for i, document in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"[{i}] "
            f"{document.metadata['section']} "
            f"| paragraph "
            f"{document.metadata['paragraph_index']}"
        )


if __name__ == "__main__":
    main()