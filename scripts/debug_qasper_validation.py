import re

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from ai_research_assistant.qasper_ingestion import load_qasper
from ai_research_assistant.qasper_vectorstore import load_qasper_store


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


papers = load_qasper(
    split="validation",
    limit=None,
)

paper = papers[0]

paper_id = paper["id"]
question = paper["qas"]["question"][0]
answer_group = paper["qas"]["answers"][0]


print("=" * 80)
print("PAPER")
print("=" * 80)

print("Paper ID:", paper_id)
print("Title:", paper["title"])
print("Question:", question)


# ----------------------------
# Gold evidence
# ----------------------------

gold_evidence = set()

for answer in answer_group["answer"]:
    for evidence in answer["evidence"]:

        if not evidence:
            continue

        if evidence.startswith("FLOAT SELECTED"):
            continue

        gold_evidence.add(normalize(evidence))


print("\nGold evidence count:", len(gold_evidence))

for evidence in list(gold_evidence)[:2]:
    print("\nGOLD:")
    print(evidence[:500])


# ----------------------------
# Load Qdrant
# ----------------------------

store = load_qasper_store()


# ----------------------------
# Test WITHOUT paper filter
# ----------------------------

unfiltered = store.similarity_search(
    query=question,
    k=5,
)

print("\n" + "=" * 80)
print("UNFILTERED SEARCH")
print("=" * 80)

print("Retrieved:", len(unfiltered))

for doc in unfiltered:
    print(
        "paper_id:",
        doc.metadata.get("paper_id")
    )


# ----------------------------
# Test WITH paper filter
# ----------------------------

paper_filter = Filter(
    must=[
        FieldCondition(
            key="metadata.paper_id",
            match=MatchValue(
                value=paper_id
            ),
        )
    ]
)

filtered = store.similarity_search(
    query=question,
    k=10,
    filter=paper_filter,
)

print("\n" + "=" * 80)
print("FILTERED SEARCH")
print("=" * 80)

print("Retrieved:", len(filtered))


for rank, doc in enumerate(filtered, start=1):

    paragraph = normalize(
        doc.metadata.get(
            "paragraph_text",
            doc.page_content,
        )
    )

    match = paragraph in gold_evidence

    print(
        f"\nRank {rank}"
        f" | paper={doc.metadata.get('paper_id')}"
        f" | GOLD={match}"
    )

    print(paragraph[:500])