import json
import re
from pathlib import Path

from ai_research_assistant.qasper_ingestion import load_qasper
from ai_research_assistant.qasper_hybrid_retrieval import retrieve_qasper_hybrid
from ai_research_assistant.qasper_hybrid_vectorstore import load_hybrid_store


KS = [1, 3, 5, 10]


def normalize(text: str) -> str:
    """Normalize whitespace for paragraph matching."""

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


def get_gold_evidence(answer_group) -> set[str]:
    """
    Collect all textual gold-evidence paragraphs
    from all annotators for one question.
    """

    evidence = set()

    for answer in answer_group["answer"]:

        for item in answer["evidence"]:

            if not item:
                continue

            # QASPER also contains table/figure evidence.
            if item.startswith("FLOAT SELECTED"):
                continue

            evidence.add(
                normalize(item)
            )

    return evidence


def main():

    # Small smoke-test benchmark for now.
    papers = load_qasper(
        split="validation",
        limit=None,
    )

    store = load_hybrid_store()

    hit_scores = {k: 0 for k in KS}
    recall_scores = {k: 0.0 for k in KS}

    reciprocal_rank_sum = 0.0

    evaluated = 0
    skipped = 0

    detailed_results = []

    for paper in papers:

        paper_id = paper["id"]

        qas = paper["qas"]

        questions = qas["question"]
        answers = qas["answers"]

        for question, answer_group in zip(
            questions,
            answers,
        ):

            gold_evidence = get_gold_evidence(
                answer_group
            )

            # Skip questions whose evidence is only
            # tables/figures or has no textual evidence.
            if not gold_evidence:
                skipped += 1
                continue

            retrieved = retrieve_qasper_hybrid(
                question=question,
                paper_id=paper_id,
                store=store,
                k=10,
            )

            retrieved_texts = [
                normalize(
                    doc.metadata["paragraph_text"]
                )
                for doc in retrieved
            ]

            evaluated += 1

            # ---------- Hit@K and Recall@K ----------

            for k in KS:

                retrieved_k = set(
                    retrieved_texts[:k]
                )

                matched = (
                    gold_evidence
                    & retrieved_k
                )

                # Did we retrieve at least ONE
                # correct evidence paragraph?
                if matched:
                    hit_scores[k] += 1

                # What fraction of ALL gold
                # evidence did we retrieve?
                recall_scores[k] += (
                    len(matched)
                    / len(gold_evidence)
                )

            # ---------- MRR@10 ----------

            first_correct_rank = None

            for rank, text in enumerate(
                retrieved_texts,
                start=1,
            ):
                if text in gold_evidence:
                    first_correct_rank = rank
                    break

            if first_correct_rank:
                reciprocal_rank_sum += (
                    1 / first_correct_rank
                )

            detailed_results.append(
                {
                    "paper_id": paper_id,
                    "question": question,
                    "gold_evidence_count": len(
                        gold_evidence
                    ),
                    "first_correct_rank":
                        first_correct_rank,
                }
            )

    print("\n" + "=" * 70)
    print("QASPER HYBRID RETRIEVAL")
    print("=" * 70)

    print(f"Evaluated questions: {evaluated}")
    print(f"Skipped questions:   {skipped}")

    if evaluated == 0:
        print("No evaluable questions found.")
        return

    print("\nHit Rate")

    for k in KS:
        score = hit_scores[k] / evaluated

        print(
            f"Hit@{k}:    {score:.4f}"
        )

    print("\nEvidence Recall")

    for k in KS:
        score = recall_scores[k] / evaluated

        print(
            f"Recall@{k}: {score:.4f}"
        )

    mrr = reciprocal_rank_sum / evaluated

    print(f"\nMRR@10:   {mrr:.4f}")

    # Save results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "system": "hybrid",
        "papers": len(papers),
        "evaluated_questions": evaluated,
        "skipped_questions": skipped,
        "hit_at_k": {
            str(k):
                hit_scores[k] / evaluated
            for k in KS
        },
        "recall_at_k": {
            str(k):
                recall_scores[k] / evaluated
            for k in KS
        },
        "mrr_at_10": mrr,
        "questions": detailed_results,
    }

    with open(
        output_dir / "hybrid.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()