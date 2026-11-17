import json
import re
import string
from collections import Counter
from pathlib import Path

from ai_research_assistant.qasper_ingestion import load_qasper
from ai_research_assistant.qasper_hybrid_vectorstore import (
    load_hybrid_store,
)
from ai_research_assistant.qasper_rag import answer_qasper


MAX_QUESTIONS = None


def normalize_answer(text: str) -> str:
    """
    SQuAD-style normalization.
    """

    text = text.lower()

    # Remove our [1], [2] citations before scoring.
    text = re.sub(r"\[\d+\]", "", text)

    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )

    words = text.split()

    # Remove English articles.
    words = [
        word
        for word in words
        if word not in {"a", "an", "the"}
    ]

    return " ".join(words)


def token_f1(
    prediction: str,
    reference: str,
) -> float:

    pred_tokens = normalize_answer(
        prediction
    ).split()

    ref_tokens = normalize_answer(
        reference
    ).split()

    if not pred_tokens and not ref_tokens:
        return 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def extract_reference_answers(
    answer_group,
) -> list[str]:
    """
    Convert QASPER annotations into textual answers.
    """

    references = []

    for annotation in answer_group["answer"]:

        if annotation["unanswerable"]:
            references.append("Unanswerable")
            continue

        spans = annotation["extractive_spans"]

        if spans:
            references.append(
                " ".join(spans)
            )
            continue

        free_form = annotation["free_form_answer"]

        if free_form:
            references.append(free_form)
            continue

        yes_no = annotation["yes_no"]

        if yes_no is not None:
            references.append(
                "Yes" if yes_no else "No"
            )

    return references


def main():

    papers = load_qasper(
        split="validation",
        limit=None,
    )

    store = load_hybrid_store()

    results = []

    total_f1 = 0.0
    evaluated = 0

    for paper in papers:

        questions = paper["qas"]["question"]
        answer_groups = paper["qas"]["answers"]

        for question, answer_group in zip(
            questions,
            answer_groups,
        ):

            references = extract_reference_answers(
                answer_group
            )

            if not references:
                continue

            result = answer_qasper(
                question=question,
                paper_id=paper["id"],
                store=store,
                context_k=5,
            )

            prediction = result["answer"]

            # QASPER may contain multiple human answers.
            f1_scores = [
                token_f1(
                    prediction,
                    reference,
                )
                for reference in references
            ]

            best_f1 = max(f1_scores)

            evaluated += 1
            total_f1 += best_f1

            print(f"\n[{evaluated}]")
            print(f"Q: {question}")
            print(f"PRED: {prediction}")
            print(f"GOLD: {references}")
            print(f"F1: {best_f1:.4f}")

            results.append(
                {
                    "paper_id": paper["id"],
                    "question": question,
                    "prediction": prediction,
                    "references": references,
                    "f1": best_f1,
                }
            )

            if MAX_QUESTIONS is not None and evaluated >= MAX_QUESTIONS:
                break

        if MAX_QUESTIONS is not None and evaluated >= MAX_QUESTIONS:
            break

    average_f1 = (
        total_f1 / evaluated
        if evaluated
        else 0.0
    )

    print("\n" + "=" * 70)
    print("QASPER GENERATION EVALUATION")
    print("=" * 70)

    print(
        f"Evaluated questions: {evaluated}"
    )

    print(
        f"Answer F1: {average_f1:.4f}"
    )

    output_dir = Path(
        "evaluation/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "generation.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model": "qwen3:4b",
                "retrieval":
                    "hybrid_plus_reranker",
                "questions": evaluated,
                "answer_f1": average_f1,
                "results": results,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()