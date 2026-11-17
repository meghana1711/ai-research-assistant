from ai_research_assistant.rag import (
    answer_question,
)


def main():

    question = input(
        "\nAsk a question: "
    ).strip()

    if not question:
        print("No question provided.")
        return

    result = answer_question(
        question=question,
        k=5,
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

    for source in result["sources"]:

        print(
            f"[{source['id']}] "
            f"{source['file_name']} "
            f"| Pages: {source['pages']} "
            f"| Headings: {source['headings']}"
        )


if __name__ == "__main__":
    main()