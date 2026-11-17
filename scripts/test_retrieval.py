from ai_research_assistant.retrieval import retrieve


def main():

    question = input(
        "\nAsk a question: "
    ).strip()

    if not question:
        print("No question provided.")
        return

    results = retrieve(
        question=question,
        k=5,
    )

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(question)

    for index, document in enumerate(
        results,
        start=1,
    ):

        print("\n" + "=" * 80)
        print(f"RESULT {index}")
        print("=" * 80)

        print("\nCONTENT:")
        print(document.page_content)

        print("\nMETADATA:")
        print(document.metadata)


if __name__ == "__main__":
    main()