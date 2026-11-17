from ai_research_assistant.summarization import (
    summarize_document,
)


FILE_PATH = "data/raw/2005.11401v4.pdf"


def main():

    summary = summarize_document(
        FILE_PATH
    )

    print("\n" + "=" * 80)
    print("PAPER SUMMARY")
    print("=" * 80)

    print(summary)


if __name__ == "__main__":
    main()