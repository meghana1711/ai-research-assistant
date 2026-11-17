from ai_research_assistant.ingestion import load_document


FILE_PATH = "data/raw/2005.11401v4.pdf"


documents = load_document(FILE_PATH)


print("=" * 80)
print("INGESTION RESULT")
print("=" * 80)

print(f"Total chunks: {len(documents)}")


for index, document in enumerate(
    documents[:10],
    start=1,
):
    print("\n" + "=" * 80)

    print(f"CHUNK {index}")

    print("=" * 80)

    print("\nCONTENT:")
    print(document.page_content)

    print("\nMETADATA:")
    print(document.metadata)