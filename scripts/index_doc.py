from ai_research_assistant.ingestion import load_document
from ai_research_assistant.vectorstore import create_vector_store


FILE_PATH = "data/raw/2005.11401v4.pdf"


def main():

    print("Loading and chunking document...")

    documents = load_document(FILE_PATH)

    print(f"Created {len(documents)} chunks.")

    print("Generating embeddings and indexing in Qdrant...")

    create_vector_store(documents)

    print("Indexing complete.")


if __name__ == "__main__":
    main()