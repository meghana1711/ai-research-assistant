from datasets import load_dataset
from huggingface_hub import hf_hub_download
from langchain_core.documents import Document


QASPER_REVISION = "cc58ffb39db7ff6ce1951e28e029996bf499304e"

QASPER_FILES = {
    "train": "qasper/qasper-train.parquet",
    "validation": "qasper/qasper-validation.parquet",
    "test": "qasper/qasper-test.parquet",
}


def load_qasper(
    split="validation",
    limit=None,
):
    local_file = hf_hub_download(
        repo_id="allenai/qasper",
        filename=QASPER_FILES[split],
        repo_type="dataset",
        revision=QASPER_REVISION,
    )

    dataset = load_dataset(
        "parquet",
        data_files=local_file,
        split="train",
    )

    if limit is not None:
        dataset = dataset.select(
            range(min(limit, len(dataset)))
        )

    return dataset


def paper_to_documents(paper) -> list[Document]:
    """
    Convert one QASPER paper into paragraph-level
    LangChain Documents.
    """

    documents = []

    paper_id = paper["id"]
    title = paper["title"]
    full_text = paper["full_text"]

    section_names = full_text["section_name"]
    section_paragraphs = full_text["paragraphs"]

    for section_index, (section_name, paragraphs) in enumerate(
        zip(section_names, section_paragraphs)
    ):

        for paragraph_index, paragraph in enumerate(paragraphs):

            if not paragraph.strip():
                continue

            documents.append(
                Document(
                    page_content=f"{section_name}\n\n{paragraph}",
                    metadata={
                        "paper_id": paper_id,
                        "title": title,
                        "section": section_name,
                        "section_index": section_index,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph,
                        "dataset": "qasper",
                    },
                )
            )

    return documents
