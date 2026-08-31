"""Load documents from a directory of mixed file types, or from URLs."""
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.structured_loaders import load_csv, load_excel, load_json

PROSE_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
SUPPORTED_EXTENSIONS = PROSE_EXTENSIONS | {".csv", ".xlsx", ".xls", ".json", ".jsonl"}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(path)).load()
    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(path)).load()
    elif suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader

        docs = TextLoader(str(path), encoding="utf-8").load()
    elif suffix == ".csv":
        docs = load_csv(path)
    elif suffix in (".xlsx", ".xls"):
        docs = load_excel(path)
    elif suffix in (".json", ".jsonl"):
        docs = load_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    for doc in docs:
        doc.metadata["source"] = path.name

    return docs


def load_directory(directory: Path) -> list[Document]:
    """Recursively load every supported file under `directory`."""
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_file(path))
    return documents


def load_prose_directory(directory: Path) -> list[Document]:
    """Like load_directory, but PDF/DOCX/TXT/MD only — for parent-document
    retrieval, which needs whole prose documents to split into parent/child
    chunks. Structured rows (CSV/Excel/JSON) are already atomic; there's no
    larger "parent" a single row would benefit from being reunited with."""
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in PROSE_EXTENSIONS:
            documents.extend(load_file(path))
    return documents


def load_urls(urls: list[str]) -> list[Document]:
    from langchain_community.document_loaders import WebBaseLoader

    docs = WebBaseLoader(urls).load()
    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", "web")
    return docs
