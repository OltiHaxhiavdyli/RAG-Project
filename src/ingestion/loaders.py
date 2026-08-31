"""Load documents from a directory of mixed file types, or from URLs."""
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.structured_loaders import load_csv, load_excel, load_json

PROSE_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
SUPPORTED_EXTENSIONS = PROSE_EXTENSIONS | {".csv", ".xlsx", ".xls", ".json", ".jsonl"}

# A line repeated on at least this fraction of a batch's pages is treated as
# site chrome rather than content. 0.5 is deliberately conservative: real
# content very rarely appears verbatim on half the pages of a site, while
# nav/footer links appear on essentially all of them.
BOILERPLATE_PAGE_FRACTION = 0.5

# Below this many pages there's no reliable signal — "appears on 1 of 2
# pages" is 50% but means nothing. Skip stripping entirely for tiny batches
# rather than guess.
BOILERPLATE_MIN_PAGES = 4


def strip_shared_boilerplate(docs: list[Document]) -> list[Document]:
    """Drop lines that repeat across most pages in the batch — nav menus,
    footer sitemaps, cookie notices. Measured on this project's own real
    corpus: **51% of all scraped web line-content was boilerplate**, the
    same ~100 nav/footer lines duplicated verbatim across all 16 ingested
    RIT Kosovo pages. That directly costs retrieval precision — every
    boilerplate-heavy chunk competing for a top-K slot is a slot not spent
    on real content, and a chunk that's mostly a nav menu can still match a
    query on the nav link text alone (see ENGINEERING.md's Evaluation
    section for the RAGAS context-precision numbers this addresses).

    Detected from the data rather than a hardcoded phrase list, so this
    generalizes to whatever site someone else ingests instead of only
    working on this one corpus. Deliberately conservative — see the two
    threshold constants above — since wrongly dropping real content is a
    correctness bug, while leaving some boilerplate in only costs the
    precision this is trying to win back."""
    pages = {}  # source -> set of its lines (per-source, so one long page can't self-inflate a count)
    for doc in docs:
        source = doc.metadata.get("source", "")
        pages.setdefault(source, set()).update(
            line.strip() for line in doc.page_content.splitlines() if line.strip()
        )

    if len(pages) < BOILERPLATE_MIN_PAGES:
        return docs

    line_page_counts = Counter()
    for lines in pages.values():
        line_page_counts.update(lines)

    threshold = len(pages) * BOILERPLATE_PAGE_FRACTION
    boilerplate = {line for line, n in line_page_counts.items() if n >= threshold}
    if not boilerplate:
        return docs

    cleaned = []
    for doc in docs:
        kept = [
            line
            for line in doc.page_content.splitlines()
            if line.strip() not in boilerplate
        ]
        text = "\n".join(kept).strip()
        if text:  # a page that was ONLY boilerplate has nothing left worth indexing
            cleaned.append(Document(page_content=text, metadata=doc.metadata))
    return cleaned


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
    return strip_shared_boilerplate(docs)
