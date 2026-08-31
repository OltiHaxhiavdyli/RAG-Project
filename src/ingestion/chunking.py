"""Split loaded documents into retrieval-sized chunks."""
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def _stable_chunk_id(source: str, page_content: str) -> str:
    """Deterministic per (source, exact text) — NOT the positional chunk_id
    below, which is only a running index within one split_documents() call
    and collides across separate calls (e.g. re-ingesting the same file
    later starts back at 0). vectorstore.add_documents() uses this as the
    Chroma document id, so re-ingesting identical content upserts in place
    instead of appending a duplicate under a fresh random id — a real bug,
    reproduced against the live store: re-running `ingest` (even just to
    add two new URLs, since it also unconditionally re-scans data/raw)
    duplicated 1340 of 1515 existing chunks, because Chroma was never told
    two additions were "the same" chunk. See README's ingestion section."""
    return hashlib.sha256(f"{source}::{page_content}".encode("utf-8")).hexdigest()


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.id = _stable_chunk_id(chunk.metadata.get("source", ""), chunk.page_content)

    return chunks
