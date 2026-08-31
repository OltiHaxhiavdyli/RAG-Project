"""Multi-representation indexing: parent-document retrieval. Small child
chunks get embedded and searched — so retrieval stays precise, matching the
right sentence/paragraph rather than a fuzzy match diluted across a whole
page — but the LARGER parent chunk they belong to is what actually gets
returned, so the model isn't stuck working from a fragment that's missing
the surrounding context that resolves "it"/cross-references between
sentences. The index (child chunks) differs from what's actually returned
(parent chunks) — that's the whole technique.

Kept as its OWN Chroma collection plus a persisted docstore for the parent
chunks, rather than sharing the main collection — child chunks carry a
doc_id pointing into that docstore, a different metadata shape than every
other retriever in this project assumes (source/sheet/etc.), and mixing the
two would make those assumptions unreliable.

Scoped to prose documents only (PDF/DOCX/TXT/MD) — structured rows (CSV/
Excel/JSON) are already atomic; there's no larger "parent" a single row
would benefit from being reunited with.
"""
import hashlib

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.rag import vectorstore

# Raw (unchunked) documents per add_documents() call — child-chunk count per
# call is unpredictable (depends on how long each source document is), so
# batching by input document count is a conservative way to stay well under
# the embedding provider's per-request limits (same issue fixed in
# vectorstore.py's add_documents, same fix shape).
ADD_BATCH_SIZE = 10


def _parent_child_vectorstore() -> Chroma:
    # Shares vectorstore.get_chroma_client() (one client per process for
    # CHROMA_PERSIST_DIR) with the main collection — see that function's
    # docstring for the real bug this avoids. Imported as a module (not
    # `from ... import get_chroma_client`) so a test's `importlib.reload
    # (vectorstore)` correctly resets this too, rather than this module
    # holding a stale pre-reload reference.
    return Chroma(
        collection_name=config.PARENT_CHILD_COLLECTION_NAME,
        embedding_function=vectorstore.get_embeddings(),
        client=vectorstore.get_chroma_client(),
    )


def build_parent_document_retriever() -> ParentDocumentRetriever:
    return ParentDocumentRetriever(
        vectorstore=_parent_child_vectorstore(),
        byte_store=LocalFileStore(config.PARENT_DOCSTORE_DIR),
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=config.CHILD_CHUNK_OVERLAP
        ),
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_CHUNK_SIZE, chunk_overlap=config.PARENT_CHUNK_OVERLAP
        ),
        search_kwargs={"k": config.PARENT_RETRIEVER_TOP_K},
    )


def _stable_source_doc_id(doc: Document) -> str:
    """Deterministic per (source, exact raw content) — e.g. distinct PDF
    pages sharing one `source` still get distinct ids, since page text
    differs. Mirrors chunking.py's _stable_chunk_id, one level up (raw
    documents here, not post-split chunks)."""
    source = doc.metadata.get("source", "")
    return hashlib.sha256(f"{source}::{doc.page_content}".encode("utf-8")).hexdigest()


def ingest_parent_documents(documents: list[Document]) -> int:
    """Adds raw (unchunked) prose documents to the parent-document index.
    Returns how many source documents were added (not child-chunk count).

    Deterministic parent ids so re-ingesting the same file upserts the SAME
    docstore entry instead of storing it again under a fresh random UUID —
    same duplication bug as chunking.py's, fixed the same way, for this
    index's raw-document store. Doesn't cover the CHILD chunks
    ParentDocumentRetriever generates internally (LangChain assigns those
    random ids itself, not exposed for override without reimplementing its
    private splitting logic) — a known, smaller-impact gap: duplicate child
    EMBEDDINGS can exist, but the PARENT text they resolve to is still
    deduped by DecomposingRetriever's final merge (same source + same
    parent text -> same dedup key), so it doesn't surface as duplicate
    answer content the way the main collection's bug did."""
    if not documents:
        return 0
    retriever = build_parent_document_retriever()
    for i in range(0, len(documents), ADD_BATCH_SIZE):
        batch = documents[i : i + ADD_BATCH_SIZE]
        ids = [_stable_source_doc_id(doc) for doc in batch]
        retriever.add_documents(batch, ids=ids)
    return len(documents)


def parent_child_chunk_count() -> int:
    return _parent_child_vectorstore()._collection.count()
