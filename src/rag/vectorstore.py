"""Chroma-backed persistent vector store, embedded via Gemini (AI Studio or
Vertex AI, depending on LLM_PROVIDER)."""
import threading

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import config

# Neither Chroma nor the embedding clients batch large requests on their own —
# they hand the whole list straight to the API in one call. Vertex AI rejects
# batches over 250 texts, AI Studio's SDK caps around 100, so chunk client-side
# to stay under both regardless of which provider is active.
EMBED_BATCH_SIZE = 100

_client_lock = threading.Lock()
_client_instance: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """One shared client per process for CHROMA_PERSIST_DIR. Letting every
    Chroma(...) call open its own independent client against the same
    on-disk directory (as happens once a second collection — the
    parent-document index — shares the directory) is a real, reproduced bug:
    concurrent/rapid access from separate client objects to the same
    persisted store raised `chromadb.errors.InternalError: ... Nothing found
    on disk`. A single shared client avoids that entirely.

    Hand-rolled double-checked locking rather than @lru_cache: lru_cache
    only makes the cache dict's own read/write atomic, not the "miss, call
    the function, store the result" sequence around it — two threads can
    both see a miss and both call chromadb.PersistentClient(...) for the
    same path concurrently. That's a real, reproduced bug too, hit via the
    FastAPI server (api/main.py) under concurrent requests when the client
    hadn't been created yet: chromadb's own shared-system registry isn't
    safe against two threads registering the same identifier at once, and
    failed with `KeyError` / `AttributeError: 'RustBindingsAPI' object has
    no attribute 'bindings'` — a corrupted, unrecoverable client for the
    rest of the process. A lock around construction closes that window."""
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return _client_instance


def get_embeddings() -> Embeddings:
    config.require_credentials()

    if config.LLM_PROVIDER == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings

        # Vertex model names don't use the "models/" prefix AI Studio uses.
        model = config.EMBEDDING_MODEL.removeprefix("models/")
        return VertexAIEmbeddings(
            model_name=model,
            project=config.VERTEX_PROJECT_ID,
            location=config.VERTEX_LOCATION,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        client=get_chroma_client(),
    )


def add_documents(chunks: list[Document]) -> Chroma:
    """Dedupes by chunk id before adding. Chunk ids are a content hash (see
    chunking.py's _stable_chunk_id) so re-ingesting the same content upserts
    instead of duplicating — but two chunks from DIFFERENT pages can still
    hash identically when their text really is identical, and Chroma rejects
    a batch containing the same id twice outright (`DuplicateIDError`)
    rather than treating it as an upsert.

    Not hypothetical: this surfaced the moment boilerplate stripping
    (loaders.py's strip_shared_boilerplate) landed. Removing shared nav/
    footer lines left several near-empty pages reduced to byte-identical
    residual text, which collided inside a single ingest batch and failed
    the whole run. Upserting once per unique id is the correct behavior
    anyway — they're genuinely the same chunk by this project's own
    definition of chunk identity."""
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        if chunk.id is not None:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
        unique.append(chunk)

    store = get_vectorstore()
    for i in range(0, len(unique), EMBED_BATCH_SIZE):
        store.add_documents(unique[i : i + EMBED_BATCH_SIZE])
    return store


def collection_count() -> int:
    return get_vectorstore()._collection.count()
