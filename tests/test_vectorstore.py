"""vectorstore tests. No API key needed — stubs chromadb.PersistentClient
and the store itself, never touches a real store."""
import threading
import time

from langchain_core.documents import Document

from src.rag import vectorstore


def test_get_chroma_client_is_created_exactly_once_under_concurrent_access(monkeypatch):
    """Real bug, found via the FastAPI server: get_chroma_client() used to
    be @lru_cache-wrapped, which only makes the cache dict's own read/write
    atomic, not the "miss, call the function, store the result" sequence
    around it. Two threads that both saw a miss before either finished
    construction both called chromadb.PersistentClient(...) for the same
    path concurrently — chromadb's own shared-system registry isn't safe
    against that, and it failed with a corrupted, unrecoverable client for
    the rest of the process (KeyError / AttributeError deep inside
    chromadb's rust bindings). Reproduces the race deterministically with a
    stub constructor that sleeps to force overlap, and asserts the
    double-checked lock closes it: constructed exactly once, every caller
    gets the same instance."""
    monkeypatch.setattr(vectorstore, "_client_instance", None)

    calls = []

    class FakeClient:
        pass

    def slow_constructor(path):
        calls.append(path)
        time.sleep(0.05)  # force overlap between concurrent callers
        return FakeClient()

    monkeypatch.setattr(vectorstore.chromadb, "PersistentClient", slow_constructor)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(vectorstore.get_chroma_client()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1  # constructed exactly once despite 8 concurrent callers
    assert len({id(r) for r in results}) == 1  # every caller got the SAME instance


def test_get_chroma_client_reuses_existing_instance(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(vectorstore, "_client_instance", sentinel)

    def fail_if_called(path):
        raise AssertionError("should reuse the existing instance, not construct a new one")

    monkeypatch.setattr(vectorstore.chromadb, "PersistentClient", fail_if_called)

    assert vectorstore.get_chroma_client() is sentinel


class _RecordingStore:
    def __init__(self):
        self.added = []

    def add_documents(self, docs):
        self.added.extend(docs)


def test_add_documents_dedupes_chunks_sharing_an_id(monkeypatch):
    """Real bug, surfaced the moment boilerplate stripping landed: chunk ids
    are a content hash, so two chunks from different pages whose text became
    byte-identical (after shared nav/footer lines were stripped) collide —
    and Chroma rejects a batch containing the same id twice outright with
    DuplicateIDError rather than upserting. Deduping by id before adding is
    correct anyway: they're the same chunk by this project's own definition
    of chunk identity."""
    store = _RecordingStore()
    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda: store)

    a = Document(page_content="same text", metadata={"source": "page-a"}, id="dup")
    b = Document(page_content="same text", metadata={"source": "page-b"}, id="dup")
    c = Document(page_content="different", metadata={"source": "page-c"}, id="unique")

    vectorstore.add_documents([a, b, c])

    assert [d.id for d in store.added] == ["dup", "unique"]  # b dropped, order kept


def test_add_documents_keeps_chunks_without_ids(monkeypatch):
    """An id of None isn't a collision — those chunks must all still be
    added, not silently collapsed into one."""
    store = _RecordingStore()
    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda: store)

    docs = [Document(page_content=f"text {i}") for i in range(3)]
    assert all(d.id is None for d in docs)

    vectorstore.add_documents(docs)

    assert len(store.added) == 3
