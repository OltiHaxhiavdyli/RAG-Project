"""vectorstore.get_chroma_client() tests. No API key needed — stubs
chromadb.PersistentClient, never touches a real store."""
import threading
import time

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
