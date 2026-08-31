"""Parent-document retrieval tests. No API key needed — these test the
empty-input short-circuit and batching, not the actual embedding/splitting."""
from langchain_core.documents import Document

from src.rag import parent_document


def test_ingest_parent_documents_skips_retriever_construction_for_empty_input(monkeypatch):
    def _fail_if_called():
        raise AssertionError("build_parent_document_retriever should not be called for empty input")

    monkeypatch.setattr(parent_document, "build_parent_document_retriever", _fail_if_called)
    assert parent_document.ingest_parent_documents([]) == 0


def test_ingest_parent_documents_batches_add_documents_calls(monkeypatch):
    calls = []

    class StubRetriever:
        def add_documents(self, batch, ids=None):
            calls.append((list(batch), ids))

    monkeypatch.setattr(parent_document, "build_parent_document_retriever", lambda: StubRetriever())
    monkeypatch.setattr(parent_document, "ADD_BATCH_SIZE", 3)

    docs = [Document(page_content=f"doc {i}") for i in range(7)]
    added = parent_document.ingest_parent_documents(docs)

    assert added == 7
    assert [len(batch) for batch, _ids in calls] == [3, 3, 1]


def test_ingest_parent_documents_uses_deterministic_ids(monkeypatch):
    """Real bug, found live against the project's own data store: without
    explicit ids, re-ingesting the same file stored it again under a fresh
    random UUID every time — 1340 of 1515 real chunks ended up exact
    duplicates after a routine re-ingest. Deterministic (source, content)
    ids make re-ingestion an upsert instead."""
    calls = []

    class StubRetriever:
        def add_documents(self, batch, ids=None):
            calls.append((list(batch), ids))

    monkeypatch.setattr(parent_document, "build_parent_document_retriever", lambda: StubRetriever())

    doc_a = Document(page_content="Refund policy text.", metadata={"source": "policy.md"})
    doc_b = Document(page_content="Widget policy text.", metadata={"source": "policy.md"})

    parent_document.ingest_parent_documents([doc_a, doc_b])
    first_ids = calls[0][1]

    calls.clear()
    parent_document.ingest_parent_documents([doc_a, doc_b])  # simulate a re-ingest
    second_ids = calls[0][1]

    assert first_ids == second_ids  # same (source, content) -> same ids, every time
    assert len(set(first_ids)) == 2  # doc_a and doc_b still get distinct ids
