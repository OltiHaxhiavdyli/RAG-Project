"""Corrective RAG tests. Mocks grade_relevance and the retrievers, so no API
key or network access is needed."""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src import config
from src.rag import corrective
from src.rag.corrective import CorrectiveRetriever, grade_relevance

LOCAL_DOC = Document(page_content="local content", metadata={"source": "local.md"})
WEB_DOC = Document(page_content="web content", metadata={"source": "https://example.com"})


class StubRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.docs


class FailIfCalledRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise AssertionError("this retriever should not have been called")


def test_grade_relevance_skips_llm_for_empty_docs(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("get_llm should not be called when there are no docs to grade")

    monkeypatch.setattr(corrective, "get_llm", _fail_if_called)
    assert grade_relevance("any question", []) == []


def test_corrective_retriever_skips_web_when_local_is_sufficient(monkeypatch):
    monkeypatch.setattr(corrective, "grade_relevance", lambda q, docs: [LOCAL_DOC])

    retriever = CorrectiveRetriever(
        local_retriever=StubRetriever(docs=[LOCAL_DOC]),
        web_retriever=FailIfCalledRetriever(),
    )
    docs = retriever.invoke("some question")
    assert docs == [LOCAL_DOC]


def test_corrective_retriever_falls_back_to_web_when_local_is_insufficient(monkeypatch):
    monkeypatch.setattr(corrective, "grade_relevance", lambda q, docs: [])

    retriever = CorrectiveRetriever(
        local_retriever=StubRetriever(docs=[LOCAL_DOC]),
        web_retriever=StubRetriever(docs=[WEB_DOC]),
    )
    docs = retriever.invoke("some question")
    assert docs == [WEB_DOC]


def test_corrective_retriever_without_web_retriever_returns_local_grade_only(monkeypatch):
    monkeypatch.setattr(corrective, "grade_relevance", lambda q, docs: [])

    retriever = CorrectiveRetriever(local_retriever=StubRetriever(docs=[LOCAL_DOC]), web_retriever=None)
    docs = retriever.invoke("some question")
    assert docs == []


def test_corrective_retriever_respects_min_relevant_docs_threshold(monkeypatch):
    monkeypatch.setattr(config, "CRAG_MIN_RELEVANT_DOCS", 2)
    monkeypatch.setattr(corrective, "grade_relevance", lambda q, docs: [LOCAL_DOC])  # only 1, below threshold

    retriever = CorrectiveRetriever(
        local_retriever=StubRetriever(docs=[LOCAL_DOC]),
        web_retriever=StubRetriever(docs=[WEB_DOC]),
    )
    docs = retriever.invoke("some question")
    assert docs == [LOCAL_DOC, WEB_DOC]
