"""Scope gate tests. No API key needed — mocks check_scope and the
retrievers, so these test the dispatch logic, not the LLM's judgment."""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag import scope_gate
from src.rag.scope_gate import ScopeGatedRetriever, check_scope

LOCAL_DOC = Document(page_content="local content", metadata={"source": "local.md"})
WEB_DOC = Document(page_content="web content", metadata={"source": "https://example.com"})


class StubRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return self.docs


class FailIfCalledRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        raise AssertionError("this retriever should not have been called")


def test_check_scope_skips_llm_for_empty_catalog(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("get_llm should not be called when there's no catalog to check against")

    monkeypatch.setattr(scope_gate, "get_llm", _fail_if_called)
    assert check_scope("any question", {}) is True


def test_scope_gated_retriever_uses_local_when_in_scope(monkeypatch):
    monkeypatch.setattr(scope_gate, "check_scope", lambda q, catalog: True)

    retriever = ScopeGatedRetriever(
        local_retriever=StubRetriever(docs=[LOCAL_DOC]),
        web_retriever=FailIfCalledRetriever(),
        catalog={"local.md": "some description"},
    )
    assert retriever.invoke("some question") == [LOCAL_DOC]


def test_scope_gated_retriever_skips_to_web_when_out_of_scope(monkeypatch):
    monkeypatch.setattr(scope_gate, "check_scope", lambda q, catalog: False)

    retriever = ScopeGatedRetriever(
        local_retriever=FailIfCalledRetriever(),
        web_retriever=StubRetriever(docs=[WEB_DOC]),
        catalog={"local.md": "some description"},
    )
    assert retriever.invoke("some question") == [WEB_DOC]


def test_scope_gated_retriever_skips_check_when_no_web_retriever(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("check_scope should not be called when there's no web fallback anyway")

    monkeypatch.setattr(scope_gate, "check_scope", _fail_if_called)

    retriever = ScopeGatedRetriever(
        local_retriever=StubRetriever(docs=[LOCAL_DOC]),
        web_retriever=None,
        catalog={"local.md": "some description"},
    )
    assert retriever.invoke("some question") == [LOCAL_DOC]
