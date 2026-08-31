"""Scope gate tests. No API key needed — mocks check_scope_and_decomposition
and the retrievers, so these test the dispatch logic, not the LLM's
judgment."""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag import scope_gate
from src.rag.scope_gate import ScopeAndDecompositionDecision, ScopeGatedRetriever

SIMPLE_DOC = Document(page_content="simple content", metadata={"source": "local.md"})
COMPLEX_DOC = Document(page_content="complex content", metadata={"source": "local.md"})
WEB_DOC = Document(page_content="web content", metadata={"source": "https://example.com"})


class StubRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return self.docs


class FailIfCalledRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        raise AssertionError("this retriever should not have been called")


def _retriever(**overrides):
    defaults = dict(
        simple_retriever=FailIfCalledRetriever(),
        complex_retriever=FailIfCalledRetriever(),
        web_retriever=FailIfCalledRetriever(),
        catalog={"local.md": "some description"},
    )
    defaults.update(overrides)
    return ScopeGatedRetriever(**defaults)


def test_scope_gated_retriever_uses_simple_retriever_when_in_scope_and_not_complex(monkeypatch):
    monkeypatch.setattr(
        scope_gate,
        "check_scope_and_decomposition",
        lambda q, catalog: ScopeAndDecompositionDecision(in_scope=True, needs_decomposition=False),
    )

    retriever = _retriever(simple_retriever=StubRetriever(docs=[SIMPLE_DOC]))
    assert retriever.invoke("some question") == [SIMPLE_DOC]


def test_scope_gated_retriever_uses_complex_retriever_when_in_scope_and_complex(monkeypatch):
    monkeypatch.setattr(
        scope_gate,
        "check_scope_and_decomposition",
        lambda q, catalog: ScopeAndDecompositionDecision(in_scope=True, needs_decomposition=True),
    )

    retriever = _retriever(complex_retriever=StubRetriever(docs=[COMPLEX_DOC]))
    assert retriever.invoke("some question") == [COMPLEX_DOC]


def test_scope_gated_retriever_skips_to_web_when_out_of_scope(monkeypatch):
    monkeypatch.setattr(
        scope_gate,
        "check_scope_and_decomposition",
        # needs_decomposition shouldn't even matter here — out-of-scope wins first
        lambda q, catalog: ScopeAndDecompositionDecision(in_scope=False, needs_decomposition=True),
    )

    retriever = _retriever(web_retriever=StubRetriever(docs=[WEB_DOC]))
    assert retriever.invoke("some question") == [WEB_DOC]


def test_scope_gated_retriever_skips_combined_check_when_no_web_retriever(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "check_scope_and_decomposition should not be called when there's no web fallback anyway"
        )

    monkeypatch.setattr(scope_gate, "check_scope_and_decomposition", _fail_if_called)
    monkeypatch.setattr(scope_gate, "needs_decomposition", lambda q: False)

    retriever = _retriever(web_retriever=None, simple_retriever=StubRetriever(docs=[SIMPLE_DOC]))
    assert retriever.invoke("some question") == [SIMPLE_DOC]


def test_scope_gated_retriever_skips_combined_check_for_empty_catalog(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "check_scope_and_decomposition should not be called with nothing to gate against"
        )

    monkeypatch.setattr(scope_gate, "check_scope_and_decomposition", _fail_if_called)
    monkeypatch.setattr(scope_gate, "needs_decomposition", lambda q: True)

    retriever = _retriever(catalog={}, complex_retriever=StubRetriever(docs=[COMPLEX_DOC]))
    assert retriever.invoke("some question") == [COMPLEX_DOC]


def test_fallback_path_still_gets_a_real_decomposition_decision(monkeypatch):
    """Even when the combined check is skipped (no catalog/web_retriever),
    the decomposition decision itself must still be made, not silently
    dropped — this is what proves the fallback calls needs_decomposition()
    rather than just always picking one retriever."""
    calls = []
    monkeypatch.setattr(
        scope_gate, "needs_decomposition", lambda q: calls.append(q) or False
    )

    retriever = _retriever(web_retriever=None, simple_retriever=StubRetriever(docs=[SIMPLE_DOC]))
    retriever.invoke("some question")

    assert calls == ["some question"]
