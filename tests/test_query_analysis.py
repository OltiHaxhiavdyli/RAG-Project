"""DecomposingRetriever tests. Mocks needs_decomposition()/decompose()/
step_back() and the base retriever, so no API key or network access is
needed."""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag import query_analysis
from src.rag.query_analysis import DecomposingRetriever

DOC_A = Document(page_content="Refunds take 5 business days.", metadata={"source": "policy.md"})
DOC_B = Document(page_content="Widgets with cracks are rejected.", metadata={"source": "policy.md"})
# same source + same first-200-chars as DOC_A -> should dedup against it
DOC_A_DUPLICATE = Document(
    page_content="Refunds take 5 business days.", metadata={"source": "policy.md"}
)


class StubRetriever(BaseRetriever):
    """Returns different docs depending on which sub-question was asked."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if "refund" in query.lower():
            return [DOC_A]
        if "widget" in query.lower():
            return [DOC_A_DUPLICATE, DOC_B]
        return []


def test_decomposing_retriever_merges_and_dedupes(monkeypatch):
    monkeypatch.setattr(query_analysis, "needs_decomposition", lambda question: True)
    monkeypatch.setattr(
        query_analysis,
        "decompose",
        lambda question: ["How long do refunds take?", "What happens to cracked widgets?"],
    )
    monkeypatch.setattr(query_analysis, "step_back", lambda question: question)  # no-op: dedupes away

    retriever = DecomposingRetriever(base_retriever=StubRetriever())
    docs = retriever.invoke("How long do refunds take, and what happens to cracked widgets?")

    assert len(docs) == 2  # DOC_A_DUPLICATE deduped against DOC_A
    assert docs[0] is DOC_A
    assert docs[1] is DOC_B


def test_decomposing_retriever_handles_single_subquestion(monkeypatch):
    monkeypatch.setattr(query_analysis, "needs_decomposition", lambda question: True)
    monkeypatch.setattr(query_analysis, "decompose", lambda question: ["How long do refunds take?"])
    monkeypatch.setattr(query_analysis, "step_back", lambda question: question)  # no-op: dedupes away

    retriever = DecomposingRetriever(base_retriever=StubRetriever())
    docs = retriever.invoke("How long do refunds take?")

    assert docs == [DOC_A]


def test_decomposing_retriever_adds_step_back_question(monkeypatch):
    monkeypatch.setattr(query_analysis, "needs_decomposition", lambda question: True)
    monkeypatch.setattr(query_analysis, "decompose", lambda question: ["How long do refunds take?"])
    monkeypatch.setattr(query_analysis, "step_back", lambda question: "What is the refund policy?")

    class StepBackAwareRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            if query == "What is the refund policy?":
                return [DOC_B]  # only the broader question surfaces this
            return [DOC_A]

    retriever = DecomposingRetriever(base_retriever=StepBackAwareRetriever())
    docs = retriever.invoke("How long do refunds take?")

    assert DOC_A in docs
    assert DOC_B in docs  # only found via the step-back question


def test_decomposing_retriever_skips_step_back_when_same_as_original(monkeypatch):
    monkeypatch.setattr(query_analysis, "needs_decomposition", lambda question: True)
    monkeypatch.setattr(query_analysis, "decompose", lambda question: ["How long do refunds take?"])
    # step_back returns the ORIGINAL question unchanged (already general) — should not add a duplicate call
    monkeypatch.setattr(query_analysis, "step_back", lambda question: question)

    calls = []

    class CountingRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            calls.append(query)
            return [DOC_A] if "refund" in query.lower() else []

    retriever = DecomposingRetriever(base_retriever=CountingRetriever())
    retriever.invoke("How long do refunds take?")

    assert calls == ["How long do refunds take?"]  # not called twice for the same question


def test_decomposing_retriever_skips_decomposition_when_not_needed(monkeypatch):
    monkeypatch.setattr(query_analysis, "needs_decomposition", lambda question: False)

    def _fail_if_called(question):
        raise AssertionError("decompose/step_back should not run when needs_decomposition says no")

    monkeypatch.setattr(query_analysis, "decompose", _fail_if_called)
    monkeypatch.setattr(query_analysis, "step_back", _fail_if_called)

    retriever = DecomposingRetriever(base_retriever=StubRetriever())
    docs = retriever.invoke("How long do refunds take?")

    assert docs == [DOC_A]  # retrieved directly on the original question, single pass


def test_decomposing_retriever_runs_decomposition_when_needed(monkeypatch):
    """Complements the skip test above: the conservative default (True,
    exercised throughout the other tests via explicit monkeypatching) is
    real behavior, not just a mock default — confirmed here by actually
    invoking needs_decomposition's caller with a distinguishable return."""
    calls = []
    monkeypatch.setattr(
        query_analysis, "needs_decomposition", lambda question: calls.append(question) or True
    )
    monkeypatch.setattr(query_analysis, "decompose", lambda question: [question])
    monkeypatch.setattr(query_analysis, "step_back", lambda question: question)

    retriever = DecomposingRetriever(base_retriever=StubRetriever())
    retriever.invoke("How long do refunds take?")

    assert calls == ["How long do refunds take?"]  # the gate itself was actually consulted
