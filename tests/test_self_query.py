"""Self-query retriever tests. No API key needed — these test the guards
(empty-store short-circuit, exception swallowing), not the LLM-driven query
construction itself."""
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.rag.self_query import ConditionalSelfQueryRetriever, SafeRetriever, build_self_query_retriever


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings — no API key/network needed to
    construct or write to a Chroma collection."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_build_self_query_retriever_returns_none_for_empty_store(tmp_path):
    store = Chroma(embedding_function=FakeEmbeddings(), persist_directory=str(tmp_path))
    assert build_self_query_retriever(store) is None


def test_build_self_query_retriever_returns_none_when_no_sources_in_metadata(tmp_path):
    store = Chroma(embedding_function=FakeEmbeddings(), persist_directory=str(tmp_path))
    store.add_documents([Document(page_content="no source key here", metadata={})])
    assert build_self_query_retriever(store) is None


class RaisingRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise ValueError("simulated malformed self-query filter")


def test_safe_retriever_swallows_exceptions_from_inner_retriever():
    retriever = SafeRetriever(inner=RaisingRetriever())
    assert retriever.invoke("anything") == []


class FailIfCalledRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise AssertionError("the inner self-query retriever should not have been invoked")


def test_conditional_self_query_skips_inner_when_no_keyword_overlap():
    retriever = ConditionalSelfQueryRetriever(
        inner=FailIfCalledRetriever(), keywords=frozenset({"calendar", "refund", "academic"})
    )
    assert retriever.invoke("what time does the library close") == []


def test_conditional_self_query_runs_inner_on_keyword_overlap():
    doc = Document(page_content="hit", metadata={"source": "calendar.pdf"})

    class StubRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            return [doc]

    retriever = ConditionalSelfQueryRetriever(
        inner=StubRetriever(), keywords=frozenset({"calendar", "refund", "academic"})
    )
    assert retriever.invoke("what does the academic calendar say about November") == [doc]


def test_conditional_self_query_always_runs_inner_without_keywords():
    """No keyword set at all (e.g. a store with sources but no catalog
    entries) means there's nothing to gate on — always attempt it, same
    fail-open default as scope_gate's empty-catalog case."""
    doc = Document(page_content="hit", metadata={"source": "x"})

    class StubRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            return [doc]

    retriever = ConditionalSelfQueryRetriever(inner=StubRetriever())
    assert retriever.invoke("anything at all") == [doc]
