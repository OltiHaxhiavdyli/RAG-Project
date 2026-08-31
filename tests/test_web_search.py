"""Web search retriever tests. Mocks the Tavily API call itself, so these
need no network access and no real TAVILY_API_KEY."""
from unittest.mock import patch

from src import config
from src.rag.web_search import WebSearchRetriever

FAKE_RESPONSE = {
    "query": "what is RAG",
    "results": [
        {
            "title": "Retrieval-Augmented Generation",
            "url": "https://example.com/rag",
            "content": "RAG combines retrieval with generation.",
            "score": 0.9,
        },
        {
            "title": "Empty content result",
            "url": "https://example.com/empty",
            "content": "",
        },
    ],
}


def test_web_search_retriever_converts_results_to_documents(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key-for-test")
    with patch(
        "langchain_tavily.tavily_search.TavilySearchAPIWrapper.raw_results",
        return_value=FAKE_RESPONSE,
    ):
        retriever = WebSearchRetriever(max_results=5)
        docs = retriever.invoke("what is RAG")

    # the empty-content result is dropped
    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content == "RAG combines retrieval with generation."
    assert doc.metadata["source"] == "https://example.com/rag"
    assert doc.metadata["title"] == "Retrieval-Augmented Generation"
    assert doc.metadata["origin"] == "web"
