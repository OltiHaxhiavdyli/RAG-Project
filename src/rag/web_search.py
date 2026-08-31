"""Web search as a retrieval source, via Tavily (a search API built for LLM
consumption — results come back as clean text, not raw HTML). Wrapped as a
BaseRetriever so it can sit in the same EnsembleRetriever as the vector/BM25
retrievers in retrieval.py: all three candidate pools get reranked together,
so the cross-encoder — not a hand-written rule — effectively decides per
query whether local docs or live web results are more relevant.
"""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src import config


class WebSearchRetriever(BaseRetriever):
    """Runs a live Tavily web search and returns results as Documents."""

    max_results: int = config.WEB_SEARCH_MAX_RESULTS

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        from langchain_tavily import TavilySearch

        search = TavilySearch(max_results=self.max_results, tavily_api_key=config.TAVILY_API_KEY)
        response = search.invoke({"query": query})
        results = response.get("results", []) if isinstance(response, dict) else response

        documents = []
        for result in results:
            content = result.get("content", "")
            if not content:
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": result.get("url", "web"),
                        "title": result.get("title", ""),
                        "origin": "web",
                    },
                )
            )
        return documents
