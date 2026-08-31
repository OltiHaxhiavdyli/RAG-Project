"""Corrective RAG (C-RAG): grade whether locally-retrieved documents are
actually good enough to answer the question, and only fall back to a live
web search if they aren't — instead of always blending web results in on
every query regardless of whether local docs already answer it fine.

This is the "Active retrieval" pattern: using a signal about retrieval
quality (the grade) to trigger re-retrieval elsewhere, rather than always
generating from whatever the first retrieval pass happened to return.
"""
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from src import config
from src.rag.chain import get_llm

GRADE_PROMPT = """You are grading whether retrieved documents are actually \
useful for answering a question. Be strict: a document only tangentially \
related does not count as relevant.

Question: {question}

Documents:
{numbered_docs}

Return the 0-based indices of the documents that would genuinely help \
answer the question. Return an empty list if none would."""


class RelevanceGrade(BaseModel):
    relevant_indices: List[int] = Field(description="0-based indices of relevant documents.")


def grade_relevance(question: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return []  # nothing to grade — skip the LLM call entirely

    numbered = "\n\n".join(f"[{i}] {doc.page_content[:500]}" for i, doc in enumerate(docs))
    llm = get_llm(temperature=0)
    grade = llm.with_structured_output(RelevanceGrade).invoke(
        GRADE_PROMPT.format(question=question, numbered_docs=numbered)
    )
    return [docs[i] for i in grade.relevant_indices if 0 <= i < len(docs)]


class CorrectiveRetriever(BaseRetriever):
    """Retrieves locally, grades relevance, and only falls back to (already
    reranked) web search if the local results don't clear the bar."""

    local_retriever: BaseRetriever
    web_retriever: Optional[BaseRetriever] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        local_docs = self.local_retriever.invoke(query)
        relevant = grade_relevance(query, local_docs)

        if len(relevant) >= config.CRAG_MIN_RELEVANT_DOCS or self.web_retriever is None:
            return relevant

        web_docs = self.web_retriever.invoke(query)
        return relevant + web_docs
