"""A cheap upfront check: does this question plausibly relate to ANY known
ingested source? If clearly not, skip the full local retrieval pass
(vector/BM25/self-query + relevance grading) and go straight to a live web
search — a cost/latency optimization layered in FRONT of Corrective RAG, not
a replacement for it. Corrective RAG still catches the case where a question
looked in-scope but the actual retrieved content wasn't useful; this only
catches the cheaper, more obvious case where the topic clearly isn't covered
by anything ingested at all.

Deliberately conservative: defaults to "in scope" whenever uncertain, since
wrongly skipping local retrieval (a false "out of scope") is a real
correctness regression, whereas wrongly running it anyway (a false "in
scope") just costs one extra retrieval+grading pass that Corrective RAG
already handles correctly.
"""
from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from src.rag.chain import get_llm

GATE_PROMPT = """Known sources and what they cover:
{catalog}

Question: {question}

Does this question plausibly relate to ANY of the sources above — even \
partially, or as background context? Default to YES whenever unsure; only \
say NO if the question is clearly about something none of these sources \
would cover at all (e.g. general knowledge, current events, or an unrelated \
topic)."""


class ScopeDecision(BaseModel):
    in_scope: bool = Field(description="True unless clearly unrelated to every known source.")


def check_scope(question: str, catalog: dict[str, str]) -> bool:
    if not catalog:
        return True  # nothing to gate against — always attempt local retrieval

    catalog_text = "\n".join(f"- {source}: {desc}" for source, desc in catalog.items())
    llm = get_llm(temperature=0)
    decision = llm.with_structured_output(ScopeDecision).invoke(
        GATE_PROMPT.format(catalog=catalog_text, question=question)
    )
    return decision.in_scope


class ScopeGatedRetriever(BaseRetriever):
    """Skips straight to web_retriever if the question is clearly out of
    scope for every known source; otherwise delegates to local_retriever
    (which may itself be the corrective/graded retriever) as normal."""

    local_retriever: BaseRetriever
    web_retriever: Optional[BaseRetriever] = None
    catalog: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.web_retriever is not None and not check_scope(query, self.catalog):
            return self.web_retriever.invoke(query)
        return self.local_retriever.invoke(query)
