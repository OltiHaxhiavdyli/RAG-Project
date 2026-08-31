"""A cheap upfront check that decides two independent things about a
question before any real retrieval happens:

1. Scope — does it plausibly relate to ANY known ingested source? If
   clearly not, skip the full local retrieval pass (vector/BM25/self-query +
   relevance grading) and go straight to a live web search — a cost/latency
   optimization layered in FRONT of Corrective RAG, not a replacement for
   it. Corrective RAG still catches the case where a question looked
   in-scope but the actual retrieved content wasn't useful; this only
   catches the cheaper, more obvious case where the topic clearly isn't
   covered by anything ingested at all.
2. Decomposition — would splitting the question into sub-questions, or also
   retrieving a broader step-back version of it, actually help? An
   already-simple question skips that work entirely (see query_analysis.py).

Both are decided by ONE structured-output LLM call
(check_scope_and_decomposition), not two — they're independent questions
about the same input, and running them as two sequential round trips was
pure latency for no extra signal, the same shape of fix already applied to
the two self-correction checks (see self_correction.py's grade_answer).

Deliberately conservative on both axes: defaults to "in scope" and "needs
decomposition" whenever uncertain, since wrongly skipping local retrieval or
wrongly skipping decomposition are both real correctness regressions,
whereas wrongly running either anyway just costs one extra pass that a
later stage (Corrective RAG, or just an unneeded retrieval) already handles
correctly.
"""
from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from src.rag.chain import get_llm
from src.rag.query_analysis import needs_decomposition

COMBINED_GATE_PROMPT = """Known sources and what they cover:
{catalog}

Question: {question}

Answer two independent questions about it.

1. in_scope — does it plausibly relate to ANY of the sources above, even \
partially or as background context? Default to YES whenever unsure; only \
say NO if it's clearly about something none of these sources would cover \
at all (e.g. general knowledge, current events, or an unrelated topic).

2. needs_decomposition — would it benefit from EITHER (a) being split into \
sub-questions [only true if it asks about more than one clearly distinct \
thing, e.g. "X, and also Y"], or (b) also retrieving a broader/more general \
version of it [only true if it names a specific, narrow entity — a course \
code, program, event, or named policy — whose answer more likely lives \
inside a broader general policy section than under that exact narrow \
phrasing]? Most simple, direct factual questions ("what is the tuition \
fee", "when does the semester start") need NEITHER — say NO for those. \
Default to YES only when genuinely unsure which way it leans, not as a \
fallback reasoning that more context never hurts."""


class ScopeDecision(BaseModel):
    in_scope: bool = Field(description="True unless clearly unrelated to every known source.")


class ScopeAndDecompositionDecision(ScopeDecision):
    needs_decomposition: bool = Field(
        description="True unless the question is clearly simple enough that "
        "decomposition/step-back would add nothing."
    )


def check_scope_and_decomposition(
    question: str, catalog: dict[str, str]
) -> ScopeAndDecompositionDecision:
    catalog_text = "\n".join(f"- {source}: {desc}" for source, desc in catalog.items())
    llm = get_llm(temperature=0)
    return llm.with_structured_output(ScopeAndDecompositionDecision).invoke(
        COMBINED_GATE_PROMPT.format(catalog=catalog_text, question=question)
    )


class ScopeGatedRetriever(BaseRetriever):
    """Dispatches to one of three retrievers based on the combined
    scope+decomposition check above:
      - out of scope       -> web_retriever directly
      - in scope, simple   -> simple_retriever (no decompose/step-back)
      - in scope, complex  -> complex_retriever (decompose/step-back)

    Skips the LLM call entirely when there's nothing to gate against (empty
    catalog) or nowhere to fall back to (no web_retriever) — but still needs
    a decomposition decision in that case, so it falls back to asking
    query_analysis.needs_decomposition() alone rather than losing that
    check too."""

    simple_retriever: BaseRetriever
    complex_retriever: BaseRetriever
    web_retriever: Optional[BaseRetriever] = None
    catalog: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.web_retriever is None or not self.catalog:
            if not needs_decomposition(query):
                return self.simple_retriever.invoke(query)
            return self.complex_retriever.invoke(query)

        decision = check_scope_and_decomposition(query, self.catalog)
        if not decision.in_scope:
            return self.web_retriever.invoke(query)
        if not decision.needs_decomposition:
            return self.simple_retriever.invoke(query)
        return self.complex_retriever.invoke(query)
