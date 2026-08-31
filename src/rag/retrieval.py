"""Hybrid retrieval: dense (vector/MMR) + sparse (BM25) fused, then reranked
with a local cross-encoder so the LLM only sees the most relevant chunks.
Live web search is a corrective fallback (see corrective.py), not blended in
unconditionally — see build_hybrid_retriever for how it all composes."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src import config
from src.rag.corrective import CorrectiveRetriever
from src.rag.parent_document import build_parent_document_retriever, parent_child_chunk_count
from src.rag.query_analysis import DecomposingRetriever
from src.rag.scope_gate import ScopeGatedRetriever
from src.rag.self_query import build_self_query_retriever
from src.rag.source_catalog import build_source_catalog
from src.rag.web_search import WebSearchRetriever

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoderReranker:
    model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    return CrossEncoderReranker(model=model, top_n=config.RERANK_TOP_K)


def _rerank(retriever: BaseRetriever) -> BaseRetriever:
    return ContextualCompressionRetriever(base_compressor=_get_reranker(), base_retriever=retriever)


def _all_documents(store: Chroma) -> list[Document]:
    raw = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def _build_local_ensemble(store: Chroma) -> BaseRetriever:
    """Vector similarity (MMR for diversity) fused with BM25 keyword search,
    self-query retrieval (auto-extracted metadata filters), and
    parent-document retrieval (precise child-chunk search, full-context
    parent-chunk return) via reciprocal rank fusion. Local documents only —
    web search is handled separately, as a corrective fallback, not a
    default ensemble member."""
    retrievers: list[BaseRetriever] = [
        store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.RETRIEVER_TOP_K, "fetch_k": config.RETRIEVER_TOP_K * 2},
        )
    ]

    documents = _all_documents(store)
    if documents:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = config.RETRIEVER_TOP_K
        retrievers.append(bm25_retriever)

    self_query_retriever = build_self_query_retriever(store)
    if self_query_retriever is not None:
        retrievers.append(self_query_retriever)

    if parent_child_chunk_count() > 0:
        retrievers.append(build_parent_document_retriever())

    if len(retrievers) == 1:
        return retrievers[0]
    return EnsembleRetriever(retrievers=retrievers, weights=[1 / len(retrievers)] * len(retrievers))


def build_hybrid_retriever(store: Chroma) -> BaseRetriever:
    """Composes:
      0. a cheap upfront scope check (scope_gate.py), against the ORIGINAL
         question only, once — not per sub-question. If the question clearly
         doesn't relate to anything ingested, skip straight to web search
         instead of paying for decomposition and a full retrieval pass on
         every sub-question. Scoping a compound question as a whole is fine
         here: the gate only needs "does ANY of this relate to something we
         have", which a single pass answers correctly, unlike reranking/
         correction below where favoring one half of the wording would
         silently drop the other half's results.
      1. query decomposition (query_analysis.py) — split into sub-questions
         + one step-back question, each retrieved independently so a
         compound question doesn't systematically starve out whichever half
         is less prominent in the combined wording (confirmed against real
         data — see README's Query decomposition section). Itself gated by
         a cheap needs_decomposition() check — an already-simple question
         skips straight to a single retrieval pass on the original wording.
      2. per (sub-)question: local retrieval (vector+BM25+self-query+parent-
         document ensemble), reranked
      3. per (sub-)question: graded for relevance (Corrective RAG) — only
         falls back to a (also reranked) live web search if the local
         results don't clear the relevance bar, instead of paying for a web
         call on every query
      4. all sub-question results merged/deduped

    Self-query construction inside the ensemble (step 2) is itself
    conditional — see self_query.py — skipped per-query when the question
    shares no vocabulary with any known source, rather than always spending
    an LLM call on it."""
    local_reranked = _rerank(_build_local_ensemble(store))

    web_reranked = _rerank(WebSearchRetriever()) if config.WEB_SEARCH_ENABLED else None
    corrective = CorrectiveRetriever(local_retriever=local_reranked, web_retriever=web_reranked)

    decomposing = DecomposingRetriever(base_retriever=corrective)

    catalog = build_source_catalog(store)
    return ScopeGatedRetriever(
        local_retriever=decomposing, web_retriever=web_reranked, catalog=catalog
    )
