"""Self-query retrieval: an LLM splits the question into (a) the actual
semantic search text and (b) structured metadata filters, instead of always
embedding the raw question as-is. "What does the academic calendar say about
November?" becomes a semantic search for "November" filtered to
source == "RITK Academic Calendar 2026-27 Final.pdf" — a filter the user
never wrote explicitly, generated from the question itself.

Only `source` and `sheet` are declared as filterable, grounded in the real
distinct values actually present in the store (so the LLM isn't guessing at
filenames) — arbitrary CSV/JSON columns aren't, since their names and meaning
vary per ingested file and can't be described generically the way a fixed
relational schema can (that's what text-to-SQL is for instead). Each source
is also tagged with a short auto-generated description (source_catalog.py),
so filtering can be grounded in what a page actually covers, not just its
name — "the mission statement" can resolve to the right file even though the
question never says the filename.

Self-query construction is itself an LLM call, made on every (sub-)question
in the ensemble. Most questions share no vocabulary with any known source at
all, and for those, the LLM call almost always constructs no filter anyway —
so ConditionalSelfQueryRetriever below skips straight to "no results" (via a
free, non-LLM keyword-overlap check against source names/descriptions)
rather than spending a round trip to confirm what a plain word-overlap check
already told us. Conservative like scope_gate: any overlap at all still runs
it, since a wrongly-skipped filter is a real regression, while an
unnecessary run just costs one call that would have contributed nothing."""
import re

from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag.chain import get_llm
from src.rag.source_catalog import build_source_catalog

DOCUMENT_CONTENT_DESCRIPTION = (
    "Text chunks from ingested documents (policies, handbooks, calendars) and "
    "flattened rows from structured data (CSV/Excel/JSON)."
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


def _distinct_metadata_values(store: Chroma, field: str) -> list[str]:
    raw = store.get(include=["metadatas"])
    values = {m.get(field) for m in raw["metadatas"] if m and m.get(field)}
    return sorted(values)


class SafeRetriever(BaseRetriever):
    """Wraps another retriever and swallows exceptions, returning no results
    instead of failing the whole ensemble. Self-query construction is an LLM
    call that occasionally produces a malformed filter for an edge-case
    question — that should degrade to "no self-query hits this time", not
    break retrieval entirely."""

    inner: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        try:
            return self.inner.invoke(query)
        except Exception:
            return []


class ConditionalSelfQueryRetriever(BaseRetriever):
    """Skips the self-query LLM call entirely unless the question shares at
    least one meaningful word with a known source's filename or catalog
    description — see the module docstring for why that's a safe skip, not
    just a cheap one."""

    inner: BaseRetriever
    keywords: frozenset = frozenset()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.keywords and not (_tokenize(query) & self.keywords):
            return []
        return self.inner.invoke(query)


def build_self_query_retriever(store: Chroma) -> BaseRetriever | None:
    """Returns None if nothing has been ingested yet — there's no metadata to
    build filterable attributes from, and nothing to filter over anyway."""
    sources = _distinct_metadata_values(store, "source")
    if not sources:
        return None

    catalog = build_source_catalog(store)
    source_lines = "\n".join(f"- {s}: {catalog[s]}" if s in catalog else f"- {s}" for s in sources)

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description=(
                "The filename/URL the chunk came from. Available sources, with "
                f"what each covers:\n{source_lines}"
            ),
            type="string",
        ),
    ]

    sheets = _distinct_metadata_values(store, "sheet")
    if sheets:
        metadata_field_info.append(
            AttributeInfo(
                name="sheet",
                description=(
                    f"For rows from an Excel file, which sheet the row came from. "
                    f"One of: {sheets}. Only set for Excel-derived rows, not for "
                    f"PDFs/DOCX/CSV/JSON."
                ),
                type="string",
            )
        )

    # SelfQueryRetriever.from_llm's auto-detection (_get_builtin_translator)
    # eagerly imports ~20 vectorstore integrations to guess which one this
    # is; on the installed langchain-community version one of those
    # (DatabricksVectorSearch) isn't exported anymore, so the whole guess
    # fails even though we already know it's Chroma. Passing the translator
    # explicitly skips that broad, fragile import entirely.
    from langchain_community.query_constructors.chroma import ChromaTranslator

    retriever = SelfQueryRetriever.from_llm(
        llm=get_llm(temperature=0),
        vectorstore=store,
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        enable_limit=True,
    )

    keywords: set[str] = set()
    for source in sources:
        keywords |= _tokenize(source)
        if source in catalog:
            keywords |= _tokenize(catalog[source])
    for sheet in sheets:
        keywords |= _tokenize(sheet)

    return ConditionalSelfQueryRetriever(
        inner=SafeRetriever(inner=retriever), keywords=frozenset(keywords)
    )
