"""Pure-logic pipeline helpers. No API key needed — deliberately kept out
of test_pipeline.py (module-skipped without GOOGLE_API_KEY) so these always
run, including in CI."""
from langchain_core.documents import Document

from src.rag.pipeline import _cited_sources

DOC_CITED = Document(page_content="cited content", metadata={"source": "cited.pdf"})
DOC_UNCITED = Document(page_content="uncited content", metadata={"source": "uncited.pdf"})


def test_cited_sources_returns_only_what_the_answer_actually_cites():
    """Real bug this fixes: sources used to be every document retrieved,
    not just what the answer cited — a broad step-back sub-question could
    pull in an irrelevant source the model correctly never cited, and it
    would still show up in `sources`. See ENGINEERING.md's Corrective RAG
    section for the real example that surfaced this."""
    answer = "The answer is X [source: cited.pdf]."
    context = [DOC_CITED, DOC_UNCITED]

    assert _cited_sources(answer, context) == ["cited.pdf"]


def test_cited_sources_handles_multiple_citations():
    answer = "First claim [source: cited.pdf]. Second claim [source: other.pdf]."
    context = [
        DOC_CITED,
        Document(page_content="other content", metadata={"source": "other.pdf"}),
        DOC_UNCITED,
    ]

    assert _cited_sources(answer, context) == ["cited.pdf", "other.pdf"]


def test_cited_sources_dedupes_repeated_citations_to_the_same_source():
    answer = "First claim [source: cited.pdf]. Second claim [source: cited.pdf]."
    assert _cited_sources(answer, [DOC_CITED]) == ["cited.pdf"]


def test_cited_sources_returns_empty_when_nothing_is_cited():
    """An honest "the context doesn't cover this" answer legitimately cites
    nothing — reporting the retrieved-but-unhelpful sources anyway would be
    exactly the bug being fixed here, not a safe fallback."""
    answer = "The context does not contain information about that."
    assert _cited_sources(answer, [DOC_CITED, DOC_UNCITED]) == []


def test_cited_sources_ignores_a_tag_that_does_not_match_any_retrieved_source():
    """Guards against a malformed/hallucinated tag — the self-correction
    hallucination check should already catch this, but this shouldn't have
    to trust that blindly to stay correct."""
    answer = "The answer is X [source: made-up-source.pdf]."
    assert _cited_sources(answer, [DOC_CITED]) == []


def test_cited_sources_accepts_a_bare_bracket_without_the_source_prefix():
    """Real case, found running the expanded RAGAS eval: the model doesn't
    always keep the "source:" prefix when citing a web URL, sometimes
    dropping straight to `[https://...]`. Rejecting that outright would
    silently under-report real citations."""
    doc = Document(page_content="web content", metadata={"source": "https://example.com/page"})
    answer = "The answer is X [https://example.com/page]."
    assert _cited_sources(answer, [doc]) == ["https://example.com/page"]
