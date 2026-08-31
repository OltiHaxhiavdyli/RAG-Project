"""End-to-end pipeline tests. Require GOOGLE_API_KEY — skipped otherwise.

Ingests tests/fixtures/, NOT data/raw/ — data/raw holds whatever real (and
possibly large/private) documents the project owner has actually loaded,
and re-embedding those on every test run would be slow and cost real API
credits."""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_ingest_and_query(tmp_path, monkeypatch):
    # Config, vectorstore, and pipeline all read settings at import time, so
    # force a reload after patching the env to keep this test's store isolated
    # from the real .chroma directory used by the CLI/API.
    import importlib

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("PARENT_DOCSTORE_DIR", str(tmp_path / ".parent_docstore"))
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    from src import config

    importlib.reload(config)

    from src.rag import pipeline, vectorstore

    importlib.reload(vectorstore)
    importlib.reload(pipeline)

    added = pipeline.ingest_directory(FIXTURES_DIR)
    assert added > 0

    session = pipeline.ChatSession()
    result = session.ask("How many business days does a refund take to process?")
    assert "5" in result["answer"]
    assert result["sources"]


def test_sql_failure_falls_back_to_vectorstore_gracefully(tmp_path, monkeypatch):
    """Regression test for a real crash: a malformed/rejected SQL query used
    to propagate all the way up and kill the whole session instead of
    degrading gracefully. See README's Text-to-SQL section."""
    import importlib

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "test_collection_fallback")
    monkeypatch.setenv("PARENT_DOCSTORE_DIR", str(tmp_path / ".parent_docstore"))
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    from src import config

    importlib.reload(config)

    from src.rag import pipeline, vectorstore

    importlib.reload(vectorstore)
    importlib.reload(pipeline)

    pipeline.ingest_directory(FIXTURES_DIR)
    session = pipeline.ChatSession()

    class RaisingSQLChain:
        def invoke(self, *_args, **_kwargs):
            raise ValueError("simulated malformed/rejected SQL")

    monkeypatch.setattr(session, "_get_sql_chain", lambda: RaisingSQLChain())

    class ForcedSQLDecision:
        destination = "sql"

    monkeypatch.setattr(pipeline, "route_question", lambda question: ForcedSQLDecision())

    result = session.ask("How many business days does a refund take to process?")
    assert result["route"] == "vectorstore"  # fell back, didn't crash
    assert "5" in result["answer"]


def test_ask_reports_real_progress_stages(tmp_path, monkeypatch):
    """The web UI shows real per-stage progress (routing -> retrieving ->
    generating -> verifying), not a fake client-side timer cycling through
    canned labels. Verifies on_stage actually fires, in order, at the real
    transitions — including "generating", which depends on
    _RetrievalDoneCallback correctly detecting when ALL retrieval (including
    nested fan-out across sub-questions) has finished, not just whichever
    retriever's on_retriever_end happens to fire first. No self-correction
    retries here (grade_answer always passes clean) so the sequence is the
    minimal, common-case path."""
    import importlib

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "test_collection_progress")
    monkeypatch.setenv("PARENT_DOCSTORE_DIR", str(tmp_path / ".parent_docstore"))
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    from src import config

    importlib.reload(config)

    from src.rag import pipeline, vectorstore

    importlib.reload(vectorstore)
    importlib.reload(pipeline)

    pipeline.ingest_directory(FIXTURES_DIR)
    session = pipeline.ChatSession()

    from src.rag import self_correction
    from src.rag.self_correction import AnswerGrade

    monkeypatch.setattr(
        self_correction,
        "grade_answer",
        lambda question, context, answer: AnswerGrade(hallucinating=False, answers_question=True),
    )

    stages = []
    result = session.ask("How long do refunds take to process?", on_stage=stages.append)

    assert stages == ["routing", "retrieving", "generating", "verifying"]
    assert "5" in result["answer"]


def test_self_correction_regenerates_on_hallucination(tmp_path, monkeypatch):
    """Verifies the hallucination-retry path actually calls document_chain
    (regenerating from the SAME context) rather than re-retrieving, and that
    it stops as soon as a regeneration passes the check."""
    import importlib

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "test_collection_hallucination")
    monkeypatch.setenv("PARENT_DOCSTORE_DIR", str(tmp_path / ".parent_docstore"))
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    from src import config

    importlib.reload(config)

    from src.rag import pipeline, vectorstore

    importlib.reload(vectorstore)
    importlib.reload(pipeline)

    pipeline.ingest_directory(FIXTURES_DIR)
    session = pipeline.ChatSession()

    from src.rag import self_correction
    from src.rag.self_correction import AnswerGrade

    grades = []

    def fake_grade(question, context, answer):
        grades.append(answer)
        # Hallucinating on the first grade only; the regeneration passes.
        return AnswerGrade(hallucinating=len(grades) == 1, answers_question=True)

    monkeypatch.setattr(self_correction, "grade_answer", fake_grade)

    def _fail_if_rewrite_called(question):
        raise AssertionError("rewrite_question should not be called for a pure hallucination retry")

    monkeypatch.setattr(self_correction, "rewrite_question", _fail_if_rewrite_called)

    # Spy on document_chain to prove regeneration reuses the ALREADY-RETRIEVED
    # context rather than re-running retrieval. Asserting on the answer text
    # changing would be wrong: at near-zero temperature with identical context,
    # a correct regeneration legitimately reproduces the same wording.
    # The chain itself is a pydantic Runnable (can't patch its methods), but
    # ChatSession is a plain dataclass, so swap the whole attribute instead.
    regenerations = []

    class SpyingDocumentChain:
        def __init__(self, inner):
            self._inner = inner

        def invoke(self, payload, *a, **kw):
            regenerations.append(payload)
            return self._inner.invoke(payload, *a, **kw)

    monkeypatch.setattr(session, "document_chain", SpyingDocumentChain(session.document_chain))

    result = session.ask("How long do refunds take to process?")

    # Graded once up front, regenerated once, then re-graded — the
    # regeneration must never be trusted without re-verification.
    assert len(grades) == 2
    assert len(regenerations) == 1  # regenerated exactly once
    assert regenerations[0]["context"], "regeneration must reuse the retrieved context"
    assert "5" in result["answer"]


def test_self_correction_rewrites_question_when_answer_insufficient(tmp_path, monkeypatch):
    """Verifies the answers-the-question check triggers a question rewrite
    and a fresh retrieval pass, bounded by MAX_QUESTION_REWRITES."""
    import importlib

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "test_collection_rewrite")
    monkeypatch.setenv("PARENT_DOCSTORE_DIR", str(tmp_path / ".parent_docstore"))
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")

    from src import config

    importlib.reload(config)

    from src.rag import pipeline, vectorstore

    importlib.reload(vectorstore)
    importlib.reload(pipeline)

    pipeline.ingest_directory(FIXTURES_DIR)
    session = pipeline.ChatSession()

    from src.rag import self_correction
    from src.rag.self_correction import AnswerGrade

    grades = []

    def fake_grade(question, context, answer):
        grades.append(answer)
        # Never hallucinating; the FIRST answer is judged not to address the
        # question, forcing a rewrite + fresh retrieval. The second passes.
        return AnswerGrade(hallucinating=False, answers_question=len(grades) > 1)

    monkeypatch.setattr(self_correction, "grade_answer", fake_grade)

    rewrites = []
    monkeypatch.setattr(
        self_correction,
        "rewrite_question",
        lambda question: rewrites.append(question) or "How many business days for a refund?",
    )

    result = session.ask("How long do refunds take to process?")

    assert len(grades) == 2  # first graded insufficient, second (post-rewrite) passed
    assert rewrites == ["How long do refunds take to process?"]  # rewrote the original
    assert "5" in result["answer"]
