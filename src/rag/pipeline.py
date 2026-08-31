"""High-level RAG operations: ingest documents, run conversational queries."""
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.ingestion.chunking import split_documents
from src.ingestion.loaders import load_directory, load_prose_directory, load_urls
from src.ingestion.structured_loaders import load_sql_table
from src.rag.chain import build_conversational_rag_chain
from src.rag.parent_document import ingest_parent_documents
from src.rag.retrieval import build_hybrid_retriever
from src.rag.router import route_question
from src.rag.vectorstore import add_documents, get_vectorstore

_NOOP_STAGE: Callable[[str], None] = lambda stage: None


class _RetrievalDoneCallback(BaseCallbackHandler):
    """Fires exactly once, when ALL retrieval work has actually finished —
    including nested fan-out across sub-questions and Corrective RAG's
    relevance grading (DecomposingRetriever/CorrectiveRetriever/etc., see
    retrieval.py) — not just whichever retriever's on_retriever_end happens
    to fire first.

    Naive "first on_retriever_end = done" is wrong here: empirically
    verified (see scripts used during development) that when a custom
    BaseRetriever is invoked as part of a LangChain-composed chain — as
    opposed to called directly — nested custom-retriever calls it makes
    internally (DecomposingRetriever's fan-out, corrective's local/web
    calls) DO get the ambient callback manager propagated, so they fire
    their own on_retriever_start/end pairs too, interleaved with the
    outermost one. Depth-tracking is required to find the true "all done"
    moment: the transition back to depth zero."""

    def __init__(self, on_all_retrieval_done: Callable[[], None]):
        self._on_done = on_all_retrieval_done
        self._depth = 0
        self._lock = threading.Lock()

    def on_retriever_start(self, *args, **kwargs) -> None:
        with self._lock:
            self._depth += 1

    def on_retriever_end(self, *args, **kwargs) -> None:
        with self._lock:
            self._depth -= 1
            done = self._depth == 0
        if done:
            self._on_done()


def ingest_directory(directory: Path = config.RAW_DATA_DIR) -> int:
    """Load every supported file under `directory`, chunk, embed, and store.
    Also feeds prose files into the separate parent-document index (see
    parent_document.py) so it stays queryable alongside the main collection.
    Returns the number of chunks added to the main collection."""
    documents = load_directory(directory)
    ingest_parent_documents(load_prose_directory(directory))
    if not documents:
        return 0
    chunks = split_documents(documents)
    add_documents(chunks)
    return len(chunks)


def ingest_urls(urls: list[str]) -> int:
    documents = load_urls(urls)
    if not documents:
        return 0
    chunks = split_documents(documents)
    add_documents(chunks)
    return len(chunks)


def ingest_sql_table(connection_string: str, table: str, query: str | None = None) -> int:
    documents = load_sql_table(connection_string, table, query)
    if not documents:
        return 0
    chunks = split_documents(documents)
    add_documents(chunks)
    return len(chunks)


@dataclass
class ChatSession:
    """A conversational RAG session that retains history across turns."""

    history: list = field(default_factory=list)

    def __post_init__(self):
        store = get_vectorstore()
        retriever = build_hybrid_retriever(store)
        self.chain, self.document_chain = build_conversational_rag_chain(retriever)
        self._sql_chain = None  # built lazily — only if a question actually routes there

    def _get_sql_chain(self):
        if self._sql_chain is None:
            from src.rag.text_to_sql import build_text_to_sql_chain

            self._sql_chain = build_text_to_sql_chain()
        return self._sql_chain

    def _ask_vectorstore(
        self, question: str, on_stage: Callable[[str], None] = _NOOP_STAGE
    ) -> tuple[str, list, list]:
        """Self-correcting generation (Self-RAG/RRR-style): grades the
        ANSWER, not just what was retrieved — distinct from Corrective RAG,
        which only grades retrieval quality before generation happens. A
        hallucinating answer gets regenerated from the same context with
        explicit feedback (re-retrieving would just return the same,
        deterministic context and likely the same answer); an answer that
        doesn't actually address the question triggers a question rewrite
        and a fresh retrieval pass instead. Both loops are bounded.

        on_stage reports real progress as the pipeline actually moves
        through it (see api/main.py's streaming endpoint) — not a fake
        client-side timer cycling through canned labels."""
        from src.rag import self_correction

        current_question = question
        answer, context = "", []

        for rewrite_attempt in range(config.MAX_QUESTION_REWRITES + 1):
            on_stage("retrieving")
            retrieval_done = _RetrievalDoneCallback(lambda: on_stage("generating"))
            result = self.chain.invoke(
                {"input": current_question, "chat_history": self.history},
                config={"callbacks": [retrieval_done]},
            )
            answer, context = result["answer"], result["context"]
            context_text = "\n\n".join(doc.page_content for doc in context)

            # Grade against the ORIGINAL question — a rewrite is a retrieval
            # aid, not a redefinition of what the user actually asked.
            on_stage("verifying")
            grade = self_correction.grade_answer(question, context_text, answer)

            for _ in range(config.MAX_HALLUCINATION_RETRIES):
                if not grade.hallucinating:
                    break
                on_stage("regenerating")
                answer = self.document_chain.invoke(
                    {
                        "input": (
                            f"{current_question}\n\n(Your previous answer included "
                            "claims not supported by the context below. Regenerate "
                            "using ONLY facts explicitly stated in the context.)"
                        ),
                        "context": context,
                        "chat_history": self.history,
                    }
                )
                # Re-grade every regeneration rather than trusting it blindly.
                # Grading before the loop and again after each retry is what
                # makes a plain range(N) correct here — an earlier version
                # graded inside the loop and needed an awkward +1 to avoid
                # leaving the final regeneration unverified.
                on_stage("verifying")
                grade = self_correction.grade_answer(question, context_text, answer)

            if grade.answers_question:
                break
            if rewrite_attempt < config.MAX_QUESTION_REWRITES:
                on_stage("rewriting_question")
                current_question = self_correction.rewrite_question(current_question)

        sources = sorted({doc.metadata.get("source", "unknown") for doc in context})
        return answer, context, sources

    def ask(self, question: str, on_stage: Callable[[str], None] = _NOOP_STAGE) -> dict:
        on_stage("routing")
        decision = route_question(question)
        route = decision.destination

        if decision.destination == "sql":
            on_stage("querying_database")
            try:
                sql_result = self._get_sql_chain().invoke({"question": question})
                answer = sql_result["answer"]
                context = []
                sources = [f"SQL: {sql_result['query']}"]
            except Exception:
                # An LLM-generated query can legitimately fail — malformed
                # SQL, a rejected unsafe statement, a real DB error — and
                # none of those should crash the whole session. Fall back to
                # the RAG path for this same question instead (real bug,
                # found and fixed: see README's Text-to-SQL notes).
                answer, context, sources = self._ask_vectorstore(question, on_stage)
                route = "vectorstore"
        else:
            answer, context, sources = self._ask_vectorstore(question, on_stage)

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "route": route,
        }
