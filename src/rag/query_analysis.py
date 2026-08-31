"""Query analysis: transform the question before retrieving, rather than
always retrieving on the literal text as asked.

- Decomposition: split a compound question into its atomic parts, so a
  question like "what's the drop deadline, and what's the mission
  statement?" doesn't only get answered for whichever half a single
  retrieval pass happened to favor.
- Step-back prompting: also generate ONE broader/more general version of the
  question (e.g. "what's the deadline to drop ACCT 110?" -> "what are the
  course drop deadlines?"), so a narrow question that happens to use
  different wording than the source document still surfaces the general
  policy context around it, not just an exact-phrasing match (or nothing).

Every (sub-)question generated here is retrieved independently through
`base_retriever` (which reranks per-query — see retrieval.py), and the
results are pooled and deduplicated.

Both are gated by needs_decomposition() first — a cheap upfront check, same
pattern as scope_gate.py: for an already-simple, self-contained question,
decompose+step_back would just add two LLM calls and double the downstream
retrieval/grading fan-out for no benefit (decompose's own atomicity check
already collapses to [question] in that case, but step-back still ran
regardless before this gate existed). Conservative like scope_gate: default
to YES (run decomposition) whenever unsure, since wrongly skipping it for a
question that needed it is a real answer-quality regression, while running
it unnecessarily only costs latency."""
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from src import config
from src.rag.chain import get_llm

DECOMPOSE_PROMPT = """Break the user's question into the minimal set of \
atomic sub-questions needed to answer it fully.

- If it's already a single, atomic question, return a list containing just \
that one question, unchanged.
- Only split on genuine distinct asks (e.g. "X, and also Y") — don't invent \
sub-questions that weren't implied.
- Return at most {max_questions} sub-questions.

Question: {question}"""

STEP_BACK_PROMPT = """Write ONE broader, more general version of the \
question below — the kind of question whose answer would provide useful \
background context for the specific one asked.

Example: "What's the deadline to drop ACCT 110 with a W?" -> "What are the \
course withdrawal deadlines and policies?"

If the question is already general, return it unchanged.

Question: {question}"""

NEEDS_DECOMPOSITION_PROMPT = """Question: {question}

Decide whether this question would benefit from EITHER:
(a) being split into sub-questions — only true if it asks about more than \
one clearly distinct thing (e.g. "X, and also Y"); or
(b) also retrieving a broader/more general version of it — only true if it \
names a specific, narrow entity (a particular course code, program, event, \
or named policy) whose answer more likely lives inside a broader general \
policy section than under that exact narrow phrasing.

Most simple, direct factual questions ("what is the tuition fee", "when \
does the semester start", "what is the mission statement") need NEITHER —  \
say NO for those. Only say YES when (a) or (b) genuinely applies, not just \
because more context could theoretically help in some general sense.
Default to YES only when genuinely unsure which way it leans, not as a \
fallback reasoning that more context never hurts."""


class SubQuestions(BaseModel):
    questions: List[str] = Field(description="Atomic sub-questions, in order.")


class StepBackQuestion(BaseModel):
    step_back_question: str = Field(description="A broader/more general version of the question.")


class DecompositionDecision(BaseModel):
    needed: bool = Field(
        description="True unless the question is clearly simple enough that "
        "decomposition/step-back would add nothing."
    )


def needs_decomposition(question: str) -> bool:
    llm = get_llm(temperature=0)
    decision = llm.with_structured_output(DecompositionDecision).invoke(
        NEEDS_DECOMPOSITION_PROMPT.format(question=question)
    )
    return decision.needed


def decompose(question: str) -> list[str]:
    llm = get_llm(temperature=0)
    structured = llm.with_structured_output(SubQuestions)
    result = structured.invoke(
        DECOMPOSE_PROMPT.format(question=question, max_questions=config.MAX_SUBQUESTIONS)
    )
    questions = [q.strip() for q in result.questions if q.strip()][: config.MAX_SUBQUESTIONS]
    return questions or [question]


def step_back(question: str) -> str:
    llm = get_llm(temperature=0)
    structured = llm.with_structured_output(StepBackQuestion)
    result = structured.invoke(STEP_BACK_PROMPT.format(question=question))
    return result.step_back_question.strip()


def _dedup_key(doc: Document) -> tuple:
    return (doc.metadata.get("source", ""), doc.page_content[:200])


class DecomposingRetriever(BaseRetriever):
    """Wraps a base retriever: decomposes the query into atomic sub-questions,
    adds one step-back (broader) question, retrieves each independently
    through base_retriever, and returns the deduplicated union of all
    results. base_retriever is expected to already rerank per-query it's
    given (see retrieval.py) — a compound/transformed question reranked once
    as a whole systematically starves out whichever part is less prominent
    in the combined wording, which is why each (sub-)question here gets
    retrieved as its own call, not merged into one.

    Independent work runs concurrently rather than sequentially: decompose
    and step_back don't depend on each other, and neither does retrieval for
    one sub-question depend on another's. Sequentially these dominated
    latency (measured: ~24s of a ~51s query was serialized grader/analysis
    calls). Order of the final merged list is kept deterministic regardless
    of which thread finishes first — dedup order matters for what survives
    into the top-N context."""

    base_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if not needs_decomposition(query):
            return self.base_retriever.invoke(query)

        with ThreadPoolExecutor(max_workers=2) as pool:
            decompose_future = pool.submit(decompose, query)
            step_back_future = pool.submit(step_back, query)
            queries = decompose_future.result()
            broader = step_back_future.result()

        normalized_existing = {q.strip().lower() for q in queries} | {query.strip().lower()}
        if broader.strip().lower() not in normalized_existing:
            queries.append(broader)

        # .batch() runs the full per-query chain (ensemble incl. conditional
        # self-query → rerank → relevance grading) concurrently across
        # sub-questions, but returns results in input order — so merge/dedup
        # below stays deterministic.
        per_query_results = self.base_retriever.batch(
            queries, config={"max_concurrency": config.RETRIEVAL_MAX_CONCURRENCY}
        )

        seen: set[tuple] = set()
        merged: list[Document] = []
        for docs in per_query_results:
            for doc in docs:
                key = _dedup_key(doc)
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged
