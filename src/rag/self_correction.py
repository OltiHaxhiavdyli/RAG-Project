"""Self-correcting generation (Self-RAG / RRR-style "active retrieval"):
grades the GENERATED ANSWER itself, not the retrieved documents — a
different mechanism from Corrective RAG (corrective.py), which only grades
retrieval quality before generation ever happens. Two checks, mirroring the
canonical LangGraph Self-RAG flow:

1. Hallucination check — is the answer actually grounded in the retrieved
   context? If not, regenerate from the SAME context with explicit
   feedback, rather than re-retrieving (retrieval is deterministic here, so
   re-running it with the same question would just return the same context
   and likely the same answer — the fix has to happen at generation time).
2. Answers-the-question check — does the answer actually address what was
   asked (an honest "the context doesn't cover this" counts as answering)?
   If not, rewrite the question and retry the FULL pipeline (fresh
   retrieval), since the problem here is more likely retrieval missing the
   right content for how the question was originally phrased.

Both loops are bounded (MAX_HALLUCINATION_RETRIES, MAX_QUESTION_REWRITES) —
a persistently bad answer degrades to "best effort so far", not an infinite
loop.
"""
from pydantic import BaseModel, Field

from src.rag.chain import get_llm

ANSWERS_QUESTION_PROMPT = """Does the answer below actually address and \
engage with the question — even if it honestly says the context doesn't \
contain the information? An honest "I don't know" or "the context doesn't \
cover this" DOES count as answering. Say NO only if the answer is \
off-topic, evasive, or fails to engage with what was actually asked.

Question: {question}

Answer:
{answer}"""

COMBINED_GRADE_PROMPT = """Grade the answer below on two INDEPENDENT criteria.

1. hallucinating — does the answer contain any factual claim NOT supported \
by the context? Be strict: every claim must be traceable to the context. A \
claim explicitly stating "the context doesn't cover this" is NOT a \
hallucination.
2. answers_question — does the answer actually address and engage with the \
question? An honest "I don't know" or "the context doesn't cover this" DOES \
count as answering. Mark false only if the answer is off-topic, evasive, or \
fails to engage with what was asked.

Question: {question}

Context:
{context}

Answer:
{answer}"""

REWRITE_PROMPT = """The question below was not well resolved by the \
documents retrieved for it. Rewrite it to be clearer or more specific, in a \
way more likely to retrieve the content that actually answers it. Return \
ONLY the rewritten question, nothing else.

Original question: {question}"""


class AnswersQuestionGrade(BaseModel):
    answers_question: bool = Field(description="True if the answer actually addresses the question.")


class AnswerGrade(BaseModel):
    """Both generation-quality checks in one structured response."""

    hallucinating: bool = Field(description="True if the answer has claims unsupported by context.")
    answers_question: bool = Field(description="True if the answer actually addresses the question.")


class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")


def answers_the_question(question: str, answer: str) -> bool:
    llm = get_llm(temperature=0)
    grade = llm.with_structured_output(AnswersQuestionGrade).invoke(
        ANSWERS_QUESTION_PROMPT.format(question=question, answer=answer)
    )
    return grade.answers_question


def grade_answer(question: str, context: str, answer: str) -> AnswerGrade:
    """Both self-correction checks in a SINGLE LLM call. They grade the same
    (question, context, answer) triple, so running them as two sequential
    round trips was pure latency for no extra signal — the context, which is
    the expensive part of the prompt, was being sent twice either way."""
    if not context:
        # Nothing to be unfaithful to, but "does it answer?" still applies.
        return AnswerGrade(
            hallucinating=False, answers_question=answers_the_question(question, answer)
        )

    llm = get_llm(temperature=0)
    return llm.with_structured_output(AnswerGrade).invoke(
        COMBINED_GRADE_PROMPT.format(question=question, context=context[:6000], answer=answer)
    )


def rewrite_question(question: str) -> str:
    llm = get_llm(temperature=0)
    result = llm.with_structured_output(RewrittenQuestion).invoke(
        REWRITE_PROMPT.format(question=question)
    )
    return result.question.strip()
