"""Self-correction tests. No API key needed — these test the guard logic
(skip the expensive grounding check when there's no context to ground
against), not the LLM's judgment."""
from src.rag import self_correction
from src.rag.self_correction import grade_answer


def test_grade_answer_skips_grounding_check_without_context(monkeypatch):
    """With no retrieved context there's nothing to be unfaithful to, so the
    combined grader must not spend a call on the grounding check — but
    "does it answer the question?" still applies and must still run."""
    monkeypatch.setattr(
        self_correction,
        "get_llm",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("the combined grounding+answers grader should not run without context")
        ),
    )
    monkeypatch.setattr(self_correction, "answers_the_question", lambda question, answer: True)

    grade = grade_answer("some question", "", "some answer")
    assert grade.hallucinating is False
    assert grade.answers_question is True
