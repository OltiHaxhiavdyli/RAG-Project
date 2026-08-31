"""Evaluate the RAG pipeline with RAGAS metrics: faithfulness, answer
relevancy, context precision, and context recall.

Usage:
    python scripts/evaluate.py [--dataset scripts/eval_dataset.json]
"""
import argparse
import json
import sys
import threading
import types
from pathlib import Path

# Generous ceiling, not a target: a normal question takes ~30-45s, and the
# self-correction loop can legitimately push a slow one well past that
# (regenerate + re-verify, or a question rewrite triggering fresh
# retrieval). This only exists to break a genuinely hung network call.
QUESTION_TIMEOUT_SECONDS = 300

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A real, current bug: ragas 0.4.3 (latest as of writing) unconditionally
# imports `langchain_community.chat_models.vertexai.ChatVertexAI` at module
# load time, just to list it in a static isinstance-check tuple (whether an
# LLM supports multi-completion sampling — a feature this project doesn't
# use). langchain-community removed that module when it "sunset" its Vertex
# AI integration into the standalone langchain-google-vertexai package (the
# one this project actually uses), so `import ragas` fails outright before
# any of our own code runs. Since the class is never instantiated — only
# checked against with isinstance(), and our real LLM is a different class
# from a different package that will never match it anyway — a harmless
# placeholder is enough to let ragas import successfully.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    shim = types.ModuleType("langchain_community.chat_models.vertexai")
    shim.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = shim

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from src.rag.chain import get_llm
from src.rag.pipeline import ChatSession
from src.rag.vectorstore import collection_count, get_embeddings


def _ask_with_timeout(session: ChatSession, question: str, timeout: float) -> dict:
    """Run session.ask() with a real timeout on a genuinely hung call.

    A ThreadPoolExecutor's `with` block calls shutdown(wait=True) on exit,
    which blocks the main thread until the worker actually finishes — so a
    truly hung call defeats the timeout entirely, just moving the hang to
    the next line instead of `.result()`. A fixed-size pool compounds this:
    with max_workers=1, one permanently-stuck worker starves every later
    question too. A fresh daemon thread per call sidesteps both: `join()`
    with a timeout returns even if the thread never finishes, and daemon
    threads never block interpreter exit, so an abandoned hung call is
    truly abandoned, not just deferred.
    """
    box: dict = {}

    def worker() -> None:
        try:
            box["result"] = session.ask(question)
        except Exception as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"question timed out after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_eval(dataset_path: Path) -> None:
    if collection_count() == 0:
        print("Vector store is empty. Run `python cli.py ingest` first.")
        return

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    # ONE session, history cleared between questions — not a fresh session per
    # question. Both give the same isolation (no cross-question memory), but a
    # fresh session re-pays the full ~13-90s cold build every time: loading
    # every document for the BM25 index, the cross-encoder, the source
    # catalog. That's the same measurement artifact already fixed in the
    # latency benchmarks (see ENGINEERING.md's Performance section), which
    # this script had independently reintroduced. It made a 16-question run
    # take 30+ minutes, which in turn made expanding the dataset — the actual
    # fix for the metrics being too noisy to interpret — impractically slow.
    session = ChatSession()

    # One flaky question must not discard the whole run. Found the hard way:
    # a transient httpx.RemoteProtocolError on question 40 of 40 threw away
    # ~20 minutes of completed API work, which is exactly the failure mode
    # that makes larger eval sets impractical — and larger sets are the whole
    # point of tightening the confidence intervals below. Matches the
    # "skip, don't crash" posture the rest of this project already has
    # (SafeRetriever, the SQL-path fallback, the corrective/scope-gate
    # fallbacks); this script was the one place that didn't.
    rows, failed = [], []
    for i, case in enumerate(cases, 1):
        session.history.clear()
        try:
            # Run in a worker so a HUNG call is survivable, not just a
            # failing one. A raised exception is catchable; a network call
            # that never returns is not, and both were hit for real running
            # this exact script — one run died on a RemoteProtocolError, a
            # later one simply stopped advancing at question 21 and never
            # resumed. Without a timeout the second case can't be recovered
            # from at all, which makes a large eval set impossible to
            # actually finish.
            result = _ask_with_timeout(session, case["question"], QUESTION_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 — any transport/API/timeout failure
            failed.append((case["question"], type(exc).__name__))
            print(
                f"  [{i}/{len(cases)}] SKIPPED ({type(exc).__name__}): {case['question'][:50]}",
                flush=True,
            )
            continue
        print(f"  [{i}/{len(cases)}] {case['question'][:60]}", flush=True)
        rows.append(
            {
                "user_input": case["question"],
                "response": result["answer"],
                "retrieved_contexts": [doc.page_content for doc in result["context"]],
                "reference": case["ground_truth"],
            }
        )

    if failed:
        # Reported loudly, never silently: a quietly-dropped row is how an
        # earlier run of this script reported 7 scores for 8 questions.
        print(f"\n!! {len(failed)} of {len(cases)} questions failed and are EXCLUDED:")
        for question, exc_name in failed:
            print(f"   - [{exc_name}] {question[:70]}")
        print("   Scores below cover only the questions that completed.")

    if not rows:
        print("\nEvery question failed — nothing to score.")
        return

    eval_dataset = EvaluationDataset.from_list(rows)

    llm = LangchainLLMWrapper(get_llm(temperature=0))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # RAGAS defaults to 16 concurrent workers, which is enough simultaneous
    # load against Vertex AI's quota to trigger 429 ResourceExhausted — a
    # real failure seen running this: retries+backoff under that much
    # concurrent contention compounded past even the (already generous)
    # 180s default per-job timeout, and that job's whole row silently
    # dropped from the final report. Lower concurrency, not a higher
    # timeout ceiling, is the actual fix — there's less to retry in the
    # first place.
    run_config = RunConfig(max_workers=4, timeout=240)

    report = evaluate(
        eval_dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    df = report.to_pandas()
    _print_scores(df)


METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _print_scores(df) -> None:
    """Report mean +- 95% confidence interval, not a bare mean.

    Bare means are how this project spent two runs unable to say whether
    faithfulness 0.93 -> 0.86 meant anything. With per-question scores in
    hand the answer is computable: the standard error of the mean is
    std/sqrt(n), so a shift smaller than roughly +-1.96*SEM is
    indistinguishable from which questions happened to land which way. The
    "min detectable shift" column makes that explicit, so a future run can
    be judged instead of squinted at. Also prints n, since all of this
    tightens with more questions and that's the real lever."""
    n = len(df)
    print(f"\nScores over {n} questions (mean +- 95% CI):\n")
    print(f"{'metric':<20} {'mean':>6}  {'95% CI':>16}  {'min detectable shift':>21}")
    print("-" * 70)
    for col in METRIC_COLUMNS:
        scores = df[col].dropna()
        mean = scores.mean()
        # ddof=1: sample std, not population — these questions are a sample
        # of possible questions, not the whole universe of them.
        sem = scores.std(ddof=1) / (len(scores) ** 0.5) if len(scores) > 1 else float("nan")
        half = 1.96 * sem
        print(
            f"{col:<20} {mean:>6.3f}  [{max(0.0, mean - half):.3f}, {min(1.0, mean + half):.3f}]"
            f"  {half:>19.3f}"
        )
    print(
        "\nA run-to-run change smaller than the last column is not evidence of "
        "anything.\nShrink it by adding questions: the interval narrows with sqrt(n)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scripts/eval_dataset.json")
    args = parser.parse_args()
    run_eval(Path(args.dataset))
