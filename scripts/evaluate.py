"""Evaluate the RAG pipeline with RAGAS metrics: faithfulness, answer
relevancy, context precision, and context recall.

Usage:
    python scripts/evaluate.py [--dataset scripts/eval_dataset.json]
"""
import argparse
import json
import sys
import types
from pathlib import Path

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


def run_eval(dataset_path: Path) -> None:
    if collection_count() == 0:
        print("Vector store is empty. Run `python cli.py ingest` first.")
        return

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        session = ChatSession()  # fresh session per question: no cross-question memory leakage
        result = session.ask(case["question"])
        rows.append(
            {
                "user_input": case["question"],
                "response": result["answer"],
                "retrieved_contexts": [doc.page_content for doc in result["context"]],
                "reference": case["ground_truth"],
            }
        )

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
    print(df.to_string(index=False))
    print("\nMean scores:")
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scripts/eval_dataset.json")
    args = parser.parse_args()
    run_eval(Path(args.dataset))
