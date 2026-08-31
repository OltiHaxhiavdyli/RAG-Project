"""Central configuration, loaded from environment variables / .env."""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# "genai"    = Google AI Studio (simple API key, GOOGLE_API_KEY)
# "vertexai" = Vertex AI (GCP project + service-account credentials, billable
#              against Cloud billing / a student credit rather than the free tier)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "genai").lower()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
# GOOGLE_APPLICATION_CREDENTIALS is read directly by google-auth (Application
# Default Credentials) once load_dotenv() puts it in the process environment,
# so it doesn't need its own config variable here.

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-flash-latest")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")

CHROMA_PERSIST_DIR = str(PROJECT_ROOT / os.environ.get("CHROMA_PERSIST_DIR", ".chroma"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag_docs")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))

RETRIEVER_TOP_K = int(os.environ.get("RETRIEVER_TOP_K", 20))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", 5))

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
WEB_SEARCH_ENABLED = os.environ.get("WEB_SEARCH_ENABLED", "true").lower() == "true" and bool(TAVILY_API_KEY)
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", 5))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SQL_DB_PATH = PROJECT_ROOT / os.environ.get("SQL_DB_PATH", "data/structured.db")
SQL_QUERY_ROW_LIMIT = int(os.environ.get("SQL_QUERY_ROW_LIMIT", 200))

MAX_SUBQUESTIONS = int(os.environ.get("MAX_SUBQUESTIONS", 4))
CRAG_MIN_RELEVANT_DOCS = int(os.environ.get("CRAG_MIN_RELEVANT_DOCS", 1))

# Parent-document retrieval — separate collection + a persisted docstore for
# the parent chunks, since child chunks carry different metadata (doc_id)
# than the rest of the pipeline's collection assumes (source/sheet/etc.).
PARENT_CHILD_COLLECTION_NAME = f"{COLLECTION_NAME}_parent_child"
PARENT_DOCSTORE_DIR = str(PROJECT_ROOT / os.environ.get("PARENT_DOCSTORE_DIR", ".parent_docstore"))
PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE", 2000))
PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP", 200))
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE", 400))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP", 40))
PARENT_RETRIEVER_TOP_K = int(os.environ.get("PARENT_RETRIEVER_TOP_K", 4))

# Source catalog: one LLM-generated sentence per ingested source, persisted
# so it's only generated once. Used to (a) enrich self-query's filtering with
# more than a bare filename list, and (b) gate whether a question is even
# plausibly answerable locally before spending a full retrieval pass.
SOURCE_CATALOG_PATH = str(PROJECT_ROOT / os.environ.get("SOURCE_CATALOG_PATH", ".source_catalog.json"))

# Self-correction (Self-RAG/RRR-style): grades GENERATION quality — is the
# answer grounded in context, does it actually address the question — as
# distinct from Corrective RAG, which only grades RETRIEVAL quality before
# generation ever happens. Bounded retries so a persistently bad answer
# degrades to "best effort", not an infinite loop.
MAX_HALLUCINATION_RETRIES = int(os.environ.get("MAX_HALLUCINATION_RETRIES", 1))
MAX_QUESTION_REWRITES = int(os.environ.get("MAX_QUESTION_REWRITES", 1))

# How many sub-questions retrieve concurrently. Deliberately modest: the whole
# per-question chain (scope gate, self-query, relevance grading) fires several
# LLM calls each, and Vertex AI returned 429 ResourceExhausted under heavy
# parallelism during eval runs. Small enough to stay well inside quota, large
# enough to stop sub-questions serializing behind each other.
RETRIEVAL_MAX_CONCURRENCY = int(os.environ.get("RETRIEVAL_MAX_CONCURRENCY", 4))


def require_credentials() -> None:
    if LLM_PROVIDER == "vertexai":
        if not VERTEX_PROJECT_ID:
            raise RuntimeError("VERTEX_PROJECT_ID is not set in .env.")
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set in .env. Point it at "
                "your downloaded service-account JSON key."
            )
    elif not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "Gemini API key from https://aistudio.google.com/apikey"
        )
