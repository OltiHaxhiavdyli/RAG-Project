"""FastAPI backend exposing the RAG pipeline over HTTP, and serving the
static chat UI (static/) at the same origin — so the UI needs no base-URL
config or CORS setup, and the whole thing is one deployable service.

Run with:
    uvicorn api.main:app --reload
"""
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rag.pipeline import ChatSession, ingest_directory, ingest_sql_table, ingest_urls
from src.rag.router import sql_db_available
from src.rag.vectorstore import collection_count

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="RAG API", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_sessions: dict[str, ChatSession] = {}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class IngestRequest(BaseModel):
    urls: list[str] = []
    sql_conn: str | None = None
    sql_table: str | None = None
    sql_query: str | None = None


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    route: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": collection_count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    added = 0
    if req.urls:
        added += ingest_urls(req.urls)
    if req.sql_conn and req.sql_table:
        added += ingest_sql_table(req.sql_conn, req.sql_table, req.sql_query)
    added += ingest_directory()
    return IngestResponse(chunks_added=added, total_chunks=collection_count())


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if collection_count() == 0 and not sql_db_available():
        raise HTTPException(400, "Nothing indexed yet. Call /ingest or build-sql-db first.")

    session_id = req.session_id or str(uuid.uuid4())
    session = _sessions.get(session_id)
    if session is None:
        session = ChatSession()
        _sessions[session_id] = session

    result = session.ask(req.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id,
        route=result["route"],
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Same as /query, but streams real progress as newline-delimited JSON
    while the pipeline runs, instead of one JSON object after the full
    30-90s wait. Each line is either {"stage": "..."} — a real transition
    reported by ChatSession.ask's on_stage callback (see pipeline.py), not
    a client-side timer guessing at canned labels — or the final result
    (same shape as QueryResponse) once the answer is ready. Used by the web
    UI (static/app.js); /query is kept as the plain single-response
    endpoint for simple API use.

    Building a brand-new ChatSession (BM25 index, cross-encoder, source
    catalog: ~13-58s cold) happens INSIDE the background thread, after the
    stream has already started, and reports its own "initializing" stage.
    Doing it before starting the stream would leave the client showing its
    initial guess-label with no real event backing it for that whole
    window - exactly the kind of unearned progress indicator this
    streaming endpoint exists to avoid."""
    if collection_count() == 0 and not sql_db_available():
        raise HTTPException(400, "Nothing indexed yet. Call /ingest or build-sql-db first.")

    session_id = req.session_id or str(uuid.uuid4())
    events: queue.Queue = queue.Queue()

    def run() -> None:
        try:
            session = _sessions.get(session_id)
            if session is None:
                events.put({"stage": "initializing"})
                session = ChatSession()
                _sessions[session_id] = session

            result = session.ask(req.question, on_stage=lambda stage: events.put({"stage": stage}))
            events.put(
                {
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "session_id": session_id,
                    "route": result["route"],
                }
            )
        except Exception as exc:
            events.put({"error": str(exc)})
        finally:
            events.put(None)  # sentinel: stream done

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.delete("/session/{session_id}")
def clear_session(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"cleared": session_id}
