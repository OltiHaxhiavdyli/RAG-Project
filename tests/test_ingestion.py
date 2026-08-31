"""Ingestion tests. No API key required — these only exercise loading and
chunking, not embeddings. Uses tests/fixtures/, not data/raw/, since data/raw
holds whatever real documents the project owner has actually ingested."""
import sqlite3
from pathlib import Path

import pandas as pd

from src.ingestion.chunking import split_documents
from src.ingestion.loaders import load_directory, load_prose_directory
from src.ingestion.structured_loaders import load_sql_table

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_load_directory_finds_sample_docs():
    docs = load_directory(FIXTURES_DIR)
    sources = {doc.metadata["source"] for doc in docs}
    assert "sample.md" in sources


def test_load_directory_handles_structured_formats():
    docs = load_directory(FIXTURES_DIR)
    sources = {doc.metadata["source"] for doc in docs}
    assert "sample.csv" in sources
    assert "sample.json" in sources

    csv_rows = [doc for doc in docs if doc.metadata["source"] == "sample.csv"]
    assert len(csv_rows) == 3  # one Document per data row
    assert "TICK-001" in csv_rows[0].page_content
    assert "severity" in csv_rows[0].page_content.lower()

    json_rows = [doc for doc in docs if doc.metadata["source"] == "sample.json"]
    assert len(json_rows) == 2
    assert any("Sprocket" in doc.page_content for doc in json_rows)


def test_load_prose_directory_excludes_structured_formats():
    docs = load_prose_directory(FIXTURES_DIR)
    sources = {doc.metadata["source"] for doc in docs}
    assert sources == {"sample.md"}  # not sample.csv or sample.json


def test_load_directory_handles_excel(tmp_path):
    xlsx_path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="sheet_one", index=False)
        pd.DataFrame({"b": [3, 4]}).to_excel(writer, sheet_name="sheet_two", index=False)

    docs = load_directory(tmp_path)
    sources = {doc.metadata["source"] for doc in docs}
    assert "sample.xlsx" in sources
    assert {doc.metadata["sheet"] for doc in docs} == {"sheet_one", "sheet_two"}


def test_load_directory_handles_docx(tmp_path):
    """Real bug, found by auditing the project for unused code: requirements.txt
    listed `python-docx`, which the app never actually imports — .docx loading
    goes through langchain's Docx2txtLoader, which needs the differently-named
    `docx2txt` package instead. That package was missing entirely, so dropping
    a real .docx into data/raw crashed with `ModuleNotFoundError: No module
    named 'docx2txt'` the moment anyone actually tried it. Undetected because
    no test exercised .docx loading at all. Fixed by swapping the dependency;
    this closes the coverage gap that let it happen. python-docx is only
    needed here, to author the fixture — the app's own runtime path never
    imports it."""
    from docx import Document as DocxDocument

    docx_path = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_paragraph("Refunds are processed within 5 business days.")
    doc.save(docx_path)

    docs = load_directory(tmp_path)
    sources = {doc.metadata["source"] for doc in docs}
    assert "sample.docx" in sources
    assert any("5 business days" in doc.page_content for doc in docs)


def test_load_sql_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE widgets (id INTEGER, name TEXT, price REAL)")
    conn.execute("INSERT INTO widgets VALUES (1, 'sprocket', 4.5), (2, 'gizmo', 9.0)")
    conn.commit()
    conn.close()

    docs = load_sql_table(f"sqlite:///{db_path.as_posix()}", table="widgets")
    assert len(docs) == 2
    assert "sprocket" in docs[0].page_content
    assert docs[0].metadata["table"] == "widgets"
    assert docs[0].metadata["source"] == "db:widgets"


def test_split_documents_respects_chunk_size():
    docs = load_directory(FIXTURES_DIR)
    chunks = split_documents(docs)
    assert len(chunks) >= len(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000 + 200  # chunk_size + slack for separators
        assert "chunk_id" in chunk.metadata
        assert "source" in chunk.metadata


def test_split_documents_assigns_deterministic_ids():
    """Real bug, found live against the project's own data store: chunks
    had no explicit Chroma id, so add_documents() generated a fresh random
    one every call — re-ingesting the same file duplicated it instead of
    upserting. 1340 of 1515 real chunks ended up exact duplicates after a
    routine re-ingest that only meant to add two new URLs. `chunk_id`
    (a running index within one split_documents() call) isn't stable
    across separate calls either, which is why the fix is a content hash,
    not that field."""
    docs = load_directory(FIXTURES_DIR)

    chunks_first_run = split_documents(docs)
    chunks_second_run = split_documents(docs)  # simulates re-ingesting the same files

    ids_first = [c.id for c in chunks_first_run]
    ids_second = [c.id for c in chunks_second_run]

    assert all(ids_first)  # every chunk actually got an id, not None
    assert ids_first == ids_second  # same (source, content) -> same ids, every time
    assert len(set(ids_first)) == len(ids_first)  # distinct chunks -> distinct ids
