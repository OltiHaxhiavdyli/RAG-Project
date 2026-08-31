"""Loaders for structured/tabular data: CSV, Excel, JSON, and SQL tables.

Vector search retrieves passages, not rows — it can't "sum column Y where
Z". So each record is flattened into a small "field: value" text block (one
Document per row) that embeds and retrieves the way prose does, while the
original field values stay in `metadata` for filtering or display. This is
good for lookup/fact questions ("what's the MRR for org X?") but not for
aggregate questions ("what's total MRR?") — those need a text-to-SQL or
pandas-agent approach instead, which is a different pipeline entirely.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.documents import Document


def _to_python_scalar(value):
    """pandas/numpy scalars (e.g. numpy.int64) aren't the Python str/int/float
    types Chroma metadata requires, and numpy.int64 isn't even an int
    subclass on most platforms, so unwrap them explicitly."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_to_text(row: dict, prefix: str = "") -> str:
    lines = [f"{prefix}" if prefix else ""]
    for key, value in row.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(line for line in lines if line)


def _dataframe_to_documents(
    df: pd.DataFrame, source: str, extra_metadata: dict | None = None
) -> list[Document]:
    documents = []
    for i, row in df.iterrows():
        record = {k: _to_python_scalar(v) for k, v in row.to_dict().items()}
        documents.append(
            Document(
                page_content=_row_to_text(record),
                metadata={
                    "source": source,
                    "row_index": int(i),
                    **{k: v for k, v in record.items() if isinstance(v, (str, int, float, bool))},
                    **(extra_metadata or {}),
                },
            )
        )
    return documents


def load_csv(path: Path) -> list[Document]:
    df = pd.read_csv(path)
    return _dataframe_to_documents(df, source=path.name)


def load_excel(path: Path) -> list[Document]:
    """Loads every sheet; each row becomes a Document tagged with its sheet name."""
    sheets = pd.read_excel(path, sheet_name=None)
    documents = []
    for sheet_name, df in sheets.items():
        documents.extend(
            _dataframe_to_documents(df, source=path.name, extra_metadata={"sheet": sheet_name})
        )
    return documents


def load_json(path: Path) -> list[Document]:
    """Handles a JSON array of flat objects, JSON Lines (.jsonl), or falls
    back to treating the whole file as one document if it's not tabular."""
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = [parsed]
        else:
            records = [{"value": parsed}]

    if records and all(isinstance(r, dict) for r in records):
        documents = []
        for i, record in enumerate(records):
            flat = {
                k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v))
                for k, v in record.items()
            }
            documents.append(
                Document(
                    page_content=_row_to_text(flat),
                    metadata={"source": path.name, "row_index": i},
                )
            )
        return documents

    return [Document(page_content=text, metadata={"source": path.name})]


def load_sql_table(connection_string: str, table: str, query: str | None = None) -> list[Document]:
    """Load rows from a SQL table (or a custom query) via SQLAlchemy. Requires
    the right DBAPI driver installed for your database (sqlite3 is built in;
    Postgres needs psycopg2-binary, MySQL needs mysqlclient, etc.)."""
    from sqlalchemy import create_engine

    engine = create_engine(connection_string)
    sql = query if query else f"SELECT * FROM {table}"
    df = pd.read_sql(sql, engine)
    return _dataframe_to_documents(df, source=f"db:{table}", extra_metadata={"table": table})
