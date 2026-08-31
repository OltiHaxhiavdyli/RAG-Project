"""Builds a real, queryable SQLite database from CSV/Excel files in a source
directory — the counterpart to structured_loaders.py, which instead flattens
those same files into text for the vector store. This one keeps the data
genuinely relational, which is what a text-to-SQL chain needs to answer
aggregate/computational questions vector search can't."""
import re
import sqlite3
from pathlib import Path

import pandas as pd


def _sanitize_identifier(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "col"


def _write_table(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> int:
    df = df.copy()
    df.columns = [_sanitize_identifier(c) for c in df.columns]
    df = df.dropna(how="all")  # drop fully-blank rows (common trailing junk in Excel exports)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    return len(df)


def build_structured_db(source_dir: Path, db_path: Path) -> dict[str, int]:
    """Loads every .csv/.xlsx/.xls under source_dir into its own table (one
    table per sheet for Excel). Returns {table_name: row_count}."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    table_row_counts: dict[str, int] = {}

    try:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() == ".csv":
                table_name = _sanitize_identifier(path.stem)
                table_row_counts[table_name] = _write_table(conn, table_name, pd.read_csv(path))

            elif path.suffix.lower() in (".xlsx", ".xls"):
                sheets = pd.read_excel(path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    table_name = _sanitize_identifier(f"{path.stem}_{sheet_name}")
                    table_row_counts[table_name] = _write_table(conn, table_name, df)

        conn.commit()
    finally:
        conn.close()

    return table_row_counts
