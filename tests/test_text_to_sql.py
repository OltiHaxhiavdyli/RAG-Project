"""Text-to-SQL tests. No API key needed — these test the SQL safety guards
and the DB builder directly, not the LLM-driven query generation."""
import sqlite3

import pandas as pd
import pytest

from src import config
from src.ingestion.sql_builder import build_structured_db
from src.rag.text_to_sql import _clean_sql, _is_safe_select, get_sql_database


def test_is_safe_select_accepts_select_and_cte():
    assert _is_safe_select("SELECT * FROM widgets")
    assert _is_safe_select("select id from widgets where price > 5")
    assert _is_safe_select("WITH t AS (SELECT 1) SELECT * FROM t")


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO widgets VALUES (1, 'x', 1.0)",
        "UPDATE widgets SET price = 0",
        "DELETE FROM widgets",
        "DROP TABLE widgets",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA table_info(widgets)",
        "SELECT * FROM widgets; DROP TABLE widgets",
        "",
    ],
)
def test_is_safe_select_rejects_unsafe_sql(sql):
    assert not _is_safe_select(sql)


def test_clean_sql_strips_fences_and_prefix():
    assert _clean_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert _clean_sql("SQLQuery: SELECT 1;") == "SELECT 1"
    assert _clean_sql("  SELECT 1;  ") == "SELECT 1"


def test_build_structured_db_creates_tables(tmp_path):
    (tmp_path / "widgets.csv").write_text("id,name,price\n1,sprocket,4.5\n2,gizmo,9.0\n")
    xlsx_path = tmp_path / "rosters.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({"Full Name": ["Ada"]}).to_excel(writer, sheet_name="Team A", index=False)

    db_path = tmp_path / "out.db"
    table_counts = build_structured_db(tmp_path, db_path)

    assert table_counts["widgets"] == 2
    assert table_counts["rosters_team_a"] == 1

    conn = sqlite3.connect(db_path)
    cols = [row[1] for row in conn.execute("PRAGMA table_info(rosters_team_a)")]
    conn.close()
    assert cols == ["full_name"]  # sanitized: lowercase, no spaces


def test_build_structured_db_drops_fully_blank_rows(tmp_path):
    xlsx_path = tmp_path / "roster.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({"name": ["Ada", None, "Grace"], "team": ["A", None, "B"]}).to_excel(
            writer, sheet_name="Sheet1", index=False
        )

    db_path = tmp_path / "out.db"
    table_counts = build_structured_db(tmp_path, db_path)

    assert table_counts["roster_sheet1"] == 2  # the fully-blank middle row is dropped


def test_get_sql_database_connection_is_read_only(tmp_path, monkeypatch):
    (tmp_path / "widgets.csv").write_text("id,name\n1,sprocket\n")
    db_path = tmp_path / "out.db"
    build_structured_db(tmp_path, db_path)

    # get_sql_database() is process-cached (see text_to_sql.py) since real
    # usage calls it on every question, not once — but that means a stale
    # entry from an earlier call (a different test, or real app startup in
    # the same process) would silently outlive this test's tmp_path. Clear
    # before AND after so this test neither reads nor leaves behind a cached
    # instance pointing at a directory pytest is about to delete.
    monkeypatch.setattr(config, "SQL_DB_PATH", db_path)
    get_sql_database.cache_clear()
    try:
        db = get_sql_database()

        with pytest.raises(Exception, match="(?i)readonly|read-only"):
            db.run("INSERT INTO widgets VALUES (2, 'gizmo')")
    finally:
        get_sql_database.cache_clear()
