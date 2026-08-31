"""Router tests. No API key needed — these test the guard that skips the LLM
call entirely when there's no SQL database to route to."""
from src import config
from src.rag import router


def test_sql_db_available_reflects_file_existence(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.db"
    monkeypatch.setattr(config, "SQL_DB_PATH", missing_path)
    assert router.sql_db_available() is False

    missing_path.touch()
    assert router.sql_db_available() is True


def test_route_question_skips_llm_when_no_sql_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQL_DB_PATH", tmp_path / "does_not_exist.db")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("get_llm should not be called when there's no SQL DB to route to")

    monkeypatch.setattr(router, "get_llm", _fail_if_called)

    decision = router.route_question("how many widgets are there?")
    assert decision.destination == "vectorstore"
