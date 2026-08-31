"""Text-to-SQL: the LLM writes a real SQL query against a real database,
instead of retrieving pre-embedded text chunks. This is what makes aggregate
and computational questions ("how many sections does ACCT have?") answerable
at all — no amount of better chunking/reranking fixes that in plain RAG,
because no single chunk contains the answer.

Security note: an LLM-generated query executed against a real database is a
genuine risk surface (a cleverly worded question could try to coerce a
DROP/DELETE, same category of concern as SQL injection even though the
"attacker" here is the model itself, not raw user input reaching SQL
directly). Two independent guards, so one being wrong isn't enough to matter:
  1. The database connection itself is opened read-only at the SQLite level.
  2. Every generated query is validated as a single SELECT statement before
     it's ever executed.
"""
import re
import sqlite3

from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from sqlalchemy import create_engine

from src import config
from src.rag.chain import get_llm

_DISALLOWED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|"
    r"TRUNCATE|GRANT|REVOKE|VACUUM)\b",
    re.IGNORECASE,
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Given the question, the SQL query used to answer it, and the query result, "
    "write a plain-language answer. Mention specific numbers/values from the "
    "result rather than being vague. If the result is empty, say so explicitly "
    "rather than guessing.\n\n"
    "Question: {question}\n"
    "SQL query: {query}\n"
    "SQL result: {result}\n\n"
    "Answer:"
)


class UnsafeSQLError(ValueError):
    pass


def _clean_sql(text: str) -> str:
    text = text.strip()
    # A real bug lived here: matching literally "```sql" left "ite\n" behind
    # when the model fenced with "```sqlite" instead — \w* strips ANY
    # language tag (sql, sqlite, SQL, etc.), not just one specific spelling.
    text = re.sub(r"^```\w*\s*|```$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^SQLQuery:\s*", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(";").strip()


def _is_safe_select(sql: str) -> bool:
    if not sql:
        return False
    if ";" in sql:  # reject stacked statements outright
        return False
    if _DISALLOWED_KEYWORDS.search(sql):
        return False
    return bool(re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE))


def get_sql_database() -> SQLDatabase:
    if not config.SQL_DB_PATH.exists():
        raise RuntimeError(
            f"{config.SQL_DB_PATH} does not exist yet. Run "
            "`python cli.py build-sql-db` first to build it from data/raw/."
        )

    db_path = config.SQL_DB_PATH

    def _connect() -> sqlite3.Connection:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    engine = create_engine("sqlite://", creator=_connect)
    return SQLDatabase(engine, sample_rows_in_table_info=2, max_string_length=300)


def build_text_to_sql_chain() -> Runnable:
    llm = get_llm(temperature=0)
    db = get_sql_database()

    write_query = create_sql_query_chain(llm, db, k=config.SQL_QUERY_ROW_LIMIT)
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    def run(inputs: dict) -> dict:
        question = inputs["question"]

        sql = _clean_sql(write_query.invoke({"question": question}))
        if not _is_safe_select(sql):
            raise UnsafeSQLError(f"Refusing to execute non-SELECT/unsafe SQL: {sql!r}")
        result = db.run(sql)

        answer = answer_chain.invoke({"question": question, "query": sql, "result": result})
        return {"query": sql, "result": result, "answer": answer}

    return RunnableLambda(run)


def ask_sql(question: str) -> dict:
    chain = build_text_to_sql_chain()
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sql_query": result["query"],
        "raw_result": result["result"],
    }
