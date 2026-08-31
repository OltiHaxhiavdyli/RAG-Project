"""Logical routing: classify each question as either a document lookup
(→ hybrid RAG over the vector store + web) or a structured/computational
question (→ text-to-SQL against a real database), and dispatch accordingly.

This is what turns three separate retrieval mechanisms (vector/BM25/web
ensemble, text-to-SQL) into one system the user can just ask things of,
instead of three tools they have to pick between themselves.
"""
from typing import Literal

from pydantic import BaseModel, Field

from src import config
from src.rag.chain import get_llm

RouteDestination = Literal["vectorstore", "sql"]

ROUTER_PROMPT = """You are routing a user's question to the system that can \
actually answer it.

- Route to "vectorstore" for questions answerable by looking something up in \
documents/policies/text — facts, definitions, explanations, "what does X say \
about Y", anything conversational.
- Route to "sql" ONLY for questions that require computing over structured \
data in the database below — counts, sums, averages, "which X has the most \
Y", filtering/ranking many records. If the database schema clearly can't \
answer the question, route to "vectorstore" instead.

Database available for the "sql" route:
{schema}

Question: {question}"""


class RouteDecision(BaseModel):
    destination: RouteDestination
    reasoning: str = Field(description="One sentence explaining the choice.")


def sql_db_available() -> bool:
    return config.SQL_DB_PATH.exists()


def route_question(question: str) -> RouteDecision:
    if not sql_db_available():
        # Nothing to route to — skip the LLM call entirely rather than let it
        # pick a destination that doesn't exist.
        return RouteDecision(
            destination="vectorstore",
            reasoning="No SQL database has been built yet (run `python cli.py build-sql-db`).",
        )

    from src.rag.text_to_sql import get_sql_database

    schema = get_sql_database().get_table_info()

    llm = get_llm(temperature=0)
    router = llm.with_structured_output(RouteDecision)
    return router.invoke(ROUTER_PROMPT.format(schema=schema, question=question))
