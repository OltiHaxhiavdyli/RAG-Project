"""Command-line interface for the RAG project.

Usage:
    python cli.py ingest [--dir data/raw] [--urls URL1 URL2 ...]
    python cli.py chat
    python cli.py build-sql-db [--dir data/raw]
    python cli.py sql "question"
"""
import argparse
from pathlib import Path

from src import config
from src.ingestion.sql_builder import build_structured_db
from src.rag.parent_document import parent_child_chunk_count
from src.rag.pipeline import ChatSession, ingest_directory, ingest_sql_table, ingest_urls
from src.rag.text_to_sql import ask_sql
from src.rag.vectorstore import collection_count


def cmd_ingest(args: argparse.Namespace) -> None:
    total_added = 0
    if args.urls:
        added = ingest_urls(args.urls)
        print(f"Ingested {added} chunks from {len(args.urls)} URL(s).")
        total_added += added

    if args.sql_conn and args.sql_table:
        added = ingest_sql_table(args.sql_conn, args.sql_table, args.sql_query)
        print(f"Ingested {added} chunks from table '{args.sql_table}'.")
        total_added += added

    directory = Path(args.dir)
    added = ingest_directory(directory)
    print(f"Ingested {added} chunks from files under {directory}.")
    total_added += added

    if total_added == 0:
        print(f"Nothing ingested. Drop files into {config.RAW_DATA_DIR} or pass --urls.")
    print(f"Vector store now holds {collection_count()} chunks total.")
    print(f"Parent-document index now holds {parent_child_chunk_count()} child chunks.")


def cmd_chat(_: argparse.Namespace) -> None:
    if collection_count() == 0:
        print(
            f"Note: vector store is empty (run `python cli.py ingest` to add documents). "
            f"Questions will still route to text-to-SQL if applicable.\n"
        )

    print("RAG chat ready. Type 'exit' to quit.\n")
    session = ChatSession()
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = session.ask(question)
        print(f"\n[routed: {result['route']}]")
        print(f"Assistant: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print()


def cmd_build_sql_db(args: argparse.Namespace) -> None:
    table_counts = build_structured_db(Path(args.dir), config.SQL_DB_PATH)
    if not table_counts:
        print(f"No .csv/.xlsx/.xls files found under {args.dir}.")
        return
    print(f"Built {config.SQL_DB_PATH}:")
    for table, count in table_counts.items():
        print(f"  {table}: {count} rows")


def cmd_sql(args: argparse.Namespace) -> None:
    result = ask_sql(args.question)
    print(f"SQL: {result['sql_query']}")
    print(f"\nAnswer: {result['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG project CLI")
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load and index documents")
    ingest_parser.add_argument("--dir", default=str(config.RAW_DATA_DIR))
    ingest_parser.add_argument("--urls", nargs="*", default=[])
    ingest_parser.add_argument("--sql-conn", help="SQLAlchemy connection string, e.g. sqlite:///data/sample.db")
    ingest_parser.add_argument("--sql-table", help="Table name to load")
    ingest_parser.add_argument("--sql-query", help="Custom SQL query instead of SELECT * FROM <table>")
    ingest_parser.set_defaults(func=cmd_ingest)

    chat_parser = subparsers.add_parser("chat", help="Interactive conversational RAG")
    chat_parser.set_defaults(func=cmd_chat)

    build_db_parser = subparsers.add_parser(
        "build-sql-db", help="Build a queryable SQLite DB from CSV/Excel files in data/raw"
    )
    build_db_parser.add_argument("--dir", default=str(config.RAW_DATA_DIR))
    build_db_parser.set_defaults(func=cmd_build_sql_db)

    sql_parser = subparsers.add_parser("sql", help="Ask a question via text-to-SQL")
    sql_parser.add_argument("question")
    sql_parser.set_defaults(func=cmd_sql)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
