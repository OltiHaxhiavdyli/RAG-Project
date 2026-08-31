"""A short, LLM-generated description per ingested source (filename or URL).

Descriptions are generated once per source and persisted to disk
(SOURCE_CATALOG_PATH), so re-opening a ChatSession doesn't re-summarize
content that hasn't changed — only genuinely new sources cost an LLM call.

Used by:
- self_query.py — enriches the `source` metadata filter with more than a
  bare filename/URL list, so filtering decisions are grounded in what's
  actually on each page, not just its name.
- scope_gate.py — a cheap upfront check for whether a question plausibly
  relates to anything ingested at all, before spending a full retrieval pass.
"""
import json
from pathlib import Path

from langchain_chroma import Chroma
from pydantic import BaseModel, Field

from src import config
from src.rag.chain import get_llm

DESCRIBE_PROMPT = """Here is an excerpt from a document/page titled "{source}":

{excerpt}

Write ONE short sentence (under 20 words) describing what topic this source \
covers, specific enough to distinguish it from other sources. Do not just \
restate the title."""


class SourceDescription(BaseModel):
    description: str = Field(description="One short sentence, under 20 words.")


def _load_catalog() -> dict[str, str]:
    path = Path(config.SOURCE_CATALOG_PATH)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_catalog(catalog: dict[str, str]) -> None:
    path = Path(config.SOURCE_CATALOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")


def _describe_source(source: str, excerpt: str) -> str:
    llm = get_llm(temperature=0)
    structured = llm.with_structured_output(SourceDescription)
    result = structured.invoke(DESCRIBE_PROMPT.format(source=source, excerpt=excerpt[:1500]))
    return result.description.strip()


def build_source_catalog(store: Chroma) -> dict[str, str]:
    """Returns {source: one-sentence description}. Generates (and caches) a
    description for any source that doesn't have one yet; drops entries for
    sources no longer present in the store."""
    raw = store.get(include=["documents", "metadatas"])
    excerpts: dict[str, str] = {}
    for text, meta in zip(raw["documents"], raw["metadatas"]):
        source = (meta or {}).get("source")
        if source and source not in excerpts:
            excerpts[source] = text

    if not excerpts:
        return {}

    catalog = _load_catalog()
    for source, excerpt in excerpts.items():
        if source not in catalog:
            catalog[source] = _describe_source(source, excerpt)

    catalog = {source: catalog[source] for source in excerpts}  # drop stale entries
    _save_catalog(catalog)  # keep the on-disk file honestly in sync, even if nothing new was described
    return catalog
