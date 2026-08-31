"""Source catalog tests. No API key needed — mocks the description-generating
LLM call, so these test the caching/staleness logic, not the LLM output."""
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import config
from src.rag import source_catalog
from src.rag.source_catalog import build_source_catalog


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def _store(tmp_path):
    return Chroma(embedding_function=FakeEmbeddings(), persist_directory=str(tmp_path / ".chroma"))


def test_build_source_catalog_empty_store_skips_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_CATALOG_PATH", str(tmp_path / "catalog.json"))

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("get_llm should not be called for an empty store")

    monkeypatch.setattr(source_catalog, "get_llm", _fail_if_called)

    store = _store(tmp_path)
    assert build_source_catalog(store) == {}


def test_build_source_catalog_generates_once_then_caches(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    monkeypatch.setattr(config, "SOURCE_CATALOG_PATH", str(catalog_path))

    calls = []
    monkeypatch.setattr(
        source_catalog, "_describe_source", lambda source, excerpt: calls.append(source) or f"about {source}"
    )

    store = _store(tmp_path)
    store.add_documents(
        [
            Document(page_content="widget policy text", metadata={"source": "widgets.md"}),
            Document(page_content="gizmo policy text", metadata={"source": "gizmos.md"}),
        ]
    )

    catalog = build_source_catalog(store)
    assert catalog == {"widgets.md": "about widgets.md", "gizmos.md": "about gizmos.md"}
    assert sorted(calls) == ["gizmos.md", "widgets.md"]
    assert json.loads(catalog_path.read_text(encoding="utf-8")) == catalog

    # second call: nothing new to describe — _describe_source must not run again
    monkeypatch.setattr(
        source_catalog,
        "_describe_source",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should be cached, not regenerated")),
    )
    assert build_source_catalog(store) == catalog


def test_build_source_catalog_drops_stale_entries(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"widgets.md": "about widgets.md", "old_removed_file.md": "stale entry"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "SOURCE_CATALOG_PATH", str(catalog_path))
    monkeypatch.setattr(source_catalog, "_describe_source", lambda *a, **k: "unused")

    store = _store(tmp_path)
    store.add_documents([Document(page_content="widget policy text", metadata={"source": "widgets.md"})])

    catalog = build_source_catalog(store)
    assert catalog == {"widgets.md": "about widgets.md"}
    assert "old_removed_file.md" not in catalog
