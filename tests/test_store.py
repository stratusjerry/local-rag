"""Tests for the vector store module."""

from pathlib import Path

import pytest

from vectorstore.store import QueryResult, VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """Create a temporary VectorStore instance."""
    return VectorStore(persist_dir=str(tmp_path / "chromadb"), collection_name="test")


class TestVectorStore:
    """Tests for the VectorStore class."""

    def test_upsert_and_count(self, store: VectorStore) -> None:
        """Upserting documents should increase the count."""
        assert store.count() == 0

        store.upsert(
            ids=["doc1::chunk_0", "doc1::chunk_1"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            documents=["Hello world", "Goodbye world"],
            metadatas=[{"filename": "doc1.txt"}, {"filename": "doc1.txt"}],
        )

        assert store.count() == 2

    def test_upsert_is_idempotent(self, store: VectorStore) -> None:
        """Upserting the same IDs should overwrite, not duplicate."""
        store.upsert(
            ids=["doc1::chunk_0"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["Original text"],
            metadatas=[{"filename": "doc1.txt"}],
        )
        store.upsert(
            ids=["doc1::chunk_0"],
            embeddings=[[0.4, 0.5, 0.6]],
            documents=["Updated text"],
            metadatas=[{"filename": "doc1.txt"}],
        )

        assert store.count() == 1

    def test_query_returns_results(self, store: VectorStore) -> None:
        """Querying with a similar embedding should return matching results."""
        store.upsert(
            ids=["doc1::chunk_0"],
            embeddings=[[1.0, 0.0, 0.0]],
            documents=["Test document about cats"],
            metadatas=[{"filename": "cats.txt"}],
        )

        results = store.query(query_embedding=[1.0, 0.0, 0.0], n_results=1)

        assert len(results) == 1
        assert isinstance(results[0], QueryResult)
        assert results[0].text == "Test document about cats"
        assert results[0].metadata["filename"] == "cats.txt"

    def test_query_empty_store(self, store: VectorStore) -> None:
        """Querying an empty store should return an empty list."""
        results = store.query(query_embedding=[1.0, 0.0, 0.0], n_results=5)
        assert results == []

    def test_reset_clears_data(self, store: VectorStore) -> None:
        """Resetting should remove all stored chunks."""
        store.upsert(
            ids=["doc1::chunk_0"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["Some text"],
            metadatas=[{"filename": "doc.txt"}],
        )
        assert store.count() == 1

        store.reset()
        assert store.count() == 0
