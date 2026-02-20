"""Tests for the text chunker module."""

import pytest

from ingestion.chunker import Chunk, chunk_document, chunk_text


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_short_text_returns_single_chunk(self) -> None:
        """Text shorter than chunk_size should return as a single chunk."""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        """Long text should be split into multiple chunks."""
        # Create text with clear paragraph boundaries
        paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        # All chunks should contain text
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_respects_chunk_size(self) -> None:
        """Chunks should not greatly exceed the specified chunk size."""
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        for chunk in chunks:
            # Allow some tolerance for overlap prepending
            assert len(chunk) <= 200, f"Chunk too large: {len(chunk)} chars"

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty or whitespace-only text should return no chunks."""
        assert chunk_text("", chunk_size=100, chunk_overlap=20) == []
        assert chunk_text("   ", chunk_size=100, chunk_overlap=20) == []

    def test_custom_separators(self) -> None:
        """Custom separators should be respected."""
        text = "part1|part2|part3"
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=0, separators=["|"])

        assert len(chunks) == 3
        assert chunks[0] == "part1"


class TestChunkDocument:
    """Tests for the chunk_document function."""

    def test_returns_chunk_objects_with_metadata(self) -> None:
        """Chunks should include original metadata plus chunk_index."""
        metadata = {"filename": "test.txt", "source": "/path/test.txt"}
        text = "A" * 500 + "\n\n" + "B" * 500

        chunks = chunk_document(text, metadata, chunk_size=300, chunk_overlap=50)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["filename"] == "test.txt"
            assert chunk.metadata["chunk_index"] == i

    def test_empty_document_returns_empty_list(self) -> None:
        """An empty document should produce no chunks."""
        chunks = chunk_document("", {"filename": "empty.txt"})
        assert chunks == []

    def test_single_chunk_document(self) -> None:
        """A short document should produce exactly one chunk."""
        chunks = chunk_document(
            "Short text.",
            {"filename": "short.txt"},
            chunk_size=1000,
        )
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_index"] == 0
