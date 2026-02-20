"""Tests for the document loader module."""

from pathlib import Path

import pytest

from ingestion.loader import (
    SUPPORTED_EXTENSIONS,
    Document,
    load_directory,
    load_file,
    load_txt,
)


@pytest.fixture
def tmp_docs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample documents."""
    # .txt file
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello, this is a test document.", encoding="utf-8")

    # Unsupported file
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2", encoding="utf-8")

    # Empty .txt file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    return tmp_path


class TestLoadTxt:
    """Tests for loading .txt files."""

    def test_load_utf8_file(self, tmp_path: Path) -> None:
        """Test loading a standard UTF-8 text file."""
        file = tmp_path / "test.txt"
        file.write_text("Hello world", encoding="utf-8")

        doc = load_txt(file)

        assert isinstance(doc, Document)
        assert doc.text == "Hello world"
        assert doc.metadata["filename"] == "test.txt"
        assert doc.metadata["file_type"] == ".txt"

    def test_load_latin1_fallback(self, tmp_path: Path) -> None:
        """Test fallback to latin-1 encoding for non-UTF-8 files."""
        file = tmp_path / "latin.txt"
        file.write_bytes(b"caf\xe9")  # 'café' in latin-1

        doc = load_txt(file)

        assert "caf" in doc.text
        assert doc.metadata["file_type"] == ".txt"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that loading a missing file raises an error."""
        file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            load_txt(file)


class TestLoadFile:
    """Tests for the generic file loader dispatcher."""

    def test_load_supported_extension(self, tmp_path: Path) -> None:
        """Test loading a file with a supported extension."""
        file = tmp_path / "test.txt"
        file.write_text("content", encoding="utf-8")

        doc = load_file(file)

        assert doc is not None
        assert doc.text == "content"

    def test_load_unsupported_extension(self, tmp_path: Path) -> None:
        """Test that unsupported extensions return None."""
        file = tmp_path / "data.csv"
        file.write_text("a,b,c", encoding="utf-8")

        result = load_file(file)

        assert result is None

    def test_supported_extensions_set(self) -> None:
        """Test that the supported extensions set contains expected types."""
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".pptx" in SUPPORTED_EXTENSIONS


class TestLoadDirectory:
    """Tests for loading all documents from a directory."""

    def test_load_directory_filters_supported(self, tmp_docs_dir: Path) -> None:
        """Test that only supported, non-empty files are loaded."""
        docs = load_directory(tmp_docs_dir)

        # Should load only sample.txt (empty.txt has no content, csv is unsupported)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "sample.txt"

    def test_load_directory_not_found(self, tmp_path: Path) -> None:
        """Test that a missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_directory(tmp_path / "nonexistent")

    def test_load_directory_not_a_directory(self, tmp_path: Path) -> None:
        """Test that passing a file path raises NotADirectoryError."""
        file = tmp_path / "file.txt"
        file.write_text("content", encoding="utf-8")

        with pytest.raises(NotADirectoryError):
            load_directory(file)
