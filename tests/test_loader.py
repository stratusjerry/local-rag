"""Tests for the document loader module."""

import struct
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.loader import (
    SUPPORTED_EXTENSIONS,
    Document,
    _extract_text_from_ppt_stream,
    load_directory,
    load_file,
    load_ppt,
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


def _build_ppt_stream(texts: list[str], record_type: int = 4008) -> bytes:
    """
    Build a fake PowerPoint Document binary stream with text records.

    Args:
        texts: Strings to encode as text records.
        record_type: 4008 for TextBytesAtom (Latin-1), 4000 for TextCharsAtom (UTF-16LE).

    Returns:
        bytes: Binary stream data.
    """
    stream = b""
    for text in texts:
        if record_type == 4000:
            encoded = text.encode("utf-16-le")
        else:
            encoded = text.encode("latin-1")
        header = struct.pack("<HHL", 0x0000, record_type, len(encoded))
        stream += header + encoded
    return stream


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


class TestExtractTextFromPptStream:
    """Tests for the low-level PPT binary stream parser."""

    def test_extracts_latin1_text_records(self) -> None:
        """TextBytesAtom records (type 4008) should be decoded as Latin-1."""
        stream = _build_ppt_stream(["Hello world", "Second text"], record_type=4008)
        texts = _extract_text_from_ppt_stream(stream)

        assert texts == ["Hello world", "Second text"]

    def test_extracts_utf16_text_records(self) -> None:
        """TextCharsAtom records (type 4000) should be decoded as UTF-16LE."""
        stream = _build_ppt_stream(["Unicode text"], record_type=4000)
        texts = _extract_text_from_ppt_stream(stream)

        assert texts == ["Unicode text"]

    def test_empty_stream(self) -> None:
        """An empty stream should produce no text."""
        assert _extract_text_from_ppt_stream(b"") == []


class TestLoadPpt:
    """Tests for loading .ppt (PowerPoint 97-2003) files."""

    @patch("ingestion.loader.olefile")
    def test_load_ppt_extracts_text(self, mock_olefile, tmp_path: Path) -> None:
        """Text should be extracted from a valid .ppt file."""
        ppt_file = tmp_path / "presentation.ppt"
        ppt_file.write_bytes(b"dummy")  # File must exist for the existence check

        stream_data = _build_ppt_stream(["Hello from slide one", "Second slide text"])
        mock_olefile.isOleFile.return_value = True
        mock_ole_instance = MagicMock()
        mock_olefile.OleFileIO.return_value = mock_ole_instance
        mock_ole_instance.exists.return_value = True
        mock_ole_instance.openstream.return_value = BytesIO(stream_data)

        doc = load_ppt(ppt_file)

        assert isinstance(doc, Document)
        assert "Hello from slide one" in doc.text
        assert "Second slide text" in doc.text
        assert doc.metadata["filename"] == "presentation.ppt"
        assert doc.metadata["file_type"] == ".ppt"

    def test_load_ppt_nonexistent(self, tmp_path: Path) -> None:
        """Loading a missing .ppt file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_ppt(tmp_path / "missing.ppt")

    def test_load_ppt_no_text_raises(self, tmp_path: Path) -> None:
        """A file with no extractable text should raise ValueError."""
        bad_file = tmp_path / "empty.ppt"
        bad_file.write_bytes(b"\x00" * 64)

        with pytest.raises(ValueError):
            load_ppt(bad_file)

    def test_load_ppt_raw_fallback(self, tmp_path: Path) -> None:
        """When OLE parsing fails, text should be extracted via raw scan."""
        stream_data = _build_ppt_stream(["Recovered text"])
        # Reason: padding must be a multiple of 8 bytes so the record headers
        # that follow stay aligned for the raw binary scanner.
        ppt_file = tmp_path / "truncated.ppt"
        ppt_file.write_bytes(b"\x00" * 104 + stream_data)

        doc = load_ppt(ppt_file)

        assert "Recovered text" in doc.text
        assert doc.metadata["file_type"] == ".ppt"


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
        assert ".ppt" in SUPPORTED_EXTENSIONS
        assert ".pptx" in SUPPORTED_EXTENSIONS


class TestLoadDirectory:
    """Tests for loading all documents from a directory."""

    def test_load_directory_filters_supported(self, tmp_docs_dir: Path) -> None:
        """Test that only supported, non-empty files are loaded."""
        docs, errors = load_directory(tmp_docs_dir)

        # Should load only sample.txt (empty.txt has no content, csv is unsupported)
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "sample.txt"
        assert errors == []

    def test_load_directory_recurses_subdirs(self, tmp_path: Path) -> None:
        """Test that files in subdirectories are loaded."""
        # Top-level file
        (tmp_path / "top.txt").write_text("top level", encoding="utf-8")

        # Nested file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content", encoding="utf-8")

        # Doubly nested file
        deep = subdir / "deep"
        deep.mkdir()
        (deep / "deep.txt").write_text("deep content", encoding="utf-8")

        docs, errors = load_directory(tmp_path)

        filenames = {d.metadata["filename"] for d in docs}
        assert filenames == {"top.txt", "nested.txt", "deep.txt"}
        assert errors == []

    def test_load_directory_skips_corrupt_files(self, tmp_path: Path) -> None:
        """Corrupt files should be skipped and reported in errors."""
        # Good file
        (tmp_path / "good.txt").write_text("valid content", encoding="utf-8")

        # Bad .ppt file (not a real OLE file, will fail to load)
        (tmp_path / "corrupt.ppt").write_text("not a real ppt", encoding="utf-8")

        docs, errors = load_directory(tmp_path)

        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "good.txt"
        assert len(errors) == 1
        assert "corrupt.ppt" in errors[0]

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
