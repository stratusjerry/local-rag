"""
Document loader for extracting text from .docx, .ppt, .pptx, and .txt files.

Supports loading individual files or entire directories of supported documents.
"""

import struct
from pathlib import Path

import olefile
from docx import Document as DocxDocument
from pptx import Presentation
from pydantic import BaseModel

SUPPORTED_EXTENSIONS = {".txt", ".docx", ".ppt", ".pptx"}


class Document(BaseModel):
    """
    Represents a loaded document with its text content and metadata.

    Attributes:
        text: The extracted text content.
        metadata: File metadata (filename, source path, file type).
    """

    text: str
    metadata: dict


def load_txt(file_path: Path) -> Document:
    """
    Load a plain text file.

    Attempts UTF-8 encoding first, falls back to latin-1 if decoding fails.

    Args:
        file_path (Path): Path to the .txt file.

    Returns:
        Document: Loaded document with text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Reason: latin-1 can decode any byte sequence, so it serves as a
        # reliable fallback for files with unknown or mixed encodings.
        text = file_path.read_text(encoding="latin-1")

    return Document(
        text=text,
        metadata={
            "filename": file_path.name,
            "source": str(file_path),
            "file_type": ".txt",
        },
    )


def load_docx(file_path: Path) -> Document:
    """
    Load a Word document (.docx) by joining all paragraph text.

    Args:
        file_path (Path): Path to the .docx file.

    Returns:
        Document: Loaded document with text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)

    return Document(
        text=text,
        metadata={
            "filename": file_path.name,
            "source": str(file_path),
            "file_type": ".docx",
        },
    )


def _extract_text_from_ppt_stream(stream_bytes: bytes) -> list[str]:
    """
    Extract text strings from a PowerPoint 97-2003 binary document stream.

    Parses the binary record headers to find TextCharsAtom (type 4000,
    UTF-16LE encoded) and TextBytesAtom (type 4008, Latin-1 encoded)
    records.

    Args:
        stream_bytes (bytes): Raw bytes of the "PowerPoint Document" OLE stream.

    Returns:
        list[str]: Extracted text strings in document order.
    """
    # Reason: PPT binary format stores text in 8-byte record headers followed
    # by the record body. recType 4000 = TextCharsAtom (UTF-16LE),
    # recType 4008 = TextBytesAtom (Latin-1/ASCII).
    texts = []
    offset = 0
    length = len(stream_bytes)

    while offset + 8 <= length:
        rec_ver_instance, rec_type, rec_len = struct.unpack_from(
            "<HHL", stream_bytes, offset
        )
        offset += 8

        if offset + rec_len > length:
            break

        if rec_type == 4000:  # TextCharsAtom — UTF-16LE
            raw = stream_bytes[offset : offset + rec_len]
            text = raw.decode("utf-16-le", errors="replace").strip()
            if text:
                texts.append(text)
        elif rec_type == 4008:  # TextBytesAtom — Latin-1
            raw = stream_bytes[offset : offset + rec_len]
            text = raw.decode("latin-1", errors="replace").strip()
            if text:
                texts.append(text)

        # Reason: container records (recVer == 0xF) hold child records inside
        # their body, so we step into them rather than skipping over.
        rec_ver = rec_ver_instance & 0x0F
        if rec_ver != 0x0F:
            offset += rec_len

    return texts


def load_ppt(file_path: Path) -> Document:
    """
    Load a PowerPoint 97-2003 presentation (.ppt) by parsing the OLE2 binary format.

    Opens the file as an OLE2 compound document, reads the "PowerPoint Document"
    stream, and extracts text from TextCharsAtom and TextBytesAtom records.

    Args:
        file_path (Path): Path to the .ppt file.

    Returns:
        Document: Loaded document with text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid OLE2 file or lacks a
            PowerPoint Document stream.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not olefile.isOleFile(str(file_path)):
        raise ValueError(f"Not a valid OLE2 file: {file_path}")

    ole = olefile.OleFileIO(str(file_path))
    try:
        if not ole.exists("PowerPoint Document"):
            raise ValueError(
                f"No 'PowerPoint Document' stream found in: {file_path}"
            )

        stream_bytes = ole.openstream("PowerPoint Document").read()
    finally:
        ole.close()

    texts = _extract_text_from_ppt_stream(stream_bytes)
    text = "\n\n".join(texts)

    return Document(
        text=text,
        metadata={
            "filename": file_path.name,
            "source": str(file_path),
            "file_type": ".ppt",
        },
    )


def load_pptx(file_path: Path) -> Document:
    """
    Load a PowerPoint presentation (.pptx) by extracting text from all slides.

    Iterates through slides, shapes, and text frames to extract all text content.

    Args:
        file_path (Path): Path to the .pptx file.

    Returns:
        Document: Loaded document with text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    prs = Presentation(str(file_path))
    slide_texts = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    paragraph_text = paragraph.text.strip()
                    if paragraph_text:
                        texts.append(paragraph_text)
        if texts:
            slide_texts.append(f"[Slide {slide_num}]\n" + "\n".join(texts))

    text = "\n\n".join(slide_texts)

    return Document(
        text=text,
        metadata={
            "filename": file_path.name,
            "source": str(file_path),
            "file_type": ".pptx",
        },
    )


# Reason: mapping extensions to loader functions avoids repeated if/elif chains
# and makes it easy to add new file types in the future.
_LOADERS = {
    ".txt": load_txt,
    ".docx": load_docx,
    ".ppt": load_ppt,
    ".pptx": load_pptx,
}


def load_file(file_path: Path) -> Document | None:
    """
    Load a single file using the appropriate loader based on extension.

    Args:
        file_path (Path): Path to the file.

    Returns:
        Document | None: Loaded document, or None if the extension is unsupported.
    """
    loader = _LOADERS.get(file_path.suffix.lower())
    if loader is None:
        return None
    return loader(file_path)


def load_directory(directory: Path) -> list[Document]:
    """
    Load all supported documents from a directory, including subdirectories.

    Args:
        directory (Path): Path to the directory to scan recursively.

    Returns:
        list[Document]: List of loaded documents.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    documents = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            doc = load_file(file_path)
            if doc and doc.text.strip():
                documents.append(doc)

    return documents
