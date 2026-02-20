"""
Recursive character text splitter for breaking documents into chunks.

Splits text by preferring semantic boundaries (paragraphs, sentences)
and falls back to smaller separators when chunks exceed the target size.
"""

from pydantic import BaseModel


class Chunk(BaseModel):
    """
    A text chunk with metadata tracking its origin.

    Attributes:
        text: The chunk text content.
        metadata: Original document metadata plus chunk_index.
    """

    text: str
    metadata: dict


# Reason: ordered from most to least semantically meaningful so splits
# happen at the most natural boundary that fits the chunk size.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """
    Recursively split text into chunks using a hierarchy of separators.

    Tries the first separator that exists in the text. Splits on it, then
    merges pieces back together up to chunk_size. If a piece still exceeds
    chunk_size, recursively splits it with the remaining separators.

    Args:
        text (str): The text to split.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        separators (list[str]): Ordered list of separators to try.

    Returns:
        list[str]: List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator that exists in this text
    separator = ""
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    # If no separator found, hard-split by character
    if not separator:
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # Split on the chosen separator
    pieces = text.split(separator)

    # Merge pieces into chunks up to chunk_size
    chunks = []
    current = ""

    for piece in pieces:
        candidate = current + separator + piece if current else piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)

            # If this single piece exceeds chunk_size, recurse with
            # finer-grained separators
            if len(piece) > chunk_size and remaining_separators:
                sub_chunks = _split_text_recursive(
                    piece, chunk_size, chunk_overlap, remaining_separators
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    # Apply overlap by prepending the tail of the previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:]
            # Reason: only prepend overlap if it doesn't duplicate the start
            if not chunks[i].startswith(overlap_text):
                overlapped.append(overlap_text + chunks[i])
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Split text into chunks with overlap.

    Args:
        text (str): The text to split.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlapping characters between consecutive chunks.
        separators (list[str] | None): Custom separators; defaults to
            paragraph, newline, sentence, and space boundaries.

    Returns:
        list[str]: List of text chunks.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    return _split_text_recursive(text, chunk_size, chunk_overlap, separators)


def chunk_document(
    text: str,
    metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Split a document's text into Chunk objects with indexed metadata.

    Args:
        text (str): Document text to split.
        metadata (dict): Original document metadata (filename, source, etc.).
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlapping characters between consecutive chunks.

    Returns:
        list[Chunk]: List of Chunk objects with chunk_index in metadata.
    """
    raw_chunks = chunk_text(text, chunk_size, chunk_overlap)

    return [
        Chunk(
            text=chunk,
            metadata={**metadata, "chunk_index": i},
        )
        for i, chunk in enumerate(raw_chunks)
    ]
