"""
Ingestion pipeline that orchestrates document loading, chunking,
embedding, and storage into the vector store.
"""

from pathlib import Path
from typing import Callable

from config.settings import Settings
from embeddings.embedder import Embedder, get_embedder
from ingestion.chunker import Chunk, chunk_document
from ingestion.loader import load_directory
from vectorstore.store import VectorStore

EMBEDDING_BATCH_SIZE = 64


def _generate_chunk_id(chunk: Chunk) -> str:
    """
    Generate a deterministic ID for a chunk based on filename and index.

    Ensures re-ingestion overwrites existing chunks instead of creating
    duplicates.

    Args:
        chunk (Chunk): The chunk to generate an ID for.

    Returns:
        str: Deterministic chunk identifier.
    """
    filename = chunk.metadata.get("filename", "unknown")
    index = chunk.metadata.get("chunk_index", 0)
    return f"{filename}::chunk_{index}"


def run_ingestion(
    settings: Settings,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Run the full ingestion pipeline: load → chunk → embed → store.

    Args:
        settings (Settings): Application configuration.
        embedder (Embedder | None): Embedder instance; created from settings if None.
        store (VectorStore | None): Vector store instance; created from settings if None.
        progress_callback (Callable[[int, int], None] | None): Optional callback
            receiving (chunks_processed, total_chunks) for progress tracking.

    Returns:
        int: Total number of chunks ingested.

    Raises:
        FileNotFoundError: If the documents directory does not exist.
    """
    documents_dir = Path(settings.documents_dir).resolve()

    if embedder is None:
        embedder = get_embedder(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
        )

    if store is None:
        store = VectorStore(
            persist_dir=settings.chroma_path,
            collection_name=settings.collection_name,
        )

    # Step 1: Load documents
    documents = load_directory(documents_dir)
    if not documents:
        return 0

    # Step 2: Chunk all documents
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(
            text=doc.text,
            metadata=doc.metadata,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    total = len(all_chunks)

    # Step 3: Embed and store in batches
    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total)
        batch = all_chunks[batch_start:batch_end]

        texts = [c.text for c in batch]
        ids = [_generate_chunk_id(c) for c in batch]
        metadatas = [c.metadata for c in batch]

        embeddings = embedder.embed_batch(texts)

        store.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        if progress_callback:
            progress_callback(batch_end, total)

    return total
