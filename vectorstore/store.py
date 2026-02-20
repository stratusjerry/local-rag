"""
ChromaDB vector store wrapper for document storage and retrieval.

Manages a persistent ChromaDB collection with explicit embedding management,
bypassing ChromaDB's built-in embedder to ensure consistency between
ingestion and query embeddings.
"""

from pathlib import Path

import chromadb
from pydantic import BaseModel


class QueryResult(BaseModel):
    """
    A single result from a vector similarity search.

    Attributes:
        text: The chunk text content.
        metadata: Chunk metadata (filename, source, chunk_index, etc.).
        distance: Cosine distance from the query vector (lower is more similar).
    """

    text: str
    metadata: dict
    distance: float


class VectorStore:
    """
    Wrapper around ChromaDB PersistentClient for document chunk storage.

    Uses explicit embeddings (not ChromaDB's built-in embedder) so the same
    embedding model is used for both ingestion and querying.

    Args:
        persist_dir (str | Path): Directory for ChromaDB persistent storage.
        collection_name (str): Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_dir: str | Path = "output/chromadb",
        collection_name: str = "local_rag",
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Insert or update document chunks in the collection.

        Uses deterministic IDs so re-ingestion overwrites rather than
        creating duplicates.

        Args:
            ids (list[str]): Unique chunk identifiers.
            embeddings (list[list[float]]): Embedding vectors.
            documents (list[str]): Chunk text content.
            metadatas (list[dict]): Chunk metadata dicts.
        """
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[QueryResult]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding (list[float]): The query embedding vector.
            n_results (int): Number of results to return.

        Returns:
            list[QueryResult]: Ranked list of similar chunks with distances.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        query_results = []
        if results["documents"] and results["documents"][0]:
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                query_results.append(
                    QueryResult(text=text, metadata=metadata, distance=distance)
                )

        return query_results

    def reset(self) -> None:
        """
        Delete and recreate the collection, removing all stored data.
        """
        collection_name = self._collection.name
        self._client.delete_collection(collection_name)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """
        Return the number of chunks stored in the collection.

        Returns:
            int: Number of stored document chunks.
        """
        return self._collection.count()
