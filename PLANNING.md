# Local RAG — Architecture & Planning

## Overview

A fully local Retrieval-Augmented Generation (RAG) system that ingests `.docx`, `.pptx`, and `.txt` files, stores embeddings in ChromaDB, and provides a Streamlit chat UI for querying documents via Ollama.

## Stack

- **Python 3.13** (3.14 not yet supported by ChromaDB)
- **ChromaDB** — persistent vector store
- **Ollama** — local LLM and embeddings
- **Streamlit** — chat UI
- **Pydantic** — settings and data validation

## Project Structure

```
local-rag/
├── config/settings.py         # Pydantic BaseSettings (env prefix RAG_)
├── ingestion/
│   ├── loader.py              # Extract text from .docx, .pptx, .txt
│   ├── chunker.py             # Recursive character text splitting
│   └── pipeline.py            # Orchestrates: load → chunk → embed → store
├── vectorstore/store.py       # ChromaDB wrapper (upsert, query, reset)
├── embeddings/embedder.py     # ABC + OllamaEmbedder + SentenceTransformerEmbedder
├── llm/ollama_client.py       # Thin wrapper for ollama.chat()
├── rag/
│   ├── chain.py               # RAG pipeline: embed query → retrieve → prompt → LLM
│   └── prompts.py             # System/user prompt templates
├── ui/app.py                  # Streamlit entry point
├── tests/                     # Pytest unit tests
├── files/                     # Drop documents here (gitignored)
└── output/                    # ChromaDB storage (gitignored)
```

## Key Design Decisions

1. **No LangChain** — all components implemented directly for simplicity and debuggability.
2. **Deterministic chunk IDs** — `{filename}::chunk_{index}` with ChromaDB `upsert()` makes re-ingestion idempotent.
3. **Explicit embedding management** — bypass ChromaDB's built-in embedder to control model and ensure ingest/query consistency.
4. **Streaming LLM responses** — local models can be slow; streaming provides immediate feedback.
5. **Embedder ABC** — `OllamaEmbedder` as primary, `SentenceTransformerEmbedder` as fallback.

## Style Guide

- PEP8 with type hints, formatted with `ruff`
- Google-style docstrings on all functions
- Pydantic models for data validation
- Relative imports within packages
- Max 500 lines per file
