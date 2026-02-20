# Local RAG

A fully local Retrieval-Augmented Generation system. Ingest `.docx`, `.pptx`, and `.txt` files into a ChromaDB vector store, then query them through a Streamlit chat UI powered by Ollama.

Everything runs locally — no data leaves your machine.

## Prerequisites

- **Python 3.13** (3.14 not yet supported by ChromaDB)
- **Ollama** installed and running — [ollama.com](https://ollama.com)
- **uv** (recommended) — [docs.astral.sh/uv](https://docs.astral.sh/uv/)

Pull the required Ollama models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Setup

```bash
# Create virtual environment
uv venv --python 3.13

# Activate it
source .venv/Scripts/activate   # Windows (Git Bash)
# or: .venv\Scripts\activate    # Windows (CMD)
# or: source .venv/bin/activate # macOS/Linux

# Install dependencies
uv pip install -r requirements.txt
```

Copy the example environment file and adjust if needed:

```bash
cp .env.example .env
```

## Usage

1. Place your `.docx`, `.pptx`, and/or `.txt` files in the `files/` directory.

2. Start the app:

```bash
streamlit run ui/app.py
```

3. In the sidebar, click **Ingest Documents** to process your files.

4. Ask questions about your documents in the chat input.

## Configuration

All settings can be configured via environment variables (prefix `RAG_`) or the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `RAG_DOCUMENTS_DIR` | `files` | Directory containing documents |
| `RAG_CHROMA_DB_PATH` | `output/chromadb` | ChromaDB storage path |
| `RAG_COLLECTION_NAME` | `local_rag` | ChromaDB collection name |
| `RAG_EMBEDDING_PROVIDER` | `ollama` | `ollama` or `sentence_transformers` |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `RAG_LLM_MODEL` | `llama3.2` | Ollama chat model |
| `RAG_CHUNK_SIZE` | `1000` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Number of results to retrieve |
| `RAG_TEMPERATURE` | `0.7` | LLM sampling temperature |

Most of these can also be adjusted in the Streamlit sidebar at runtime.

## Running Tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

## Project Structure

```
config/settings.py         — Pydantic settings (single source of truth)
ingestion/loader.py        — Document text extraction (.docx, .pptx, .txt)
ingestion/chunker.py       — Recursive character text splitting
ingestion/pipeline.py      — Load → chunk → embed → store orchestration
vectorstore/store.py       — ChromaDB wrapper
embeddings/embedder.py     — Ollama + SentenceTransformer embedding providers
llm/ollama_client.py       — Ollama chat wrapper with streaming
rag/chain.py               — RAG query pipeline
rag/prompts.py             — Prompt templates
ui/app.py                  — Streamlit chat interface
tests/                     — Unit tests
```
