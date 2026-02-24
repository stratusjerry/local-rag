# Local RAG

A fully local Retrieval-Augmented Generation system. Ingest `.docx`, `.ppt`, `.pptx`, and `.txt` files into a ChromaDB vector store, then query them through a Streamlit chat UI powered by your choice of LLM provider.

Supports three LLM backends: **Ollama** (local), **LM Studio** (local), and **AWS Bedrock** (Claude Sonnet 4.6).

## Prerequisites

- **Python 3.13** (3.14 not yet supported by ChromaDB)
- **uv** (recommended) — [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- At least one LLM provider:
  - **Ollama** — [ollama.com](https://ollama.com) — run `ollama pull llama3.2` and `ollama pull nomic-embed-text`
  - **LM Studio** — [lmstudio.ai](https://lmstudio.ai) — load a model and start the local server
  - **AWS Bedrock** — an AWS account with Bedrock model access enabled

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

1. Place your `.docx`, `.ppt`, `.pptx`, and/or `.txt` files in the `files/` directory (subdirectories are supported).

2. Start the app:

```bash
streamlit run ui/app.py
```

3. In the sidebar, select your **LLM Provider** and configure its settings.

4. Click **Ingest Documents** in the sidebar to process your files.

5. Ask questions about your documents in the chat input.

## LLM Providers

### Ollama (default)

Runs models locally via Ollama. Set `RAG_LLM_PROVIDER=ollama` and `RAG_LLM_MODEL` to the model name (e.g., `llama3.2`).

### LM Studio

Connects to LM Studio's OpenAI-compatible local API. Set `RAG_LLM_PROVIDER=lmstudio` and optionally change `RAG_LMSTUDIO_URL` (defaults to `http://localhost:1234/v1`).

### AWS Bedrock

Calls Claude Sonnet 4.6 (or other models) via AWS Bedrock. Set `RAG_LLM_PROVIDER=bedrock` and provide your Bedrock API key:

```bash
RAG_LLM_PROVIDER=bedrock
RAG_BEDROCK_REGION=us-east-1
RAG_BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
RAG_BEDROCK_API_KEY=your-bedrock-api-key
```

If `RAG_BEDROCK_API_KEY` is left blank, the default AWS credential chain is used (environment variables, IAM role, `~/.aws/credentials`).

**Isolated / air-gapped regions:** If your environment uses a custom Certificate Authority or a non-standard endpoint, set these variables (available under **Advanced Settings** in the sidebar):

```bash
RAG_BEDROCK_ENDPOINT_URL=https://bedrock-runtime.us-iso-east-1.c2s.ic.gov
RAG_BEDROCK_CA_BUNDLE=/path/to/custom-ca-bundle.pem
```

## Configuration

All settings can be configured via environment variables (prefix `RAG_`) or the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `RAG_DOCUMENTS_DIR` | `files` | Directory containing documents |
| `RAG_CHROMA_DB_PATH` | `output/chromadb` | ChromaDB storage path |
| `RAG_COLLECTION_NAME` | `local_rag` | ChromaDB collection name |
| `RAG_EMBEDDING_PROVIDER` | `ollama` | `ollama` or `sentence_transformers` |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `RAG_LLM_PROVIDER` | `lmstudio` | `lmstudio`, `bedrock`, or `ollama` |
| `RAG_LLM_MODEL` | `llama3.2` | Model name (Ollama / LM Studio) |
| `RAG_CHUNK_SIZE` | `1000` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Number of results to retrieve |
| `RAG_TEMPERATURE` | `0.7` | LLM sampling temperature |
| `RAG_LMSTUDIO_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `RAG_BEDROCK_REGION` | `us-east-1` | AWS region |
| `RAG_BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID |
| `RAG_BEDROCK_API_KEY` | | Bedrock API key (blank = default AWS chain) |
| `RAG_BEDROCK_ENDPOINT_URL` | | Custom Bedrock endpoint for isolated regions |
| `RAG_BEDROCK_CA_BUNDLE` | | Path to custom CA bundle `.pem` file |

Most of these can also be adjusted in the Streamlit sidebar at runtime.

## Running Tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

## Project Structure

```
config/settings.py         — Pydantic settings (single source of truth)
ingestion/loader.py        — Document text extraction (.docx, .ppt, .pptx, .txt)
ingestion/chunker.py       — Recursive character text splitting
ingestion/pipeline.py      — Load → chunk → embed → store orchestration
vectorstore/store.py       — ChromaDB wrapper
embeddings/embedder.py     — Ollama + SentenceTransformer embedding providers
llm/base.py               — LLM client ABC
llm/ollama_client.py      — Ollama LLM provider
llm/lmstudio_client.py    — LM Studio LLM provider (OpenAI-compatible)
llm/bedrock_client.py     — AWS Bedrock LLM provider (Claude Sonnet 4.6)
rag/chain.py               — RAG query pipeline
rag/prompts.py             — Prompt templates
ui/app.py                  — Streamlit chat interface
tests/                     — Unit tests
```
