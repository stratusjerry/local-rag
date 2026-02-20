# Task Tracking

## Completed

- [x] **Project scaffolding** — directory structure, requirements.txt, .env.example, .gitignore, config/settings.py (2026-02-20)
- [x] **Document loader** — ingestion/loader.py with .txt, .docx, .pptx support and load_directory() (2026-02-20)
- [x] **Text chunker** — ingestion/chunker.py with recursive character splitting (2026-02-20)
- [x] **Embeddings module** — embeddings/embedder.py with Ollama and SentenceTransformer providers (2026-02-20)
- [x] **Vector store** — vectorstore/store.py wrapping ChromaDB PersistentClient (2026-02-20)
- [x] **Ingestion pipeline** — ingestion/pipeline.py orchestrating load/chunk/embed/store (2026-02-20)
- [x] **RAG query chain** — rag/chain.py, rag/prompts.py, llm/ollama_client.py (2026-02-20)
- [x] **Streamlit UI** — ui/app.py with sidebar config, chat interface, streaming, source attribution (2026-02-20)
- [x] **Unit tests** — tests for loader, chunker, store, and chain (31 tests passing) (2026-02-20)
- [x] **Documentation** — PLANNING.md, TASK.md, README.md (2026-02-20)

## Discovered During Work

- ChromaDB (v1.5.1) is not compatible with Python 3.14 due to pydantic v1 dependency. Must use Python 3.13.
