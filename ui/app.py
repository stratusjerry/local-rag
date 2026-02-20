"""
Streamlit chat UI for the Local RAG application.

Provides a sidebar for configuration and ingestion controls, and a main
chat area for querying documents with source attribution.
"""

import sys
from pathlib import Path

import streamlit as st

# Reason: add project root to sys.path so imports work when running
# with `streamlit run ui/app.py` from the project root.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.settings import get_settings
from embeddings.embedder import Embedder, get_embedder
from ingestion.pipeline import run_ingestion
from llm import get_llm_client
from llm.base import LLMClient
from rag.chain import query_rag_stream
from vectorstore.store import VectorStore

# --- Page config ---
st.set_page_config(
    page_title="Local RAG",
    page_icon="📄",
    layout="wide",
)

LLM_PROVIDERS = ["ollama", "lmstudio", "bedrock"]


# --- Cached resources ---
@st.cache_resource
def get_embedder_cached(provider: str, model: str) -> Embedder:
    """
    Create and cache the embedder to avoid re-initialization on reruns.

    Args:
        provider (str): Embedding provider name.
        model (str): Embedding model name.

    Returns:
        Embedder: Cached embedder instance.
    """
    return get_embedder(provider=provider, model=model)


@st.cache_resource
def get_store_cached(persist_dir: str, collection_name: str) -> VectorStore:
    """
    Create and cache the vector store to avoid re-initialization on reruns.

    Args:
        persist_dir (str): ChromaDB persistent storage path.
        collection_name (str): ChromaDB collection name.

    Returns:
        VectorStore: Cached vector store instance.
    """
    return VectorStore(persist_dir=persist_dir, collection_name=collection_name)


@st.cache_resource
def get_llm_cached(
    provider: str,
    lmstudio_url: str = "",
    bedrock_region: str = "",
    bedrock_access_key: str = "",
    bedrock_secret_key: str = "",
    bedrock_session_token: str = "",
    bedrock_endpoint_url: str = "",
    bedrock_ca_bundle: str = "",
) -> LLMClient:
    """
    Create and cache the LLM client to avoid re-initialization on reruns.

    Args:
        provider (str): LLM provider name.
        lmstudio_url (str): LM Studio API URL.
        bedrock_region (str): AWS region for Bedrock.
        bedrock_access_key (str): AWS access key.
        bedrock_secret_key (str): AWS secret key.
        bedrock_session_token (str): AWS session token.
        bedrock_endpoint_url (str): Custom Bedrock endpoint URL.
        bedrock_ca_bundle (str): Path to custom CA bundle .pem file.

    Returns:
        LLMClient: Cached LLM client instance.
    """
    return get_llm_client(
        provider=provider,
        lmstudio_url=lmstudio_url,
        bedrock_region=bedrock_region,
        bedrock_access_key=bedrock_access_key,
        bedrock_secret_key=bedrock_secret_key,
        bedrock_session_token=bedrock_session_token,
        bedrock_endpoint_url=bedrock_endpoint_url,
        bedrock_ca_bundle=bedrock_ca_bundle,
    )


# --- Session state initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


def _render_sidebar() -> None:
    """
    Render the sidebar with configuration controls and ingestion buttons.
    """
    settings = get_settings()

    with st.sidebar:
        st.header("Configuration")

        # Document directory
        docs_dir = st.text_input(
            "Documents Directory",
            value=settings.documents_dir,
            help="Path to the folder containing documents to ingest.",
        )

        # --- LLM provider ---
        llm_provider = st.selectbox(
            "LLM Provider",
            options=LLM_PROVIDERS,
            index=LLM_PROVIDERS.index(settings.llm_provider)
            if settings.llm_provider in LLM_PROVIDERS
            else 0,
        )

        # Provider-specific settings
        llm_model = settings.llm_model
        lmstudio_url = settings.lmstudio_url
        bedrock_region = settings.bedrock_region
        bedrock_model_id = settings.bedrock_model_id
        bedrock_access_key = settings.bedrock_access_key
        bedrock_secret_key = settings.bedrock_secret_key
        bedrock_session_token = settings.bedrock_session_token
        bedrock_endpoint_url = settings.bedrock_endpoint_url
        bedrock_ca_bundle = settings.bedrock_ca_bundle

        if llm_provider == "ollama":
            llm_model = st.text_input(
                "LLM Model",
                value=settings.llm_model,
                help="Ollama model name (e.g., llama3.2).",
            )
        elif llm_provider == "lmstudio":
            lmstudio_url = st.text_input(
                "LM Studio URL",
                value=settings.lmstudio_url,
                help="LM Studio API endpoint.",
            )
            llm_model = st.text_input(
                "LLM Model",
                value=settings.llm_model,
                help="Model identifier loaded in LM Studio.",
            )
        elif llm_provider == "bedrock":
            bedrock_region = st.text_input(
                "AWS Region",
                value=settings.bedrock_region,
            )
            bedrock_model_id = st.text_input(
                "Bedrock Model ID",
                value=settings.bedrock_model_id,
                help="e.g., us.anthropic.claude-sonnet-4-5-v1",
            )
            bedrock_access_key = st.text_input(
                "AWS Access Key",
                value=settings.bedrock_access_key,
                type="password",
                help="Leave blank to use default AWS credential chain.",
            )
            bedrock_secret_key = st.text_input(
                "AWS Secret Key",
                value=settings.bedrock_secret_key,
                type="password",
            )
            bedrock_session_token = st.text_input(
                "AWS Session Token",
                value=settings.bedrock_session_token,
                type="password",
                help="Optional. For temporary credentials.",
            )
            bedrock_endpoint_url = st.text_input(
                "Custom Endpoint URL",
                value=settings.bedrock_endpoint_url,
                help="Optional. For isolated/air-gapped regions.",
            )
            bedrock_ca_bundle = st.text_input(
                "Custom CA Bundle Path",
                value=settings.bedrock_ca_bundle,
                help="Optional. Path to a .pem CA bundle for custom CAs.",
            )

        # Embedding provider
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["ollama", "sentence_transformers"],
            index=0 if settings.embedding_provider == "ollama" else 1,
        )

        # Embedding model
        default_embed = (
            "nomic-embed-text"
            if embedding_provider == "ollama"
            else "all-MiniLM-L6-v2"
        )
        embedding_model = st.text_input(
            "Embedding Model",
            value=default_embed,
        )

        # Retrieval settings
        top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=settings.top_k)
        chunk_size = st.number_input(
            "Chunk Size", min_value=100, max_value=5000, value=settings.chunk_size, step=100
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap", min_value=0, max_value=1000, value=settings.chunk_overlap, step=50
        )

        # Store all overrides in session state
        st.session_state.docs_dir = docs_dir
        st.session_state.llm_provider = llm_provider
        st.session_state.llm_model = llm_model
        st.session_state.lmstudio_url = lmstudio_url
        st.session_state.bedrock_region = bedrock_region
        st.session_state.bedrock_model_id = bedrock_model_id
        st.session_state.bedrock_access_key = bedrock_access_key
        st.session_state.bedrock_secret_key = bedrock_secret_key
        st.session_state.bedrock_session_token = bedrock_session_token
        st.session_state.bedrock_endpoint_url = bedrock_endpoint_url
        st.session_state.bedrock_ca_bundle = bedrock_ca_bundle
        st.session_state.embedding_provider = embedding_provider
        st.session_state.embedding_model = embedding_model
        st.session_state.top_k = top_k
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

        st.divider()

        # LLM connectivity check
        llm_client = _get_current_llm_client()
        connected, status_msg = llm_client.check_connectivity()
        if connected:
            st.success(status_msg)
        else:
            st.error(status_msg)

        # Show chunk count
        store = get_store_cached(
            persist_dir=str(settings.chroma_path),
            collection_name=settings.collection_name,
        )
        st.session_state.chunk_count = store.count()
        st.metric("Stored Chunks", st.session_state.chunk_count)

        st.divider()

        # Ingest button
        if st.button("Ingest Documents", type="primary", use_container_width=True):
            _run_ingestion(settings)

        # Clear chat
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Reset database
        if st.button("Reset Database", use_container_width=True):
            store.reset()
            st.session_state.chunk_count = 0
            st.success("Database cleared.")
            st.rerun()


def _get_current_llm_client() -> LLMClient:
    """
    Build an LLM client from the current session state settings.

    Returns:
        LLMClient: The configured LLM client.
    """
    provider = st.session_state.get("llm_provider", "ollama")
    return get_llm_cached(
        provider=provider,
        lmstudio_url=st.session_state.get("lmstudio_url", "http://localhost:1234/v1"),
        bedrock_region=st.session_state.get("bedrock_region", "us-east-1"),
        bedrock_access_key=st.session_state.get("bedrock_access_key", ""),
        bedrock_secret_key=st.session_state.get("bedrock_secret_key", ""),
        bedrock_session_token=st.session_state.get("bedrock_session_token", ""),
        bedrock_endpoint_url=st.session_state.get("bedrock_endpoint_url", ""),
        bedrock_ca_bundle=st.session_state.get("bedrock_ca_bundle", ""),
    )


def _run_ingestion(settings) -> None:
    """
    Execute the ingestion pipeline with a progress bar in the sidebar.

    Args:
        settings: Application settings (used as base, overridden by sidebar values).
    """
    # Override settings with sidebar values
    settings.documents_dir = st.session_state.docs_dir
    settings.embedding_provider = st.session_state.embedding_provider
    settings.embedding_model = st.session_state.embedding_model
    settings.chunk_size = st.session_state.chunk_size
    settings.chunk_overlap = st.session_state.chunk_overlap

    docs_path = Path(settings.documents_dir).resolve()
    if not docs_path.exists():
        st.error(f"Directory not found: {docs_path}")
        return

    embedder = get_embedder_cached(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
    )
    store = get_store_cached(
        persist_dir=str(settings.chroma_path),
        collection_name=settings.collection_name,
    )

    progress_bar = st.progress(0, text="Ingesting documents...")

    def update_progress(current: int, total: int) -> None:
        progress_bar.progress(current / total, text=f"Processing chunk {current}/{total}")

    try:
        count = run_ingestion(
            settings=settings,
            embedder=embedder,
            store=store,
            progress_callback=update_progress,
        )
        progress_bar.progress(1.0, text="Ingestion complete!")
        st.session_state.chunk_count = store.count()
        st.success(f"Ingested {count} chunks from documents.")
    except Exception as e:
        st.error(f"Ingestion failed: {e}")


def _render_chat() -> None:
    """
    Render the main chat area with message history and input.
    """
    st.title("Local RAG Chat")
    st.caption("Ask questions about your ingested documents.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            _generate_response(prompt)


def _generate_response(query: str) -> None:
    """
    Generate and stream an LLM response for the user's query.

    Args:
        query (str): The user's question.
    """
    settings = get_settings()
    settings.llm_provider = st.session_state.get("llm_provider", settings.llm_provider)
    settings.llm_model = st.session_state.get("llm_model", settings.llm_model)
    settings.top_k = st.session_state.get("top_k", settings.top_k)
    settings.embedding_provider = st.session_state.get(
        "embedding_provider", settings.embedding_provider
    )
    settings.embedding_model = st.session_state.get(
        "embedding_model", settings.embedding_model
    )
    settings.bedrock_model_id = st.session_state.get(
        "bedrock_model_id", settings.bedrock_model_id
    )

    embedder = get_embedder_cached(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
    )
    store = get_store_cached(
        persist_dir=str(settings.chroma_path),
        collection_name=settings.collection_name,
    )

    if store.count() == 0:
        response = "No documents have been ingested yet. Please ingest documents using the sidebar before asking questions."
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        return

    # Build chat history for context (exclude sources metadata)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # Exclude current user message
        if m["role"] in ("user", "assistant")
    ]

    llm_client = _get_current_llm_client()

    try:
        stream, sources = query_rag_stream(
            query=query,
            embedder=embedder,
            store=store,
            llm_client=llm_client,
            settings=settings,
            chat_history=chat_history[-10:],  # Limit history to last 10 messages
        )

        # Stream the response
        response = st.write_stream(stream)

        # Show sources
        source_data = [
            {"filename": s.metadata.get("filename", "Unknown"), "distance": s.distance}
            for s in sources
        ]
        _render_sources(source_data)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": source_data,
        })

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


def _render_sources(sources: list[dict]) -> None:
    """
    Render source attribution below a response.

    Args:
        sources (list[dict]): List of source dicts with filename and distance.
    """
    if not sources:
        return

    with st.expander("Sources"):
        for src in sources:
            filename = src.get("filename", "Unknown")
            distance = src.get("distance", 0)
            similarity = max(0, 1 - distance)  # Convert distance to similarity
            st.markdown(f"- **{filename}** (relevance: {similarity:.1%})")


def main() -> None:
    """
    Main entry point for the Streamlit application.
    """
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
