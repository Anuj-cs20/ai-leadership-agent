"""Configuration loader for the AI Leadership Insight Agent.

Uses Pydantic BaseModel for configuration because:
  1. **Type safety** — every config value is validated at load time (e.g. temperature must be float).
  2. **Defaults** — each field has a sensible default so the system works even without config.yaml.
  3. **Nested structure** — BaseModel subclasses mirror the YAML sections, making YAML deserialization
     trivial with ``RAGConfig(**yaml_data)``.
  4. **IDE support** — auto-complete and type hints for every config field.
  5. **Serialization** — ``.model_dump()`` / ``.model_json_schema()`` for logging and debugging.

LlamaIndex Settings (global singleton):
  LlamaIndex uses a global ``Settings`` object to inject the LLM and embedding model into every
  component (index, retriever, query engine) without passing them explicitly.  We call
  ``initialize_llama_index()`` once at startup to configure ``Settings.llm``,
  ``Settings.embed_model``, ``Settings.chunk_size``, and ``Settings.chunk_overlap``.
  After that, any LlamaIndex component automatically uses these models.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# ---------------------------------------------------------------------------
# Pydantic config models — one per YAML section
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """OpenAI LLM settings."""
    model: str = Field("gpt-4o-mini", description="OpenAI model name")
    temperature: float = Field(0.1, description="Sampling temperature (low = deterministic)")


class EmbeddingConfig(BaseModel):
    """OpenAI embedding model settings."""
    model: str = Field("text-embedding-3-small", description="OpenAI embedding model (1536 dims)")


class RetrievalConfig(BaseModel):
    """Retrieval and chunking parameters."""
    top_k: int = Field(10, description="Number of chunks to retrieve per query")
    chunk_size: int = Field(1024, description="Target chunk size in tokens for SentenceSplitter")
    chunk_overlap: int = Field(200, description="Overlap between consecutive text chunks")
    mmr_threshold: float = Field(
        0.5,
        description=(
            "MMR (Maximal Marginal Relevance) diversity threshold. "
            "0.0 = maximum diversity, 1.0 = pure similarity."
        ),
    )


class ChromaDBConfig(BaseModel):
    """ChromaDB vector store settings."""
    collection: str = Field("company_docs", description="ChromaDB collection name")
    persist_directory: str = Field("./chroma_db", description="Path to persistent ChromaDB storage")
    space: str = Field(
        "cosine",
        description="HNSW distance metric: 'cosine', 'l2', or 'ip' (inner product)",
    )


class RAGConfig(BaseModel):
    """Top-level configuration — mirrors config.yaml structure.

    Example::
        config = load_config("config.yaml")
        config.llm.model          # "gpt-4o-mini"
        config.retrieval.top_k    # 10
        config.chromadb.space     # "cosine"
    """
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chromadb: ChromaDBConfig = ChromaDBConfig()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> RAGConfig:
    """Load configuration from a YAML file, falling back to Pydantic defaults."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return RAGConfig(**data)
    return RAGConfig()


# ---------------------------------------------------------------------------
# LlamaIndex global initialization
# ---------------------------------------------------------------------------

def initialize_llama_index(config: RAGConfig) -> None:
    """Set up LlamaIndex's global Settings singleton.

    Configures the LLM, embedding model, and chunking parameters once.
    Every LlamaIndex component created afterwards (VectorStoreIndex,
    retriever, query engine) automatically uses these settings.
    """
    Settings.llm = OpenAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )
    print(f"✓ LLM: OpenAI ({config.llm.model})")

    Settings.embed_model = OpenAIEmbedding(model=config.embedding.model)
    print(f"✓ Embedding: OpenAI ({config.embedding.model})")

    Settings.chunk_size = config.retrieval.chunk_size
    Settings.chunk_overlap = config.retrieval.chunk_overlap
