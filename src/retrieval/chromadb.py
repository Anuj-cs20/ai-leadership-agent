"""ChromaDB vector store wrapper.

Provides functions to create/connect to a ChromaDB collection,
load an existing vector index, list collections, and clear data.
The HNSW distance metric (``space``) is read from config.yaml.
"""

import logging

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import RAGConfig

logger = logging.getLogger(__name__)


def get_vector_store(config: RAGConfig) -> ChromaVectorStore:
    """Create or connect to a ChromaDB vector store.

    Uses ``config.chromadb.space`` for the HNSW distance metric (default: cosine).
    """
    client = chromadb.PersistentClient(path=config.chromadb.persist_directory)
    collection = client.get_or_create_collection(
        name=config.chromadb.collection,
        metadata={"hnsw:space": config.chromadb.space},
    )
    logger.info(
        f"ChromaDB collection '{config.chromadb.collection}': "
        f"{collection.count()} existing vectors (space={config.chromadb.space})"
    )
    return ChromaVectorStore(chroma_collection=collection)


def load_index(config: RAGConfig) -> VectorStoreIndex:
    """Load an existing vector index from ChromaDB."""
    vector_store = get_vector_store(config)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def list_collections(config: RAGConfig) -> list:
    """Return a list of all collection names in the ChromaDB instance."""
    client = chromadb.PersistentClient(path=config.chromadb.persist_directory)
    collections = client.list_collections()
    return [(c.name, c.count()) for c in collections]


def clear_collection(config: RAGConfig) -> None:
    """Delete and recreate the ChromaDB collection (for re-ingestion)."""
    client = chromadb.PersistentClient(path=config.chromadb.persist_directory)
    try:
        client.delete_collection(config.chromadb.collection)
        logger.info(f"Deleted collection '{config.chromadb.collection}'")
    except Exception:
        pass
