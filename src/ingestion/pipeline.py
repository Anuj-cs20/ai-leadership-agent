"""Ingestion pipeline: parse → chunk → embed → index.

Provides:
  - ``filter_files()``  — discover supported documents in a directory
  - ``get_indexed_files()`` — query ChromaDB metadata for already-indexed filenames
  - ``ingest_documents()`` — full pipeline with smart re-indexing (skips already-indexed files)
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex

from src.config import RAGConfig
from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import parse_documents
from src.retrieval.chromadb import get_vector_store

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".md", ".txt"}


def filter_files(data_dir: str = "data/sample") -> List[str]:
    """Find all supported documents in the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(
        str(f) for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def get_indexed_files(config: RAGConfig) -> Set[str]:
    """Return the set of filenames already indexed in ChromaDB.

    Queries the ChromaDB collection metadata to find unique ``filename`` values.
    This is used by ``ingest_documents()`` to skip already-indexed files.
    """
    try:
        client = chromadb.PersistentClient(path=config.chromadb.persist_directory)
        collection = client.get_or_create_collection(name=config.chromadb.collection)
        if collection.count() == 0:
            return set()
        # Fetch all metadata entries and extract unique filenames
        result = collection.get(include=["metadatas"])
        filenames = {m.get("filename", "") for m in result["metadatas"] if m}
        filenames.discard("")
        return filenames
    except Exception as e:
        logger.warning(f"Could not query indexed files: {e}")
        return set()


def ingest_documents(
    config: RAGConfig,
    data_dir: str = "data/sample",
    file_paths: Optional[List[str]] = None,
    reindex: bool = False,
) -> VectorStoreIndex:
    """Run the full ingestion pipeline.

    1. Discover / accept document file paths
    2. **Smart re-indexing**: skip files already in ChromaDB (unless ``reindex=True``)
    3. Parse documents (DoclingReader for DOCX/XLSX/PPTX, pypdf for PDF, direct for TXT)
    4. Chunk into text/table nodes with metadata
    5. Embed and build VectorStoreIndex backed by ChromaDB

    Args:
        config: RAGConfig with retrieval and ChromaDB settings
        data_dir: directory to scan for documents
        file_paths: explicit list of files (overrides directory scan)
        reindex: if True, re-ingest ALL files even if already indexed

    Returns:
        VectorStoreIndex ready for querying.
    """
    # Step 1: Collect file paths
    if file_paths is None:
        file_paths = filter_files(data_dir)

    if not file_paths:
        raise ValueError(f"No documents found in {data_dir}")

    # Step 2: Smart re-indexing — skip already indexed files
    if not reindex:
        indexed = get_indexed_files(config)
        if indexed:
            original_count = len(file_paths)
            file_paths = [
                fp for fp in file_paths
                if Path(fp).name not in indexed
            ]
            skipped = original_count - len(file_paths)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already-indexed files")
            if not file_paths:
                logger.info("All files already indexed — loading existing index")
                vector_store = get_vector_store(config)
                return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    logger.info(f"Ingesting {len(file_paths)} documents from {data_dir}")
    for fp in file_paths:
        logger.info(f"  → {Path(fp).name}")

    # Step 3: Parse
    documents = parse_documents(file_paths)
    if not documents:
        raise ValueError("No documents were successfully parsed")

    # Step 4: Chunk
    nodes = chunk_documents(
        documents,
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
    )
    logger.info(f"Created {len(nodes)} chunks from {len(documents)} documents")

    # Step 5: Build vector index
    vector_store = get_vector_store(config)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("Vector index built successfully")
    return index
