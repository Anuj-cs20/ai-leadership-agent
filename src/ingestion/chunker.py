"""Table-aware chunking using LlamaIndex's MarkdownNodeParser.

**What does MarkdownNodeParser do?**
  DoclingReader exports every document as markdown.  MarkdownNodeParser splits that
  markdown by *structure* — headers, code blocks, and tables — instead of raw character
  counts.  This produces semantically coherent chunks where each chunk corresponds to
  one section or table from the original document.

  Example — given this markdown input:
      ## Executive Summary
      Revenue grew 23% YoY to $188.5M …
      ## Risk Factors
      1. Client Concentration …
  MarkdownNodeParser produces TWO nodes:
      Node 1 → "Executive Summary\nRevenue grew 23% YoY …"
      Node 2 → "Risk Factors\n1. Client Concentration …"

**SentenceSplitter post-pass:**
  Some markdown sections can be very long (e.g. a 3-page narrative section).
  After the markdown-aware split, SentenceSplitter breaks any node that exceeds
  ``chunk_size`` tokens, respecting sentence boundaries so no sentence is cut mid-way.

**Assumption — atomic table chunks:**
  Markdown pipe-tables (e.g. KPI dashboards, risk registers) are kept as single
  atomic chunks and are NEVER split by SentenceSplitter.  This ensures the LLM
  always sees a complete table with headers and all rows.  The trade-off is that
  an oversized table may exceed ``chunk_size``, but splitting it would destroy
  the tabular structure and make the data meaningless.
"""

import logging
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


def is_table_chunk(text: str) -> bool:
    """Detect whether a chunk is primarily a markdown pipe-table.

    Heuristic: if ≥ 2 lines start with ``|`` and contain ≥ 2 pipe characters,
    and those lines make up > 40 % of the chunk, it's a table.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return False
    table_lines = sum(1 for line in lines if line.startswith("|") and line.count("|") >= 2)
    return table_lines >= 2 and table_lines / len(lines) > 0.4


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
) -> List[TextNode]:
    """Split documents into chunks with markdown-aware handling.

    Pipeline:
      1. **MarkdownNodeParser** — splits on markdown structure (headers, tables, etc.)
      2. **SentenceSplitter** — post-pass for any text chunk exceeding ``chunk_size``
      3. Table chunks (markdown pipe-tables) are tagged ``chunk_type=table`` and kept atomic

    All original document metadata is preserved on every chunk.
    """
    # Step 1: Markdown-aware splitting
    md_parser = MarkdownNodeParser()
    md_nodes = md_parser.get_nodes_from_documents(documents)

    # Step 2: Post-split oversized text chunks with SentenceSplitter
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_nodes: List[TextNode] = []

    for node in md_nodes:
        text = node.get_content()

        # Detect table chunks (markdown pipe-tables)
        is_table = is_table_chunk(text)

        if is_table:
            # ASSUMPTION: tables are kept atomic — never split them.
            # Risk: an oversized table may exceed chunk_size, but splitting
            # would destroy the tabular structure (headers separated from rows).
            tnode = TextNode(
                text=text,
                metadata={
                    **node.metadata,
                    "chunk_type": "table",
                },
                excluded_llm_metadata_keys=["parser", "file_type"],
                excluded_embed_metadata_keys=["parser", "file_type"],
            )
            all_nodes.append(tnode)
        else:
            # Split text chunks that exceed chunk_size
            sub_nodes = splitter.get_nodes_from_documents(
                [Document(text=text, metadata=node.metadata)]
            )
            for snode in sub_nodes:
                snode.metadata["chunk_type"] = "text"
                snode.excluded_llm_metadata_keys = ["parser", "file_type"]
                snode.excluded_embed_metadata_keys = ["parser", "file_type"]
                all_nodes.append(snode)

    table_count = sum(1 for n in all_nodes if n.metadata.get("chunk_type") == "table")
    text_count = len(all_nodes) - table_count
    logger.info(f"Chunking complete: {len(all_nodes)} total ({text_count} text, {table_count} table)")
    return all_nodes


