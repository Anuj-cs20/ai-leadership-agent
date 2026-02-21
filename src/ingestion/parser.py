"""Document parser using DoclingReader with fallback chain.

**Why DoclingReader (IBM Docling)?**
  - Converts complex formats (DOCX tables, XLSX multi-sheet, PPTX slides) into clean
    markdown, which downstream MarkdownNodeParser can split semantically.
  - Preserves table structure as markdown pipe-tables — critical for financial KPI data.
  - Single unified interface for DOCX / XLSX / PPTX / CSV.

**Limitation — PDF excluded from Docling:**
  Docling's PDF pipeline uses OpenCV internally, which requires ``libGL.so.1``.
  This shared library is absent in lightweight containers (GitHub Codespaces, Docker slim).
  Rather than installing system packages, we use ``pypdf`` as a lightweight PDF fallback.

**Parser strategy (per document):**
  1. If format is in ``DOCLING_FORMATS`` → try DoclingReader first.
  2. If DoclingReader fails or format is not supported → try format-specific fallback
     (pypdf for PDF, pandas for XLSX/CSV, direct read for TXT/MD).
  3. Every successfully parsed Document is enriched with custom metadata via
     ``create_metadata()``.

**Metadata enrichment:**
  - ``doc_type``: classified from filename (annual, quarterly, strategy, …)
  - ``year`` / ``quarter``: extracted from filename patterns like ``2024`` or ``Q3``
  - ``source_path``: absolute path to original file on disk
  - ``ingestion_date``: ISO-8601 timestamp of when the document was parsed
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.readers.docling import DoclingReader
from pypdf import PdfReader
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata helpers (public — used by pipeline and notebook cells)
# ---------------------------------------------------------------------------

def classify_doc_type(filename: str) -> str:
    """Classify document type from filename patterns.

    Assumption: filenames follow a naming convention where the document purpose
    is encoded in the name (e.g. ``annual_report_2024.pdf``, ``q3_quarterly_report.pdf``).
    This is a heuristic — if filenames don't match any pattern, "general" is returned.
    """
    name = filename.lower()
    if "annual" in name or "yearly" in name:
        return "annual"
    if any(f"q{i}" in name for i in range(1, 5)) or "quarterly" in name:
        return "quarterly"
    if "strateg" in name:
        return "strategy"
    if any(w in name for w in ("kpi", "dashboard", "metric")):
        return "operational"
    if "operational" in name or "update" in name:
        return "operational"
    if "risk" in name:
        return "risk"
    if "board" in name or "meeting" in name:
        return "meeting"
    if "directive" in name or "memo" in name:
        return "directive"
    if "roadmap" in name or "presentation" in name:
        return "presentation"
    return "general"


def extract_time_info(filename: str) -> dict:
    """Extract year and quarter from filename.

    Assumption: years are 4-digit strings starting with '20' (e.g. 2024),
    quarters are Q1–Q4 (case-insensitive).
    """
    meta = {}
    year = re.search(r"20\d{2}", filename)
    if year:
        meta["year"] = year.group()
    quarter = re.search(r"[Qq]([1-4])", filename)
    if quarter:
        meta["quarter"] = f"Q{quarter.group(1)}"
    return meta


def create_metadata(doc: Document, file_path: Path, parser: str) -> Document:
    """Attach custom metadata to a parsed Document.

    Added fields:
      - "filename": base filename for source citation
      - "source_path": absolute path on disk (for re-indexing / dedup)
      - "doc_type": heuristic classification
      - "file_type": file extension
      - "parser": which parser produced this document
      - "ingestion_date": ISO-8601 UTC timestamp
      - "year" / "quarter": extracted from filename (if present)

    "parser" and "file_type" are excluded from LLM/embedding context
    to keep token usage efficient.
    """
    doc.metadata.update({
        "filename": file_path.name,
        "source_path": str(file_path.resolve()),
        "doc_type": classify_doc_type(file_path.name),
        "file_type": file_path.suffix.lower(),
        "parser": parser,
        "ingestion_date": datetime.now(timezone.utc).isoformat(),
        **extract_time_info(file_path.name),
    })
    doc.excluded_llm_metadata_keys = ["parser", "file_type", "source_path", "ingestion_date"]
    doc.excluded_embed_metadata_keys = ["parser", "file_type", "source_path", "ingestion_date"]
    return doc


# ---------------------------------------------------------------------------
# Primary parser: DoclingReader
# ---------------------------------------------------------------------------

def load_docling_reader():
    """Try to instantiate a DoclingReader (LlamaIndex wrapper around IBM Docling).

    Returns None if neither llama-index-readers-docling nor raw docling is installed.
    """
    try:
        return DoclingReader(export_type="markdown")
    except ImportError:
        logger.info("llama-index-readers-docling not installed, trying raw docling")

    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
        return RawDoclingAdapter()
    except ImportError:
        return None


class RawDoclingAdapter:
    """Minimal adapter so raw docling has the same ``.load_data()`` API as DoclingReader."""

    def load_data(self, file_path: str) -> List[Document]:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
        return [Document(text=text)]


# ---------------------------------------------------------------------------
# Fallback parsers
# ---------------------------------------------------------------------------

def parse_with_pypdf(file_path: str) -> List[Document]:
    """Parse PDF using pypdf as lightweight fallback (no libGL dependency)."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError("pypdf extracted no text")
    return [Document(text=text)]


def parse_with_pandas(file_path: str) -> List[Document]:
    """Parse Excel/CSV files using pandas as fallback.

    For XLSX: reads every sheet and formats each as a markdown table.
    For CSV: reads the single file as a markdown table.
    """
    path = Path(file_path)
    if path.suffix.lower() == ".xlsx":
        xls = pd.ExcelFile(path)
        sheets = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheets.append(f"## Sheet: {sheet_name}\n\n{df.to_markdown(index=False)}")
        return [Document(text="\n\n".join(sheets))]
    else:
        df = pd.read_csv(path)
        return [Document(text=df.to_markdown(index=False))]


def parse_direct(file_path: str) -> List[Document]:
    """Read text/markdown files directly — no conversion needed."""
    text = Path(file_path).read_text(encoding="utf-8")
    return [Document(text=text)]


# ---------------------------------------------------------------------------
# Parser configuration
# ---------------------------------------------------------------------------

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".md", ".txt"}

# Mapping: suffix → ordered list of (parser_name, parser_function) to try as fallback
FALLBACKS = {
    ".pdf":  [("pypdf", parse_with_pypdf), ("direct", parse_direct)],
    ".docx": [("direct", parse_direct)],
    ".pptx": [],
    ".xlsx": [("pandas", parse_with_pandas)],
    ".csv":  [("pandas", parse_with_pandas)],
    ".md":   [("direct", parse_direct)],
    ".txt":  [("direct", parse_direct)],
}

# Formats that DoclingReader handles natively.
# NOTE: PDF is excluded because Docling's PDF pipeline needs libGL.so.1 (OpenCV),
# which is missing in GitHub Codespaces / lightweight containers.
# pypdf handles PDFs without any system dependencies.
DOCLING_FORMATS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_documents(file_paths: List[str]) -> List[Document]:
    """Parse a list of file paths into LlamaIndex Document objects.

    Strategy per file:
      1. Try DoclingReader if format is in DOCLING_FORMATS
      2. Try TXT/MD directly (no value in sending plain text through docling)
      3. Try fallback parsers in order
      4. Skip file if all parsers fail

    Every successful Document gets metadata via ``create_metadata()``.
    """
    docling = load_docling_reader()
    if docling:
        logger.info("DoclingReader available — using AI-based document parsing")
    else:
        logger.warning("docling not available — using fallback parsers only")

    documents = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED:
            logger.warning(f"Unsupported format: {suffix} ({path.name})")
            continue

        docs = None
        parser_used = None

        # 1. Try DoclingReader for rich formats (DOCX, XLSX, PPTX, CSV)
        if docling and suffix in DOCLING_FORMATS:
            try:
                docs = docling.load_data(str(path))
                parser_used = "docling"
            except Exception as e:
                logger.warning(f"DoclingReader failed for {path.name}: {e}")

        # 2. Try TXT/MD directly (plain text doesn't benefit from docling)
        if docs is None and suffix in (".md", ".txt"):
            try:
                docs = parse_direct(str(path))
                parser_used = "direct"
            except Exception as e:
                logger.warning(f"Direct read failed for {path.name}: {e}")

        # 3. Try fallback parsers in configured order
        if docs is None:
            for name, fn in FALLBACKS.get(suffix, []):
                try:
                    docs = fn(str(path))
                    parser_used = name
                    break
                except Exception as e:
                    logger.warning(f"{name} fallback failed for {path.name}: {e}")

        if not docs:
            logger.error(f"Cannot parse {path.name} — skipping")
            continue

        # Enrich metadata on all returned Documents
        for doc in docs:
            create_metadata(doc, path, parser_used)
            documents.append(doc)

        total_chars = sum(len(d.text) for d in docs)
        logger.info(f"Parsed: {path.name} ({total_chars:,} chars, parser={parser_used})")

    logger.info(f"Total documents parsed: {len(documents)}")
    return documents
