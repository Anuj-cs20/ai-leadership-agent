"""Main RAG generation pipeline for the AI Leadership Agent.

**Query workflow (3 GPT calls per question):**

  Call 1 — Query Rewrite (``rewrite_query`` in query_processor.py):
    Rewrites the user's natural-language question into a search-optimized sentence
    (max 30 words) with synonyms.  Uses ``QUERY_REWRITE_PROMPT``.

  Call 2 — Main Generation:
    Sends the retrieved context + original question + query_type hint to the LLM
    with ``SYSTEM_PROMPT`` (defines persona, JSON schema, rules) and ``CONTEXT_PROMPT``
    (context docs, metadata, chart instructions).  Returns a structured JSON response
    matching the ``QueryResponse`` Pydantic model from output_schema.py.

  Call 3 — Fallback Chart (conditional):
    If the LLM returns ``visualization: null`` for a trend/comparison query, a focused
    follow-up call (``_CHART_FALLBACK_PROMPT``) asks the LLM to produce ONLY the chart
    JSON from the same context.  This is the safety net so charts aren't missed.

**Output schema usage:**
  The LLM is instructed to return JSON matching ``QueryResponse.model_json_schema()``.
  After parsing the JSON, we validate it with ``QueryResponse(**data)`` — Pydantic
  catches any missing or invalid fields.  If JSON parsing fails entirely, a best-effort
  fallback response is built from the raw LLM text.

**MMR (Maximal Marginal Relevance) retrieval:**
  Instead of returning the top-k most similar chunks (which may be near-duplicates),
  MMR iteratively selects chunks that are both relevant to the query AND diverse from
  each other.  The ``mmr_threshold`` controls this trade-off:
    - 0.0 = maximum diversity (most different chunks)
    - 1.0 = pure similarity (standard top-k behavior)
  We read ``mmr_threshold`` from ``config.retrieval.mmr_threshold``.
"""

import json
import logging
from typing import List

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import MetadataMode, NodeWithScore

from src.config import RAGConfig
from src.generation.output_schema import ChartData, QueryResponse, Source
from src.generation.prompts import CONTEXT_PROMPT, SYSTEM_PROMPT
from src.retrieval.query_processor import rewrite_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart fallback prompt — used when GPT call #2 returns visualization: null
# for a trend/comparison query.  GPT call #3 asks for ONLY the chart JSON.
# ---------------------------------------------------------------------------

_CHART_FALLBACK_PROMPT = """Based on the following context, generate ONLY a chart visualization as JSON.

CONTEXT:
{context}

QUESTION: {question}
QUERY TYPE: {query_type}

RULES:
- For "trend" queries: chart_type MUST be "line", labels = time periods, values = numeric metrics
- For "comparison" queries: chart_type MUST be "bar", labels = categories, values = numeric scores/percentages
- Use EXACT numbers from the documents
- Output ONLY a JSON object with keys: chart_type, title, data (with "labels" and "values"), description
- No markdown fences, no explanation — just the JSON object"""


# ---------------------------------------------------------------------------
# Helpers (public — no underscore prefix)
# ---------------------------------------------------------------------------

def classify_query_type(question: str) -> str:
    """Heuristic classification of query type.

    Assumption: this is a simple keyword-based classifier.  It does NOT use the LLM
    (to avoid an extra API call).  The trade-off is that ambiguous questions may be
    misclassified — e.g. "How are departments doing on risk?" could be "comparison"
    or "risk".  The heuristic checks risk-related keywords first, so risk wins.

    Possible values: trend, comparison, risk, general.
    """
    q = question.lower()
    if any(w in q for w in ("trend", "growth", "over time", "trajectory", "revenue", "progress")):
        return "trend"
    if any(w in q for w in ("compare", "underperform", "versus", "department", "which", "best", "worst")):
        return "comparison"
    if any(w in q for w in ("risk", "threat", "challenge", "concern", "vulnerability")):
        return "risk"
    return "general"


def build_context(nodes: List[NodeWithScore]) -> tuple:
    """Build context string and source metadata from retrieved nodes.

    Returns (context_text, metadata_text) where:
      - context_text: numbered source blocks separated by ``---``
      - metadata_text: one line per source with file, type, section info
    """
    context_parts = []
    metadata_parts = []

    for i, node_with_score in enumerate(nodes, 1):
        text = node_with_score.node.get_content(metadata_mode=MetadataMode.NONE)
        meta = node_with_score.node.metadata

        filename = meta.get("filename", "Unknown")
        doc_type = meta.get("doc_type", "general")
        section = meta.get("section_header", "")
        chunk_type = meta.get("chunk_type", "text")

        context_parts.append(f"[Source {i}: {filename}]\n{text}")
        metadata_parts.append(
            f"Source {i}: file={filename}, type={doc_type}, "
            f"section='{section}', chunk_type={chunk_type}"
        )

    return "\n\n---\n\n".join(context_parts), "\n".join(metadata_parts)


def parse_llm_json(text: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def extract_sources_from_evidence(result: QueryResponse) -> List[Source]:
    """Extract unique sources from evidence quotes when the sources array is empty.

    Safety net for when the LLM returns an empty sources [] array
    but still cites documents in the evidence quotes.
    """
    seen = set()
    sources = []
    for ev in result.answer.evidence:
        name = ev.source_document
        if name and name not in seen:
            seen.add(name)
            sources.append(Source(document_name=name))
    return sources


def generate_fallback_chart(
    question: str,
    query_type: str,
    context: str,
) -> ChartData | None:
    """GPT call #3 — generate chart JSON when the main call returned null.

    Only called for trend/comparison queries.  Returns a ChartData object
    or None if the LLM still can't produce valid chart JSON.
    """
    prompt = _CHART_FALLBACK_PROMPT.format(
        context=context,
        question=question,
        query_type=query_type,
    )
    try:
        response = Settings.llm.complete(prompt)
        data = parse_llm_json(response.text)
        chart = ChartData(**data)
        logger.info(f"Fallback chart generated: {chart.chart_type}")
        return chart
    except Exception as e:
        logger.warning(f"Fallback chart generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------

def query(
    question: str,
    index: VectorStoreIndex,
    config: RAGConfig,
    rewrite: bool = True,
) -> QueryResponse:
    """Run the full RAG pipeline for a leadership question.

    Steps:
      1. Classify query type (heuristic — no GPT call)
      2. Rewrite query for better retrieval          ← GPT call #1
      3. Retrieve top-k chunks using MMR
      4. Build prompt with context + query_type hint
      5. Generate structured JSON response           ← GPT call #2
      6. Parse into QueryResponse (Pydantic model from output_schema.py)
      7. (Optional) fallback chart generation         ← GPT call #3

    Returns a ``QueryResponse`` with answer, sources, and optional chart data.
    """
    query_type = classify_query_type(question)

    # 1. Query rewriting (GPT call #1)
    search_query = rewrite_query(question) if rewrite else question

    # 2. Retrieve with MMR (Maximal Marginal Relevance) for diversity
    #    mmr_threshold from config: 0.0 = max diversity, 1.0 = pure similarity
    retriever = index.as_retriever(
        similarity_top_k=config.retrieval.top_k,
        vector_store_query_mode="mmr",
        vector_store_kwargs={"mmr_threshold": config.retrieval.mmr_threshold},
    )
    nodes = retriever.retrieve(search_query)

    if not nodes:
        logger.warning("No relevant chunks retrieved")
        return QueryResponse(
            question=question,
            query_type=query_type,
            answer={
                "summary": "No relevant information found in the available documents.",
                "key_points": ["No matching documents were found for this query."],
                "evidence": [],
            },
            sources=[],
        )

    unique_sources = len({n.node.metadata.get("filename") for n in nodes})
    logger.info(
        f"Retrieved {len(nodes)} chunks via MMR from {unique_sources} sources "
        f"(scores: {[f'{n.score:.3f}' for n in nodes]})"
    )

    # 3. Build context
    context, metadata_str = build_context(nodes)

    # 4. Build prompt with query_type hint
    #    SYSTEM_PROMPT: defines agent persona, JSON schema, and rules
    #    CONTEXT_PROMPT: injects context, metadata, question, and chart instructions
    schema_json = json.dumps(QueryResponse.model_json_schema(), indent=2)
    system = SYSTEM_PROMPT.format(schema=schema_json)
    user_prompt = CONTEXT_PROMPT.format(
        context=context,
        metadata=metadata_str,
        question=question,
        query_type=query_type,
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]

    # 5. Generate (GPT call #2)
    response = Settings.llm.chat(messages)
    raw_text = response.message.content

    # 6. Parse into structured response using output_schema.py models
    try:
        data = parse_llm_json(raw_text)
        # Override query_type with heuristic to ensure correct chart behavior
        data["query_type"] = query_type
        result = QueryResponse(**data)
        logger.info(f"Structured response parsed: query_type={result.query_type}")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"JSON parse failed ({e}), building fallback response")
        unique_source_names = list({
            n.node.metadata.get("filename", "Unknown") for n in nodes
        })
        result = QueryResponse(
            question=question,
            query_type=query_type,
            answer={
                "summary": raw_text[:500],
                "key_points": [raw_text[:300]],
                "evidence": [],
            },
            sources=[Source(document_name=s) for s in unique_source_names],
        )

    # --- Post-processing safety nets ---

    # 7a. Source extraction fallback: if the LLM returned empty sources
    #     but cited documents in evidence, populate sources from evidence.
    if not result.sources and result.answer.evidence:
        result.sources = extract_sources_from_evidence(result)
        logger.info(
            f"Extracted {len(result.sources)} sources from evidence: "
            f"{[s.document_name for s in result.sources]}"
        )

    # 7b. Fallback chart generation (GPT call #3): if the query is
    #     trend/comparison and the main call returned visualization=null,
    #     make a focused follow-up call for chart data only.
    if result.visualization is None and query_type in ("trend", "comparison"):
        logger.info(f"visualization is null for {query_type} query — attempting fallback chart")
        result.visualization = generate_fallback_chart(question, query_type, context)

    return result
