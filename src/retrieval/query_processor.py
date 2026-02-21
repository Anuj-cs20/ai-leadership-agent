"""Query rewriting for improved retrieval."""

import logging
from llama_index.core import Settings
from src.generation.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


def rewrite_query(question: str) -> str:
    """Rewrite the user question for better retrieval.

    Uses the LLM to expand the query with synonyms,
    related business terms, and financial vocabulary.
    """
    prompt = QUERY_REWRITE_PROMPT.format(question=question)

    try:
        response = Settings.llm.complete(prompt)
        rewritten = response.text.strip().strip('"').strip("'")
        logger.info(f"Query rewrite: '{question}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed ({e}), using original")
        return question
