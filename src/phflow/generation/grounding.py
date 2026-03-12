"""
Post-generation grounding verification.

Checks whether each claim in the LLM's answer is actually supported by
the retrieved context — using the same embedding model already in memory.

Approach:
  1. Split answer into sentences.
  2. Encode each sentence + each context section with the existing encoder.
  3. For each answer sentence, find its max cosine similarity to any section.
  4. Sentences below the threshold are "ungrounded" — likely hallucinated.
  5. If the fraction of ungrounded sentences exceeds a limit, reject the answer.

Cost: ~50ms on CPU for a typical answer (5-8 sentences, 5-10 sections).
No additional model needed — reuses the BGE encoder already loaded.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from phflow.embeddings.encoder import RegulationEncoder

logger = logging.getLogger(__name__)

# Sentences shorter than this are typically connectors ("Yes.", "It depends.")
# that can't meaningfully be grounded against regulatory text.
_MIN_SENTENCE_LENGTH = 20

# Similarity threshold below which a sentence is considered ungrounded.
_GROUNDING_THRESHOLD = 0.65

# If more than this fraction of substantive sentences are ungrounded,
# the entire answer is rejected.
_MAX_UNGROUNDED_RATIO = 0.5


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.  Simple regex-based splitter that handles
    common abbreviations and numbered steps.
    """
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    # Merge numbered step prefixes back with their content
    sentences = []
    for s in raw:
        s = s.strip()
        if s and len(s) > 3:  # skip tiny fragments
            sentences.append(s)
    return sentences


def check_grounding(
    answer: str,
    sections: list[dict],
    encoder: RegulationEncoder,
    threshold: float = _GROUNDING_THRESHOLD,
    max_ungrounded_ratio: float = _MAX_UNGROUNDED_RATIO,
) -> tuple[bool, float, list[dict]]:
    """
    Verify that the answer is grounded in the retrieved context.

    Args:
        answer: The LLM-generated answer text.
        sections: Retrieved sections (each with 'full_text' key).
        encoder: The embedding encoder (already loaded in memory).
        threshold: Min cosine similarity for a sentence to be "grounded".
        max_ungrounded_ratio: Max fraction of ungrounded sentences before rejection.

    Returns:
        (is_grounded, grounding_score, details)
        - is_grounded: True if the answer passes the grounding check.
        - grounding_score: Average max-similarity across all substantive sentences (0-1).
        - details: Per-sentence breakdown for logging/debugging.
    """
    sentences = _split_sentences(answer)
    # Filter to substantive sentences (skip short connectors)
    substantive = [s for s in sentences if len(s) >= _MIN_SENTENCE_LENGTH]

    if not substantive:
        # Very short answer — can't meaningfully ground, let other guards handle it
        return True, 1.0, []

    # Get context texts from sections
    context_texts = [s.get("full_text", "") for s in sections if s.get("full_text")]
    if not context_texts:
        return False, 0.0, []

    # Encode everything with the passage encoder (no query prefix — both are content)
    sentence_vecs = np.array(
        encoder.encode_passages(substantive, show_progress=False),
        dtype=np.float32,
    )
    context_vecs = np.array(
        encoder.encode_passages(context_texts, show_progress=False),
        dtype=np.float32,
    )

    # Compute similarity matrix: (num_sentences x num_sections)
    sim_matrix = sentence_vecs @ context_vecs.T

    details = []
    ungrounded_count = 0

    for i, sentence in enumerate(substantive):
        max_sim = float(np.max(sim_matrix[i]))
        grounded = max_sim >= threshold
        if not grounded:
            ungrounded_count += 1
        details.append({
            "sentence": sentence[:100],
            "max_similarity": round(max_sim, 4),
            "grounded": grounded,
        })

    grounding_score = float(np.mean(np.max(sim_matrix, axis=1)))
    ungrounded_ratio = ungrounded_count / len(substantive)
    is_grounded = ungrounded_ratio <= max_ungrounded_ratio

    if not is_grounded:
        logger.warning(
            "Grounding check FAILED: %.0f%% sentences ungrounded (score=%.3f)",
            ungrounded_ratio * 100,
            grounding_score,
        )

    return is_grounded, grounding_score, details
