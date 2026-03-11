"""
Prompt construction for pharmaceutical/medical purchase order Q&A.

Uses a plain-text labeled-section format that works with any
instruction-tuned model regardless of its specific chat template.

The system prompt is loaded from the file configured in settings.system_prompt_path
(default: prompts/system_prompt.txt at the project root).  Edit that file to change
the model's instructions without touching code.
"""

from __future__ import annotations

import re
from pathlib import Path

from phflow.settings import settings


def _load_system_prompt() -> str:
    """
    Load the system prompt from the configured file path.
    Falls back to a minimal inline prompt if the file is missing.
    """
    path = Path(settings.system_prompt_path)
    if not path.is_absolute():
        # Resolve relative paths from the project root (3 levels up from this file)
        path = Path(__file__).parents[3] / path
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"System prompt file not found: {path}\n"
            "Create it at prompts/system_prompt.txt or set SYSTEM_PROMPT_PATH in .env."
        )


# Loaded once at import time.  Restart the API server after editing the file.
SYSTEM_PROMPT: str = _load_system_prompt()


# Number of conversation turns (user + assistant pairs) to include
# Currently set to 0 to prevent prompt contamination and hallucination loops
HISTORY_WINDOW = 0

# Detect back-references in the current query that require prior conversation context
_REFERENCE_RE = re.compile(
    r"\b(that|it|its|same|above|those|this|the\s+section|the\s+requirement|"
    r"the\s+rule|the\s+regulation|mentioned|referred|previous|prior)\b",
    re.IGNORECASE,
)


def _has_reference_signals(question: str) -> bool:
    """Return True if the question contains pronouns or back-reference words."""
    return bool(_REFERENCE_RE.search(question))


def build_prompt(
    context: str,
    question: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build a single prompt string with clearly labelled sections.

    Structure:
        INSTRUCTIONS:
        {system}

        CONVERSATION HISTORY:   (optional, last HISTORY_WINDOW turns)
        User: ...
        Assistant: ...

        CONTEXT:
        {retrieved sections labelled [SOURCE N]}

        QUESTION:
        {question}

        Answer:
    """
    parts: list[str] = []

    parts.append(f"INSTRUCTIONS:\n{SYSTEM_PROMPT}")

    if conversation_history and HISTORY_WINDOW > 0:
        # Expand to 2 turns when the question contains a back-reference pronoun
        # (e.g. "that section", "it", "same requirement") so the model has enough
        # context to resolve the reference. Default stays at HISTORY_WINDOW = 1.
        window = 2 if _has_reference_signals(question) else HISTORY_WINDOW
        recent = conversation_history[-(window * 2):]
        if recent:
            history_lines = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {msg['content']}")
            parts.append("CONVERSATION HISTORY:\n" + "\n\n".join(history_lines))

    parts.append(f"CONTEXT:\n{context}")
    parts.append(f"QUESTION:\n{question}")
    parts.append("Answer: ")

    return "\n\n".join(parts)


def build_retrieval_query(messages: list[dict], current_query: str) -> str:
    """
    Return the current query for vector retrieval.

    History disambiguation (prepending previous user question) was removed
    because it created long noisy retrieval queries that hurt vector search
    quality by mixing unrelated questions.
    """
    return current_query


def build_retrieval_rewrite_prompt(
    current_query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build a short prompt asking the LLM to rewrite a colloquial warehouse/operations
    query into formal CFR regulatory terminology suitable for vector retrieval.

    Used when ENABLE_QUERY_REWRITE=true. The result replaces the raw query
    for retrieval only; the original question is still shown to the user.
    """
    parts: list[str] = []
    parts.append(
        "INSTRUCTIONS:\n"
        "You are a query rewriting assistant for an FDA Title 21 CFR regulatory "
        "search system used by pharmaceutical warehouse and manufacturing staff.\n\n"
        "Rewrite the user's operational question into a concise search query using "
        "formal CFR regulatory terminology so it matches the language of the regulations.\n\n"
        "Vocabulary mapping to apply:\n"
        "- pallet / carton / box / shipment → container / outer container / shipment\n"
        "- leakage / leak / spill → contaminated / damaged container / deteriorated\n"
        "- accept / reject a delivery → receipt / receiving / acceptance criteria\n"
        "- temperature logger / data logger → temperature monitoring / temperature recording equipment\n"
        "- power failure / power outage → equipment failure / loss of temperature control\n"
        "- lot mix-up / mix together → commingling / mixup / segregation\n"
        "- bin / shelf / rack → storage area / storage conditions\n"
        "- handwritten label → labeling / label legibility\n"
        "- quarantine area / hold area → quarantine / withheld from use\n"
        "- quality team / QA / quality unit → quality control unit\n"
        "- approved supplier → qualified / approved supplier / vendor qualification\n"
        "- temperature-sensitive → temperature-controlled / cold chain / storage requirements\n"
        "- damaged seal → compromised container / opened container / container integrity\n"
        "- relabel / re-label → repackaging / labeling / relabeling\n"
        "- inventory count → physical inventory / inventory reconciliation\n"
        "- ship to customers → distribution / release for distribution\n\n"
        "IMPORTANT CONSTRAINTS:\n"
        "- Only substitute operational terms with CFR equivalents from the vocabulary above.\n"
        "- Do NOT add new concepts, conditions, or terms not present in the original question.\n"
        "- Do NOT rephrase the entire question — only replace non-CFR words with CFR equivalents.\n"
        "- Keep the rewritten query as close to the original structure as possible.\n"
        "- If the original question already uses formal terms, return it unchanged.\n"
        "- Output ONLY the rewritten query — no explanation, no extra words. Keep it under 20 words."
    )
    if conversation_history:
        recent = conversation_history[-4:]
        history_lines = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ]
        parts.append("CONVERSATION HISTORY:\n" + "\n\n".join(history_lines))
    parts.append(f"CURRENT QUESTION:\n{current_query}")
    parts.append("Rewritten query:")
    return "\n\n".join(parts)
