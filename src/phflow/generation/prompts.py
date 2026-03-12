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


def _resolve_prompt_path() -> Path:
    """Resolve the system prompt file path."""
    path = Path(settings.system_prompt_path)
    if not path.is_absolute():
        path = Path(__file__).parents[3] / path
    return path


def get_system_prompt() -> str:
    """
    Load the system prompt from disk on every call.
    Edit the file and the next query picks up the change — no restart needed.
    """
    path = _resolve_prompt_path()
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"System prompt file not found: {path}\n"
            "Create it at prompts/system_prompt.txt or set SYSTEM_PROMPT_PATH in .env."
        )


# Number of conversation turns (user + assistant pairs) to include
# Increased to 1 per user request
HISTORY_WINDOW = 1

# Text used to indicate that retrieved context was not sufficient for a real answer.
# CRITICAL: This exact string is used to filter out insufficient turns from chat history.
INSUFFICIENT_MESSAGE = (
    "The provided context does not contain sufficient information "
    "to answer this question."
)

# Prompt shown when query is too vague to produce a useful answer.
VAGUE_QUERY_MESSAGE = (
    "Your question is quite broad. Could you be more specific? For example:\n"
    "- What are the temperature storage requirements for components?\n"
    "- What should I do if a shipment arrives with a broken seal?\n"
    "- Who is responsible for approving received materials?"
)

# Minimum word count for a query to be considered specific enough.
# "What is X?" (4 words) is too vague; "What are the requirements for X?" (6) is fine.
_MIN_QUERY_WORDS = 5

# Detect back-references in the current query that require prior conversation context
_REFERENCE_RE = re.compile(
    r"\b(that|it|its|same|above|those|this|the\s+section|the\s+requirement|"
    r"the\s+rule|the\s+regulation|mentioned|referred|previous|prior)\b",
    re.IGNORECASE,
)


def _has_reference_signals(question: str) -> bool:
    """Return True if the question contains pronouns or back-reference words."""
    return bool(_REFERENCE_RE.search(question))


def _filter_insufficient_history(history: list[dict]) -> list[dict]:
    """
    Filter the conversation history to remove turns where the assistant 
    provided an 'insufficient information' response.
    """
    if not history:
        return []
        
    filtered: list[dict] = []
    # Conversation is expected to be [User, Assistant, User, Assistant, ...]
    # We iterate by pairs to ensure we remove the question alongside the insufficient answer.
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            user_msg = history[i]
            assistant_msg = history[i+1]
            if INSUFFICIENT_MESSAGE not in assistant_msg.get("content", ""):
                filtered.append(user_msg)
                filtered.append(assistant_msg)
        else:
            # Trailing message (likely the last user query which hasn't been answered yet)
            filtered.append(history[i])
            
    return filtered


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

    parts.append(f"INSTRUCTIONS:\n{get_system_prompt()}")

    if conversation_history and HISTORY_WINDOW > 0:
        # Filter history to remove turns with insufficient content
        clean_history = _filter_insufficient_history(conversation_history)
        
        # Take the last HISTORY_WINDOW turns (each turn is 2 messages)
        recent = clean_history[-(HISTORY_WINDOW * 2):]
        
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
        "- ship to customers → distribution / release for distribution\n"
        "- material / raw material / ingredient → component / drug product component\n"
        "- what do I do / what should I do / how to handle → regulatory requirements for / procedures for deviation from\n"
        "- not stored / not in storage / left out / found outside → deviation from storage requirements / failure to store\n"
        "- wrong temperature / too hot / too cold / not refrigerated → deviation from temperature storage conditions\n\n"
        "IMPORTANT CONSTRAINTS:\n"
        "- Only substitute operational terms with CFR equivalents from the vocabulary above.\n"
        "- If the user asks what to do or how to handle a situation, rewrite as a query for the regulatory requirements that apply to that situation.\n"
        "- Do NOT add new concepts, conditions, or terms not present in the original question.\n"
        "- Do NOT rephrase the entire question — only replace non-CFR words with CFR equivalents.\n"
        "- Keep the rewritten query as close to the original structure as possible.\n"
        "- If the original question already uses formal terms, return it unchanged.\n"
        "- Output ONLY the rewritten query — no explanation, no extra words. Keep it under 20 words."
    )
    if conversation_history:
        # Also filter rewrite history to keep only high-quality interaction context
        clean_history = _filter_insufficient_history(conversation_history)
        recent = clean_history[-4:]
        history_lines = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ]
        parts.append("CONVERSATION HISTORY:\n" + "\n\n".join(history_lines))
    parts.append(f"CURRENT QUESTION:\n{current_query}")
    parts.append("Rewritten query:")
    return "\n\n".join(parts)


def build_query_decomposition_prompt(rewritten_query: str) -> str:
    """
    Build a prompt asking the LLM to decompose a complex regulatory query
    into 2-3 independent sub-queries that each target a different regulatory concept.

    This improves retrieval coverage for multi-faceted questions where a single
    query vector can only point toward one region of embedding space.
    """
    return (
        "INSTRUCTIONS:\n"
        "You are a search query decomposition assistant for an FDA Title 21 CFR regulatory system.\n\n"
        "Given a regulatory search query, decide if it covers multiple distinct regulatory concepts. "
        "If it does, decompose it into 2-3 independent sub-queries, each targeting one concept.\n\n"
        "Rules:\n"
        "- If the query is simple and targets one concept, return it unchanged on a single line.\n"
        "- If the query covers multiple concepts, return 2-3 sub-queries, one per line.\n"
        "- Each sub-query should be a complete, self-contained search query.\n"
        "- Do NOT add concepts not present in the original query.\n"
        "- Output ONLY the sub-queries, one per line. No numbering, no explanation.\n\n"
        f"QUERY:\n{rewritten_query}\n\n"
        "Sub-queries:"
    )


# The scope gate response when a query is off-topic.
OFF_TOPIC_MESSAGE = (
    "This system only handles questions about pharmaceutical warehouse "
    "dock-to-stock operations under FDA 21 CFR Parts 210 and 211. "
    "Please rephrase your question in that context."
)


def build_scope_and_rewrite_prompt(query: str) -> str:
    """
    Combined scope gate + query rewrite in a single LLM call.

    Returns a prompt that:
    - If in-scope: outputs "IN_SCOPE: <rewritten query>"
    - If out-of-scope: outputs "OUT_OF_SCOPE"
    """
    return (
        "INSTRUCTIONS:\n"
        "You are a scope classifier AND query rewriter for a pharmaceutical "
        "regulatory search system used by warehouse and manufacturing staff.\n\n"
        "STEP 1 — SCOPE CHECK:\n"
        "The system answers questions about pharmaceutical GMP requirements "
        "under FDA 21 CFR Parts 210 and 211.  This covers the ENTIRE manufacturing "
        "and warehouse lifecycle: receiving, inspection, sampling, quarantine, "
        "storage, temperature control, labeling, rejection, release, production, "
        "process controls, packaging, distribution, shipping, and record-keeping.\n\n"
        "In-scope: receiving shipments, temperature excursions, quarantine, "
        "rejected materials, storage conditions, equipment cleaning/maintenance, "
        "component testing, container integrity, deviation handling, batch records, "
        "production/process controls, written procedures, reprocessing, packaging, "
        "distribution records, shipping, expiration dating, quality control unit, "
        "personnel training, building/facility requirements, calibration.\n"
        "Out-of-scope: weather, cooking, general medical advice, general science, "
        "non-pharmaceutical regulations, clinical trials, drug efficacy, patient treatment.\n\n"
        "STEP 2 — REWRITE (only if in-scope):\n"
        "Rewrite the question into formal CFR regulatory terminology.\n\n"
        "Vocabulary mapping:\n"
        "- pallet / carton / box / shipment → container / outer container / shipment\n"
        "- leakage / leak / spill → contaminated / damaged container / deteriorated\n"
        "- accept / reject a delivery → receipt / receiving / acceptance criteria\n"
        "- temperature logger / data logger → temperature monitoring / temperature recording equipment\n"
        "- power failure / power outage → equipment failure / loss of temperature control\n"
        "- lot mix-up / mix together → commingling / mixup / segregation\n"
        "- bin / shelf / rack → storage area / storage conditions\n"
        "- quarantine area / hold area → quarantine / withheld from use\n"
        "- quality team / QA / quality unit → quality control unit\n"
        "- temperature-sensitive → temperature-controlled / cold chain / storage requirements\n"
        "- material / raw material / ingredient → component / drug product component\n"
        "- what do I do / what should I do / how to handle → regulatory requirements for / procedures for deviation from\n"
        "- not stored / not in storage / left out / found outside → deviation from storage requirements / failure to store\n"
        "- wrong temperature / too hot / too cold / not refrigerated → deviation from temperature storage conditions\n\n"
        "If the user asks what to do or how to handle a situation, rewrite as a query "
        "for the regulatory requirements that apply to that situation.\n\n"
        "OUTPUT FORMAT (strictly one line, no explanation):\n"
        "- If in-scope: IN_SCOPE: <rewritten query in under 20 words>\n"
        "- If out-of-scope: OUT_OF_SCOPE\n\n"
        f"QUESTION:\n{query}\n\n"
        "Output:"
    )
