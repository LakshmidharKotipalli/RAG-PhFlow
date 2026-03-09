"""
Prompt construction for Q&A mode.

Uses a plain-text labeled-section format that works with any
instruction-tuned model regardless of its specific chat template.
"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are an FDA regulatory assistant. Your task is to answer the user's question \
using ONLY the regulatory context provided below. \
Do not use any knowledge outside the provided context. \
If the context does not contain enough information to answer, say so clearly. \
Cite every factual claim with the source number shown in the context, e.g. [SOURCE 1]."""


# Number of conversation turns (user + assistant pairs) to include
HISTORY_WINDOW = 3


def build_prompt(
    context: str,
    question: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build a single prompt string with clearly labelled sections.

    Structure:
        ### INSTRUCTIONS
        {system}

        ### CONVERSATION HISTORY   (optional, last HISTORY_WINDOW turns)
        User: ...
        Assistant: ...

        ### CONTEXT
        {retrieved sections}

        ### QUESTION
        {question}

        ### ANSWER

    The plain-text section labels work with any instruction-tuned model
    regardless of its specific chat template (Llama-3, Mistral, Qwen, etc.).
    """
    parts: list[str] = []

    parts.append(f"### INSTRUCTIONS\n{SYSTEM_PROMPT}")

    if conversation_history:
        recent = conversation_history[-(HISTORY_WINDOW * 2):]
        if recent:
            history_lines = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {msg['content']}")
            parts.append("### CONVERSATION HISTORY\n" + "\n\n".join(history_lines))

    parts.append(f"### CONTEXT\n{context}")
    parts.append(f"### QUESTION\n{question}")
    parts.append("### ANSWER\n")

    return "\n\n".join(parts)


def build_retrieval_query(messages: list[dict], current_query: str) -> str:
    """
    Build a retrieval-optimized query from conversation history + current turn.
    Prepends the last user message for context-aware retrieval in multi-turn sessions.
    """
    recent_user = [m["content"] for m in messages if m.get("role") == "user"]
    if len(recent_user) >= 2:
        return recent_user[-1] + " " + current_query
    return current_query
