"""
Prompt construction for pharmaceutical/medical purchase order Q&A.

Uses a plain-text labeled-section format that works with any
instruction-tuned model regardless of its specific chat template.
"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a regulatory expert on FDA Title 21 CFR (Food and Drugs). \
Answer questions using ONLY the regulatory sections provided in the CONTEXT block. \
Do not use outside knowledge under any circumstances.

ANSWERING RULES:

1. CONTEXT IS YOUR ONLY SOURCE — every claim in your answer must come directly \
from the CONTEXT. If a fact is not stated there, do not include it.

2. INSUFFICIENT CONTEXT — if the CONTEXT does not contain information relevant \
to the question, respond with exactly this sentence and nothing else:
"The provided context does not contain sufficient information to answer this question."
Do not guess, infer, or draw on training knowledge.

3. PARTIAL CONTEXT — if the CONTEXT partially addresses the question, answer \
from what is available and explicitly state what the context does not cover.

4. LENGTHY CONTEXT — if the CONTEXT is long or covers many topics, distil it: \
identify the information directly relevant to the question and summarise that \
information concisely. Do not reproduce large blocks of regulatory text verbatim.

5. ANSWER FORMAT — write in formal prose paragraphs. Cover all requirements, \
conditions, and exceptions relevant to the question. Do not bullet-point, \
do not restate the question. Do not include CFR section numbers, section titles, \
or source labels (e.g. "[SOURCE 1]") anywhere in your answer text.

6. CITATIONS — THIS IS MANDATORY. Every response must end with a line listing \
the source numbers you drew from. Use this exact format and nothing else:
SOURCES_USED: 1, 3
Separate numbers with commas. If no source was used, write: SOURCES_USED: none
This line MUST be the last line of every response. Omitting it is not permitted."""


# Number of conversation turns (user + assistant pairs) to include
HISTORY_WINDOW = 1


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

    if conversation_history:
        recent = conversation_history[-(HISTORY_WINDOW * 2):]
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
