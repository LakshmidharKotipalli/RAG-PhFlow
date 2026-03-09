"""
Prompt construction for pharmaceutical/medical purchase order Q&A.

Uses a plain-text labeled-section format that works with any
instruction-tuned model regardless of its specific chat template.
"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a regulatory and procurement expert specialising in medical and \
pharmaceutical purchase orders, FDA 21 CFR compliance, drug/device labelling, \
storage, handling, and supply chain requirements.

Your task is to read the CONTEXT section carefully and write a clear, \
well-reasoned answer to the QUESTION. Follow these rules without exception:

1. SYNTHESISE — do not copy sentences directly from the context. Read the \
relevant regulations, understand what they mean, and explain them in your own words.
2. BE THOROUGH — your answer MUST consist of at least three complete sentences \
that explain the regulation clearly. Never use symbols, checkmarks (✓ ✘), \
emojis, or single-character responses. Never answer with only a number or \
unit (e.g. "2–8 °C" alone is not an acceptable answer — you must explain \
the full requirement in plain English). Cover all applicable requirements, \
conditions, and exceptions present in the context.
3. STAY IN CONTEXT — use ONLY information from the provided CONTEXT. Do not \
add outside knowledge.
4. NO CITATION MARKERS — do not write [SOURCE 1] or any similar reference \
tags anywhere in your answer.
5. LANGUAGE AND FORMAT — always write in formal, professional English. \
Use complete sentences with correct grammar and spelling. \
Do not use casual or colloquial expressions such as "you know", "basically", \
"kind of", "sort of", "pretty much", "a lot", "stuff", "things", "okay", \
"alright", "yeah", "nope", or similar informal language. \
Write in plain prose paragraphs. Start directly with the answer; do not \
restate the question or repeat section headers.
6. EMPTY OR INSUFFICIENT CONTEXT:
   - If the CONTEXT section contains only "No relevant regulatory sections found.", \
you MUST respond with exactly the following sentence and nothing else: \
"The provided context does not contain sufficient information to answer this question." \
Do NOT attempt to answer from your own training knowledge in this case.
   - If the context partially addresses the question, answer from what IS \
available and note what the context does not cover.
   - Use the insufficient-information message only when the context has absolutely \
no bearing on the question. If the context is imperfect or covers a related \
aspect, use what you have.
   - Do not reproduce section headers, section titles, or bare CFR section numbers \
as your answer.
7. DECLARE SOURCES — this step is REQUIRED. After your answer, write a new line \
containing ONLY the source numbers you actually used. Use this exact format:
SOURCES_USED: 1, 3
Write only the numbers separated by commas. No other text on that line. \
If you used no sources, write: SOURCES_USED: none
This line MUST appear at the end of every response, no exceptions."""


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


def build_correction_prompt(query: str) -> str:
    """
    Build a short prompt that asks the LLM to fix spelling mistakes and
    rephrase informal/colloquial input into formal professional language.
    Returns only the corrected question — no explanation, no extra text.
    """
    return (
        "Fix any spelling mistakes and rewrite the following question in formal, "
        "professional English suitable for a pharmaceutical regulatory database query. "
        "Return ONLY the corrected question — no explanation, no extra text, no punctuation changes beyond what is needed.\n\n"
        f"Question: {query}\n\n"
        "Corrected question:"
    )


def build_retrieval_query(messages: list[dict], current_query: str) -> str:
    """
    Return the current query for vector retrieval.

    History disambiguation (prepending previous user question) was removed
    because it created long noisy retrieval queries that hurt vector search
    quality by mixing unrelated questions.
    """
    return current_query
