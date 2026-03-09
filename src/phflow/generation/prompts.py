"""
Prompt template for Q&A mode.

The system message defines the role and strict answering rules.
The user message carries the retrieved regulatory context + current question.
Conversation history (last 3 turns) is injected between the system message
and the current user message so the LLM has multi-turn context.
"""

from __future__ import annotations


QA_SYSTEM = """\
You are an expert FDA regulatory analyst with deep knowledge of Title 21 \
of the Code of Federal Regulations (21 CFR — Food and Drugs).

Your job is to answer questions accurately using ONLY the regulatory context provided. \
Follow these rules strictly:

1. READ the full context before answering.
2. IDENTIFY every CFR section in the context that is relevant to the question.
3. ANSWER directly and concisely — lead with the key requirement or answer, \
then support it with the regulation.
4. CITE precisely using the format "21 CFR § X.XX" or "21 CFR § X.XX(a)(1)" \
for specific paragraphs. Never cite a section you did not read in the context.
5. If the context covers the topic partially, answer what is supported and \
explicitly state what is not covered.
6. If the context does not address the question at all, respond: \
"The provided regulatory context does not contain information sufficient to answer this question."
7. Never speculate, infer beyond the text, or use outside knowledge."""

QA_USER = """\
REGULATORY CONTEXT:
{context}

QUESTION: {question}"""


# Number of conversation turns (user + assistant pairs) to include
HISTORY_WINDOW = 3


def build_messages(
    context: str,
    question: str,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """
    Return a chat messages list for /v1/chat/completions.

    Structure:
        [system]      role + instructions
        [user]        turn N-2  (if history available)
        [assistant]   answer N-2
        [user]        turn N-1
        [assistant]   answer N-1
        [user]        REGULATORY CONTEXT + current question  ← always last

    Args:
        context: Formatted regulatory context block for the current query
        question: Current user question
        conversation_history: Full session message list (role/content dicts)
    """
    messages: list[dict] = [{"role": "system", "content": QA_SYSTEM}]

    if conversation_history:
        recent = conversation_history[-(HISTORY_WINDOW * 2):]
        messages.extend(recent)

    messages.append({
        "role": "user",
        "content": QA_USER.format(context=context, question=question),
    })

    return messages


def build_retrieval_query(messages: list[dict], current_query: str) -> str:
    """
    Build a retrieval-optimized query from conversation history + current turn.
    Prepends the last user message for context-aware retrieval in multi-turn sessions.
    """
    recent_user = [m["content"] for m in messages if m.get("role") == "user"]
    if len(recent_user) >= 2:
        return recent_user[-1] + " " + current_query
    return current_query
