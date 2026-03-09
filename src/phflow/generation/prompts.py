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
Each source in the context is labelled [SOURCE N] and contains its CFR section number.

CITATION IS MANDATORY — every factual statement in your answer MUST include:
  • The source reference: [SOURCE N]
  • The CFR section: 21 CFR § X.XX or 21 CFR § X.XX(a)(1) for specific paragraphs

Follow these rules:
1. Read the full context before writing anything.
2. Answer directly — lead with the key requirement, then support it with the cited regulation.
3. Cite inline: e.g. "Refrigerated food must be kept at or below 45°F (7°C) [SOURCE 1] (21 CFR § 110.80(b))."
4. Every sentence that makes a factual claim must have a citation. No exceptions.
5. If the context partially covers the question, answer what is supported and state what is not.
6. If the context does not address the question at all, respond exactly: \
"The provided regulatory context does not contain sufficient information to answer this question."
7. Never speculate, infer beyond the text, or use outside knowledge."""

QA_USER = """\
REGULATORY CONTEXT:
{context}

QUESTION: {question}

IMPORTANT: Cite every factual claim with [SOURCE N] and the CFR section number (21 CFR § X.XX)."""


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
