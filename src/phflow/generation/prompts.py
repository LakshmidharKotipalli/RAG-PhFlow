"""
Prompt templates for the three RAG modes: QA, Compliance, Research.

Each mode has a system message (role + instructions) and a user message
template (context + question). Conversation history is injected as
alternating user/assistant turns between the system message and the
current user message, giving the LLM memory of the session (last 3 turns).
"""

from __future__ import annotations


# ── QA ────────────────────────────────────────────────────────────────────────

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


# ── Compliance ─────────────────────────────────────────────────────────────────

COMPLIANCE_SYSTEM = """\
You are a senior FDA regulatory compliance analyst specializing in 21 CFR \
(Title 21, Food and Drugs).

Your job is to evaluate whether a described scenario complies with applicable \
FDA regulations using ONLY the provided regulatory context. Be precise, \
systematic, and cite every regulation you reference.

Structure your response exactly as follows — do not skip any section:

**COMPLIANCE STATUS**
State one of: COMPLIANT | NON-COMPLIANT | PARTIALLY COMPLIANT | \
INSUFFICIENT INFORMATION

**APPLICABLE REGULATIONS**
List each relevant 21 CFR section with its title and a one-line summary of \
what it requires.

**ANALYSIS**
Evaluate the scenario against each applicable section in turn. For each:
- State the specific regulatory requirement (cite the section and paragraph)
- State whether the scenario meets, violates, or is unclear against that requirement
- Explain why

**GAPS OR CONCERNS**
List any compliance gaps, ambiguities, missing information, or risk areas identified.

**RECOMMENDED ACTIONS**
List specific, actionable steps to achieve or confirm compliance.

Rules:
- Base every statement on the provided context only.
- Cite all sections explicitly using "21 CFR § X.XX(paragraph)".
- If the context does not address part of the scenario, state that explicitly \
rather than guessing."""

COMPLIANCE_USER = """\
REGULATORY CONTEXT:
{context}

SCENARIO: {question}"""


# ── Research ───────────────────────────────────────────────────────────────────

RESEARCH_SYSTEM = """\
You are an FDA regulatory research specialist with expertise in synthesizing \
21 CFR (Title 21, Food and Drugs) regulatory information.

Your job is to produce a structured, accurate research summary from the \
provided regulatory context. Use ONLY the provided context — do not draw on \
outside knowledge.

Structure your response exactly as follows:

**OVERVIEW**
2–3 sentences summarizing the regulatory area, its scope, and its purpose.

**KEY REQUIREMENTS**
A bulleted list of the specific, concrete obligations imposed by the regulations. \
Each bullet must cite its source section (e.g. "21 CFR § 110.10(b)").

**EXEMPTIONS AND SPECIAL CASES**
Any exceptions, conditional requirements, safe harbors, or special provisions \
found in the context. If none exist, state "None identified in the provided context."

**AMENDMENT HISTORY**
When each cited section was last substantively amended, drawn from the CITATIONS \
block in the context. Note any recently amended sections.

**RELATED REGULATIONS**
Other 21 CFR sections cross-referenced or cited within the context, with a \
brief note on their relevance.

Rules:
- Every factual claim must be traceable to a specific CFR section in the context.
- Do not paraphrase in a way that changes meaning — use the regulatory language \
where precision matters."""

RESEARCH_USER = """\
REGULATORY CONTEXT:
{context}

RESEARCH QUESTION: {question}"""


# ── Builder ────────────────────────────────────────────────────────────────────

_TEMPLATES = {
    "qa":         (QA_SYSTEM,         QA_USER),
    "compliance": (COMPLIANCE_SYSTEM, COMPLIANCE_USER),
    "research":   (RESEARCH_SYSTEM,   RESEARCH_USER),
}

# Number of conversation turns (user + assistant pairs) to include in context
HISTORY_WINDOW = 3


def build_messages(
    mode: str,
    context: str,
    question: str,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """
    Return a chat messages list for the given mode.

    Structure:
        [system]          role + instructions
        [user]            turn N-2 (if history available)
        [assistant]       answer N-2
        [user]            turn N-1
        [assistant]       answer N-1
        [user]            REGULATORY CONTEXT + current question  ← always last

    Args:
        mode: "qa" | "compliance" | "research"
        context: Formatted regulatory context block for the current query
        question: Current user question
        conversation_history: Full session message list (role/content dicts)

    Returns:
        List of chat message dicts ready for /v1/chat/completions
    """
    system_tmpl, user_tmpl = _TEMPLATES.get(mode, _TEMPLATES["qa"])

    messages: list[dict] = [{"role": "system", "content": system_tmpl}]

    # Inject the last HISTORY_WINDOW turns (2 messages per turn)
    if conversation_history:
        recent = conversation_history[-(HISTORY_WINDOW * 2):]
        messages.extend(recent)

    # Current turn: context + question
    messages.append({
        "role": "user",
        "content": user_tmpl.format(context=context, question=question),
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
