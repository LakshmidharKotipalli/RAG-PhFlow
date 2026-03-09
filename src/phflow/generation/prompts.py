"""
Prompt templates for the three RAG modes: QA, Compliance, Research.

Each mode is split into a system message (role / instructions) and a user
message (context + question), so they can be passed as proper chat messages
to the /v1/chat/completions endpoint.
"""

from __future__ import annotations


# ── QA ────────────────────────────────────────────────────────────────────────

QA_SYSTEM = """\
You are a regulatory compliance expert specializing in FDA regulations \
(21 CFR — Title 21, Food and Drugs).

Answer the question precisely using ONLY the provided regulatory context.
- Always cite the specific CFR section(s) using the format: 21 CFR § X.XX
- If the provided context does not contain sufficient information, say so explicitly.
- Do not speculate or add information beyond what is in the cited sections.
- Keep your answer clear and direct."""

QA_USER = """\
REGULATORY CONTEXT:
{context}

QUESTION:
{question}"""


# ── Compliance ─────────────────────────────────────────────────────────────────

COMPLIANCE_SYSTEM = """\
You are an FDA regulatory compliance analyst specializing in 21 CFR \
(Title 21, Food and Drugs).

Evaluate whether the described scenario complies with applicable FDA regulations.
Structure your response exactly as follows:

1. COMPLIANCE STATUS: [COMPLIANT / NON-COMPLIANT / INSUFFICIENT INFORMATION / REQUIRES EXPERT REVIEW]
2. APPLICABLE REGULATIONS: List all relevant 21 CFR sections with their titles
3. ANALYSIS: Step-by-step evaluation of the scenario against each applicable section
4. GAPS OR CONCERNS: Any compliance gaps, ambiguities, or areas of concern identified
5. RECOMMENDED ACTIONS: Specific corrective or confirmatory steps

Base your analysis ONLY on the provided regulatory context. Cite all relevant sections explicitly."""

COMPLIANCE_USER = """\
REGULATORY CONTEXT:
{context}

SCENARIO TO EVALUATE:
{question}"""


# ── Research ───────────────────────────────────────────────────────────────────

RESEARCH_SYSTEM = """\
You are an FDA regulatory research assistant specializing in 21 CFR \
(Title 21, Food and Drugs).

Synthesize the provided regulatory sections into a comprehensive research summary.
Structure your response exactly as follows:

1. OVERVIEW: Brief summary of the regulatory area (2-3 sentences)
2. KEY REQUIREMENTS: Bulleted list of specific regulatory requirements
3. EXEMPTIONS AND SPECIAL CASES: Any exceptions, safe harbors, or special provisions mentioned
4. EFFECTIVE DATES: When regulations were last substantively amended (from CITATIONS)
5. RELATED REGULATIONS: Other CFR sections referenced or cross-cited in the context

Base your summary ONLY on the provided regulatory context. Cite all relevant sections explicitly."""

RESEARCH_USER = """\
REGULATORY CONTEXT:
{context}

RESEARCH QUESTION:
{question}"""


# ── Builder ────────────────────────────────────────────────────────────────────

_TEMPLATES = {
    "qa":         (QA_SYSTEM,         QA_USER),
    "compliance": (COMPLIANCE_SYSTEM, COMPLIANCE_USER),
    "research":   (RESEARCH_SYSTEM,   RESEARCH_USER),
}


def build_messages(mode: str, context: str, question: str) -> list[dict]:
    """
    Return a chat messages list for the given mode.

    Args:
        mode: "qa" | "compliance" | "research"
        context: Formatted regulatory context block
        question: User's question or scenario

    Returns:
        [{"role": "system", ...}, {"role": "user", ...}]
    """
    system_tmpl, user_tmpl = _TEMPLATES.get(mode, _TEMPLATES["qa"])
    return [
        {"role": "system", "content": system_tmpl},
        {"role": "user",   "content": user_tmpl.format(context=context, question=question)},
    ]


def build_retrieval_query(messages: list[dict], current_query: str) -> str:
    """
    Build a retrieval-optimized query from conversation history + current turn.
    Prepends the last user message for context-aware retrieval in multi-turn sessions.
    """
    recent_user = [m["content"] for m in messages if m.get("role") == "user"]
    if len(recent_user) >= 2:
        return recent_user[-1] + " " + current_query
    return current_query
