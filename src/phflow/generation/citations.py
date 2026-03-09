"""
CFR citation formatting utilities for RAG responses.
"""

from __future__ import annotations


def format_section_context(section: dict, index: int) -> str:
    """
    Format a retrieved section as a numbered context block for the LLM prompt.
    """
    meta = section.get("metadata", {})
    section_number = meta.get("section_number", section.get("section_number", ""))
    section_title = meta.get("section_title", section.get("section_title", ""))
    part_number = meta.get("part_number", "")
    part_name = meta.get("part_name", "")
    subpart_letter = meta.get("subpart_letter", "")
    subpart_name = meta.get("subpart_name", "")
    amendment_date = meta.get("latest_amendment_date", "")

    header_parts = [f"[SOURCE {index + 1}] 21 CFR {section_number} — {section_title}"]
    if part_number:
        loc = f"Part {part_number}: {part_name}"
        if subpart_letter:
            loc += f" | Subpart {subpart_letter}: {subpart_name}"
        header_parts.append(loc)
    if amendment_date:
        header_parts.append(f"Last amended: {amendment_date}")

    header = "\n".join(header_parts)
    separator = "—" * 60
    full_text = section.get("full_text", "")

    return f"{header}\n{separator}\n{full_text}"


def format_context_block(sections: list[dict]) -> str:
    """
    Format all retrieved sections into the REGULATORY CONTEXT block
    that is injected into LLM prompts.
    """
    if not sections:
        return "No relevant regulatory sections found."
    parts = [format_section_context(s, i) for i, s in enumerate(sections)]
    return "\n\n".join(parts)



def source_summary(sections: list[dict]) -> list[dict]:
    """
    Produce a compact list of source dicts for API responses.
    """
    results = []
    for section in sections:
        meta = section.get("metadata", {})
        results.append({
            "section_number": meta.get("section_number", section.get("section_number", "")),
            "section_title": meta.get("section_title", section.get("section_title", "")),
            "section_node": section.get("section_node", ""),
            "part_number": meta.get("part_number"),
            "part_name": meta.get("part_name", ""),
            "latest_amendment_date": meta.get("latest_amendment_date", ""),
            "score": section.get("score", 0.0),
            "full_text": section.get("full_text", ""),
        })
    return results
