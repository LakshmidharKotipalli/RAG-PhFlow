"""
Build Qdrant Filter objects from user-supplied query parameters.
"""

from __future__ import annotations

import re

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

# Patterns for extracting filter hints from natural-language queries
_PART_RE = re.compile(r"\bpart\s+(\d+)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"§\s*(\d+)\.\d+")           # § 211.42 → part 211
_CHAPTER_RE = re.compile(r"\bchapter\s+([IVXivx]+)\b", re.IGNORECASE)
_AMENDED_RE = re.compile(r"amended\s+after\s+(\d{4})", re.IGNORECASE)


def extract_filters_from_query(query: str, existing: dict) -> dict:
    """
    Scan a natural-language query for explicit filter hints and return a dict
    of filter kwargs to merge with caller-supplied filters.

    Only populates fields that are None in `existing` — never overwrites
    an explicit caller-supplied value.

    Args:
        query: The user's raw query string.
        existing: Dict of currently set filter kwargs (None values = unset).

    Returns:
        Updated dict with auto-extracted values filled in where possible.
    """
    result = dict(existing)

    if result.get("part_number") is None:
        m = _PART_RE.search(query) or _SECTION_RE.search(query)
        if m:
            result["part_number"] = int(m.group(1))

    if result.get("chapter_number") is None:
        m = _CHAPTER_RE.search(query)
        if m:
            result["chapter_number"] = m.group(1).upper()

    if result.get("amended_after") is None:
        m = _AMENDED_RE.search(query)
        if m:
            result["amended_after"] = f"{m.group(1)}-01-01"

    return result


def build_filter(
    part_number: int | None = None,
    part_numbers: list[int] | None = None,
    chapter_number: str | None = None,
    subpart_letter: str | None = None,
    chunk_types: list[str] | None = None,
    exclude_removed: bool = True,
    amended_after: str | None = None,   # ISO date string e.g. "2020-01-01"
    has_table: bool | None = None,
) -> Filter | None:
    """
    Construct a Qdrant Filter from optional parameters.
    Returns None if no conditions are specified (disables filtering).

    Part filtering precedence:
      - part_number (single int, exact match) takes priority over part_numbers.
      - part_numbers (list[int], MatchAny) is used when no specific single part
        is targeted — e.g. a default scope of [210, 211].
    """
    conditions: list = []

    if exclude_removed:
        conditions.append(FieldCondition(key="is_removed", match=MatchValue(value=False)))

    if part_number is not None:
        # Specific part wins — exact single-value match
        conditions.append(FieldCondition(key="part_number", match=MatchValue(value=part_number)))
    elif part_numbers:
        # Scope filter — match any of the listed parts
        conditions.append(FieldCondition(key="part_number", match=MatchAny(any=part_numbers)))

    if chapter_number:
        conditions.append(
            FieldCondition(key="chapter_number", match=MatchValue(value=chapter_number))
        )

    if subpart_letter:
        conditions.append(
            FieldCondition(key="subpart_letter", match=MatchValue(value=subpart_letter.upper()))
        )

    if chunk_types:
        conditions.append(FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types)))

    if amended_after:
        # Lexicographic comparison works for ISO date strings (YYYY-MM-DD)
        conditions.append(
            FieldCondition(key="latest_amendment_date", range=Range(gte=amended_after))
        )

    if has_table is not None:
        conditions.append(FieldCondition(key="has_table", match=MatchValue(value=has_table)))

    return Filter(must=conditions) if conditions else None
