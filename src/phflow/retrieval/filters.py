"""
Build Qdrant Filter objects from user-supplied query parameters.
"""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range


def build_filter(
    part_number: int | None = None,
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
    """
    conditions: list = []

    if exclude_removed:
        conditions.append(FieldCondition(key="is_removed", match=MatchValue(value=False)))

    if part_number is not None:
        conditions.append(FieldCondition(key="part_number", match=MatchValue(value=part_number)))

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
