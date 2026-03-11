"""
Build filter predicates from user-supplied query parameters.

Replaces qdrant_client.models.Filter with plain Python callables
that can be applied to FAISS metadata dicts.
"""

from __future__ import annotations

import re
from typing import Callable

# Patterns for extracting filter hints from natural-language queries
_PART_RE = re.compile(r"\bpart\s+(\d+)\b", re.IGNORECASE)
_EXACT_SECTION_RE = re.compile(r"\bsection\s+(\d+\.\d+)\b", re.IGNORECASE)  # "section 211.80"
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

    if result.get("section_number") is None:
        ms = _EXACT_SECTION_RE.search(query)
        if ms:
            result["section_number"] = f"§ {ms.group(1)}"
            # If they specify a section, implicitly set the part number
            result["part_number"] = int(ms.group(1).split(".")[0])

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
    section_number: str | None = None,
    chapter_number: str | None = None,
    subpart_letter: str | None = None,
    chunk_types: list[str] | None = None,
    exclude_removed: bool = True,
    amended_after: str | None = None,   # ISO date string e.g. "2020-01-01"
    has_table: bool | None = None,
) -> Callable[[dict], bool] | None:
    """
    Construct a filter predicate from optional parameters.

    Returns a callable ``filter_fn(payload) -> bool`` that returns True
    for payloads matching ALL conditions, or None if no conditions are
    specified.

    Part filtering precedence:
      - part_number (single int, exact match) takes priority over part_numbers.
      - part_numbers (list[int]) is used when no specific single part
        is targeted — e.g. a default scope of [210, 211].
    """
    conditions: list[Callable[[dict], bool]] = []

    if exclude_removed:
        conditions.append(lambda p: not p.get("is_removed", False))

    if section_number is not None:
        conditions.append(lambda p, sn=section_number: p.get("section_number") == sn)
        
    if part_number is not None:
        conditions.append(lambda p, pn=part_number: p.get("part_number") == pn)
    elif part_numbers:
        _pn_set = set(part_numbers)
        conditions.append(lambda p: p.get("part_number") in _pn_set)

    if chapter_number:
        conditions.append(lambda p, cn=chapter_number: p.get("chapter_number") == cn)

    if subpart_letter:
        _sl = subpart_letter.upper()
        conditions.append(lambda p: p.get("subpart_letter") == _sl)

    if chunk_types:
        _ct_set = set(chunk_types)
        conditions.append(lambda p: p.get("chunk_type") in _ct_set)

    if amended_after:
        conditions.append(
            lambda p, aa=amended_after: (p.get("latest_amendment_date") or "") >= aa
        )

    if has_table is not None:
        conditions.append(lambda p, ht=has_table: p.get("has_table") == ht)

    if not conditions:
        return None

    def _combined(payload: dict) -> bool:
        return all(cond(payload) for cond in conditions)

    return _combined
