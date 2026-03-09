"""
Structural chunker for CFR sections.

Strategy (parent-child model):
  - Parent: full section text stored in payload, NOT embedded
  - Children: one chunk per top-level CFR paragraph label (a), (b), (c), ...
    Each chunk includes the top-level paragraph and all its sub-paragraphs
    (1), (2), (i), (ii), (A), etc.

  Special cases:
    - Intro text (no label, before first (a)) → own "intro" chunk
    - Single-paragraph sections → single chunk
    - Tables → dedicated chunk with chunk_type="table"
    - AUTH/SOURCE blocks → dedicated chunk with chunk_type="authority"
    - Overflow groups (>max_tokens) → split at (1)/(2)/... number boundaries

Tokenizer: BAAI/bge-base-en-v1.5 (matches the embedding model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from phflow.ingestion.parser import SectionData

_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

# Roman numeral words that are ambiguous with single-letter top-level labels.
# When one of these appears as a label AND we've already seen a top-level letter,
# treat it as a sub-level roman numeral, not a new top-level group.
_ROMAN_SINGLES = frozenset({"i", "v", "x", "l", "c", "d", "m"})

# Regex to detect a CFR label at the very start of a paragraph, e.g. "(a)", "(1)", "(ii)"
_LABEL_RE = re.compile(r"^\s*\(([a-zA-Z0-9]+)\)")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def count_tokens(text: str) -> int:
    """Public API: count tokens using the embedding model tokenizer."""
    return _count_tokens(text)


@dataclass
class Chunk:
    text: str
    chunk_type: str       # "text" | "table" | "authority"
    chunk_index: int
    section_number: str
    section_node: str
    section_title: str
    paragraph_label: str = ""   # top-level label: "a", "b", "intro", or "" for table/authority


def _build_header(section: SectionData) -> str:
    """Context header prepended to every chunk."""
    header = f"21 CFR {section.section_number} — {section.section_title}"
    if section.part_number:
        header += f"\nPart {section.part_number}: {section.part_name}"
    if section.subpart_letter:
        header += f" | Subpart {section.subpart_letter}: {section.subpart_name}"
    return header + "\n\n"


def _detect_label(text: str) -> tuple[str, str]:
    """
    Detect a CFR paragraph label at the start of text.

    Returns (label, kind) where kind is one of:
      'letter'  — single lowercase letter, e.g. "a", "b" (may be roman numeral)
      'number'  — digit string, e.g. "1", "2"
      'roman'   — multi-char roman numeral, e.g. "ii", "iii"
      'upper'   — uppercase letter(s), e.g. "A", "B"
      ''        — no label detected
    """
    m = _LABEL_RE.match(text)
    if not m:
        return "", ""
    label = m.group(1)
    if re.fullmatch(r"\d+", label):
        return label, "number"
    if re.fullmatch(r"[A-Z]+", label):
        return label, "upper"
    if re.fullmatch(r"[a-z]+", label):
        if len(label) == 1:
            return label, "letter"
        # multi-char lowercase = roman numeral (ii, iii, iv, ...)
        return label, "roman"
    return label, ""


def _is_new_top_level(label: str, kind: str, last_top: str) -> bool:
    """
    Decide whether this label starts a new top-level paragraph group.

    Rules:
    - Only 'letter' kind can be top-level
    - "(a)" always starts a new group (resets the sequence)
    - After "(a)" has been seen, the next expected letter starts a new group
    - Single-char roman numeral singles (i, v, x, l, c, d, m) are treated as
      sub-level when we are already inside a top-level sequence
    """
    if kind != "letter":
        return False
    if label == "a":
        return True
    if not last_top:
        # No top-level seen yet — treat as top-level unless it looks roman
        return label not in _ROMAN_SINGLES
    # We're inside a sequence: accept the next expected letter
    expected = chr(ord(last_top) + 1)
    return label == expected


def _group_by_structure(
    paragraphs: list[str],
) -> list[tuple[str, list[str]]]:
    """
    Group paragraphs into structural units based on CFR label hierarchy.

    Returns a list of (group_label, [paragraph, ...]) tuples:
      - group_label == "intro"       for unlabeled intro text
      - group_label == "a"/"b"/...   for top-level labeled paragraphs
      - group_label == "__table__"   for markdown table paragraphs
      - group_label == "__auth__"    for [AUTH]/[SOURCE] paragraphs
    """
    groups: list[tuple[str, list[str]]] = []
    current_label: str = "intro"
    current_group: list[str] = []
    last_top: str = ""   # last confirmed top-level letter

    def flush() -> None:
        if current_group:
            groups.append((current_label, list(current_group)))
            current_group.clear()

    for para in paragraphs:
        # --- Special block types: always their own group ---
        if para.startswith("|"):
            flush()
            groups.append(("__table__", [para]))
            continue
        if para.startswith("[AUTH]") or para.startswith("[SOURCE]"):
            flush()
            groups.append(("__auth__", [para]))
            continue

        label, kind = _detect_label(para)

        if _is_new_top_level(label, kind, last_top):
            flush()
            current_label = label
            last_top = label
            current_group.append(para)
        else:
            # sub-paragraph, roman numeral, number, or unlabeled continuation
            current_group.append(para)

    flush()
    return groups


def _overflow_split(
    paras: list[str],
    header: str,
    max_tokens: int,
) -> list[str]:
    """
    Split a paragraph group that exceeds max_tokens.
    Splits at (1)/(2)/... number-label boundaries.
    Returns a list of chunk text strings (each includes the header).
    """
    # Find number-label split points
    split_indices: list[int] = [0]
    for i, para in enumerate(paras[1:], start=1):
        _, kind = _detect_label(para)
        if kind == "number":
            split_indices.append(i)

    if len(split_indices) == 1:
        # No number-label splits found — return as single chunk despite overflow
        return [header + "\n".join(paras)]

    chunks: list[str] = []
    split_indices.append(len(paras))   # sentinel

    for start, end in zip(split_indices, split_indices[1:]):
        segment = paras[start:end]
        chunks.append(header + "\n".join(segment))

    return chunks


def chunk_section(
    section: SectionData,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Produce ordered Chunk objects for a SectionData using structural grouping."""
    header = _build_header(section)
    chunks: list[Chunk] = []

    groups = _group_by_structure(section.paragraphs)

    for group_label, paras in groups:
        # --- Table chunk ---
        if group_label == "__table__":
            chunks.append(Chunk(
                text=header + paras[0],
                chunk_type="table",
                chunk_index=len(chunks),
                section_number=section.section_number,
                section_node=section.section_node,
                section_title=section.section_title,
                paragraph_label="",
            ))
            continue

        # --- Authority chunk ---
        if group_label == "__auth__":
            chunks.append(Chunk(
                text=header + "\n".join(paras),
                chunk_type="authority",
                chunk_index=len(chunks),
                section_number=section.section_number,
                section_node=section.section_node,
                section_title=section.section_title,
                paragraph_label="",
            ))
            continue

        # --- Text chunk (intro or labeled paragraph group) ---
        full_text = header + "\n".join(paras)
        if _count_tokens(full_text) > max_tokens:
            # Overflow: split at sub-paragraph number boundaries
            texts = _overflow_split(paras, header, max_tokens)
        else:
            texts = [full_text]

        for text in texts:
            chunks.append(Chunk(
                text=text,
                chunk_type="text",
                chunk_index=len(chunks),
                section_number=section.section_number,
                section_node=section.section_node,
                section_title=section.section_title,
                paragraph_label=group_label,
            ))

    return chunks


def build_parent_text(section: SectionData) -> str:
    """Concatenate all paragraphs into the full parent text (not chunked)."""
    return "\n\n".join(section.paragraphs)
