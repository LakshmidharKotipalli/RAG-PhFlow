"""
Streaming XML parser for ECFR Title 21 XML using lxml iterparse.

Walks the DIV hierarchy maintaining an ancestor stack, emitting one
structured dict per DIV8 (section) element without loading the full
21.7MB DOM into memory.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from phflow.ingestion.cleaner import (
    extract_auth_source,
    extract_cita,
    extract_paragraphs,
    normalize_whitespace,
)


@dataclass
class HierarchyFrame:
    """One level in the ancestor stack."""
    div_type: str      # TITLE, CHAPTER, SUBCHAP, PART, SUBPART
    number: str        # N attribute value
    head: str          # HEAD element text


@dataclass
class SectionData:
    """All data extracted from a single DIV8 section element."""
    # Section identity
    section_number: str      # e.g. "§ 1.1"
    section_node: str        # e.g. "21:1.0.1.1.1.1.49.1"
    section_title: str       # HEAD text

    # Hierarchy (from ancestor stack)
    title_number: int
    title_name: str
    chapter_number: str
    chapter_name: str
    subchapter_letter: str
    subchapter_name: str
    part_number: int
    part_name: str
    subpart_letter: str
    subpart_name: str

    # Content
    paragraphs: list[str] = field(default_factory=list)
    authority_text: str = ""
    source_text: str = ""
    cita_text: str = ""

    # Flags
    has_table: bool = False


def _local_name(tag: str) -> str:
    if "{" in tag:
        return tag.split("}", 1)[1]
    return tag


def _get_head(elem: etree._Element) -> str:
    """Return text of the first HEAD child element."""
    for child in elem:
        if _local_name(child.tag) == "HEAD":
            return normalize_whitespace("".join(child.itertext()))
    return ""


def _parse_part_number(n_attr: str) -> int:
    """Extract integer part number from N attribute like 'PART 1' or '1'."""
    digits = re.search(r"\d+", n_attr)
    return int(digits.group()) if digits else 0


def _parse_subpart_letter(n_attr: str) -> str:
    """Extract letter from subpart N attribute like 'Subpart A' or 'A'."""
    m = re.search(r"[A-Z]", n_attr, re.IGNORECASE)
    return m.group().upper() if m else ""


def parse_sections(xml_path: Path) -> Generator[SectionData, None, None]:
    """
    Stream-parse ECFR XML and yield one SectionData per DIV8 section.

    Uses lxml iterparse with explicit memory management (elem.clear())
    so the full 21.7MB DOM is never resident in memory simultaneously.
    """
    # Stack of HierarchyFrame objects; one per open DIV1/3/4/5/6
    stack: list[HierarchyFrame] = []

    # Mapping from div element -> frame (so we can pop on "end" event)
    div_map: dict[int, HierarchyFrame] = {}

    ANCESTOR_DIVS = {"DIV1", "DIV3", "DIV4", "DIV5", "DIV6"}
    DIV_TYPE_MAP = {
        "DIV1": "TITLE",
        "DIV3": "CHAPTER",
        "DIV4": "SUBCHAP",
        "DIV5": "PART",
        "DIV6": "SUBPART",
    }

    context = etree.iterparse(
        str(xml_path),
        events=("start", "end"),
        recover=True,
        huge_tree=True,
    )

    prev_sibling: etree._Element | None = None

    for event, elem in context:
        tag = _local_name(elem.tag)

        if event == "start":
            if tag in ANCESTOR_DIVS:
                div_type = DIV_TYPE_MAP[tag]
                n_attr = elem.get("N", "")
                frame = HierarchyFrame(
                    div_type=div_type,
                    number=n_attr,
                    head="",  # populated when HEAD is parsed; use lazy approach
                )
                stack.append(frame)
                div_map[id(elem)] = frame

        elif event == "end":
            if tag in ANCESTOR_DIVS:
                # Populate head now that the element is fully parsed
                frame = div_map.pop(id(elem), None)
                if frame:
                    frame.head = _get_head(elem)
                    # Remove from stack
                    if frame in stack:
                        stack.remove(frame)

                # Memory management: clear ancestor divs after processing their sections
                # (DIV8s are processed before their parent DIV ends)
                if tag in ("DIV5", "DIV6"):
                    # Keep elem alive until after section processing — done via stack
                    pass

            elif tag == "DIV8":
                section = _build_section(elem, stack)
                if section:
                    yield section

                # Clear the DIV8 element from memory
                elem.clear()
                # Also eliminate references from the parent's child list
                if prev_sibling is not None:
                    prev_sibling.clear()
                prev_sibling = elem

    del context


def _build_section(elem: etree._Element, stack: list[HierarchyFrame]) -> SectionData | None:
    """Build a SectionData from a DIV8 element and current hierarchy stack."""
    if elem.get("TYPE", "") != "SECTION":
        return None

    section_number = elem.get("N", "")
    section_node = elem.get("NODE", "")
    section_title = _get_head(elem)

    # Walk stack to extract hierarchy levels
    title_number = 21
    title_name = ""
    chapter_number = ""
    chapter_name = ""
    subchapter_letter = ""
    subchapter_name = ""
    part_number = 0
    part_name = ""
    subpart_letter = ""
    subpart_name = ""

    for frame in stack:
        if frame.div_type == "TITLE":
            title_name = frame.head
            m = re.search(r"\d+", frame.number)
            if m:
                title_number = int(m.group())
        elif frame.div_type == "CHAPTER":
            chapter_number = frame.number.replace("CHAPTER", "").strip()
            chapter_name = frame.head
        elif frame.div_type == "SUBCHAP":
            subchapter_letter = frame.number.replace("SUBCHAPTER", "").strip()
            subchapter_name = frame.head
        elif frame.div_type == "PART":
            part_number = _parse_part_number(frame.number)
            part_name = frame.head
        elif frame.div_type == "SUBPART":
            subpart_letter = _parse_subpart_letter(frame.number)
            subpart_name = frame.head

    # Check for tables
    has_table = any(_local_name(child.tag) == "TABLE" for child in elem.iter())

    # Extract content
    paragraphs = extract_paragraphs(elem)
    authority_text, source_text = extract_auth_source(elem)
    cita_text = extract_cita(elem)

    return SectionData(
        section_number=section_number,
        section_node=section_node,
        section_title=section_title,
        title_number=title_number,
        title_name=title_name,
        chapter_number=chapter_number,
        chapter_name=chapter_name,
        subchapter_letter=subchapter_letter,
        subchapter_name=subchapter_name,
        part_number=part_number,
        part_name=part_name,
        subpart_letter=subpart_letter,
        subpart_name=subpart_name,
        paragraphs=paragraphs,
        authority_text=authority_text,
        source_text=source_text,
        cita_text=cita_text,
        has_table=has_table,
    )
