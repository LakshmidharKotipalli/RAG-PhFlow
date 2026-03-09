"""Text normalization and XML content cleaning utilities."""

import re
from lxml import etree


# Tags whose text content should be preserved inline
INLINE_TAGS = {"I", "E", "SU", "FR", "BCAP", "TCAP", "AC"}

# Tags to skip entirely (no text content)
SKIP_TAGS = {"AMDDATE", "EDNOTE", "EFFDNOT", "REVTXT", "SECAUTH", "APPRO", "FTREF", "FTNT"}


def element_to_text(elem: etree._Element) -> str:
    """Recursively extract clean text from an XML element, handling inline tags."""
    tag = _local_name(elem.tag)

    if tag in SKIP_TAGS:
        return ""

    if tag == "TABLE":
        return _table_to_markdown(elem)

    # Build text by walking the element tree.
    # Inline children (e.g. <I>, <E>) are accumulated and joined with spaces.
    # Block children (e.g. nested <P>) flush the inline buffer first, then are
    # separated with double-newlines so paragraph structure is preserved.
    inline_parts: list[str] = []
    block_parts: list[str] = []

    if elem.text:
        inline_parts.append(elem.text.strip())

    for child in elem:
        child_tag = _local_name(child.tag)
        child_text = element_to_text(child)

        if child_tag in INLINE_TAGS:
            if child_text:
                inline_parts.append(child_text)
        else:
            # Flush accumulated inline text as one block piece
            if inline_parts:
                block_parts.append(" ".join(p for p in inline_parts if p))
                inline_parts = []
            if child_text:
                block_parts.append(child_text)

        if child.tail:
            inline_parts.append(child.tail.strip())

    # Flush any remaining inline text
    if inline_parts:
        block_parts.append(" ".join(p for p in inline_parts if p))

    return "\n\n".join(p for p in block_parts if p)


def _local_name(tag: str) -> str:
    """Strip namespace prefix from tag name."""
    if "{" in tag:
        return tag.split("}", 1)[1]
    return tag


def _table_to_markdown(table_elem: etree._Element) -> str:
    """Convert a TABLE element to pipe-delimited markdown."""
    rows: list[list[str]] = []

    for tr in table_elem.iter("TR"):
        cells: list[str] = []
        for cell in tr:
            cell_tag = _local_name(cell.tag)
            if cell_tag in ("TH", "TD"):
                # Flatten any nested elements in cells
                text = "".join(cell.itertext()).strip()
                text = re.sub(r"\s+", " ", text)
                cells.append(text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            # Header separator
            lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(lines)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace, strip leading/trailing."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_paragraphs(section_elem: etree._Element) -> list[str]:
    """
    Extract a list of paragraph strings from a DIV8 section element.
    Each <P>, <FP>, <FP-2>, <EXTRACT> becomes one paragraph string.
    Tables are included as their own paragraph.
    """
    paragraphs: list[str] = []
    para_tags = {"P", "P1", "P2", "FP", "FP1", "FP2", "FP-2", "EXTRACT"}

    for child in section_elem:
        tag = _local_name(child.tag)
        if tag == "HEAD":
            continue  # Already captured as section_title
        if tag == "TABLE":
            md = _table_to_markdown(child)
            if md:
                paragraphs.append(md)
        elif tag in para_tags:
            text = element_to_text(child)
            text = normalize_whitespace(text)
            if text:
                paragraphs.append(text)
        elif tag in ("AUTH", "SOURCE"):
            # Auth/source: capture as a block
            text = "".join(child.itertext()).strip()
            text = normalize_whitespace(text)
            if text:
                paragraphs.append(f"[{tag}] {text}")
        # CITA is captured separately in metadata; skip here

    return paragraphs


def extract_auth_source(section_elem: etree._Element) -> tuple[str, str]:
    """Extract AUTH and SOURCE element text."""
    auth = ""
    source = ""
    for child in section_elem:
        tag = _local_name(child.tag)
        if tag == "AUTH":
            auth = normalize_whitespace("".join(child.itertext()))
        elif tag == "SOURCE":
            source = normalize_whitespace("".join(child.itertext()))
    return auth, source


def extract_cita(section_elem: etree._Element) -> str:
    """Extract CITA element text (Federal Register amendment history)."""
    for child in section_elem:
        if _local_name(child.tag) == "CITA":
            return normalize_whitespace("".join(child.itertext()))
    return ""
