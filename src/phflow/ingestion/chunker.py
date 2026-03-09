"""
Hierarchical paragraph-boundary chunker for CFR sections.

Strategy (parent-child model):
  - Parent: full section text stored in payload, NOT embedded
  - Children: paragraph-group chunks, 400 token target / 512 max
    - Short sections (<150 tokens total) → single chunk
    - Tables → dedicated chunk with chunk_type="table"
    - AUTH/SOURCE blocks → dedicated chunk with chunk_type="authority"
    - Regular paragraphs → grouped to target size, split at sentence boundary
      if a single paragraph exceeds max

Each chunk text is prefixed with a regulatory context header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from phflow.ingestion.parser import SectionData

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '. ' boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    text: str
    chunk_type: str  # "text" | "table" | "authority"
    chunk_index: int
    section_number: str
    section_node: str
    section_title: str


def _build_header(section: SectionData) -> str:
    """Context header prepended to every chunk."""
    header = f"21 CFR {section.section_number} — {section.section_title}"
    if section.part_number:
        header += f"\nPart {section.part_number}: {section.part_name}"
    if section.subpart_letter:
        header += f" | Subpart {section.subpart_letter}: {section.subpart_name}"
    return header + "\n\n"


def _merge_groups(
    paragraphs: list[str],
    target: int,
    maximum: int,
    header: str,
) -> list[str]:
    """
    Group paragraphs into chunks targeting `target` tokens (with header).
    Single paragraphs exceeding `maximum` are split at sentence boundaries.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = _count_tokens(header)

    def flush():
        if buffer:
            chunks.append(header + "\n".join(buffer))
            buffer.clear()

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        if para_tokens > maximum:
            # Must split this paragraph at sentence boundaries
            flush()
            sentences = _split_sentences(para)
            sub_buf: list[str] = []
            sub_tok = _count_tokens(header)
            for sent in sentences:
                sent_tok = _count_tokens(sent)
                if sub_tok + sent_tok + 1 > maximum and sub_buf:
                    chunks.append(header + " ".join(sub_buf))
                    sub_buf = []
                    sub_tok = _count_tokens(header)
                sub_buf.append(sent)
                sub_tok += sent_tok + 1
            if sub_buf:
                chunks.append(header + " ".join(sub_buf))
        elif buffer_tokens + para_tokens + 1 > target and buffer:
            flush()
            buffer.append(para)
            buffer_tokens = _count_tokens(header) + para_tokens
        else:
            buffer.append(para)
            buffer_tokens += para_tokens + 1

    flush()
    return chunks


def chunk_section(
    section: SectionData,
    target_tokens: int = 400,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Produce ordered Chunk objects for a SectionData."""
    header = _build_header(section)
    chunks: list[Chunk] = []

    # Separate paragraph types
    table_paras: list[str] = []
    auth_paras: list[str] = []
    text_paras: list[str] = []

    for para in section.paragraphs:
        if para.startswith("|"):  # markdown table
            table_paras.append(para)
        elif para.startswith("[AUTH]") or para.startswith("[SOURCE]"):
            auth_paras.append(para)
        else:
            text_paras.append(para)

    # --- Text chunks ---
    full_text = "\n".join(text_paras)
    full_tokens = _count_tokens(header + full_text)

    if full_tokens <= max_tokens or not text_paras:
        # Single chunk for the whole section text
        if text_paras:
            chunks.append(
                Chunk(
                    text=header + full_text,
                    chunk_type="text",
                    chunk_index=len(chunks),
                    section_number=section.section_number,
                    section_node=section.section_node,
                    section_title=section.section_title,
                )
            )
    else:
        for text in _merge_groups(text_paras, target_tokens, max_tokens, header):
            chunks.append(
                Chunk(
                    text=text,
                    chunk_type="text",
                    chunk_index=len(chunks),
                    section_number=section.section_number,
                    section_node=section.section_node,
                    section_title=section.section_title,
                )
            )

    # --- Table chunks ---
    for table_text in table_paras:
        chunks.append(
            Chunk(
                text=header + table_text,
                chunk_type="table",
                chunk_index=len(chunks),
                section_number=section.section_number,
                section_node=section.section_node,
                section_title=section.section_title,
            )
        )

    # --- Authority chunks ---
    if auth_paras:
        auth_text = "\n".join(auth_paras)
        chunks.append(
            Chunk(
                text=header + auth_text,
                chunk_type="authority",
                chunk_index=len(chunks),
                section_number=section.section_number,
                section_node=section.section_node,
                section_title=section.section_title,
            )
        )

    return chunks


def build_parent_text(section: SectionData) -> str:
    """Concatenate all paragraphs into the full parent text (not chunked)."""
    return "\n\n".join(section.paragraphs)
