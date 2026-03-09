"""Unit tests for the chunker module."""

import pytest
from phflow.ingestion.chunker import Chunk, chunk_section
from phflow.ingestion.parser import SectionData


def _make_section(**kwargs) -> SectionData:
    defaults = dict(
        section_number="§ 1.1",
        section_node="21:1.0.1.1.1.1.49.1",
        section_title="§ 1.1   General.",
        title_number=21,
        title_name="Food and Drugs",
        chapter_number="I",
        chapter_name="FDA",
        subchapter_letter="A",
        subchapter_name="General",
        part_number=1,
        part_name="General Enforcement",
        subpart_letter="A",
        subpart_name="General Provisions",
        paragraphs=["(a) This is a short paragraph.", "(b) This is another paragraph."],
        authority_text="",
        source_text="",
        cita_text="",
        has_table=False,
    )
    defaults.update(kwargs)
    return SectionData(**defaults)


def test_short_section_single_chunk():
    section = _make_section(paragraphs=["Short text."])
    chunks = chunk_section(section, target_tokens=400, max_tokens=512)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "text"
    assert chunks[0].chunk_index == 0
    assert "§ 1.1" in chunks[0].text


def test_table_paragraph_gets_table_type():
    section = _make_section(paragraphs=["| Col1 | Col2 |\n|---|---|\n| A | B |"])
    chunks = chunk_section(section)
    assert any(c.chunk_type == "table" for c in chunks)


def test_authority_paragraph_gets_authority_type():
    section = _make_section(paragraphs=["[AUTH] 21 U.S.C. 321"])
    chunks = chunk_section(section)
    assert any(c.chunk_type == "authority" for c in chunks)


def test_chunk_indices_are_sequential():
    section = _make_section(paragraphs=["Para one.", "Para two.", "| A | B |"])
    chunks = chunk_section(section)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_context_header_in_every_chunk():
    section = _make_section()
    chunks = chunk_section(section)
    for chunk in chunks:
        assert "21 CFR" in chunk.text
        assert "§ 1.1" in chunk.text
