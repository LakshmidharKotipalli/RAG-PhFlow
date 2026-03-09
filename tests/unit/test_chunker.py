"""Unit tests for the structural chunker module."""

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


# --- Structural grouping ---

def test_labeled_paragraphs_grouped_by_top_level():
    """(a)+(1)+(2) in one chunk, (b)+(1) in another."""
    section = _make_section(paragraphs=[
        "(a) First rule.",
        "(1) Sub-item one.",
        "(2) Sub-item two.",
        "(b) Second rule.",
        "(1) Sub-item of b.",
    ])
    chunks = chunk_section(section)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) == 2
    assert text_chunks[0].paragraph_label == "a"
    assert "(a) First rule." in text_chunks[0].text
    assert "(1) Sub-item one." in text_chunks[0].text
    assert "(2) Sub-item two." in text_chunks[0].text
    assert text_chunks[1].paragraph_label == "b"
    assert "(b) Second rule." in text_chunks[1].text
    assert "(1) Sub-item of b." in text_chunks[1].text


def test_intro_text_gets_own_chunk():
    """Unlabeled text before (a) becomes a separate intro chunk."""
    section = _make_section(paragraphs=[
        "This section establishes general requirements.",
        "(a) Requirement one.",
        "(b) Requirement two.",
    ])
    chunks = chunk_section(section)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) == 3
    assert text_chunks[0].paragraph_label == "intro"
    assert "This section establishes" in text_chunks[0].text
    assert text_chunks[1].paragraph_label == "a"
    assert text_chunks[2].paragraph_label == "b"


def test_single_paragraph_section_is_one_chunk():
    section = _make_section(paragraphs=["Short text with no labels."])
    chunks = chunk_section(section)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "text"
    assert chunks[0].paragraph_label == "intro"


def test_two_labeled_paragraphs_become_two_chunks():
    section = _make_section(paragraphs=["(a) First.", "(b) Second."])
    chunks = chunk_section(section)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) == 2
    assert text_chunks[0].paragraph_label == "a"
    assert text_chunks[1].paragraph_label == "b"


def test_sub_paragraphs_stay_with_parent():
    """(1) and (2) sub-items do not become separate top-level chunks."""
    section = _make_section(paragraphs=[
        "(a) Parent rule.",
        "(1) First sub.",
        "(2) Second sub.",
        "(i) Third sub.",
        "(ii) Fourth sub.",
    ])
    chunks = chunk_section(section)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) == 1
    assert text_chunks[0].paragraph_label == "a"
    full = text_chunks[0].text
    assert "(1) First sub." in full
    assert "(2) Second sub." in full
    assert "(i) Third sub." in full


# --- Special chunk types ---

def test_table_paragraph_gets_table_type():
    section = _make_section(paragraphs=["| Col1 | Col2 |\n|---|---|\n| A | B |"])
    chunks = chunk_section(section)
    assert any(c.chunk_type == "table" for c in chunks)


def test_authority_paragraph_gets_authority_type():
    section = _make_section(paragraphs=["[AUTH] 21 U.S.C. 321"])
    chunks = chunk_section(section)
    assert any(c.chunk_type == "authority" for c in chunks)


def test_table_does_not_absorb_surrounding_text():
    """Table appears between two labeled paragraphs; each stays separate."""
    section = _make_section(paragraphs=[
        "(a) See the table below.",
        "| Header |\n|---|\n| Value |",
        "(b) After the table.",
    ])
    chunks = chunk_section(section)
    types = [c.chunk_type for c in chunks]
    assert "table" in types
    assert types.count("text") == 2


# --- Chunk metadata ---

def test_chunk_indices_are_sequential():
    section = _make_section(paragraphs=[
        "(a) Para one.",
        "(b) Para two.",
        "| A | B |",
    ])
    chunks = chunk_section(section)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_context_header_in_every_chunk():
    section = _make_section()
    chunks = chunk_section(section)
    for chunk in chunks:
        assert "21 CFR" in chunk.text
        assert "§ 1.1" in chunk.text


def test_section_number_on_every_chunk():
    section = _make_section(paragraphs=["(a) Text.", "[AUTH] Auth text."])
    chunks = chunk_section(section)
    for chunk in chunks:
        assert chunk.section_number == "§ 1.1"
        assert chunk.section_node == "21:1.0.1.1.1.1.49.1"


# --- Overflow splitting ---

def test_overflow_group_splits_at_sub_paragraphs():
    """A group exceeding max_tokens is split at (1)/(2) boundaries."""
    long_word = "regulation " * 30
    section = _make_section(paragraphs=[
        f"(a) {long_word}",
        f"(1) {long_word}",
        f"(2) {long_word}",
    ])
    chunks = chunk_section(section, max_tokens=50)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert len(text_chunks) > 1
    for c in text_chunks:
        assert c.paragraph_label == "a"
