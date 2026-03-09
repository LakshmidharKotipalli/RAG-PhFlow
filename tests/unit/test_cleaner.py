"""Unit tests for the XML cleaner module."""

from lxml import etree
from phflow.ingestion.cleaner import extract_cita, extract_paragraphs, normalize_whitespace


def _xml(s: str) -> etree._Element:
    return etree.fromstring(s)


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"


def test_extract_paragraphs_basic():
    elem = _xml(
        "<DIV8><HEAD>§ 1.1 General.</HEAD>"
        "<P>(a) First paragraph.</P>"
        "<P>(b) Second paragraph.</P>"
        "</DIV8>"
    )
    paras = extract_paragraphs(elem)
    assert len(paras) == 2
    assert "(a)" in paras[0]
    assert "(b)" in paras[1]


def test_extract_paragraphs_skips_head():
    elem = _xml("<DIV8><HEAD>§ 1.1</HEAD><P>Content.</P></DIV8>")
    paras = extract_paragraphs(elem)
    assert all("§" not in p or "1.1" not in p for p in paras)


def test_extract_cita():
    elem = _xml(
        "<DIV8><P>Text.</P>"
        "<CITA TYPE='N'>[42 FR 15553, Mar. 22, 1977]</CITA>"
        "</DIV8>"
    )
    cita = extract_cita(elem)
    assert "42 FR" in cita
