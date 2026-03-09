"""Shared pytest fixtures."""

import pytest


@pytest.fixture(scope="session")
def sample_xml_path():
    """Path to the real ECFR XML file."""
    from pathlib import Path
    p = Path("data/ecfr-title-21/ECFR-title21.xml")
    if not p.exists():
        pytest.skip("ECFR XML file not found")
    return p


@pytest.fixture(scope="session")
def versions_path():
    from pathlib import Path
    p = Path("data/ecfr-title-21/versions-title-21.json")
    if not p.exists():
        pytest.skip("versions JSON file not found")
    return p
