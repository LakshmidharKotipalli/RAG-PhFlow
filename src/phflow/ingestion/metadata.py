"""
Metadata enrichment: joins parsed section data with versions-title-21.json
to add amendment dates, substantive change flags, and removal status.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_versions(versions_path: Path) -> dict[str, dict]:
    """
    Load versions-title-21.json and return a dict keyed by section identifier.

    The identifier in versions JSON is the section number without "§ " prefix,
    e.g. "1.1", "101.36".
    """
    with open(versions_path, encoding="utf-8") as f:
        data = json.load(f)

    versions: dict[str, dict] = {}
    for entry in data.get("content_versions", []):
        identifier = entry.get("identifier", "")
        if not identifier:
            continue
        # Keep the most recent entry per identifier (versions are chronologically ordered)
        existing = versions.get(identifier)
        if existing is None:
            versions[identifier] = entry
        else:
            # Prefer the latest amendment date
            if entry.get("amendment_date", "") > existing.get("amendment_date", ""):
                versions[identifier] = entry

    return versions


def section_identifier(section_number: str) -> str:
    """Convert '§ 1.1' → '1.1'."""
    return section_number.replace("§", "").strip()


def enrich_with_version(
    section_number: str, versions: dict[str, dict]
) -> dict:
    """
    Look up version metadata for a section and return enrichment fields.
    Returns defaults if section not found in versions index.
    """
    ident = section_identifier(section_number)
    entry = versions.get(ident, {})

    return {
        "latest_amendment_date": entry.get("amendment_date", ""),
        "issue_date": entry.get("issue_date", ""),
        "substantive_change": entry.get("substantive", False),
        "is_removed": entry.get("removed", False),
    }
