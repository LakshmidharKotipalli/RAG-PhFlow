#!/usr/bin/env python3
"""
Quick spot-check of the Qdrant collection after ingestion.
Prints stats and 5 random sample payloads.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phflow.config.settings import settings
from phflow.vectorstore.client import get_qdrant_client


def main() -> None:
    client = get_qdrant_client()

    if not client.collection_exists(settings.collection_name):
        print(f"Collection '{settings.collection_name}' does not exist. Run ingest.py first.")
        sys.exit(1)

    info = client.get_collection(settings.collection_name)
    print(f"Collection: {settings.collection_name}")
    print(f"  Total points     : {info.points_count:,}")
    print(f"  Vectors count    : {info.vectors_count:,}")
    print(f"  Status           : {info.status}")
    print()

    # Sample 5 random points
    results, _ = client.scroll(
        collection_name=settings.collection_name,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    print("Sample payloads:")
    print("=" * 60)
    for i, point in enumerate(results, 1):
        p = point.payload
        print(f"\n[{i}] {p.get('section_number')} — {p.get('section_title', '')[:60]}")
        print(f"     Part {p.get('part_number')}: {p.get('part_name', '')[:50]}")
        print(f"     Chunk {p.get('chunk_index')}/{p.get('total_chunks')} | type={p.get('chunk_type')}")
        print(f"     Amended: {p.get('latest_amendment_date', 'N/A')} | removed={p.get('is_removed')}")
        print(f"     Tokens: {p.get('token_count')} | has_table={p.get('has_table')}")
        print(f"     Text preview: {str(p.get('chunk_text', ''))[:120]}...")


if __name__ == "__main__":
    main()
