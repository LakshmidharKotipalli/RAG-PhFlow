#!/usr/bin/env python3
"""
Quick spot-check of the FAISS index after ingestion.
Prints stats and 5 sample payloads.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phflow.settings import settings
from phflow.vectorstore.client import get_faiss_store


def main() -> None:
    store = get_faiss_store()

    if not store.exists():
        print(f"FAISS index does not exist at '{settings.faiss_index_dir}'. Run ingest.py first.")
        sys.exit(1)

    print(f"FAISS Index: {settings.faiss_index_dir}")
    print(f"  Total points     : {store.total_points:,}")
    print(f"  Metadata entries : {len(store._metadata):,}")
    print(f"  Sparse entries   : {len(store._sparse_data):,}")
    print()

    # Sample 5 points
    samples = store._metadata[:5]

    print("Sample payloads:")
    print("=" * 60)
    for i, p in enumerate(samples, 1):
        print(f"\n[{i}] {p.get('section_number')} — {p.get('section_title', '')[:60]}")
        print(f"     Part {p.get('part_number')}: {p.get('part_name', '')[:50]}")
        print(f"     Chunk {p.get('chunk_index')}/{p.get('total_chunks')} | type={p.get('chunk_type')}")
        print(f"     Amended: {p.get('latest_amendment_date', 'N/A')} | removed={p.get('is_removed')}")
        print(f"     Tokens: {p.get('token_count')} | has_table={p.get('has_table')}")
        print(f"     Text preview: {str(p.get('chunk_text', ''))[:120]}...")


if __name__ == "__main__":
    main()
