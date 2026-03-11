#!/usr/bin/env python3
"""
Ingestion CLI.

Usage:
    python scripts/ingest.py                  # incremental (skip if collection exists)
    python scripts/ingest.py --reset          # drop and recreate collection
    python scripts/ingest.py --batch-size 32  # custom batch size
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phflow.ingestion.pipeline import run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ECFR Title 21 XML into FAISS")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the FAISS index before ingestion",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        dest="batch_size",
        help="Number of chunks per upsert batch (default: 64)",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Override XML file path",
    )
    parser.add_argument(
        "--versions",
        type=Path,
        default=None,
        help="Override versions JSON path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PhFlow Ingestion Pipeline")
    print("=" * 60)
    if args.reset:
        print("Mode: RESET (collection will be dropped and recreated)")
    else:
        print("Mode: INCREMENTAL (skips if collection already exists)")
    print(f"Batch size: {args.batch_size}")
    print()

    start = time.time()
    stats = run_ingestion(
        xml_path=args.xml,
        versions_path=args.versions,
        reset=args.reset,
        ingest_batch_size=args.batch_size,
    )
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print("Ingestion Complete")
    print("=" * 60)
    print(f"  Sections processed : {stats['sections_processed']:,}")
    print(f"  Chunks created     : {stats['chunks_created']:,}")
    print(f"  Errors             : {stats['errors']}")
    print(f"  Elapsed time       : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print()

    if stats["errors"] > 0:
        print(f"WARNING: {stats['errors']} errors occurred. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
