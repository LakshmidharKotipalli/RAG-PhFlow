"""
Full ingestion pipeline orchestrator.

Execution order:
  1. Load versions-title-21.json into memory dict
  2. Stream-parse ECFR XML section by section
  3. Enrich each section with version metadata
  4. Chunk each section into paragraph-group chunks
  5. Batch-encode dense vectors (sentence-transformers)
  6. Batch-encode sparse vectors (fastembed BM25)
  7. Upsert batches into Qdrant
"""

from __future__ import annotations

import logging

from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
from tqdm import tqdm

from phflow.config.settings import settings
from phflow.embeddings.encoder import RegulationEncoder
from phflow.ingestion.chunker import Chunk, build_parent_text, chunk_section, count_tokens
from phflow.ingestion.metadata import enrich_with_version, load_versions
from phflow.ingestion.parser import SectionData, parse_sections
from phflow.vectorstore.client import build_point, get_qdrant_client, upsert_batch
from phflow.vectorstore.schema import collection_exists, create_collection

logger = logging.getLogger(__name__)


def _build_payload(section: SectionData, chunk: Chunk, version_meta: dict, total_chunks: int) -> dict:
    token_count = count_tokens(chunk.text)

    return {
        # Section identity
        "section_number": section.section_number,
        "section_node": section.section_node,
        "section_title": section.section_title,
        # Hierarchy
        "title_number": section.title_number,
        "title_name": section.title_name,
        "chapter_number": section.chapter_number,
        "chapter_name": section.chapter_name,
        "subchapter_letter": section.subchapter_letter,
        "subchapter_name": section.subchapter_name,
        "part_number": section.part_number,
        "part_name": section.part_name,
        "subpart_letter": section.subpart_letter,
        "subpart_name": section.subpart_name,
        # Legal metadata
        "authority_text": section.authority_text,
        "source_text": section.source_text,
        "cita_text": section.cita_text,
        # Version metadata (from versions-title-21.json)
        **version_meta,
        # Chunk metadata
        "chunk_index": chunk.chunk_index,
        "chunk_type": chunk.chunk_type,
        "paragraph_label": chunk.paragraph_label,
        "total_chunks": total_chunks,
        "has_table": section.has_table,
        "token_count": token_count,
        # Parent text for context expansion (not embedded)
        "parent_text": build_parent_text(section),
    }


def run_ingestion(
    xml_path: Path | None = None,
    versions_path: Path | None = None,
    reset: bool = False,
    ingest_batch_size: int = 64,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        xml_path: Path to ECFR XML. Defaults to settings.xml_file.
        versions_path: Path to versions JSON. Defaults to settings.versions_file.
        reset: If True, drop and recreate the Qdrant collection first.
        ingest_batch_size: Number of chunks per upsert batch.

    Returns:
        Stats dict with sections_processed, chunks_created, errors.
    """
    xml_path = xml_path or settings.xml_file
    versions_path = versions_path or settings.versions_file

    logger.info("Loading versions metadata from %s", versions_path)
    versions = load_versions(versions_path)
    logger.info("Loaded %d version records", len(versions))

    # Initialize components
    encoder = RegulationEncoder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    qdrant = get_qdrant_client()

    # Collection setup
    if reset and collection_exists(qdrant, settings.collection_name):
        logger.info("Dropping existing collection '%s'", settings.collection_name)
        qdrant.delete_collection(settings.collection_name)

    if not collection_exists(qdrant, settings.collection_name):
        logger.info("Creating collection '%s' (dim=%d)", settings.collection_name, encoder.embedding_dim)
        create_collection(qdrant, settings.collection_name, encoder.embedding_dim)

    # Accumulation buffers for batching
    pending_chunks: list[Chunk] = []
    pending_sections: list[SectionData] = []
    pending_version_metas: list[dict] = []

    stats = {"sections_processed": 0, "chunks_created": 0, "errors": 0}

    def flush_batch(chunks: list[Chunk], sections_map: list[SectionData], vmetas: list[dict]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]

        try:
            dense_vecs = encoder.encode_passages(texts, show_progress=False)
            sparse_vecs = list(sparse_model.embed(texts))

            points = []
            for chunk, dense, sparse, section, vmeta in zip(
                chunks, dense_vecs, sparse_vecs, sections_map, vmetas
            ):
                total_chunks = sum(1 for c in chunks if c.section_node == chunk.section_node)
                sv = SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
                payload = _build_payload(section, chunk, vmeta, total_chunks)
                points.append(build_point(chunk.text, dense, sv, payload))

            upsert_batch(qdrant, settings.collection_name, points)
            stats["chunks_created"] += len(points)
        except Exception as exc:
            logger.error("Batch upsert failed: %s", exc)
            stats["errors"] += 1

    logger.info("Streaming XML from %s", xml_path)
    section_iter = parse_sections(xml_path)

    with tqdm(desc="Ingesting sections", unit="section") as pbar:
        for section in section_iter:
            try:
                version_meta = enrich_with_version(section.section_number, versions)
                chunks = chunk_section(
                    section,
                    max_tokens=settings.chunk_max_tokens,
                )

                for chunk in chunks:
                    pending_chunks.append(chunk)
                    pending_sections.append(section)
                    pending_version_metas.append(version_meta)

                if len(pending_chunks) >= ingest_batch_size:
                    flush_batch(pending_chunks, pending_sections, pending_version_metas)
                    pending_chunks.clear()
                    pending_sections.clear()
                    pending_version_metas.clear()

                stats["sections_processed"] += 1
                pbar.update(1)

            except Exception as exc:
                logger.error("Failed to process section %s: %s", section.section_number, exc)
                stats["errors"] += 1

    # Flush remaining
    flush_batch(pending_chunks, pending_sections, pending_version_metas)

    logger.info(
        "Ingestion complete: %d sections, %d chunks, %d errors",
        stats["sections_processed"],
        stats["chunks_created"],
        stats["errors"],
    )
    return stats
