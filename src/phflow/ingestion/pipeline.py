"""
Full ingestion pipeline orchestrator.

Execution order:
  1. Load versions-title-21.json into memory dict
  2. Stream-parse ECFR XML section by section
  3. Enrich each section with version metadata
  4. Chunk each section into paragraph-group chunks
  5. Batch-encode dense vectors (sentence-transformers)
  6. Batch-encode sparse vectors (fastembed BM25)
  7. Add batches to FAISS index + metadata store
  8. Save FAISS index to disk
"""

from __future__ import annotations

import logging

from pathlib import Path

from fastembed import SparseTextEmbedding
from tqdm import tqdm

from phflow.settings import settings
from phflow.embeddings.encoder import RegulationEncoder
from phflow.ingestion.chunker import Chunk, build_parent_text, chunk_section, count_tokens
from phflow.ingestion.metadata import enrich_with_version, load_versions
from phflow.ingestion.parser import SectionData, parse_sections
from phflow.vectorstore.client import get_faiss_store
from phflow.vectorstore.schema import create_index, index_exists

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
        reset: If True, drop and recreate the FAISS index first.
        ingest_batch_size: Number of chunks per batch.

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
    store = get_faiss_store(dim=encoder.embedding_dim)

    # Index setup
    if reset and index_exists(store):
        logger.info("Dropping existing FAISS index at '%s'", settings.faiss_index_dir)
        create_index(store)

    if not index_exists(store):
        logger.info("Creating FAISS index (dim=%d) at '%s'", encoder.embedding_dim, settings.faiss_index_dir)
        create_index(store)

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

            payloads = []
            sparse_data_list = []
            for chunk, sparse, section, vmeta in zip(
                chunks, sparse_vecs, sections_map, vmetas
            ):
                total_chunks = sum(1 for c in chunks if c.section_node == chunk.section_node)
                payload = _build_payload(section, chunk, vmeta, total_chunks)
                payload["chunk_text"] = chunk.text
                payloads.append(payload)
                sparse_data_list.append({
                    "indices": sparse.indices.tolist(),
                    "values": sparse.values.tolist(),
                })

            store.add(
                dense_vectors=dense_vecs,
                payloads=payloads,
                sparse_data=sparse_data_list,
            )
            stats["chunks_created"] += len(payloads)
        except Exception as exc:
            logger.error("Batch add failed: %s", exc)
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

    # Persist to disk
    store.save()

    logger.info(
        "Ingestion complete: %d sections, %d chunks, %d errors",
        stats["sections_processed"],
        stats["chunks_created"],
        stats["errors"],
    )
    return stats
