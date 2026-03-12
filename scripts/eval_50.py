#!/usr/bin/env python3
"""
50-question end-to-end evaluation of the RAG pipeline.

Runs each query through the full chain (scope gate → rewrite → decomposition →
retrieval → generation → grounding) and collects metrics:

  - answer_rate:      % of queries that got a substantive answer (not INSUFFICIENT)
  - grounding_pass:   % of answers that passed the grounding check
  - avg_sources:      average number of declared sources per answer
  - avg_latency:      average wall-clock seconds per query
  - scope_reject:     % of in-scope queries incorrectly rejected as out-of-scope

Usage:
    python scripts/eval_50.py               # full run, results to stdout + JSON
    python scripts/eval_50.py --quick 10    # run only first N questions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phflow.settings import settings
from phflow.embeddings.encoder import RegulationEncoder
from phflow.vectorstore.client import get_faiss_store
from phflow.generation.llm import LlamaCppClient
from phflow.retrieval.hybrid import HybridRetriever
from phflow.generation.chain import RAGChain
from phflow.generation.prompts import INSUFFICIENT_MESSAGE, OFF_TOPIC_MESSAGE, VAGUE_QUERY_MESSAGE

# ── 50 evaluation questions ──────────────────────────────────────────────
# Mix of:
#   - formal regulatory language (easy for retrieval)
#   - operational/warehouse language (tests rewrite)
#   - multi-faceted questions (tests decomposition)
#   - edge cases (tests scope gate + guards)

EVAL_QUESTIONS = [
    # --- Operational language (dock-to-stock) ---
    {"id": "e01", "query": "A pallet arrived with a broken seal. What do I do?",
     "expect_answer": True, "category": "receiving"},
    {"id": "e02", "query": "The temperature logger on a shipment shows it went above 25°C for 3 hours. Can I accept this?",
     "expect_answer": True, "category": "temperature"},
    {"id": "e03", "query": "I found a box of raw material left outside the cold room overnight. What should I do?",
     "expect_answer": True, "category": "temperature"},
    {"id": "e04", "query": "Can I relabel a container if the original label is damaged?",
     "expect_answer": True, "category": "labeling"},
    {"id": "e05", "query": "What do I do if two different lots got mixed together on the same pallet?",
     "expect_answer": True, "category": "segregation"},
    {"id": "e06", "query": "A material failed its identity test. Can we still use it?",
     "expect_answer": True, "category": "testing"},
    {"id": "e07", "query": "Who decides if a rejected material can be reprocessed?",
     "expect_answer": True, "category": "quality_unit"},
    {"id": "e08", "query": "How long do we need to keep batch records?",
     "expect_answer": True, "category": "records"},
    {"id": "e09", "query": "The power went out in our warehouse for 4 hours. What are the requirements?",
     "expect_answer": True, "category": "temperature"},
    {"id": "e10", "query": "Can I store approved and quarantined materials in the same area?",
     "expect_answer": True, "category": "storage"},

    # --- Formal regulatory language ---
    {"id": "e11", "query": "What are the requirements for the quality control unit regarding component acceptance?",
     "expect_answer": True, "category": "quality_unit"},
    {"id": "e12", "query": "What testing is required before a component can be released for manufacturing?",
     "expect_answer": True, "category": "testing"},
    {"id": "e13", "query": "What are the storage conditions required under 21 CFR 211.80?",
     "expect_answer": True, "category": "storage"},
    {"id": "e14", "query": "What documentation is required when receiving drug product containers and closures?",
     "expect_answer": True, "category": "receiving"},
    {"id": "e15", "query": "What are the requirements for equipment cleaning and maintenance?",
     "expect_answer": True, "category": "equipment"},
    {"id": "e16", "query": "What are the written procedures required for production and process control?",
     "expect_answer": True, "category": "production"},
    {"id": "e17", "query": "What are the container and closure requirements under GMP?",
     "expect_answer": True, "category": "containers"},
    {"id": "e18", "query": "What records must be kept for distribution of drug products?",
     "expect_answer": True, "category": "distribution"},
    {"id": "e19", "query": "What are the responsibilities of the quality control unit?",
     "expect_answer": True, "category": "quality_unit"},
    {"id": "e20", "query": "What are the personnel qualification and training requirements?",
     "expect_answer": True, "category": "personnel"},

    # --- Multi-faceted questions (test decomposition) ---
    {"id": "e21", "query": "What are both the temperature monitoring and quarantine requirements for received components?",
     "expect_answer": True, "category": "multi"},
    {"id": "e22", "query": "What identity testing and storage segregation is required for incoming materials?",
     "expect_answer": True, "category": "multi"},
    {"id": "e23", "query": "How do equipment maintenance records and batch production records relate?",
     "expect_answer": True, "category": "multi"},
    {"id": "e24", "query": "What are the sampling procedures and container integrity requirements for received shipments?",
     "expect_answer": True, "category": "multi"},
    {"id": "e25", "query": "What labeling and quarantine requirements apply to rejected materials?",
     "expect_answer": True, "category": "multi"},

    # --- Scenario-based (warehouse floor) ---
    {"id": "e26", "query": "A delivery truck arrived 2 hours late and the driver says the refrigeration unit stopped working. What are the regulatory requirements?",
     "expect_answer": True, "category": "scenario"},
    {"id": "e27", "query": "I'm doing inventory and found material with no label on it. What do I do?",
     "expect_answer": True, "category": "scenario"},
    {"id": "e28", "query": "A vendor shipped us the wrong material but the paperwork says it's correct. How do I handle this?",
     "expect_answer": True, "category": "scenario"},
    {"id": "e29", "query": "We need to move quarantined material to a different building. What requirements apply?",
     "expect_answer": True, "category": "scenario"},
    {"id": "e30", "query": "An approved supplier just got a warning letter from FDA. Can we still accept their shipments?",
     "expect_answer": True, "category": "scenario"},

    # --- Specific CFR sections ---
    {"id": "e31", "query": "What does 21 CFR 211.82 require for receipt and storage of components?",
     "expect_answer": True, "category": "specific"},
    {"id": "e32", "query": "What does Part 211.42 say about building design and features?",
     "expect_answer": True, "category": "specific"},
    {"id": "e33", "query": "What are the requirements under 211.142 for warehousing procedures?",
     "expect_answer": True, "category": "specific"},
    {"id": "e34", "query": "What does 211.68 cover regarding automatic equipment?",
     "expect_answer": True, "category": "specific"},
    {"id": "e35", "query": "What are the reprocessing requirements under 211.115?",
     "expect_answer": True, "category": "specific"},

    # --- Edge cases (should still get answers) ---
    {"id": "e36", "query": "quarantine",
     "expect_answer": "vague", "category": "edge"},
    {"id": "e37", "query": "What is temperature control?",
     "expect_answer": "vague", "category": "edge"},
    {"id": "e38", "query": "tell me about receiving procedures",
     "expect_answer": True, "category": "edge"},
    {"id": "e39", "query": "sampling",
     "expect_answer": "vague", "category": "edge"},
    {"id": "e40", "query": "How should I handle a deviation from storage conditions?",
     "expect_answer": True, "category": "edge"},

    # --- Out of scope (should be rejected) ---
    {"id": "e41", "query": "What is the weather forecast for tomorrow?",
     "expect_answer": False, "category": "out_of_scope"},
    {"id": "e42", "query": "How do I cook pasta?",
     "expect_answer": False, "category": "out_of_scope"},
    {"id": "e43", "query": "What is the capital of France?",
     "expect_answer": False, "category": "out_of_scope"},
    {"id": "e44", "query": "Tell me about clinical trial Phase 3 requirements",
     "expect_answer": False, "category": "out_of_scope"},
    {"id": "e45", "query": "What medications should I take for a headache?",
     "expect_answer": False, "category": "out_of_scope"},

    # --- More operational ---
    {"id": "e46", "query": "What cleaning records do I need to keep for warehouse equipment?",
     "expect_answer": True, "category": "equipment"},
    {"id": "e47", "query": "Can the quality team delegate approval of incoming materials?",
     "expect_answer": True, "category": "quality_unit"},
    {"id": "e48", "query": "What's required when we reject a batch of components?",
     "expect_answer": True, "category": "receiving"},
    {"id": "e49", "query": "How often do we need to calibrate temperature monitoring equipment?",
     "expect_answer": True, "category": "equipment"},
    {"id": "e50", "query": "What documentation is needed for shipping finished drug products?",
     "expect_answer": True, "category": "distribution"},
]


async def run_eval(chain: RAGChain, llm: LlamaCppClient, questions: list[dict]) -> dict:
    """Run all questions and collect results."""
    results = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["query"]
        expect_answer = q["expect_answer"]
        category = q["category"]

        print(f"\n[{i}/{total}] {qid}: {query[:70]}...", flush=True)

        t0 = time.time()
        try:
            result = await chain.run(
                query=query,
                mode="qa",
                conversation_history=[],
                expand_to_parent=True,
                stream=False,
            )
            elapsed = time.time() - t0

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            steps = result.get("pipeline_steps", [])

            is_insufficient = answer == INSUFFICIENT_MESSAGE
            is_off_topic = answer == OFF_TOPIC_MESSAGE
            is_vague = answer == VAGUE_QUERY_MESSAGE
            got_answer = not is_insufficient and not is_off_topic and not is_vague

            # Determine correctness
            if expect_answer == "vague":
                correct = is_vague  # should have asked for clarification
            elif expect_answer:
                correct = got_answer  # should have answered
            else:
                correct = is_off_topic  # should have been rejected

            status = "PASS" if correct else "FAIL"
            answer_preview = answer[:120].replace("\n", " ")
            print(f"  {status} ({elapsed:.1f}s) | sources: {len(sources)} | {answer_preview}")

            results.append({
                "id": qid,
                "query": query,
                "category": category,
                "expect_answer": expect_answer,
                "got_answer": got_answer,
                "is_off_topic": is_off_topic,
                "is_insufficient": is_insufficient,
                "is_vague": is_vague,
                "correct": correct,
                "num_sources": len(sources),
                "source_sections": [s.get("section_number", "") for s in sources],
                "answer_length": len(answer),
                "answer_preview": answer[:200],
                "pipeline_steps": steps,
                "latency_s": round(elapsed, 2),
            })

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR ({elapsed:.1f}s): {exc}")
            results.append({
                "id": qid,
                "query": query,
                "category": category,
                "expect_answer": expect_answer,
                "got_answer": False,
                "is_off_topic": False,
                "is_insufficient": False,
                "correct": False,
                "error": str(exc),
                "latency_s": round(elapsed, 2),
            })

    return results


def print_report(results: list[dict]) -> None:
    """Print a summary report."""
    total = len(results)
    errors = [r for r in results if "error" in r]

    # Split by expected behavior
    in_scope = [r for r in results if r["expect_answer"] is True and "error" not in r]
    out_scope = [r for r in results if r["expect_answer"] is False and "error" not in r]
    expect_vague = [r for r in results if r["expect_answer"] == "vague" and "error" not in r]

    # In-scope metrics
    answered = [r for r in in_scope if r["got_answer"]]
    insufficient = [r for r in in_scope if r["is_insufficient"]]
    wrongly_rejected = [r for r in in_scope if r["is_off_topic"]]

    # Out-of-scope metrics
    correctly_rejected = [r for r in out_scope if r["is_off_topic"]]
    wrongly_answered = [r for r in out_scope if r["got_answer"]]

    # Latency
    latencies = [r["latency_s"] for r in results if "error" not in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    in_scope_latencies = [r["latency_s"] for r in in_scope]
    avg_in_scope_latency = sum(in_scope_latencies) / len(in_scope_latencies) if in_scope_latencies else 0

    # Sources
    source_counts = [r["num_sources"] for r in answered]
    avg_sources = sum(source_counts) / len(source_counts) if source_counts else 0

    # Category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "answered": 0}
        categories[cat]["total"] += 1
        if r.get("correct"):
            categories[cat]["correct"] += 1
        if r.get("got_answer"):
            categories[cat]["answered"] += 1

    print("\n" + "=" * 70)
    print("RAG PIPELINE EVALUATION REPORT")
    print("=" * 70)

    print(f"\nTotal questions:     {total}")
    print(f"Errors:              {len(errors)}")

    print(f"\n--- IN-SCOPE QUESTIONS ({len(in_scope)}) ---")
    print(f"Answered:            {len(answered)}/{len(in_scope)} ({100*len(answered)/len(in_scope):.0f}%)")
    print(f"Insufficient:        {len(insufficient)}/{len(in_scope)} ({100*len(insufficient)/len(in_scope):.0f}%)")
    print(f"Wrongly rejected:    {len(wrongly_rejected)}/{len(in_scope)} ({100*len(wrongly_rejected)/len(in_scope):.0f}%)")

    print(f"\n--- OUT-OF-SCOPE QUESTIONS ({len(out_scope)}) ---")
    if out_scope:
        print(f"Correctly rejected:  {len(correctly_rejected)}/{len(out_scope)} ({100*len(correctly_rejected)/len(out_scope):.0f}%)")
        print(f"Wrongly answered:    {len(wrongly_answered)}/{len(out_scope)} ({100*len(wrongly_answered)/len(out_scope):.0f}%)")
    else:
        print("  (no out-of-scope questions in this run)")

    if expect_vague:
        correctly_vague = [r for r in expect_vague if r.get("is_vague")]
        print(f"\n--- VAGUE QUERIES ({len(expect_vague)}) ---")
        print(f"Correctly prompted:  {len(correctly_vague)}/{len(expect_vague)} ({100*len(correctly_vague)/len(expect_vague):.0f}%)")

    print(f"\n--- QUALITY ---")
    print(f"Avg sources/answer:  {avg_sources:.1f}")
    print(f"Avg latency:         {avg_latency:.1f}s")
    print(f"Avg in-scope latency:{avg_in_scope_latency:.1f}s")

    print(f"\n--- CATEGORY BREAKDOWN ---")
    for cat in sorted(categories.keys()):
        c = categories[cat]
        pct = 100 * c["correct"] / c["total"] if c["total"] else 0
        print(f"  {cat:20s}  {c['correct']}/{c['total']} correct ({pct:.0f}%)")

    # List failures
    failures = [r for r in results if not r.get("correct")]
    if failures:
        print(f"\n--- FAILURES ({len(failures)}) ---")
        for r in failures:
            reason = "error" if "error" in r else (
                "wrongly rejected" if r["is_off_topic"] and r["expect_answer"] else
                "insufficient" if r["is_insufficient"] and r["expect_answer"] else
                "answered out-of-scope" if r["got_answer"] and not r["expect_answer"] else
                "unknown"
            )
            print(f"  {r['id']}: [{reason}] {r['query'][:60]}")

    overall_correct = sum(1 for r in results if r.get("correct"))
    print(f"\n{'=' * 70}")
    print(f"OVERALL ACCURACY: {overall_correct}/{total} ({100*overall_correct/total:.0f}%)")
    print(f"{'=' * 70}")


async def main():
    parser = argparse.ArgumentParser(description="RAG 50-question evaluation")
    parser.add_argument("--quick", type=int, default=0, help="Run only first N questions")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    questions = EVAL_QUESTIONS
    if args.quick:
        questions = questions[:args.quick]

    print(f"Initializing pipeline...", flush=True)

    encoder = RegulationEncoder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    store = get_faiss_store(dim=encoder.embedding_dim)
    llm = LlamaCppClient()
    retriever = HybridRetriever(store=store, encoder=encoder)
    chain = RAGChain(retriever=retriever, llm=llm)

    # Health check
    healthy = await llm.health()
    if not healthy:
        print(f"ERROR: LLM server not reachable at {settings.llama_cpp_base_url}")
        print("Start it with: ollama serve  (or llama-server)")
        sys.exit(1)
    print(f"LLM server OK ({settings.llm_backend} at {settings.llama_cpp_base_url})")
    print(f"FAISS index: {store.total_points} vectors")
    print(f"Running {len(questions)} questions...\n")

    results = await run_eval(chain, llm, questions)
    print_report(results)

    # Save detailed results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to: {output_path}")

    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
