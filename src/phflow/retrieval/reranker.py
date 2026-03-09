"""
Cross-encoder reranking using BAAI/bge-reranker-base.

BGE reranker is a standard AutoModelForSequenceClassification cross-encoder.
It takes (query, passage) pairs and outputs a single relevance logit per pair.
sigmoid(logit) gives a calibrated confidence score in [0, 1].
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, local_files_only=True
            ).to(device)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(device)

        # Ensure a pad token exists — required for batch inference.
        # BERT-based models (bge-reranker-base) already have [PAD];
        # decoder models (Qwen etc.) use EOS as fallback.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.train(False)
        self.device = device
        self.max_length = max_length

    def rerank(
        self,
        query: str,
        candidates: list,  # list of ScoredPoint from Qdrant
        top_n: int = 5,
        score_threshold: float = 0.25,
    ) -> list:
        """
        Rerank candidate chunks by relevance to the query.

        Scores each (query, passage) pair with a cross-encoder and applies
        sigmoid to produce a confidence value in [0, 1].

        Returns top_n (score, candidate) tuples sorted by confidence descending,
        filtered to score >= score_threshold.
        """
        if not candidates:
            return []

        queries = [query] * len(candidates)
        passages = [c.payload.get("chunk_text", "") for c in candidates]

        with torch.no_grad():
            inputs = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Shape: [batch, 1] → sigmoid → confidence [0, 1]
            logits = self.model(**inputs).logits
            scores = torch.sigmoid(logits[:, 0]).tolist()

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [(score, c) for score, c in ranked[:top_n] if score >= score_threshold]
