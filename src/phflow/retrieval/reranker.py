"""
Cross-encoder reranking using Qwen3-Reranker-0.6B.

Qwen3-Reranker uses a generative classification format:
the model outputs "yes"/"no" tokens whose logits are used as relevance scores.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RERANKER_INSTRUCTION = (
    "Given a regulatory query about FDA/CFR regulations, "
    "retrieve relevant sections that directly answer the query."
)


class Reranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        # Load with local_files_only=True when model is already cached.
        # This skips the HuggingFace hub check that writes a .no_exist sentinel
        # file on every startup, which triggers uvicorn's --reload loop.
        # Falls back to a full download on first run.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,
            ).to(device)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                trust_remote_code=True,
            ).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.train(False)  # inference mode: disables dropout
        self.device = device
        self.max_length = max_length

        # Token IDs for "yes" / "no" classification
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

    def _format_input(self, query: str, passage: str) -> str:
        return (
            f"<|im_start|>system\n{RERANKER_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {RERANKER_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {passage}"
            f"<|im_end|>"
        )

    def rerank(
        self,
        query: str,
        candidates: list,  # list of ScoredPoint from Qdrant
        top_n: int = 5,
        score_threshold: float = 0.5,
    ) -> list:
        """
        Rerank candidate chunks by relevance to the query.
        Returns top_n candidates sorted by reranker score (descending).
        """
        if not candidates:
            return []

        pairs = [
            self._format_input(query, c.payload.get("chunk_text", ""))
            for c in candidates
        ]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # CausalLM outputs [batch, seq_len, vocab_size]; score from last token
            last_token_logits = self.model(**inputs).logits[:, -1, :]
            # Softmax over [false_id, true_id] → relevance probability for "yes"
            scores = torch.nn.functional.softmax(
                last_token_logits[:, [self.token_false_id, self.token_true_id]], dim=-1
            )[:, 1].tolist()

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for score, c in ranked[:top_n] if score >= score_threshold]
