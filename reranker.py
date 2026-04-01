from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SUPPORTED_RERANK_MODELS = {
    "bge-base": {
        "model_name": "BAAI/bge-reranker-base",
        "description": "BGE base reranker, default baseline for Chinese rerank experiments",
    },
    "bge-large": {
        "model_name": "BAAI/bge-reranker-large",
        "description": "Larger BGE reranker, usually stronger but slower",
    },
    "bce-base": {
        "model_name": "maidalun1020/bce-reranker-base_v1",
        "description": "BCE reranker, often competitive on Chinese retrieval tasks",
    },
}


@dataclass
class RerankCandidate:
    source: str | None
    chunk_id: int | None
    page_content: str
    original_score: float
    original_rank: int
    preview: str


class HuggingFaceReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_n: int | None = None,
    ) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, candidate.page_content) for candidate in candidates]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        if logits.ndim == 2 and logits.shape[1] > 1:
            scores = logits[:, 1]
        else:
            scores = logits.reshape(-1)

        reranked = []
        for candidate, score in zip(candidates, scores.tolist(), strict=True):
            reranked.append(
                {
                    "source": candidate.source,
                    "chunk_id": candidate.chunk_id,
                    "page_content": candidate.page_content,
                    "preview": candidate.preview,
                    "original_score": candidate.original_score,
                    "original_rank": candidate.original_rank,
                    "rerank_score": float(score),
                }
            )

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        for idx, item in enumerate(reranked, start=1):
            item["rank"] = idx

        if top_n is not None:
            return reranked[:top_n]
        return reranked

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "cpu").lower()
        if normalized == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return normalized


def list_supported_rerank_models() -> dict[str, dict[str, str]]:
    return SUPPORTED_RERANK_MODELS


def resolve_rerank_model_name(model_name_or_alias: str) -> str:
    if model_name_or_alias in SUPPORTED_RERANK_MODELS:
        return SUPPORTED_RERANK_MODELS[model_name_or_alias]["model_name"]
    return model_name_or_alias
