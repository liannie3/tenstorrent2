"""Single-process subsystem contracts. Python 3.10; no ML imports needed."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import math


@dataclass(frozen=True)
class VisualOutput:
    # Current: [1, 576 or 577, 2048]. Retrieved: [1, 101, 2048].
    projected_tokens: Any
    # Optional normalized CLIP vector used only for the clip RAG query mode.
    retrieval_embedding: Any = None


@dataclass(frozen=True)
class RagHit:
    entry_id: str
    similarity: float
    description: str
    image_path: str

    def validate(self) -> None:
        if not math.isfinite(self.similarity) or not -1 <= self.similarity <= 1:
            raise ValueError('RAG similarity must be a finite cosine score in [-1, 1].')
        if not all(isinstance(x, str) and x.strip()
                   for x in (self.entry_id, self.description, self.image_path)):
            raise ValueError('RAG hit requires entry_id, description, and image_path.')


class YoloStage(Protocol):
    def hybrid_prompt(self, image_path: str, question: str) -> str:
        """Return question plus detected class names, NOT ground-truth claims."""
        ...


class VisionStage(Protocol):
    alignment_id: str
    def encode(self, image_path: str, *, role: str) -> VisualOutput:
        """role is current or retrieved; use the trained feature selection policy."""
        ...


class RagStage(Protocol):
    query_mode: str
    embedding_id: str
    def retrieve(self, hybrid_prompt: str, query_embedding: Any) -> RagHit | None:
        """Return top-1 cosine match, or None; load the bank once in __init__."""
        ...


@dataclass(frozen=True)
class StageAnswer:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str


@dataclass(frozen=True)
class PipelineAnswer:
    answer: StageAnswer
    hybrid_prompt: str
    retrieved_entry_id: str | None
