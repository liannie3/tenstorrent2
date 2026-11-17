"""Wrap teammates' real functions in the common pipeline interface.

These wrappers call supplied functions; they do not implement or mock other models.
For a C++ RAG backend, pass its existing Python binding as retrieve_fn.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class YoloAdapter:
    hybrid_prompt_fn: Callable

    def hybrid_prompt(self, image_path, question):
        return self.hybrid_prompt_fn(image_path, question)


@dataclass
class VisionAdapter:
    encode_fn: Callable
    alignment_id: str

    def encode(self, image_path, *, role):
        return self.encode_fn(image_path, role=role)


@dataclass
class RagAdapter:
    retrieve_fn: Callable
    query_mode: str
    embedding_id: str

    def retrieve(self, hybrid_prompt, query_embedding):
        return self.retrieve_fn(hybrid_prompt, query_embedding)
