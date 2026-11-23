"""Portable correctness model for the fused TT-Metal retrieval reduction.

This is shared specification code, not a hardware backend.  It fixes the work
partition and winner semantics before those rules are translated into kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EntryRange:
    begin: int
    end: int


@dataclass(frozen=True)
class Candidate:
    score: float
    id: int
    valid: bool = True


@dataclass(frozen=True)
class RetrievalResult:
    accepted: bool
    id: int
    score: float
    features: tuple[float, ...] | None = None


def partition_entries(valid_entries: int, workers: int) -> list[EntryRange]:
    if valid_entries <= 0 or workers <= 0:
        raise ValueError("valid_entries and workers must be positive")
    active = min(valid_entries, workers)
    base, extra = divmod(valid_entries, active)
    result: list[EntryRange] = []
    begin = 0
    for worker in range(active):
        size = base + (1 if worker < extra else 0)
        result.append(EntryRange(begin, begin + size))
        begin += size
    return result


def better(left: Candidate, right: Candidate) -> Candidate:
    """Return the deterministic winner; invalid/padded candidates never win."""
    if not left.valid:
        return right
    if not right.valid:
        return left
    if not math.isfinite(left.score) or not math.isfinite(right.score):
        raise ValueError("candidate scores must be finite")
    if right.score > left.score or (
        right.score == left.score and right.id < left.id
    ):
        return right
    return left


def reduce_candidates(candidates: list[Candidate], threshold: float) -> Candidate:
    if not candidates or not math.isfinite(threshold):
        raise ValueError("candidates must be non-empty and threshold finite")
    winner = Candidate(float("-inf"), 0, False)
    for candidate in candidates:
        winner = better(winner, candidate)
    if not winner.valid:
        raise ValueError("at least one candidate must be valid")
    # Equality is accepted by the retrieval contract.
    return Candidate(winner.score, winner.id, winner.score >= threshold)


def normalize_query(query: Sequence[float]) -> list[float]:
    if not query or not all(math.isfinite(value) for value in query):
        raise ValueError("query must contain finite values")
    squared_norm = math.fsum(value * value for value in query)
    if not math.isfinite(squared_norm) or squared_norm <= 0.0:
        raise ValueError("query norm must be finite and positive")
    inverse_norm = 1.0 / math.sqrt(squared_norm)
    return [value * inverse_norm for value in query]


def fused_search(
    query: Sequence[float],
    normalized_keys: Sequence[Sequence[float]],
    threshold: float,
    workers: int,
    features: Sequence[Sequence[float]] | None = None,
) -> RetrievalResult:
    """Executable oracle for the fused kernels and device-side feature gather."""
    if not normalized_keys:
        raise ValueError("bank must be non-empty")
    dimension = len(query)
    if dimension == 0 or any(len(key) != dimension for key in normalized_keys):
        raise ValueError("query and bank dimensions must agree")
    if features is not None and len(features) != len(normalized_keys):
        raise ValueError("feature table must have one row per bank entry")

    normalized_query = normalize_query(query)
    local_winners: list[Candidate] = []
    for entry_range in partition_entries(len(normalized_keys), workers):
        local = Candidate(float("-inf"), 0, False)
        for entry_id in range(entry_range.begin, entry_range.end):
            key = normalized_keys[entry_id]
            if not all(math.isfinite(value) for value in key):
                raise ValueError("bank keys must contain finite values")
            score = math.fsum(
                left * right for left, right in zip(normalized_query, key)
            )
            local = better(local, Candidate(score, entry_id))
        local_winners.append(local)

    winner = reduce_candidates(local_winners, threshold)
    selected = None
    if winner.valid and features is not None:
        selected = tuple(features[winner.id])
    return RetrievalResult(winner.valid, winner.id, winner.score, selected)
