"""Portable correctness model for the fused TT-Metal retrieval reduction.

This is shared specification code, not a hardware backend.  It fixes the work
partition and winner semantics before those rules are translated into kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EntryRange:
    begin: int
    end: int


@dataclass(frozen=True)
class Candidate:
    score: float
    id: int
    valid: bool = True


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
