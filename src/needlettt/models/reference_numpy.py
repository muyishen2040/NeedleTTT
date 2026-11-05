"""Deterministic, UNTRAINED teaching model for API and contract smoke tests.

This is NOT the GRU two-tower, learned fast memory or Needle implementation.
Scores have no experimental meaning. Embeddings are random with a fixed seed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from needlettt.backend.reference_numpy import matrix_step, vector_step
from needlettt.contracts import ItemCatalog, validate_rating
from needlettt.models.interfaces import ModelState, ModelVariant


class TeachingRecommender:
    model_status = "toy_untrained"

    def __init__(
        self,
        catalog: ItemCatalog,
        variant: ModelVariant = "base",
        dim: int = 8,
        eta: float = 0.5,
        seed: int = 17,
    ) -> None:
        if variant not in ("base", "vector", "matrix"):
            raise ValueError(f"Unknown variant: {variant}")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.catalog = catalog
        self.variant = variant
        self.dim = dim
        self.eta = eta
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(len(catalog.items), dim))
        self.embeddings = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
        self.keys = self.embeddings.copy()  # identity projection; NOT a learned P_key

    def init_state(self, batch_size: int = 1) -> ModelState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.variant == "base":
            fast = None
        elif self.variant == "vector":
            fast = np.zeros((batch_size, self.dim), dtype=np.float64)
        else:
            fast = np.zeros((batch_size, self.dim, self.dim), dtype=np.float64)
        return ModelState(histories=tuple(() for _ in range(batch_size)), fast=fast)

    def _base_embedding(self, history: tuple[tuple[int, int], ...]) -> np.ndarray:
        # Deliberately simple history average, NOT the final GRU user tower.
        if not history:
            return np.zeros(self.dim)
        return sum(
            (((rating - 3) / 2) * self.embeddings[item_idx] for item_idx, rating in history),
            np.zeros(self.dim),
        ) / len(history)

    def recommend(
        self, state: ModelState, candidate_ids: Sequence[Sequence[int]]
    ) -> np.ndarray:
        if len(candidate_ids) != state.batch_size:
            raise ValueError("candidate_ids must contain one list per user")
        if len({len(row) for row in candidate_ids}) != 1:
            raise ValueError("candidate_ids must have shape [B,C]")
        result: list[np.ndarray] = []
        for b, row in enumerate(candidate_ids):
            indices = np.asarray(row, dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= len(self.catalog.items)):
                raise ValueError("Unknown item index")
            e = self.embeddings[indices]
            scores = e @ self._base_embedding(state.histories[b])
            if self.variant == "vector":
                assert state.fast is not None
                scores = scores + e @ state.fast[b]
            elif self.variant == "matrix":
                assert state.fast is not None
                k = self.keys[indices]
                mem = (state.fast[b] @ k.T).T
                adjustment = np.einsum("nd,nd->n", e, mem) / (
                    np.einsum("nd,nd->n", e, e) + 1e-8
                )
                scores = scores + adjustment
            result.append(np.asarray(scores, dtype=np.float64))
        return np.stack(result, axis=0)

    def update(
        self, state: ModelState, item_ids: Sequence[int], ratings: Sequence[int]
    ) -> ModelState:
        if len(item_ids) != state.batch_size or len(ratings) != state.batch_size:
            raise ValueError("item_ids and ratings must both have shape [B]")
        fast = None if state.fast is None else state.fast.copy()
        histories = []
        for b, (item_idx, rating) in enumerate(zip(item_ids, ratings)):
            item_idx = int(item_idx)
            rating = int(rating)
            validate_rating(rating)
            if not 0 <= item_idx < len(self.catalog.items):
                raise ValueError("Unknown item index")
            if self.variant == "vector":
                assert fast is not None
                fast[b] = vector_step(
                    fast[b], self.embeddings[item_idx], float(rating - 3), eta=self.eta
                )
            elif self.variant == "matrix":
                assert fast is not None
                fast[b] = matrix_step(
                    fast[b], self.keys[item_idx],
                    ((rating - 3) / 2) * self.embeddings[item_idx],
                    eta=self.eta,
                )
            histories.append((*state.histories[b], (item_idx, rating)))
        return ModelState(histories=tuple(histories), fast=fast)

    def replay(self, histories: Sequence[Sequence[tuple[int, int]]]) -> ModelState:
        if not histories:
            raise ValueError("Need at least one user history")
        final_histories = []
        memory = []
        for history in histories:
            one = self.init_state(batch_size=1)
            for item_idx, rating in history:
                one = self.update(one, [item_idx], [rating])
            final_histories.append(one.histories[0])
            if one.fast is not None:
                memory.append(one.fast[0])
        return ModelState(
            histories=tuple(final_histories),
            fast=np.stack(memory, axis=0) if memory else None,
        )
