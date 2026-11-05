"""Variant-neutral inference boundary, based on Linear DEE-5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np


ModelVariant = Literal["base", "vector", "matrix"]


@dataclass(frozen=True)
class ModelState:
    """Separate ordered histories and optional fast memory for each batch member.

    The array is treated as immutable: methods return a new array instead of
    mutating the old one. The trained Needle implementation will use Needle tensors.
    """

    histories: tuple[tuple[tuple[int, int], ...], ...]
    fast: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return len(self.histories)


class Recommender(Protocol):
    variant: ModelVariant

    def init_state(self, batch_size: int = 1) -> ModelState: ...

    def recommend(
        self, state: ModelState, candidate_ids: Sequence[Sequence[int]]
    ) -> np.ndarray: ...

    def update(
        self, state: ModelState, item_ids: Sequence[int], ratings: Sequence[int]
    ) -> ModelState: ...

    def replay(
        self, histories: Sequence[Sequence[tuple[int, int]]]
    ) -> ModelState: ...
