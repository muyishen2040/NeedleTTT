"""Checkpoint-specific catalog and feedback data contracts.

External IDs are always strings. Internal indices are contiguous integers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    item_idx: int
    title: str
    categories: tuple[str, ...]
    image_url: str | None
    source: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "CatalogItem":
        return cls(
            item_id=str(row["item_id"]),
            item_idx=int(row["item_idx"]),
            title=str(row["title"]),
            categories=tuple(str(c) for c in row.get("categories", [])),
            image_url=str(row["image_url"]) if row.get("image_url") else None,
            source=str(row.get("source", "unknown")),
        )

    def payload(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "item_idx": self.item_idx,
            "title": self.title,
            "categories": list(self.categories),
            "image_url": self.image_url,
            "source": self.source,
        }


class ItemCatalog:
    """A stable item_id ↔ item_idx mapping shared by model, evaluator and API."""

    def __init__(self, items: Iterable[CatalogItem]):
        ordered = sorted(items, key=lambda item: item.item_idx)
        if not ordered:
            raise ValueError("Catalog cannot be empty")
        if [item.item_idx for item in ordered] != list(range(len(ordered))):
            raise ValueError("item_idx must be unique and contiguous starting at zero")
        if len({item.item_id for item in ordered}) != len(ordered):
            raise ValueError("Duplicate external item_id")
        self.items: tuple[CatalogItem, ...] = tuple(ordered)
        self.by_id = {item.item_id: item for item in self.items}

    @classmethod
    def from_json(cls, path: str | Path) -> "ItemCatalog":
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Expected a JSON list of catalog rows")
        return cls(CatalogItem.from_mapping(row) for row in rows)

    def by_index(self, idx: int) -> CatalogItem:
        return self.items[idx]

    def get(self, item_id: str) -> CatalogItem:
        return self.by_id[item_id]

    def search(self, query: str = "", limit: int = 50) -> list[dict[str, object]]:
        q = query.casefold().strip()
        return [
            item.payload()
            for item in self.items
            if not q or q in item.title.casefold() or any(q in x.casefold() for x in item.categories)
        ][:limit]


def validate_rating(rating: int) -> None:
    if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 5:
        raise ValueError("rating must be an integer between 1 and 5")
