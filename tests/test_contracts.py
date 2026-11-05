from pathlib import Path

import pytest

from needlettt.contracts import CatalogItem, ItemCatalog, validate_rating


def test_catalog_external_ids_and_search():
    root = Path(__file__).resolve().parents[1]
    catalog = ItemCatalog.from_json(root / "data/examples/catalog.json")
    assert catalog.get("demo-astro").item_idx == 0
    assert catalog.by_index(0).item_id == "demo-astro"
    assert len(catalog.search("Racing")) == 2


def test_duplicate_or_missing_contiguous_indices_are_rejected():
    def item(idx: int, external: str) -> CatalogItem:
        return CatalogItem(external, idx, "Example", (), None, "test")

    with pytest.raises(ValueError):
        ItemCatalog([item(0, "a"), item(0, "b")])
    with pytest.raises(ValueError):
        ItemCatalog([item(0, "a"), item(2, "b")])
    with pytest.raises(ValueError):
        ItemCatalog([item(0, "a"), item(1, "a")])


@pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
def test_valid_ratings(rating: int):
    validate_rating(rating)


@pytest.mark.parametrize("rating", [0, 6, True, 2.5])
def test_invalid_ratings(rating):
    with pytest.raises(ValueError):
        validate_rating(rating)
