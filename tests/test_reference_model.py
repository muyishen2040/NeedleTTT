from pathlib import Path

import numpy as np
import pytest

from needlettt.contracts import ItemCatalog
from needlettt.models.reference_numpy import TeachingRecommender


@pytest.fixture
def catalog() -> ItemCatalog:
    path = Path(__file__).resolve().parents[1] / "data/examples/catalog.json"
    return ItemCatalog.from_json(path)


@pytest.mark.parametrize("variant", ["base", "vector", "matrix"])
def test_update_is_nonmutating_and_replay_matches(catalog, variant):
    model = TeachingRecommender(catalog, variant=variant)
    initial = model.init_state()
    first = model.update(initial, [0], [5])
    second = model.update(first, [1], [1])
    replayed = model.replay([[(0, 5), (1, 1)]])
    assert initial.histories == ((),)
    assert first.histories == (((0, 5),),)
    assert second.histories == replayed.histories
    if variant != "base":
        assert initial.fast is not None
        np.testing.assert_array_equal(initial.fast, np.zeros_like(initial.fast))
        np.testing.assert_allclose(second.fast, replayed.fast)
    np.testing.assert_allclose(
        model.recommend(second, [[2, 3]]),
        model.recommend(replayed, [[2, 3]]),
    )


@pytest.mark.parametrize("variant", ["base", "vector", "matrix"])
def test_independent_batched_user_histories(catalog, variant):
    model = TeachingRecommender(catalog, variant=variant)
    state = model.init_state(batch_size=2)
    changed = model.update(state, item_ids=[0, 1], ratings=[5, 1])
    assert changed.histories == (((0, 5),), ((1, 1),))
    np.testing.assert_allclose(
        model.recommend(changed, [[2, 3], [2, 3]]),
        model.recommend(model.replay([[(0, 5)], [(1, 1)]]), [[2, 3], [2, 3]]),
    )


@pytest.mark.parametrize("variant", ["base", "vector", "matrix"])
def test_predict_then_reveal_changes_only_later_scores(catalog, variant):
    model = TeachingRecommender(catalog, variant=variant)
    state = model.init_state()
    before = model.recommend(state, [[2, 3]])
    after_state = model.update(state, [0], [5])
    np.testing.assert_array_equal(before, model.recommend(state, [[2, 3]]))
    after = model.recommend(after_state, [[2, 3]])
    assert not np.allclose(before, after)
