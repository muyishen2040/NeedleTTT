import numpy as np
import pytest

from needlettt.backend.reference_numpy import (
    matrix_step,
    matrix_step_batched,
    vector_step,
    vector_step_batched,
)


def test_batched_updates_match_individual_reference_without_cross_user_leaks():
    w = np.zeros((2, 3))
    x = np.array([[1., 0., 0.], [0., 0., 2.]])
    y = np.array([2., -2.])
    vec = vector_step_batched(w, x, y)
    expected_vec = np.stack([vector_step(w[i], x[i], y[i]) for i in range(2)])
    np.testing.assert_allclose(vec, expected_vec)
    np.testing.assert_allclose(w, 0)

    W = np.zeros((2, 3, 3))
    key = x.copy()
    target = np.array([[1., 0., 0.], [0., 0., -1.]])
    mat = matrix_step_batched(W, key, target)
    expected_mat = np.stack([matrix_step(W[i], key[i], target[i]) for i in range(2)])
    np.testing.assert_allclose(mat, expected_mat)
    np.testing.assert_allclose(W, 0)
    assert not np.allclose(mat[0], mat[1])


@pytest.mark.parametrize(
    "w,x,y",
    [
        (np.zeros((2, 3)), np.zeros((1, 3)), np.zeros(2)),
        (np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(3)),
    ],
)
def test_vector_step_batched_rejects_wrong_shapes(w, x, y):
    with pytest.raises(ValueError):
        vector_step_batched(w, x, y)


def test_matrix_step_batched_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        matrix_step_batched(np.zeros((2, 3, 3)), np.zeros((2, 4)), np.zeros((2, 3)))
