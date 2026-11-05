import numpy as np
import pytest

from needlettt.backend.reference_numpy import matrix_scan, matrix_step, vector_step


def test_vector_step_moves_prediction_toward_positive_target():
    w = np.zeros(2)
    x = np.array([1.0, 0.0])
    out = vector_step(w, x, centered_rating=2.0, eta=0.5)
    np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(w, [0.0, 0.0])


def test_matrix_step_matches_hand_calculation_without_mutation():
    W = np.zeros((2, 2))
    k = np.array([1.0, 0.0])
    v = np.array([1.0, 0.0])
    out = matrix_step(W, k, v, eta=0.5)
    np.testing.assert_allclose(out, [[0.5, 0.0], [0.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(W, np.zeros_like(W))


def test_matrix_scan_honors_lengths_and_separate_users():
    W0 = np.zeros((2, 2, 2))
    keys = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    ])
    values = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[-1.0, 0.0], [999.0, 999.0]],
    ])
    got = matrix_scan(W0, keys, values, lengths=np.array([2, 1]), eta=0.5)
    np.testing.assert_allclose(got[0], [[0.5, 0], [0, 0.5]], atol=1e-7)
    np.testing.assert_allclose(got[1], [[-0.5, 0], [0, 0]], atol=1e-7)
    np.testing.assert_array_equal(W0, np.zeros_like(W0))


def test_zero_length_and_invalid_lengths():
    W0 = np.zeros((1, 2, 2))
    keys = np.zeros((1, 0, 2))
    np.testing.assert_array_equal(
        matrix_scan(W0, keys, keys, np.array([0])), W0
    )
    with pytest.raises(ValueError):
        matrix_scan(W0, keys, keys, np.array([1]))
