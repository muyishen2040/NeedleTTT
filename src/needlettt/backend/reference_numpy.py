"""Small numerical oracle for the DEE-5 fast-weight updates.

NumPy is used here only for testing formulas and the local teaching demo.
Real training must implement these operations in Needle with autodiff.
"""

from __future__ import annotations

import numpy as np


def vector_step(
    w: np.ndarray, features: np.ndarray, centered_rating: float,
    eta: float = 0.5, eps: float = 1e-8,
) -> np.ndarray:
    """Normalized gradient step for scalar regression; inputs shape [D]."""
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if w.ndim != 1 or x.shape != w.shape:
        raise ValueError("w and features must both be shape [D]")
    error = float(w @ x) - float(centered_rating)
    return w - (eta * error / (float(x @ x) + eps)) * x


def matrix_step(
    W: np.ndarray, key: np.ndarray, target: np.ndarray,
    eta: float = 0.5, eps: float = 1e-8,
) -> np.ndarray:
    """Normalized MSE gradient step for a vector-valued response; shapes [d,d],[d],[d]."""
    W = np.asarray(W, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(target, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or k.shape != (W.shape[1],):
        raise ValueError("Expected W[d,d] and key[d]")
    if v.shape != (W.shape[0],):
        raise ValueError("Expected target[d]")
    error = W @ k - v
    return W - (eta / (float(k @ k) + eps)) * np.outer(error, k)


def matrix_scan(
    initial: np.ndarray, keys: np.ndarray, targets: np.ndarray,
    lengths: np.ndarray, eta: float = 0.5, eps: float = 1e-8,
) -> np.ndarray:
    """Replay padded batch in user order; shapes [B,d,d], [B,T,d], [B,T,d], [B]."""
    W = np.asarray(initial, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if W.ndim != 3 or W.shape[1] != W.shape[2]:
        raise ValueError("initial must be shape [B,d,d]")
    batch, dim, _ = W.shape
    if keys.ndim != 3 or keys.shape != targets.shape or keys.shape[0] != batch:
        raise ValueError("keys and targets must both be [B,T,d]")
    if keys.shape[-1] != dim or lengths.shape != (batch,):
        raise ValueError("Inconsistent dimensions or lengths")
    if np.any(lengths < 0) or np.any(lengths > keys.shape[1]):
        raise ValueError("lengths must fall within [0,T]")
    result = W.copy()
    for b in range(batch):
        for t in range(int(lengths[b])):
            result[b] = matrix_step(result[b], keys[b, t], targets[b, t], eta, eps)
    return result
