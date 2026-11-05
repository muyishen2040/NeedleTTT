"""DEE-13: differentiable TTT operators must use Needle tensors.

The public interface is BATCHED throughout. These functions are intentionally
unimplemented until the real course framework is available locally.
The separate NumPy oracle is for correctness, never for claimed Needle training.
"""

from typing import Any


def vector_step(
    w: Any, features: Any, centered_ratings: Any, eta: Any = 0.5, eps: float = 1e-8
) -> Any:
    """Needle: w/features [B,D], centered_ratings [B] -> updated w [B,D]."""
    raise NotImplementedError("DEE-13: implement differentiable batched vector_step in Needle")


def matrix_step(
    W: Any, keys: Any, targets: Any, eta: Any = 0.5, eps: float = 1e-8
) -> Any:
    """Needle: W [B,d,d], keys/targets [B,d] -> updated W [B,d,d]."""
    raise NotImplementedError("DEE-13: implement differentiable batched matrix_step in Needle")


def matrix_scan(
    W0: Any, keys: Any, targets: Any, lengths: Any, eta: Any = 0.5, eps: float = 1e-8
) -> Any:
    """Needle: W0 [B,d,d], keys/targets [B,T,d], lengths [B] -> [B,d,d]."""
    raise NotImplementedError("DEE-13: implement differentiable padded matrix_scan in Needle")
