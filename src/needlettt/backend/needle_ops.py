"""DEE-13 implementation landing zone.

These functions deliberately raise until a teammate wires real differentiable
Needle operations. The NumPy oracle is in reference_numpy.py.
Never silently fall back to NumPy during claimed Needle training.
"""

from typing import Any


def vector_step(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("DEE-13: implement vector fast-weight update with Needle autodiff")


def matrix_step(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("DEE-13: implement matrix fast-weight update with Needle autodiff")


def matrix_scan(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("DEE-13: implement differentiable padded matrix scan in Needle")
