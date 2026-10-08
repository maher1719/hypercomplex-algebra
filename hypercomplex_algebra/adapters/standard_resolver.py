"""Adapter: resolves basis products via hypercomplex-engine's O(1) engine.

This is the ONLY module that imports hypercomplex.
"""
from hypercomplex_engine import FastStandard

from ..core.base.resolver import BasisProductResolver


class StandardResolver(BasisProductResolver):
    """Standard Cayley-Dickson basis products. Dimension-independent."""

    def __init__(self):
        self._engine = FastStandard()

    def resolve(self, i: int, j: int) -> tuple[int, int]:
        if i < 0 or j < 0:
            raise ValueError(f"Basis indices must be >= 0, got i={i}, j={j}")
        # Coefficients carry input signs -> use the sign-free index API.
        return self._engine.multiply_indices(i, j)