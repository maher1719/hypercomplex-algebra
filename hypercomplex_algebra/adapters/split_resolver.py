"""Adapter: split basis products via hypercomplex-engine's O(1) engine."""
from hypercomplex_engine import FastSplit

from ..core.base.resolver import BasisProductResolver


class SplitResolver(BasisProductResolver):
    """Split Cayley-Dickson basis products, pinned to a dimension level.

    The split structure depends on which doubling is 'split', so this resolver
    is fixed to a specific dim. Mixing split dimensions is not allowed.
    """

    def __init__(self, dim: int):
        if dim < 0:
            raise ValueError(f"dim must be >= 0, got {dim}")
        self._dim = dim
        self._limit = 1 << dim
        self._engine = FastSplit()

    @property
    def dim(self) -> int:
        return self._dim

    def resolve(self, i: int, j: int) -> tuple[int, int]:
        if i < 0 or j < 0:
            raise ValueError(f"Basis indices must be >= 0, got i={i}, j={j}")
        if i >= self._limit or j >= self._limit:
            raise ValueError(
                f"Index out of range for split dim={self._dim} "
                f"(valid: 0..{self._limit - 1}). Split dimensions cannot be "
                f"mixed. Create a new SplitResolver with the dimension you need."
            )
        return self._engine.multiply_indices(i, j, self._dim)