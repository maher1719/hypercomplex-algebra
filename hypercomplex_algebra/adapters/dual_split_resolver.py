"""Adapter: dual basis products over a split parent algebra."""
from hypercomplex_engine import FastSplit

from ..core.dual.resolver import DualResolver


class DualSplitResolver(DualResolver):
    """Dual Cayley-Dickson over a split parent, pinned to a dimension level.

    The split parent's structure depends on which doubling is 'split', so this
    resolver is fixed to a specific dim. Mixing dual_split dimensions is not
    allowed. The nilpotency rule (ε² = 0) is identical to dual ordinary — only
    the base products differ (they follow split sign rules).
    """

    def __init__(self, dim: int):
        if dim < 0:
            raise ValueError(f"dim must be >= 0, got {dim}")
        self._dim = dim
        self._limit = 1 << dim
        self._parent = FastSplit()

    @property
    def dim(self) -> int:
        return self._dim

    def resolve(self, i: int, eps_i: int, j: int, eps_j: int) -> tuple[int, int, int]:
        """Resolve (index_a, eps_a) * (index_b, eps_b).

        Returns (sign, result_index, result_eps_flag).
        Returns (0, 0, 0) for nilpotent zero products.
        """
        if i < 0 or j < 0:
            raise ValueError(f"Indices must be >= 0, got i={i}, j={j}")
        if i >= self._limit or j >= self._limit:
            raise ValueError(
                f"Index out of range for dual_split dim={self._dim} "
                f"(valid: 0..{self._limit - 1}). Dual_split dimensions cannot "
                f"be mixed. Create a new DualSplitResolver with the dimension you need."
            )

        # Nilpotency: ε² = 0 (same as dual ordinary)
        if eps_i and eps_j:
            return (0, 0, 0)

        # Parent algebra product — SPLIT sign rules (differs from standard)
        parent_sign, parent_idx = self._parent.multiply_indices(i, j, self._dim)

        # Result eps flag: exactly one ε survives
        result_eps = eps_i ^ eps_j

        return (parent_sign, parent_idx, result_eps)