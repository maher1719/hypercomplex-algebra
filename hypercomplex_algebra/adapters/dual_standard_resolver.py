"""Adapter: dual basis products over a standard parent algebra."""
from hypercomplex_engine import FastStandard

from ..core.dual.resolver import DualResolver as DualResolverBase


class DualStandardResolver(DualResolverBase):
    """Dual Cayley-Dickson over a standard parent.

    Elements are (index, eps_flag) pairs. The product rules:
        e_i * e_j       = parent product (eps stays 0)
        e_i * (ε·e_j)   = ε · (parent product)
        (ε·e_i) * e_j   = ε · (parent product)
        (ε·e_i) * (ε·e_j) = 0  (nilpotent: ε² = 0)
    """

    def __init__(self):
        self._parent = FastStandard()

    def resolve(self, i: int, eps_i: int, j: int, eps_j: int) -> tuple[int, int, int]:
        """Resolve (index_a, eps_a) * (index_b, eps_b).

        Returns (sign, result_index, result_eps_flag).
        Returns (0, 0, 0) for nilpotent zero products.
        """
        if i < 0 or j < 0:
            raise ValueError(f"Indices must be >= 0, got i={i}, j={j}")

        # Nilpotency: ε² = 0
        if eps_i and eps_j:
            return (0, 0, 0)

        # Parent algebra product
        parent_sign, parent_idx = self._parent.multiply_indices(i, j)

        # Result eps flag: exactly one ε survives (XOR, but both can't be 1 here)
        result_eps = eps_i ^ eps_j

        return (parent_sign, parent_idx, result_eps)