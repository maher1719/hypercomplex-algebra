"""Sparse dual multiplication. Keys are (index, eps_flag) tuples."""

import math
from ...helpers.numerics import add_finite
from .resolver import DualResolver


class DualSparseMultiplier:
    """Multiplies two sparse dual elements.

    Elements are dicts: {(index, eps_flag): coefficient}
    Example: {(2, 0): 3.0, (1, 1): -2.5}  means  3·e₂ - 2.5·ε·e₁
    """

    def __init__(self, resolver: DualResolver, tolerance: float = 0.0):
        self._resolver = resolver
        self._tolerance = tolerance

    def multiply(self, a: dict[tuple[int, int], float],
                 b: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for (i, eps_i), ci in a.items():
            if self._tolerance > 0.0 and abs(ci) <= self._tolerance:
                continue
            for (j, eps_j), cj in b.items():
                if self._tolerance > 0.0 and abs(cj) <= self._tolerance:
                    continue
                sign, idx, eps = self._resolver.resolve(i, eps_i, j, eps_j)
                if sign == 0:  # nilpotent zero, skip
                    continue
                key = (idx, eps)
                contribution = sign * ci * cj
                if not math.isfinite(contribution):
                    raise OverflowError("Non-finite coefficient produced by multiplication.")
                add_finite(result, key, contribution)
        if self._tolerance > 0.0:
            return {k: c for k, c in result.items() if abs(c) > self._tolerance}
        return result
