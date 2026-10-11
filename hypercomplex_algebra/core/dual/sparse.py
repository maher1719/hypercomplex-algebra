"""Sparse dual multiplication. Keys are (index, eps_flag) tuples."""
from .resolver import DualResolver


class DualSparseMultiplier:
    """Multiplies two sparse dual elements.

    Elements are dicts: {(index, eps_flag): coefficient}
    Example: {(2, 0): 3.0, (1, 1): -2.5}  means  3·e₂ - 2.5·ε·e₁
    """

    def __init__(self, resolver: DualResolver, tolerance: float = 1e-12):
        self._resolver = resolver
        self._tolerance = tolerance

    def multiply(self, a: dict[tuple[int, int], float],
                 b: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for (i, eps_i), ci in a.items():
            if abs(ci) < self._tolerance:
                continue
            for (j, eps_j), cj in b.items():
                if abs(cj) < self._tolerance:
                    continue
                sign, idx, eps = self._resolver.resolve(i, eps_i, j, eps_j)
                if sign == 0:  # nilpotent zero, skip
                    continue
                key = (idx, eps)
                result[key] = result.get(key, 0.0) + sign * ci * cj
        return {k: c for k, c in result.items() if abs(c) > self._tolerance}