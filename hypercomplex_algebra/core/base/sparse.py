"""Sparse hypercomplex multiplication. Depends only on the resolver contract."""
from .resolver import BasisProductResolver


class SparseMultiplier:
    """Multiplies two sparse elements ({index: coefficient}).

    Since e_i * e_j = ±e_k (a single element), the product of elements with
    k1 and k2 terms has at most k1*k2 terms before combining like indices.
    """

    def __init__(self, resolver: BasisProductResolver, tolerance: float = 1e-12):
        self._resolver = resolver
        self._tolerance = tolerance

    def multiply(self, a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
        result: dict[int, float] = {}
        for i, ci in a.items():
            if abs(ci) < self._tolerance:
                continue
            for j, cj in b.items():
                if abs(cj) < self._tolerance:
                    continue
                sign, idx = self._resolver.resolve(i, j)
                result[idx] = result.get(idx, 0.0) + sign * ci * cj
        return {idx: c for idx, c in result.items() if abs(c) > self._tolerance}