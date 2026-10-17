"""Sparse hypercomplex multiplication. Depends only on the resolver contract."""

import math
from .resolver import BasisProductResolver
from ...helpers.numerics import add_finite


class SparseMultiplier:
    """Multiplies two sparse elements ({index: coefficient}).

    Since e_i * e_j = ±e_k (a single element), the product of elements with
    k1 and k2 terms has at most k1*k2 terms before combining like indices.
    """

    def __init__(self, resolver: BasisProductResolver, tolerance: float = 0.0):
        self._resolver = resolver
        self._tolerance = tolerance

    def multiply(self, a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
        result: dict[int, float] = {}
        for i, ci in a.items():
            if self._tolerance > 0.0 and abs(ci) <= self._tolerance:
                continue
            for j, cj in b.items():
                if self._tolerance > 0.0 and abs(cj) <= self._tolerance:
                    continue
                sign, idx = self._resolver.resolve(i, j)
                contribution = sign * ci * cj
                if not math.isfinite(contribution):
                    raise OverflowError("Non-finite coefficient produced by multiplication.")
                add_finite(result, idx, contribution)
        if self._tolerance > 0.0:
            return {idx: c for idx, c in result.items() if abs(c) > self._tolerance}
        return result
