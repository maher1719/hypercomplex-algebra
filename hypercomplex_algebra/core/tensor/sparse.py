"""Sparse tensor multiplication. Keys are tuples of indices (one per slot)."""

import math
from ...helpers.numerics import add_finite
from .resolver import TensorResolver


class TensorSparseMultiplier:
    """Multiplies two sparse tensor elements.

    Elements are dicts: {(i, j, k, ...): coefficient}
    """

    def __init__(self, resolver: TensorResolver, tolerance: float = 0.0):
        self._resolver = resolver
        self._tolerance = tolerance

    def multiply(self, a: dict, b: dict) -> dict:
        result: dict = {}
        for key_a, ca in a.items():
            if self._tolerance > 0.0 and abs(ca) <= self._tolerance:
                continue
            for key_b, cb in b.items():
                if self._tolerance > 0.0 and abs(cb) <= self._tolerance:
                    continue
                sign, result_key = self._resolver.resolve(key_a, key_b)
                contribution = sign * ca * cb
                if not math.isfinite(contribution):
                    raise OverflowError("Non-finite coefficient produced by multiplication.")
                add_finite(result, result_key, contribution)
        if self._tolerance > 0.0:
            return {k: c for k, c in result.items() if abs(c) > self._tolerance}
        return result
