"""Inverse for standard and split algebras: x⁻¹ = x* / N(x)."""

import math

from ..conjugate.base import conjugate_base
from ..norm.base import norm_base


def inverse_base(terms: dict[int, float], multiplier) -> dict[int, float]:
    """Compute the inverse while scaling before evaluating the norm.

    Scaling avoids squaring very large or very small input coefficients. A zero
    norm (including split-algebra zero divisors) is not invertible.
    """
    if not terms:
        raise ZeroDivisionError("Cannot invert zero.")

    scale = max(abs(value) for value in terms.values())
    if scale == 0.0:
        raise ZeroDivisionError("Cannot invert zero.")

    scaled = {idx: value / scale for idx, value in terms.items()}
    norm = norm_base(scaled, multiplier)
    if norm == 0.0:
        raise ZeroDivisionError(
            "Cannot invert an element with zero norm. "
            "The element may be zero or a zero divisor."
        )
    conjugate = conjugate_base(scaled)
    result = {}
    for idx, coeff in conjugate.items():
        value = (coeff / norm) / scale
        if not math.isfinite(value):
            raise OverflowError("Inverse coefficient is outside the finite float range.")
        if value != 0.0:
            result[idx] = value
    return result
