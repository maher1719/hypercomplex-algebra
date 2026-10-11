"""Inverse for standard and split algebras: x⁻¹ = x* / N(x)

Uses the Hypercomplex Conjugate (same as norm).
"""
from ..conjugate.base import conjugate_base
from ..norm.base import norm_base

_TOLERANCE = 1e-15


def inverse_base(terms: dict[int, float], multiplier) -> dict[int, float]:
    """
    Computes x⁻¹ = x* / N(x) for standard/split algebras.

    Raises ZeroDivisionError if the norm is zero (element is not invertible).
    """
    n = norm_base(terms, multiplier)
    if abs(n) < _TOLERANCE:
        raise ZeroDivisionError(
            "Cannot invert an element with a norm of zero. "
            "Element may be a zero divisor or the zero element."
        )

    conj = conjugate_base(terms)
    return {idx: coeff / n for idx, coeff in conj.items()}