"""Inverse for dual algebras, with scale-aware coefficient handling."""

import math

from ..conjugate.dual import conjugate_dual
from ..norm.dual import norm_dual


def inverse_dual(
    terms: dict[tuple[int, int], float],
    multiplier,
    mode: str = "hypercomplex",
) -> dict[tuple[int, int], float]:
    """Compute x⁻¹ = x* / N(x), scaling x before computing its norm."""
    if not terms:
        raise ZeroDivisionError("Cannot invert zero.")

    scale = max(abs(value) for value in terms.values())
    if scale == 0.0:
        raise ZeroDivisionError("Cannot invert zero.")
    scaled = {key: value / scale for key, value in terms.items()}

    norm_dict = norm_dual(scaled, multiplier, mode=mode)
    A = norm_dict.get((0, 0), 0.0)
    B = norm_dict.get((0, 1), 0.0)
    if A == 0.0:
        raise ZeroDivisionError(
            "Cannot invert a dual element whose norm has a zero real part. "
            "The element is not invertible."
        )

    inv_A = 1.0 / A
    inv_B = (-B / A) / A
    if not math.isfinite(inv_A) or not math.isfinite(inv_B):
        raise OverflowError("Dual norm inverse is outside the finite float range.")

    inv_norm_dict: dict[tuple[int, int], float] = {(0, 0): inv_A}
    if inv_B != 0.0:
        inv_norm_dict[(0, 1)] = inv_B

    conjugate = conjugate_dual(scaled, mode=mode)
    normalized_result = multiplier.multiply(conjugate, inv_norm_dict)
    result = {}
    for key, coeff in normalized_result.items():
        value = coeff / scale
        if not math.isfinite(value):
            raise OverflowError("Inverse coefficient is outside the finite float range.")
        if value != 0.0:
            result[key] = value
    return result
