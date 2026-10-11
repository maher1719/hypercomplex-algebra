"""Inverse for dual algebras: x⁻¹ = x* / N(x)

Uses the Hypercomplex Conjugate by default (must match norm's conjugate).
Division by a dual scalar (A + Bε) uses: (A + Bε)⁻¹ = A⁻¹ - B·A⁻²·ε
"""
from ..conjugate.dual import conjugate_dual
from ..norm.dual import norm_dual

# Use a very small tolerance to allow inversion of tiny but valid floats.
_TOLERANCE = 1e-30


def inverse_dual(
    terms: dict[tuple[int, int], float],
    multiplier,
    mode: str = "hypercomplex",
) -> dict[tuple[int, int], float]:
    """
    Computes x⁻¹ = x* / N(x) for dual algebras.

    The conjugate mode MUST match the norm's mode for x · x⁻¹ = 1 to hold.
    """
    # Compute norm (uses the same conjugate mode)
    norm_dict = norm_dual(terms, multiplier, mode=mode)

    # Extract real part A and dual part B from the norm (A + B·ε)
    A = norm_dict.get((0, 0), 0.0)
    B = norm_dict.get((0, 1), 0.0)

    if abs(A) < _TOLERANCE:
        raise ZeroDivisionError(
            "Cannot invert a dual element whose norm has a zero real part. "
            "The element is not invertible."
        )

    # Inverse of dual scalar (A + Bε) is: A⁻¹ - B·A⁻²·ε
    inv_A = 1.0 / A
    inv_B = -B / (A * A)

    # Build the inverse norm dict
    inv_norm_dict: dict[tuple[int, int], float] = {(0, 0): inv_A}
    if abs(inv_B) > 1e-12:  # Use aligned global tolerance for pruning
        inv_norm_dict[(0, 1)] = inv_B

    # Compute conjugate (same mode as norm)
    conj = conjugate_dual(terms, mode=mode)

    # x⁻¹ = x* · (N(x))⁻¹
    return multiplier.multiply(conj, inv_norm_dict)