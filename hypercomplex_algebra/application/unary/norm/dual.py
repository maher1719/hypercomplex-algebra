"""Norm for dual algebras: N(x) = x · x*

Uses the Hypercomplex Conjugate by default (W*).
The result collapses to a dual scalar (real + eps parts, no spatial e_i terms).

The "total" conjugate mode is also supported as an advanced option.
The "dual" conjugate mode is REJECTED because it does not collapse to a scalar.
"""
from ..conjugate.dual import conjugate_dual

_VALID_NORM_MODES = ("hypercomplex", "total")


def norm_dual(
    terms: dict[tuple[int, int], float],
    multiplier,
    mode: str = "hypercomplex",
) -> dict[tuple[int, int], float]:
    """
    Computes N(x) = x · x* for dual algebras.

    Returns the dual scalar norm as a dict (e.g., {(0,0): 13, (0,1): 6}).

    Args:
        terms: The dual sparse dict to compute the norm of.
        multiplier: The DualSparseMultiplier instance.
        mode: "hypercomplex" (default) or "total".
              "dual" is rejected because it doesn't collapse to a scalar.

    Raises:
        ValueError: If mode is "dual" or if the result doesn't collapse to a dual scalar.
    """
    if mode not in _VALID_NORM_MODES:
        raise ValueError(
            f"Conjugate mode {mode!r} cannot be used for norm computation. "
            f"The dual conjugate does not collapse to a scalar. "
            f"Valid modes for norm: {_VALID_NORM_MODES}"
        )

    conj = conjugate_dual(terms, mode=mode)
    product = multiplier.multiply(terms, conj)

    if not product:
        return {}

    # Verify it collapsed to a pure dual scalar (no spatial e_i terms)
    for (idx, eps) in product.keys():
        if idx != 0:
            raise ValueError(
                f"Norm did not collapse to a dual scalar. "
                f"Found spatial term at index {idx}. "
                f"Element may be a zero divisor."
            )

    return product