"""Norm for standard and split algebras: N(x) = x · x*

Uses the Hypercomplex Conjugate (negates index > 0).
The result collapses to a real scalar.
"""
from ..conjugate.base import conjugate_base


def norm_base(terms: dict[int, float], multiplier) -> float:
    """
    Computes N(x) = x · x* for standard/split algebras.

    Returns the real scalar norm as a float.
    Raises ValueError if the result doesn't collapse to a scalar (zero divisor).
    """
    conj = conjugate_base(terms)
    product = multiplier.multiply(terms, conj)

    if not product:
        return 0.0
    if len(product) != 1 or 0 not in product:
        raise ValueError(
            "Norm did not collapse to a scalar. "
            "Element may be a zero divisor (common in split algebras)."
        )

    return product[0]