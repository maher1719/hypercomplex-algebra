"""Hypercomplex conjugate for standard and split algebras.

Negates all imaginary basis elements (index > 0). Keeps scalar (index 0) positive.
This is equivalent to the Hypercomplex Conjugate (W*) from types.csv,
applied to algebras without an epsilon dimension.
"""


def conjugate_base(terms: dict[int, float]) -> dict[int, float]:
    """Negates all imaginary basis elements (index > 0). Keeps scalar positive."""
    return {
        idx: (coeff if idx == 0 else -coeff)
        for idx, coeff in terms.items()
    }