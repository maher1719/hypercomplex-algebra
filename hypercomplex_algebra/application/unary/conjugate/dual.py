"""Dual conjugates: hypercomplex, dual, and total.

Matches types.csv exactly:
    Original (W):          Real +, Imag +, PureEps +, Mixed +
    Dual Conjugate (W̅):    Real +, Imag +, PureEps -, Mixed -
    Hypercomplex Conj (W*): Real +, Imag -, PureEps +, Mixed -
    Total Conjugate (W†):   Real +, Imag -, PureEps -, Mixed +
"""

_TOLERANCE = 1e-15


def conjugate_dual(
    terms: dict[tuple[int, int], float],
    mode: str = "hypercomplex",
) -> dict[tuple[int, int], float]:
    """
    Applies the specified conjugate mode to a dual sparse dict.

    Keys are (index, eps_flag) tuples.

    Modes:
        "hypercomplex" (default): Negates imaginary and mixed. Keeps real and pure eps.
        "dual":                   Negates pure eps and mixed. Keeps real and imaginary.
        "total":                  Negates imaginary and pure eps. Mixed stays positive.
    """
    valid_modes = ("hypercomplex", "dual", "total")
    if mode not in valid_modes:
        raise ValueError(f"Unknown conjugate mode: {mode!r}. Use {valid_modes}.")
        
    result = {}
    for (idx, eps), coeff in terms.items():
        if mode == "hypercomplex":
            # Real (0,0): + | Imag (i>0,0): - | PureEps (0,1): + | Mixed (i>0,1): -
            new_coeff = -coeff if idx > 0 else coeff

        elif mode == "dual":
            # Real (0,0): + | Imag (i>0,0): + | PureEps (0,1): - | Mixed (i>0,1): -
            new_coeff = -coeff if eps == 1 else coeff

        elif mode == "total":
            # Real (0,0): + | Imag (i>0,0): - | PureEps (0,1): - | Mixed (i>0,1): +
            if idx == 0 and eps == 0:
                new_coeff = coeff       # Real: positive
            elif idx > 0 and eps == 0:
                new_coeff = -coeff      # Imaginary: negative
            elif idx == 0 and eps == 1:
                new_coeff = -coeff      # Pure eps: negative
            else:
                new_coeff = coeff       # Mixed: positive (double negation)
        else:
            raise ValueError(
                f"Unknown conjugate mode: {mode!r}. "
                f"Use 'hypercomplex', 'dual', or 'total'."
            )

        result[(idx, eps)] = new_coeff

    return {k: v for k, v in result.items() if abs(v) > _TOLERANCE}