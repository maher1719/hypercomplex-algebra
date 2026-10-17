"""Format {(i, j, k, ...): coefficient} back into tensor expression strings."""

from ...helpers.numerics import format_float


class TensorElementFormatter:
    """Formats tensor sparse dicts into strings like '3e[1,2,0] - e[0,1,3]'."""

    def __init__(self, tolerance: float = 0.0):
        self._tolerance = tolerance

    def format(self, terms: dict[tuple, float]) -> str:
        items = sorted(
            (key, c) for key, c in terms.items() if c != 0.0
        )
        if not items:
            return "0"

        parts = []
        for pos, (key, coeff) in enumerate(items):
            body = self._magnitude(key, abs(coeff))
            if pos == 0:
                parts.append(f"-{body}" if coeff < 0 else body)
            else:
                parts.append(f"- {body}" if coeff < 0 else f"+ {body}")
        return " ".join(parts)

    def _magnitude(self, key: tuple, mag: float) -> str:
        """Format |coefficient| * basis without sign."""
        if all(idx == 0 for idx in key):
            return format_float(mag)  # identity / scalar
        basis = "e[" + ",".join(str(i) for i in key) + "]"
        if mag == 1.0:
            return basis
        return f"{format_float(mag)}{basis}"