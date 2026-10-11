"""Format {index: coefficient} back into an expression string."""

class ElementFormatter:
    """Formats standard/split sparse dicts into expression strings."""

    def __init__(self, tolerance: float = 1e-12):
        self._tolerance = tolerance

    def format(self, terms: dict[int, float]) -> str:
        items = sorted(
            (idx, c) for idx, c in terms.items() if abs(c) > self._tolerance
        )
        if not items:
            return "0"

        parts = []
        for pos, (idx, coeff) in enumerate(items):
            body = self._magnitude(idx, abs(coeff))
            if pos == 0:
                parts.append(f"-{body}" if coeff < 0 else body)
            else:
                parts.append(f"- {body}" if coeff < 0 else f"+ {body}")
        return " ".join(parts)

    def _magnitude(self, index: int, mag: float) -> str:
        if index == 0:
            return self._format_coeff(mag)
        basis = f"e{index}"
        if abs(mag - 1.0) < self._tolerance:
            return basis
        return f"{self._format_coeff(mag)}{basis}"

    @staticmethod
    def _format_coeff(mag: float) -> str:
        """Format a coefficient without scientific notation."""
        if mag == int(mag) and abs(mag) < 1e15:
            return str(int(mag))
        # Use fixed-point notation with 15 decimal places, strip trailing zeros
        return f"{mag:.15f}".rstrip('0').rstrip('.')