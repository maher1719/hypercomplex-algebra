"""Format {(index, eps_flag): coefficient} back into a dual expression string."""


class DualElementFormatter:
    """Formats dual sparse dicts, grouping base terms before dual terms (a + ε·b)."""

    def __init__(self, tolerance: float = 1e-9):
        self._tolerance = tolerance

    def format(self, terms: dict[tuple[int, int], float]) -> str:
        base_terms = [
            (idx, c) for (idx, eps), c in terms.items()
            if eps == 0 and abs(c) > self._tolerance
        ]
        dual_terms = [
            (idx, c) for (idx, eps), c in terms.items()
            if eps == 1 and abs(c) > self._tolerance
        ]
        base_terms.sort(key=lambda item: item[0])
        dual_terms.sort(key=lambda item: item[0])

        ordered = base_terms + dual_terms
        if not ordered:
            return "0"

        parts = []
        for pos, (idx, coeff) in enumerate(ordered):
            eps_flag = 0 if pos < len(base_terms) else 1
            body = self._magnitude(idx, abs(coeff), eps_flag)
            if pos == 0:
                parts.append(f"-{body}" if coeff < 0 else body)
            else:
                parts.append(f"- {body}" if coeff < 0 else f"+ {body}")
        return " ".join(parts)

    def _magnitude(self, index: int, mag: float, eps_flag: int = 0) -> str:
        if eps_flag == 0:
            if index == 0:
                return self._format_coeff(mag)
            basis = f"e{index}"
        else:
            if index == 0:
                basis = "eps"
            else:
                basis = f"eps_e{index}"
        if abs(mag - 1.0) < self._tolerance:
            return basis
        return f"{self._format_coeff(mag)}{basis}"

    @staticmethod
    def _format_coeff(mag: float) -> str:
        """Format a coefficient without scientific notation."""
        if mag == int(mag) and abs(mag) < 1e15:
            return str(int(mag))
        return f"{mag:.15f}".rstrip('0').rstrip('.')