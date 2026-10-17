from ..tokenizer import tokenize
from ...helpers.numerics import add_finite, parse_finite_float

class DualElementParser:
    """Parses expressions for dual/dual_split algebras. Recognizes eps tokens."""

    def parse(self, s: str) -> dict[tuple[int, int], float]:
        terms: dict[tuple[int, int], float] = {}
        for sign, body in tokenize(s):
            coeff, index, eps = self._parse_body(body)
            key = (index, eps)
            add_finite(terms, key, sign * coeff)
        return terms

    @staticmethod
    def _parse_body(body: str) -> tuple[float, int, int]:
        """Returns (coefficient, index, eps_flag). 3-tuple — dual-aware."""
        # Bare "eps" means ε·e₀
        if body == "eps":
            return 1.0, 0, 1

        # Dual element with coefficient: {coeff}eps (means coeff * ε·e₀)
        if body.endswith("eps") and "eps_e" not in body:
            coeff_str = body[:-3]
            magnitude = 1.0 if coeff_str == "" else parse_finite_float(coeff_str)
            return magnitude, 0, 1

        # Dual element: eps_e{n} or {coeff}eps_e{n}
        if "eps_e" in body:
            coeff_str, index_str = body.split("eps_e", 1)
            magnitude = 1.0 if coeff_str == "" else parse_finite_float(coeff_str)
            index = int(index_str)
            if index < 0:
                raise ValueError(f"Basis index must be >= 0, got {index}")
            return magnitude, index, 1

        # Base element: e{n}  (eps_flag = 0)
        if "e" in body:
            coeff_str, index_str = body.split("e", 1)
            if not index_str or index_str[0] in "+-":
                raise ValueError(
                    f"Cannot parse '{body}': nothing recognizable after 'e'. "
                    f"Basis indices are plain non-negative integers, e.g. 'e3', "
                    f"not signed ('e-1') or exponential ('1e-3' — use decimal "
                    f"form like 0.001 for scientific-notation coefficients)."
                )
            magnitude = 1.0 if coeff_str == "" else parse_finite_float(coeff_str)
            index = int(index_str)
            if index < 0:
                raise ValueError(f"Basis index must be >= 0, got {index}")
            return magnitude, index, 0

        # Pure scalar
        magnitude = parse_finite_float(body)
        return magnitude, 0, 0