from ..tokenizer import tokenize
from ...helpers.numerics import add_finite, parse_finite_float

class ElementParser:
    """Parses expressions for standard and split algebras."""

    def parse(self, s: str) -> dict[int, float]:
        terms: dict[int, float] = {}
        for sign, body in tokenize(s):
            coeff, index = self._parse_body(body)
            add_finite(terms, index, sign * coeff)
        return terms

    @staticmethod
    def _parse_body(body: str) -> tuple[float, int]:
        """Returns (coefficient, index). 2-tuple — no eps for base algebras."""
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
            return magnitude, index

        # Pure scalar
        magnitude = parse_finite_float(body)
        return magnitude, 0