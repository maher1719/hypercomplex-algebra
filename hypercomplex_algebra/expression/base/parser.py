import math
from ..tokenizer import tokenize

class ElementParser:
    """Parses expressions for standard and split algebras."""

    _PRUNE = 1e-12  # Aligned tolerance

    def parse(self, s: str) -> dict[int, float]:
        terms: dict[int, float] = {}
        for sign, body in tokenize(s):
            coeff, index = self._parse_body(body)
            terms[index] = terms.get(index, 0.0) + sign * coeff
        return {k: c for k, c in terms.items() if abs(c) > self._PRUNE}

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
            magnitude = 1.0 if coeff_str == "" else float(coeff_str)
            if not math.isfinite(magnitude):
                raise ValueError(f"Non-finite coefficient {magnitude} is not allowed.")
                
            index = int(index_str)
            if index < 0:
                raise ValueError(f"Basis index must be >= 0, got {index}")
            return magnitude, index

        # Pure scalar
        magnitude = float(body)
        if not math.isfinite(magnitude):
            raise ValueError(f"Non-finite coefficient {magnitude} is not allowed.")
        return magnitude, 0