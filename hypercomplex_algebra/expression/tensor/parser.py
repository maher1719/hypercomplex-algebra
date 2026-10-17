"""Parse tensor expression strings into {(i, j, k, ...): coefficient}."""
from ..tokenizer import tokenize
from ...helpers.numerics import add_finite, parse_finite_float


class TensorElementParser:
    """Parses tensor expressions using e[i,j,k] multi-index notation.

    The parser must know the number of slots (num_slots) to parse scalars
    into the correct-length zero tuple and to validate index counts.
    """

    def __init__(self, num_slots: int):
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self._num_slots = num_slots

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def parse(self, s: str) -> dict[tuple, float]:
        terms: dict[tuple, float] = {}
        for sign, body in tokenize(s):
            magnitude, key = self._parse_body(body)
            add_finite(terms, key, sign * magnitude)
        return terms

    def _parse_body(self, body: str) -> tuple[float, tuple]:
        """Returns (coefficient, key) where key is a tuple of indices."""
        if "e[" in body:
            coeff_str, rest = body.split("e[", 1)
            if not rest.endswith("]"):
                raise ValueError(f"Malformed tensor basis (missing ']'): '{body}'")
            indices_str = rest[:-1]
            try:
                indices = tuple(int(x.strip()) for x in indices_str.split(","))
            except ValueError:
                raise ValueError(f"Invalid indices in '{body}'")
            if len(indices) != self._num_slots:
                raise ValueError(
                    f"Expected {self._num_slots} indices, got {len(indices)} in '{body}'"
                )
            if any(i < 0 for i in indices):
                raise ValueError(f"Indices must be >= 0 in '{body}'")
            magnitude = 1.0 if coeff_str == "" else parse_finite_float(coeff_str)
            return magnitude, indices
        else:
            # Pure scalar -> identity in all slots
            magnitude = parse_finite_float(body)
            return magnitude, (0,) * self._num_slots