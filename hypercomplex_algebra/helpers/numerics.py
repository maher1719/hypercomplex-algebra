"""Numerical helpers shared by parsers, formatters, and sparse arithmetic."""

import math
from decimal import Decimal, InvalidOperation


def parse_finite_float(text: str) -> float:
    """Parse a float and reject NaN and infinities."""
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric coefficient: {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite coefficient {text!r} is not allowed.")
    # Do not silently turn a nonzero decimal literal into 0.0 through underflow.
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        decimal_value = None
    if value == 0.0 and decimal_value is not None and decimal_value != 0:
        raise OverflowError(f"Coefficient {text!r} underflows the float range.")
    return value


def add_finite(terms: dict, key, value: float) -> None:
    """Accumulate one coefficient, removing exact cancellation only."""
    if not math.isfinite(value):
        raise OverflowError(f"Non-finite arithmetic contribution for {key!r}.")
    total = terms.get(key, 0.0) + value
    if not math.isfinite(total):
        raise OverflowError(f"Coefficient accumulation overflowed for {key!r}.")
    if total == 0.0:
        terms.pop(key, None)
    else:
        terms[key] = total


def format_float(value: float) -> str:
    """Format a finite float with a shortest round-trip representation.

    Uppercase E is intentional: lowercase e is reserved by this package's
    basis-index syntax.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Cannot format non-finite value: {value!r}")
    if value == 0.0:
        return "0"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value).replace("e", "E")
