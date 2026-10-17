"""Regression tests for preserving coefficients and numerical boundaries."""

import math
import pytest

from hypercomplex_algebra import ExpressionMultiplier


def _assert_identity_product(multiplier, expr):
    inverse = multiplier.inverse(expr)
    product = multiplier.multiply(expr, inverse)
    terms = multiplier._parser.parse(product)
    scalar_key = (0, 0) if multiplier.kind in ("dual", "dual_split") else 0
    assert math.isclose(terms.get(scalar_key, 0.0), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    for key, value in terms.items():
        if key != scalar_key:
            assert abs(value) <= 1e-9


def test_large_scalar_inverse_is_not_pruned():
    m = ExpressionMultiplier(kind="standard")
    assert m.inverse("10000000000000") == "1E-13"


def test_small_scalar_survives_parse_multiply_and_format():
    m = ExpressionMultiplier(kind="standard")
    assert m.multiply("1E-13", "1") == "1E-13"


def test_small_basis_coefficient_round_trip():
    m = ExpressionMultiplier(kind="standard")
    expr = "1E-13e1"
    parsed = m._parser.parse(expr)
    reparsed = m._parser.parse(m._formatter.format(parsed))
    assert math.isclose(reparsed[1], 1e-13, rel_tol=1e-14, abs_tol=0.0)


def test_standard_inverse_round_trip_for_large_scalar():
    _assert_identity_product(ExpressionMultiplier(kind="standard"), "10000000000000")


def test_split_zero_divisor_is_not_invertible():
    m = ExpressionMultiplier(kind="split", dim=1)
    assert m.norm("1 + e1") == "0"
    with pytest.raises(ZeroDivisionError):
        m.inverse("1 + e1")


def test_negative_split_norm_is_invertible():
    m = ExpressionMultiplier(kind="split", dim=1)
    assert m.norm("1 + 2e1") == "-3"
    _assert_identity_product(m, "1 + 2e1")


def test_small_dual_coefficient_survives_inverse():
    m = ExpressionMultiplier(kind="dual")
    result = m.inverse("1 + 1E-13eps")
    parsed = m._parser.parse(result)
    assert math.isclose(parsed.get((0, 0), 0.0), 1.0, abs_tol=1e-15)
    assert math.isclose(parsed.get((0, 1), 0.0), -1e-13, rel_tol=1e-12, abs_tol=0.0)


def test_dual_inverse_round_trip():
    _assert_identity_product(ExpressionMultiplier(kind="dual"), "3 + 2e1 + eps")


def test_pure_epsilon_is_not_invertible():
    with pytest.raises(ZeroDivisionError):
        ExpressionMultiplier(kind="dual").inverse("eps")


@pytest.mark.parametrize("expr", ["nan", "inf", "-inf", "nane1", "infe1"])
def test_non_finite_base_coefficients_are_rejected(expr):
    with pytest.raises(ValueError, match="Non-finite"):
        ExpressionMultiplier(kind="standard").multiply(expr, "1")


@pytest.mark.parametrize("expr", ["nan", "inf", "-inf"])
def test_non_finite_tensor_scalars_are_rejected(expr):
    m = ExpressionMultiplier(kind="tensor", slots=[("standard", None), ("standard", None)])
    with pytest.raises(ValueError, match="Non-finite"):
        m.multiply(expr, "1")


def test_tensor_unary_operations_fail_explicitly():
    m = ExpressionMultiplier(kind="tensor", slots=[("standard", None), ("standard", None)])
    with pytest.raises(NotImplementedError):
        m.conjugate("1")
    with pytest.raises(NotImplementedError):
        m.norm("1")
    with pytest.raises(NotImplementedError):
        m.inverse("1")
