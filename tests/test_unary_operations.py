"""
Test suite for unary operations: conjugate, norm, inverse.

Validates:
    1. Base conjugate (standard/split)
    2. Dual conjugate modes (hypercomplex, dual, total)
    3. Norm computation (real scalar for base, dual scalar for duals)
    4. Inverse computation (x · x⁻¹ = 1)
    5. Mode rejection (dual mode rejected for norm/inverse)
    6. Expression simplification before unary ops
"""
import math
import pytest
from hypercomplex_algebra import ExpressionMultiplier


class TestBaseConjugate:
    def test_standard_conjugate(self):
        m = ExpressionMultiplier(kind="standard")
        assert m.conjugate("1 + 2e1 + 3e2") == "1 - 2e1 - 3e2"

    def test_standard_conjugate_scalar_only(self):
        m = ExpressionMultiplier(kind="standard")
        assert m.conjugate("5") == "5"

    def test_standard_conjugate_pure_imaginary(self):
        m = ExpressionMultiplier(kind="standard")
        assert m.conjugate("3e1") == "-3e1"


class TestDualConjugateModes:
    def setup_method(self):
        self.m = ExpressionMultiplier(kind="dual")

    def test_hypercomplex_mode(self):
        # Real +, Imag -, PureEps +, Mixed -
        expr = "3 + 2e1 + eps + 4eps_e2"
        assert self.m.conjugate(expr, mode="hypercomplex") == "3 - 2e1 + eps - 4eps_e2"

    def test_dual_mode(self):
        # Real +, Imag +, PureEps -, Mixed -
        expr = "3 + 2e1 + eps + 4eps_e2"
        assert self.m.conjugate(expr, mode="dual") == "3 + 2e1 - eps - 4eps_e2"

    def test_total_mode(self):
        # Real +, Imag -, PureEps -, Mixed +
        expr = "3 + 2e1 + eps + 4eps_e2"
        assert self.m.conjugate(expr, mode="total") == "3 - 2e1 - eps + 4eps_e2"

    def test_default_is_hypercomplex(self):
        expr = "3 + 2e1 + eps + 4eps_e2"
        assert self.m.conjugate(expr) == self.m.conjugate(expr, mode="hypercomplex")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown conjugate mode"):
            self.m.conjugate("e1", mode="invalid")


class TestBaseNorm:
    def test_standard_norm(self):
        m = ExpressionMultiplier(kind="standard")
        # |1 + 2e1 + 3e2|² = 1 + 4 + 9 = 14
        assert m.norm("1 + 2e1 + 3e2") == "14"

    def test_standard_norm_scalar(self):
        m = ExpressionMultiplier(kind="standard")
        assert m.norm("5") == "25"

    def test_standard_norm_pure_imaginary(self):
        m = ExpressionMultiplier(kind="standard")
        # |3e1|² = 9
        assert m.norm("3e1") == "9"


class TestDualNorm:
    def setup_method(self):
        self.m = ExpressionMultiplier(kind="dual")

    def test_hypercomplex_norm(self):
        # x = 3 + 2e1 + eps
        # x* = 3 - 2e1 + eps
        # x·x* = (3+eps)² - (2e1)² = 9 + 6eps + 4 = 13 + 6eps
        assert self.m.norm("3 + 2e1 + eps") == "13 + 6eps"

    def test_norm_pure_real(self):
        # x = 5, x* = 5, x·x* = 25
        assert self.m.norm("5") == "25"

    def test_norm_pure_dual(self):
        # x = eps, x* = eps, x·x* = eps² = 0
        assert self.m.norm("eps") == "0"

    def test_dual_mode_rejected_for_norm(self):
        with pytest.raises(ValueError, match="cannot be used for norm"):
            self.m.norm("3 + 2e1 + eps", mode="dual")



class TestDualInverse:
    def setup_method(self):
        self.m = ExpressionMultiplier(kind="dual")

    def test_inverse_and_multiply_back(self):
        expr = "3 + 2e1 + eps"
        inv_result = self.m.inverse(expr)
        assert inv_result is not None

        # Verify x · x⁻¹ ≈ 1 (approximate due to floating point)
        result = self.m.multiply(expr, inv_result)
        parsed = self.m._parser.parse(result)

        # Scalar part should be approximately 1
        assert abs(parsed.get((0, 0), 0.0) - 1.0) < 1e-9

        # All other terms should be approximately 0
        for key, val in parsed.items():
            if key != (0, 0):
                assert abs(val) < 1e-9, f"Residual term {key} = {val} is too large"

    def test_inverse_pure_real(self):
        # x = 5, x⁻¹ = 1/5 = 0.2
        result = self.m.inverse("5")
        assert result == "0.2"

    def test_inverse_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            self.m.inverse("0")

    def test_dual_mode_rejected_for_inverse(self):
        with pytest.raises(ValueError, match="cannot be used for norm"):
            self.m.inverse("3 + 2e1 + eps", mode="dual")


class TestExpressionSimplification:
    def test_simplification_before_conjugate(self):
        m = ExpressionMultiplier(kind="dual")
        # Parser combines -eps_e1 + 2eps_e1 = +eps_e1
        expr = "3 + e3 - eps_e1 + eps + 2eps_e1"
        # Simplified: 3 + e3 + eps + eps_e1
        # Hypercomplex conjugate: 3 - e3 + eps - eps_e1
        assert m.conjugate(expr) == "3 - e3 + eps - eps_e1"

class TestEdgeCasesAndSplit:
    def test_split_norm_and_zero_divisor(self):
        m = ExpressionMultiplier(kind="split", dim=1)
        # Split-complex: e1^2 = +1
        # Norm of (1 + e1) = (1 + e1)(1 - e1) = 1 - e1^2 = 1 - 1 = 0
        # This is a zero divisor and cannot be inverted.
        assert m.norm("1 + e1") == "0"
        with pytest.raises(ZeroDivisionError):
            m.inverse("1 + e1")


    def test_tiny_invertible_scalar(self):
        # Values whose norm falls below the pruning tolerance (1e-12)
        # cannot be inverted — this is a documented numerical limitation.
        # 0.0001 has norm = 1e-8, safely above the tolerance.
        m = ExpressionMultiplier(kind="standard")
        inv = m.inverse("0.0001")
        # 1 / 0.0001 = 10000
        parsed = m._parser.parse(inv)
        assert abs(parsed.get(0, 0.0) - 10000.0) < 1e-6

    def test_non_finite_coefficients_rejected(self):
        m = ExpressionMultiplier(kind="standard")
        # Use "nane1" and "infe1" (no spaces) so the tokenizer passes 
        # them as a single body to the parser.
        with pytest.raises(ValueError, match="Non-finite"):
            m.multiply("nane1", "e1")
        with pytest.raises(ValueError, match="Non-finite"):
            m.multiply("infe1", "e1")
        with pytest.raises(ValueError, match="Non-finite"):
            m.multiply("nan", "1")
            
    def test_invalid_conjugate_mode_on_zero(self):
        m = ExpressionMultiplier(kind="dual")
        # Empty expression "0" should still validate the mode
        with pytest.raises(ValueError, match="Unknown conjugate mode"):
            m.conjugate("0", mode="invalid")

    def test_dual_split_nilpotent_inverse_rejection(self):
        m = ExpressionMultiplier(kind="dual_split", dim=1)
        # Pure epsilon has norm 0 (nilpotent)
        with pytest.raises(ZeroDivisionError):
            m.inverse("eps")

    def test_round_trip_small_dual_coefficient(self):
        m = ExpressionMultiplier(kind="dual")
        # With aligned tolerances (1e-12), 1e-10 should survive round-trip
        expr = "1e-10eps"
        parsed = m._parser.parse(expr)
        formatted = m._formatter.format(parsed)
        reparsed = m._parser.parse(formatted)
        assert abs(reparsed.get((0, 1), 0.0) - 1e-10) < 1e-15
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])