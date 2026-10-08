"""
Full test suite for hypercomplex_algebra (sparse expression multiplier).

Coverage:
    1. ElementParser     — string → sparse dict, incl. malformed input
    2. ElementFormatter  — sparse dict → string, ordering & pruning
    3. SparseMultiplier     — domain math, single-element-product property
    4. StandardResolver         — engine adapter, contract conformance
    5. ExpressionMultiplier — facade integration
    6. Cross-validation     — verified against hypercomplex.build_table
    7. Edge cases           — associativity, cancellation, identities

Run:
    pytest tests/test_hypercomplex_algebra.py -v
"""

import pytest

from hypercomplex_engine import build_table

from hypercomplex_algebra import (
    BasisProductResolver,
    ElementFormatter,
    ExpressionMultiplier,
    ElementParser,
    StandardResolver,
    SparseMultiplier,
    multiply_expressions,
)


# ======================================================================
# 1. PARSER
# ======================================================================

class TestStandardAlgebra:
    def setup_method(self):
        self.parser = ElementParser()

    # --- basic forms ---
    def test_pure_scalar(self):
        assert self.parser.parse("5") == {0: 5.0}

    def test_negative_scalar(self):
        assert self.parser.parse("-3") == {0: -3.0}

    def test_float_scalar(self):
        assert self.parser.parse("2.5") == {0: 2.5}

    def test_single_basis_implicit_coeff(self):
        assert self.parser.parse("e1") == {1: 1.0}

    def test_single_basis_negative_implicit(self):
        assert self.parser.parse("-e2") == {2: -1.0}

    def test_single_basis_explicit_plus(self):
        assert self.parser.parse("+e3") == {3: 1.0}

    def test_explicit_coefficient(self):
        assert self.parser.parse("3e1") == {1: 3.0}

    def test_float_coefficient(self):
        assert self.parser.parse("2.5e3") == {3: 2.5}

    def test_multiple_terms(self):
        assert self.parser.parse("1 + 2e1 - 3e2") == {0: 1.0, 1: 2.0, 2: -3.0}

    def test_scalar_as_e0(self):
        # '3e0' is 3 * e0 = 3 (basis element 0)
        assert self.parser.parse("3e0") == {0: 3.0}

    # --- whitespace & sign handling ---
    def test_whitespace_ignored(self):
        assert self.parser.parse("  1  +  e1  ") == {0: 1.0, 1: 1.0}

    def test_no_leading_sign_gets_plus(self):
        assert self.parser.parse("3e0 + e1") == {0: 3.0, 1: 1.0}

    # --- combining & cancellation ---
    def test_combines_like_terms(self):
        assert self.parser.parse("e1 + e1") == {1: 2.0}

    def test_cancels_opposite_terms(self):
        assert self.parser.parse("e1 - e1") == {}

    def test_scientific_notation_raises_clear_error(self):
        with pytest.raises(ValueError, match="scientific"):
            self.parser.parse("1e-3 e1")


    def test_scientific_notation_raises_clear_error(self):
        with pytest.raises(ValueError):
            self.parser.parse("1e")


    def test_scientific_notation_raises_clear_error(self):
        with pytest.raises(ValueError):
            self.parser.parse("e")


    # --- large indices ---
    def test_large_index(self):
        assert self.parser.parse("e5000") == {5000: 1.0}

    # --- malformed input raises ---
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self.parser.parse("")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            self.parser.parse("abc")

    def test_bare_e_raises(self):
        with pytest.raises(ValueError):
            self.parser.parse("e")

    def test_trailing_e_raises(self):
        with pytest.raises(ValueError):
            self.parser.parse("1e")


# ======================================================================
# 2. FORMATTER
# ======================================================================

class TestElementFormatter:
    def setup_method(self):
        self.formatter = ElementFormatter()

    def test_empty_is_zero(self):
        assert self.formatter.format({}) == "0"

    def test_scalar(self):
        assert self.formatter.format({0: 3.0}) == "3"

    def test_negative_scalar(self):
        assert self.formatter.format({0: -2.5}) == "-2.5"

    def test_unit_basis_omits_coeff(self):
        assert self.formatter.format({1: 1.0}) == "e1"

    def test_negative_unit_basis(self):
        assert self.formatter.format({1: -1.0}) == "-e1"

    def test_coefficient_basis(self):
        assert self.formatter.format({1: 2.5}) == "2.5e1"

    def test_terms_sorted_by_index(self):
        result = self.formatter.format({2: -3.0, 0: 1.0, 1: 2.0})
        assert result == "1 + 2e1 - 3e2"

    def test_first_term_negative_no_leading_plus(self):
        assert self.formatter.format({1: -2.0, 2: 1.0}) == "-2e1 + e2"

    def test_near_zero_pruned(self):
        assert self.formatter.format({1: 1e-20}) == "0"

    def test_large_index(self):
        assert self.formatter.format({5000: 1.0}) == "e5000"


# ======================================================================
# 3. SPARSE MULTIPLIER (domain logic)
# ======================================================================

class TestSparseMultiplier:
    def setup_method(self):
        self.mult = SparseMultiplier(StandardResolver())

    def test_scalar_times_scalar(self):
        assert self.mult.multiply({0: 2.0}, {0: 3.0}) == {0: 6.0}

    def test_basis_times_basis(self):
        # e1 * e2 = e3
        assert self.mult.multiply({1: 1.0}, {2: 1.0}) == {3: 1.0}

    def test_basis_squared_is_negative_scalar(self):
        # e1 * e1 = -e0
        assert self.mult.multiply({1: 1.0}, {1: 1.0}) == {0: -1.0}

    def test_distributive_complex_square(self):
        # (1 + e1)^2 = 1 + 2e1 + e1^2 = 1 + 2e1 - 1 = 2e1
        a = {0: 1.0, 1: 1.0}
        assert self.mult.multiply(a, a) == {1: 2.0}

    def test_coefficient_multiplication(self):
        # (2 e1) * (-3 e2) = -6 e3
        assert self.mult.multiply({1: 2.0}, {2: -3.0}) == {3: -6.0}

    def test_single_element_product_property(self):
        # e_i * e_j ALWAYS yields exactly one basis element (±e_k), never a sum.
        for i in range(16):
            for j in range(16):
                result = self.mult.multiply({i: 1.0}, {j: 1.0})
                assert len(result) == 1, f"e{i}*e{j} must yield one element"

    def test_zero_input_yields_empty(self):
        assert self.mult.multiply({}, {1: 1.0}) == {}
        assert self.mult.multiply({1: 1.0}, {}) == {}


# ======================================================================
# 4. FAST RESOLVER (engine adapter)
# ======================================================================

class TestStandardResolver:
    def setup_method(self):
        self.resolver = StandardResolver()

    def test_implements_contract(self):
        assert isinstance(self.resolver, BasisProductResolver)

    def test_contract_is_abstract(self):
        with pytest.raises(TypeError):
            BasisProductResolver()

    def test_identity(self):
        assert self.resolver.resolve(0, 0) == (1, 0)

    def test_e1_squared(self):
        assert self.resolver.resolve(1, 1) == (-1, 0)

    def test_quaternion_ij(self):
        assert self.resolver.resolve(1, 2) == (1, 3)

    def test_quaternion_ji_anticommute(self):
        assert self.resolver.resolve(2, 1) == (-1, 3)

    def test_high_index(self):
        sign, idx = self.resolver.resolve(5000, 5000)
        assert (sign, idx) == (-1, 0)


# ======================================================================
# 5. FACADE / INTEGRATION
# ======================================================================

class TestExpressionMultiplier:
    def test_scalar_multiplication(self):
        assert multiply_expressions("2", "3") == "6"

    def test_basis_multiplication(self):
        assert multiply_expressions("e1", "e2") == "e3"

    def test_complex_square(self):
        assert multiply_expressions("1 + e1", "1 + e1") == "2e1"

    def test_high_index_no_ceiling(self):
        # Would explode a dense array; sparse handles it instantly.
        assert multiply_expressions("e5000", "e5000") == "-1"

    def test_coefficient_signs(self):
        assert multiply_expressions("2e1", "-3e2") == "-6e3"

    def test_multiply_by_one_is_identity(self):
        assert multiply_expressions("1", "3e1 + 2e2") == "3e1 + 2e2"

    def test_multiply_by_zero(self):
        assert multiply_expressions("0", "e1 + e2") == "0"

    def test_negative_times_negative(self):
        assert multiply_expressions("-e1", "-e1") == "-1"

    def test_class_and_function_agree(self):
        m = ExpressionMultiplier()
        assert m.multiply("e1", "e2") == multiply_expressions("e1", "e2")


# ======================================================================
# 6. CROSS-VALIDATION against build_table  (the authoritative check)
# ======================================================================

class TestCrossValidation:
    """Verify every wrapper layer against the engine's build_table."""

    @staticmethod
    def _expected_str(sign: int, idx: int) -> str:
        if idx == 0:
            return "1" if sign > 0 else "-1"
        return f"e{idx}" if sign > 0 else f"-e{idx}"

    @pytest.mark.parametrize("n", range(6))  # n = 0..5 → up to 32-D
    def test_resolver_matches_table(self, n):
        signs, indices = build_table("standard", n)
        dim = 1 << n
        resolver = StandardResolver()
        for i in range(dim):
            for j in range(dim):
                exp_sign, exp_idx = int(signs[i, j]), int(indices[i, j])
                got_sign, got_idx = resolver.resolve(i, j)
                assert got_sign == exp_sign, f"n={n}: e{i}*e{j} sign"
                assert got_idx == exp_idx, f"n={n}: e{i}*e{j} index"

    @pytest.mark.parametrize("n", range(6))
    def test_sparse_multiplier_matches_table(self, n):
        signs, indices = build_table("standard", n)
        dim = 1 << n
        mult = SparseMultiplier(StandardResolver())
        for i in range(dim):
            for j in range(dim):
                exp_sign, exp_idx = int(signs[i, j]), int(indices[i, j])
                result = mult.multiply({i: 1.0}, {j: 1.0})
                assert len(result) == 1, f"n={n}: e{i}*e{j} not single element"
                got_idx = next(iter(result))
                got_coeff = result[got_idx]
                assert got_idx == exp_idx
                assert got_coeff == float(exp_sign)

    @pytest.mark.parametrize("n", range(6))
    def test_expression_matches_table(self, n):
        signs, indices = build_table("standard", n)
        dim = 1 << n
        for i in range(dim):
            for j in range(dim):
                exp_sign, exp_idx = int(signs[i, j]), int(indices[i, j])
                got = multiply_expressions(f"e{i}", f"e{j}")
                assert got == self._expected_str(exp_sign, exp_idx), \
                    f"n={n}: e{i}*e{j} -> {got}"


# ======================================================================
# 7. EDGE CASES
# ======================================================================

class TestEdgeCases:
    def test_quaternion_associativity(self):
        # Quaternions (n=2) are associative: (e1*e2)*e3 == e1*(e2*e3)
        left = multiply_expressions(multiply_expressions("e1", "e2"), "e3")
        right = multiply_expressions("e1", multiply_expressions("e2", "e3"))
        assert left == right

    def test_difference_of_squares(self):
        # (e1 + e2)(e1 - e2) = e1² - e1e2 + e2e1 - e2²
        #                    = -1 - e3 - e3 + 1 = -2e3
        assert multiply_expressions("e1 + e2", "e1 - e2") == "-2e3"

    def test_full_cancellation(self):
        # (1 + e1)(1 - e1) = 1 - e1 + e1 - e1² = 1 + 1 = 2
        assert multiply_expressions("1 + e1", "1 - e1") == "2"

    def test_result_can_have_more_terms(self):
        # (e1 + e2)(e1 + e2) = e1² + e1e2 + e2e1 + e2²
        #                     = -1 + e3 - e3 - 1 = -2
        assert multiply_expressions("e1 + e2", "e1 + e2") == "-2"

    def test_scalar_distributes(self):
        assert multiply_expressions("2", "e1 + e2") == "2e1 + 2e2"

    def test_repeated_multiplication_accumulates(self):
        # e1 * e1 * e1 = (e1*e1)*e1 = (-1)*e1 = -e1
        step1 = multiply_expressions("e1", "e1")   # "-1"
        step2 = multiply_expressions(step1, "e1")  # "-e1"
        assert step2 == "-e1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])