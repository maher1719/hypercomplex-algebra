"""
Dual algebra test suite for hypercomplex_algebra.

Coverage:
    1. DualResolver core       — nilpotency, base/dual products, ε-commutation
    2. Parent consistency      — base×base matches standard build_table
    3. Table cross-validation  — resolver vs build_table("dual", n)
    4. Dual parser             — eps_e{n}, eps shorthand, coefficients
    5. Dual formatter          — output formatting, eps shorthand
    6. Round-trip              — parse(format(d)) == d
    7. Dual expressions        — end-to-end string multiplication
    8. Dual facade             — multiply_dual_expressions, multiply_many_dual_expressions
    9. Edge cases              — zero, large indices, isolation from standard

Run:
    pytest tests/test_dual_algebra.py -v
"""

import pytest

from hypercomplex_engine import build_table

from hypercomplex_algebra import (
    DualStandardResolver,
    DualElementParser,
    DualElementFormatter,
    ExpressionMultiplier,
    multiply_dual_expressions,
    multiply_many_dual_expressions,
    multiply_expressions,
)


# ======================================================================
# 1. DUAL RESOLVER — core algebraic logic
# ======================================================================

class TestDualResolverCore:
    def setup_method(self):
        self.r = DualStandardResolver()

    # --- base × base (matches standard parent) ---
    def test_base_times_base(self):
        # e1 * e2 = e3 (standard quaternion product)
        assert self.r.resolve(1, 0, 2, 0) == (1, 3, 0)

    def test_base_times_base_anticommute(self):
        # e2 * e1 = -e3
        assert self.r.resolve(2, 0, 1, 0) == (-1, 3, 0)

    def test_base_squared(self):
        # e1 * e1 = -e0
        assert self.r.resolve(1, 0, 1, 0) == (-1, 0, 0)

    def test_identity(self):
        assert self.r.resolve(0, 0, 5, 0) == (1, 5, 0)
        assert self.r.resolve(5, 0, 0, 0) == (1, 5, 0)

    # --- base × dual and dual × base ---
    def test_base_times_dual(self):
        # e1 * (ε·e2) = ε·(e1*e2) = ε·e3
        assert self.r.resolve(1, 0, 2, 1) == (1, 3, 1)

    def test_dual_times_base(self):
        # (ε·e1) * e2 = ε·(e1*e2) = ε·e3
        assert self.r.resolve(1, 1, 2, 0) == (1, 3, 1)

    def test_epsilon_commutes_with_basis(self):
        # e1 * ε = ε * e1 = ε·e1  (ε is central)
        assert self.r.resolve(1, 0, 0, 1) == (1, 1, 1)
        assert self.r.resolve(0, 1, 1, 0) == (1, 1, 1)

    # --- nilpotency (ε² = 0) ---
    def test_nilpotent_eps_eps(self):
        # ε * ε = 0
        assert self.r.resolve(0, 1, 0, 1) == (0, 0, 0)

    def test_nilpotent_dual_dual(self):
        # (ε·e1) * (ε·e2) = ε²·(e1*e2) = 0
        assert self.r.resolve(1, 1, 2, 1) == (0, 0, 0)

    def test_nilpotent_same_dual(self):
        # (ε·e1) * (ε·e1) = 0
        assert self.r.resolve(1, 1, 1, 1) == (0, 0, 0)

    # --- validation ---
    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            self.r.resolve(-1, 0, 2, 0)
        with pytest.raises(ValueError):
            self.r.resolve(1, 0, -2, 0)

    def test_single_element_or_zero_property(self):
        # Product of two dual basis elements is a single dual basis element or zero.
        for i in range(8):
            for ei in (0, 1):
                for j in range(8):
                    for ej in (0, 1):
                        sign, idx, eps = self.r.resolve(i, ei, j, ej)
                        if sign == 0:
                            assert (idx, eps) == (0, 0)
                        else:
                            assert sign in (1, -1)


# ======================================================================
# 2. PARENT CONSISTENCY — base×base matches the standard algebra
# ======================================================================

class TestDualParentConsistency:
    @pytest.mark.parametrize("n", range(1, 5))
    def test_base_products_match_standard_table(self, n):
        signs, indices = build_table("standard", n)
        dim = 1 << n
        resolver = DualStandardResolver()
        for i in range(dim):
            for j in range(dim):
                got_sign, got_idx, got_eps = resolver.resolve(i, 0, j, 0)
                assert got_sign == int(signs[i, j]), f"n={n}: e{i}*e{j} sign"
                assert got_idx == int(indices[i, j]), f"n={n}: e{i}*e{j} index"
                assert got_eps == 0, f"n={n}: e{i}*e{j} must have eps=0"


# ======================================================================
# 3. TABLE CROSS-VALIDATION — resolver vs build_table("dual", n)
# ======================================================================

class TestDualTableCrossValidation:
    """Verify the dual resolver against the engine's authoritative dual table.

    Interpretation: build_table("dual", n) returns arrays indexed by GLOBAL
    dual indices. The first half of indices are base elements e_i, the second
    half are dual elements ε·e_i. Results are stored as (sign, local_index, eps).
    """

    @pytest.mark.parametrize("n", range(1, 4))
    def test_resolver_matches_dual_table(self, n):
        signs, indices, eps_table = build_table("dual", n)
        dim = signs.shape[0]
        half = dim // 2
        resolver = DualStandardResolver()

        for gi in range(dim):
            i, eps_i = gi % half, gi // half
            for gj in range(dim):
                j, eps_j = gj % half, gj // half

                got_sign, got_idx, got_eps = resolver.resolve(i, eps_i, j, eps_j)
                exp_sign = int(signs[gi, gj])

                if exp_sign == 0:
                    # Nilpotent zero product
                    assert got_sign == 0, (
                        f"dual n={n}: ({i},eps={eps_i})*({j},eps={eps_j}) "
                        f"expected zero, got sign={got_sign}"
                    )
                else:
                    assert got_sign == exp_sign, (
                        f"dual n={n}: ({i},{eps_i})*({j},{eps_j}) sign"
                    )
                    assert got_idx == int(indices[gi, gj]), (
                        f"dual n={n}: ({i},{eps_i})*({j},{eps_j}) index"
                    )
                    assert got_eps == int(eps_table[gi, gj]), (
                        f"dual n={n}: ({i},{eps_i})*({j},{eps_j}) eps"
                    )


# ======================================================================
# 4. DUAL PARSER
# ======================================================================

class TestDualParser:
    def setup_method(self):
        self.parser = DualElementParser()

    def test_parse_base_element(self):
        assert self.parser.parse("e3") == {(3, 0): 1.0}

    def test_parse_dual_element(self):
        assert self.parser.parse("eps_e3") == {(3, 1): 1.0}

    def test_parse_eps_shorthand(self):
        # bare "eps" means ε·e₀
        assert self.parser.parse("eps") == {(0, 1): 1.0}

    def test_parse_coefficient_dual(self):
        assert self.parser.parse("3eps_e2") == {(2, 1): 3.0}

    def test_parse_negative_dual(self):
        assert self.parser.parse("-eps_e1") == {(1, 1): -1.0}

    def test_parse_mixed_base_and_dual(self):
        result = self.parser.parse("e1 + 2eps_e3")
        assert result == {(1, 0): 1.0, (3, 1): 2.0}

    def test_parse_scalar(self):
        assert self.parser.parse("5") == {(0, 0): 5.0}

    def test_parse_combines_like_terms(self):
        assert self.parser.parse("eps_e1 + eps_e1") == {(1, 1): 2.0}

    def test_base_and_dual_same_index_are_distinct(self):
        # e2 and eps_e2 are DIFFERENT basis elements
        result = self.parser.parse("e2 + eps_e2")
        assert result == {(2, 0): 1.0, (2, 1): 1.0}

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError):
            self.parser.parse("")


# ======================================================================
# 5. DUAL FORMATTER
# ======================================================================

class TestDualFormatter:
    def setup_method(self):
        self.formatter = DualElementFormatter()
        self.parser = DualElementParser()

    def test_format_base(self):
        assert self.formatter.format({(3, 0): 1.0}) == "e3"

    def test_format_dual(self):
        assert self.formatter.format({(3, 1): 1.0}) == "eps_e3"

    def test_format_eps_shorthand(self):
        # ε·e₀ formats as "eps"
        assert self.formatter.format({(0, 1): 1.0}) == "eps"

    def test_format_coefficient_dual(self):
        assert self.formatter.format({(2, 1): 3.0}) == "3eps_e2"

    def test_format_negative_dual(self):
        assert self.formatter.format({(1, 1): -1.0}) == "-eps_e1"

    def test_format_scalar(self):
        assert self.formatter.format({(0, 0): 5.0}) == "5"

    def test_format_mixed(self):
        result = self.formatter.format({(1, 0): 1.0, (3, 1): 2.0})
        assert result == "e1 + 2eps_e3"

    def test_format_empty_is_zero(self):
        assert self.formatter.format({}) == "0"
    # TestDualFormatter
    def test_format_mixed(self):
        # base first, then dual
        result = self.formatter.format({(1, 0): 1.0, (3, 1): 2.0})
        assert result == "e1 + 2eps_e3"   # unchanged here (already base-first)

    def test_format_base_and_dual_interleaved_input(self):
        # even if input alternates, output groups base then dual
        result = self.formatter.format({(2, 1): 1.0, (1, 0): 1.0, (3, 0): 1.0})
        assert result == "e1 + e3 + eps_e2"

    # TestDualRoundTrip — add an interleaved case
    @pytest.mark.parametrize("expr", [
        "e1 + 2eps_e3",
        "eps",
        "3e2 - eps_e1",
        "5",
        "e1 + eps_e1",
        "2e0 - 3eps_e2 + eps",
        "e3 + eps_e1 + e1 + eps_e3",   # new: verifies regrouping
    ])
    def test_parse_format_roundtrip(self, expr):
        parsed = self.parser.parse(expr)
        formatted = self.formatter.format(parsed)
        reparsed = self.parser.parse(formatted)
        assert reparsed == parsed
    def test_parsed_dual_elements(self):
        exp="2eps_e3 + 1 - e3 + eps + e23 - 12eps_e1 + e4 - e231"
        parsed_exp="1 - e3 + e4 + e23 - e231 + eps - 12eps_e1 + 2eps_e3"
        try_parsed=self.parser.parse(exp)
        try_parsed_formatted= self.formatter.format(try_parsed)
        try_reparsed=self.parser.parse(try_parsed_formatted)
        assert(parsed_exp==self.formatter.format(try_parsed))
        assert(parsed_exp==self.formatter.format(try_reparsed))


# ======================================================================
# 6. ROUND-TRIP — parse(format(d)) == d
# ======================================================================

class TestDualRoundTrip:
    def setup_method(self):
        self.parser = DualElementParser()
        self.formatter = DualElementFormatter()

    @pytest.mark.parametrize("expr", [
        "e1 + 2eps_e3",
        "eps",
        "3e2 - eps_e1",
        "5",
        "e1 + eps_e1",
        "2e0 - 3eps_e2 + eps",
    ])
    def test_parse_format_roundtrip(self, expr):
        parsed = self.parser.parse(expr)
        formatted = self.formatter.format(parsed)
        reparsed = self.parser.parse(formatted)
        assert reparsed == parsed


# ======================================================================
# 7. DUAL EXPRESSIONS — end-to-end string multiplication
# ======================================================================

class TestDualExpressions:
    def setup_method(self):
        self.m = ExpressionMultiplier(kind="dual")

    def test_eps_squared_is_zero(self):
        assert self.m.multiply("eps", "eps") == "0"

    def test_base_times_eps(self):
        # e1 * ε = ε·e1
        assert self.m.multiply("e1", "eps") == "eps_e1"

    def test_eps_times_base_commutes(self):
        # ε * e1 = ε·e1
        assert self.m.multiply("eps", "e1") == "eps_e1"

    def test_dual_binomial_square(self):
        # (1 + eps)² = 1 + 2eps + eps² = 1 + 2eps
        assert self.m.multiply("1 + eps", "1 + eps") == "1 + 2eps"

    def test_square_cross_terms_cancel(self):
        # (e1 + eps_e2)² = e1² + e1·eps_e2 + eps_e2·e1 + eps_e2²
        #                = -1 + eps_e3 - eps_e3 + 0 = -1
        assert self.m.multiply("e1 + eps_e2", "e1 + eps_e2") == "-1"

    def test_square_cross_terms_add(self):
        # (e1 + eps)² = e1² + e1·eps + eps·e1 + eps²
        #             = -1 + eps_e1 + eps_e1 + 0 = -1 + 2eps_e1
        assert self.m.multiply("e1 + eps", "e1 + eps") == "-1 + 2eps_e1"

    def test_scalar_times_dual(self):
        assert self.m.multiply("3", "eps_e2") == "3eps_e2"

    def test_dual_times_dual_is_zero(self):
        assert self.m.multiply("eps_e1", "eps_e1") == "0"

    def test_kind_property(self):
        assert self.m.kind == "dual"


# ======================================================================
# 8. DUAL FACADE — convenience functions
# ======================================================================

class TestDualFacade:
    def test_multiply_dual_base(self):
        assert multiply_dual_expressions("e1", "e2") == "e3"

    def test_multiply_dual_eps(self):
        assert multiply_dual_expressions("eps", "e1") == "eps_e1"

    def test_multiply_dual_nilpotent(self):
        assert multiply_dual_expressions("eps", "eps") == "0"

    def test_multiply_many_dual(self):
        assert multiply_many_dual_expressions(["e1", "e2"]) == "e3"

    def test_multiply_many_dual_nilpotent(self):
        # eps * eps * e1 = 0 * e1 = 0
        assert multiply_many_dual_expressions(["eps", "eps", "e1"]) == "0"

    def test_multiply_many_dual_single(self):
        assert multiply_many_dual_expressions(["e1 + eps"]) == "e1 + eps"

    def test_multiply_many_dual_empty_raises(self):
        with pytest.raises(ValueError):
            multiply_many_dual_expressions([])


# ======================================================================
# 9. EDGE CASES
# ======================================================================

class TestDualEdgeCases:
    def test_zero_times_dual(self):
        assert multiply_dual_expressions("0", "e1 + eps") == "0"

    def test_dual_and_standard_agree_on_base(self):
        # Base products are identical in standard and dual (same parent)
        assert multiply_expressions("e1", "e1") == "-1"
        assert multiply_dual_expressions("e1", "e1") == "-1"
        assert multiply_expressions("e1", "e2") == "e3"
        assert multiply_dual_expressions("e1", "e2") == "e3"

    def test_dual_has_eps_structure_standard_does_not(self):
        # Dual can represent ε·e1; standard would treat "eps_e1" as invalid
        assert multiply_dual_expressions("e1", "eps") == "eps_e1"

    def test_large_index_dual(self):
        # Standard parent is dimension-independent, so large indices work
        assert multiply_dual_expressions("e100", "eps") == "eps_e100"

    def test_zero_short_circuit_in_mm(self):
        # Once product hits zero (nilpotent), MM short-circuits
        assert multiply_many_dual_expressions(["eps", "eps", "e5", "e7"]) == "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])