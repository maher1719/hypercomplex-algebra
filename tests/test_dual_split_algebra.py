"""
Dual-Split algebra test suite for hypercomplex_algebra.

Dual over a SPLIT parent: base products follow split sign rules,
nilpotency (ε²=0) is identical to dual ordinary.

Coverage:
    1. DualSplitResolver core    — split base products, nilpotency, dim pinning
    2. Split-vs-standard contrast— proves base products use split (not standard) rules
    3. Table cross-validation    — resolver vs build_table("dual_split", n)
    4. Dual-split expressions    — end-to-end string multiplication
    5. Facade                    — multiply_dual_split_expressions / multiply_many
    6. Edge cases                — dim mixing, isolation, zero handling

Run:
    pytest tests/test_dual_split_algebra.py -v
"""

import pytest

from hypercomplex_engine import build_table

from hypercomplex_algebra import (
    DualSplitResolver,
    ExpressionMultiplier,
    multiply_dual_split_expressions,
    multiply_many_dual_split_expressions,
    multiply_dual_expressions,
    multiply_split_expressions,
)


# ======================================================================
# 1. DUAL-SPLIT RESOLVER — core logic
# ======================================================================

class TestDualSplitResolver:
    def setup_method(self):
        self.r = DualSplitResolver(dim=2)

    def test_split_base_product(self):
        # In split dim=2, e2 is a split element: e2*e2 = +e0
        assert self.r.resolve(2, 0, 2, 0) == (1, 0, 0)

    def test_parent_element_squares_negative(self):
        # e1 is in the standard parent: e1*e1 = -e0
        assert self.r.resolve(1, 0, 1, 0) == (-1, 0, 0)

    def test_base_times_dual(self):
        # e1 * (ε·e2) = ε·(e1*e2)
        sign, idx, _ = self.r.resolve(1, 0, 2, 0)
        _, expected_idx, _ = self.r.resolve(1, 0, 2, 0)
        result = self.r.resolve(1, 0, 2, 1)
        assert result == (sign, idx, 1)

    def test_nilpotent(self):
        # (ε·e1) * (ε·e2) = 0
        assert self.r.resolve(1, 1, 2, 1) == (0, 0, 0)

    def test_eps_squared(self):
        assert self.r.resolve(0, 1, 0, 1) == (0, 0, 0)

    def test_epsilon_commutes(self):
        # e1 * ε = ε * e1
        assert self.r.resolve(1, 0, 0, 1) == self.r.resolve(0, 1, 1, 0)

    def test_out_of_range_raises(self):
        # dim=2 → limit 4, so index 4 is out of range
        with pytest.raises(ValueError):
            self.r.resolve(4, 0, 1, 0)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            self.r.resolve(-1, 0, 2, 0)

    def test_dim_property(self):
        assert self.r.dim == 2

    def test_negative_dim_raises(self):
        with pytest.raises(ValueError):
            DualSplitResolver(dim=-1)


# ======================================================================
# 2. SPLIT vs STANDARD contrast — proves base products use split rules
# ======================================================================

class TestDualSplitVsDualOrdinary:
    def test_e1_square_differs(self):
        # dual ordinary (standard parent): e1*e1 = -1
        # dual_split dim=1 (split parent): e1*e1 = +1
        assert multiply_dual_expressions("e1", "e1") == "-1"
        assert multiply_dual_split_expressions("e1", "e1", dim=1) == "1"

    def test_split_element_square_positive(self):
        # In dual_split dim=2, e2 is a split element: e2*e2 = +1
        assert multiply_dual_split_expressions("e2", "e2", dim=2) == "1"

    def test_nilpotency_same_in_both(self):
        # ε²=0 regardless of parent algebra
        assert multiply_dual_expressions("eps", "eps") == "0"
        assert multiply_dual_split_expressions("eps", "eps", dim=2) == "0"

    def test_epsilon_commutes_in_both(self):
        assert multiply_dual_expressions("e1", "eps") == "eps_e1"
        assert multiply_dual_split_expressions("e1", "eps", dim=2) == "eps_e1"


# ======================================================================
# 3. TABLE CROSS-VALIDATION
# ======================================================================

class TestDualSplitCrossValidation:
    @pytest.mark.parametrize("n", range(1, 4))
    def test_resolver_matches_dual_split_table(self, n):
        signs, indices, eps_table = build_table("dual_split", n)
        dim = signs.shape[0]
        half = dim // 2
        resolver = DualSplitResolver(dim=n)

        for gi in range(dim):
            i, eps_i = gi % half, gi // half
            for gj in range(dim):
                j, eps_j = gj % half, gj // half

                got_sign, got_idx, got_eps = resolver.resolve(i, eps_i, j, eps_j)
                exp_sign = int(signs[gi, gj])

                if exp_sign == 0:
                    assert got_sign == 0, (
                        f"dual_split n={n}: ({i},{eps_i})*({j},{eps_j}) should be zero"
                    )
                else:
                    assert got_sign == exp_sign
                    assert got_idx == int(indices[gi, gj])
                    assert got_eps == int(eps_table[gi, gj])

    @pytest.mark.parametrize("n", range(1, 4))
    def test_base_products_match_split_table(self, n):
        # Base×base in dual_split must match the split parent algebra
        signs, indices = build_table("split", n)
        dim = 1 << n
        resolver = DualSplitResolver(dim=n)
        for i in range(dim):
            for j in range(dim):
                got_sign, got_idx, got_eps = resolver.resolve(i, 0, j, 0)
                assert got_sign == int(signs[i, j])
                assert got_idx == int(indices[i, j])
                assert got_eps == 0


# ======================================================================
# 4. DUAL-SPLIT EXPRESSIONS — end-to-end
# ======================================================================

class TestDualSplitExpressions:
    def setup_method(self):
        self.m = ExpressionMultiplier(kind="dual_split", dim=1)

    def test_eps_squared_is_zero(self):
        assert self.m.multiply("eps", "eps") == "0"

    def test_split_complex_square(self):
        # split-complex: e1*e1 = +1
        assert self.m.multiply("e1", "e1") == "1"

    def test_split_binomial_square(self):
        # (1+eps)² = 1 + 2eps + eps² = 1 + 2eps  (nilpotency unchanged)
        assert self.m.multiply("1 + eps", "1 + eps") == "1 + 2eps"

    def test_split_element_binomial(self):
        # (e1 + eps)² in split-complex:
        # e1² + e1·eps + eps·e1 + eps² = +1 + eps_e1 + eps_e1 + 0 = 1 + 2eps_e1
        assert self.m.multiply("e1 + eps", "e1 + eps") == "1 + 2eps_e1"

    def test_kind_property(self):
        assert self.m.kind == "dual_split"


# ======================================================================
# 5. DUAL-SPLIT FACADE
# ======================================================================

class TestDualSplitFacade:
    def test_me_split_base(self):
        assert multiply_dual_split_expressions("e1", "e1", dim=1) == "1"

    def test_me_nilpotent(self):
        assert multiply_dual_split_expressions("eps", "eps", dim=2) == "0"

    def test_mm_basic(self):
        assert multiply_many_dual_split_expressions(["e1", "e1"], dim=1) == "1"

    def test_mm_nilpotent_short_circuit(self):
        # eps*eps = 0, then 0*e1 = 0
        assert multiply_many_dual_split_expressions(["eps", "eps", "e1"], dim=2) == "0"

    def test_mm_single(self):
        assert multiply_many_dual_split_expressions(["e1 + eps"], dim=1) == "e1 + eps"

    def test_mm_empty_raises(self):
        with pytest.raises(ValueError):
            multiply_many_dual_split_expressions([], dim=2)

    def test_requires_dim(self):
        with pytest.raises(ValueError):
            ExpressionMultiplier(kind="dual_split")  # no dim


# ======================================================================
# 6. EDGE CASES
# ======================================================================

class TestDualSplitEdgeCases:
    def test_dim_mixing_raises(self):
        # dim=2 resolver rejects index that needs dim=3
        with pytest.raises(ValueError):
            multiply_dual_split_expressions("e5", "e1", dim=2)

    def test_zero_times_dual_split(self):
        assert multiply_dual_split_expressions("0", "e1 + eps", dim=2) == "0"

    def test_isolation_from_dual_ordinary(self):
        # Same expression, different parent → different base result
        assert multiply_dual_expressions("e1", "e1") == "-1"           # standard parent
        assert multiply_dual_split_expressions("e1", "e1", dim=1) == "1"  # split parent

    def test_large_index_within_dim(self):
        # dim=3 → limit 8, so e7 is valid
        assert multiply_dual_split_expressions("e7", "eps", dim=3) == "eps_e7"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])