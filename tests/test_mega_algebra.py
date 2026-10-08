"""
MEGA TEST — adversarial + comprehensive suite for hypercomplex_algebra.

The goal is to BREAK everything: parser, formatter, resolvers, sparse
multiplier, facade, and the MM/ME composition. Randomized property tests
use fixed seeds for reproducibility (no external dependency).

Sections:
    1. Parser attack          — malformed / hostile inputs
    2. Formatter attack       — degenerate coefficient dicts
    3. Resolver attack        — negatives, out-of-range, huge indices
    4. Sparse multiplier      — empty, near-zero, explosion bounds
    5. Facade attack          — empty MM, dim violations, bad kinds
    6. Property-based         — identity, zero, distributivity, associativity, round-trip
    7. Cross-validation stress— random pairs vs build_table
    8. Engine zero rejection  — (0,0) input raises

Run:
    pytest tests/test_mega_algebra.py -v
"""

import random

import pytest

from hypercomplex_engine import FastStandard, build_table

from hypercomplex_algebra import (
    ElementFormatter,
    ExpressionMultiplier,
    ElementParser,
    SparseMultiplier,
    SplitResolver,
    StandardResolver,
    multiply_expressions,
    multiply_many_expressions,
    multiply_many_split_expressions,
    multiply_split_expressions,
)


# ======================================================================
# Helpers
# ======================================================================

def _random_sparse(rng, max_index=8, max_terms=4):
    d = {}
    for _ in range(rng.randint(1, max_terms)):
        idx = rng.randint(0, max_index)
        coeff = round(rng.uniform(-5, 5), 2)
        if abs(coeff) < 0.01:
            coeff = 1.0
        d[idx] = d.get(idx, 0.0) + coeff
    return {k: v for k, v in d.items() if abs(v) > 1e-15}


def _add_dicts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0.0) + v
    return {k: v for k, v in result.items() if abs(v) > 1e-15}


def _dicts_close(a, b, tol=1e-9):
    if set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) < tol for k in a)


# ======================================================================
# 1. PARSER ATTACK
# ======================================================================

class TestParserAttack:
    def setup_method(self):
        self.parser = ElementParser()

    @pytest.mark.parametrize("bad", [
        "",              # empty
        "   ",           # whitespace only
        "e",             # bare e
        "1e",            # trailing e
        "abc",           # non-numeric
        "1e-3 e1",       # scientific notation
        "e-3",           # negative index / sci-notation hybrid
        "1.2.3e4",       # malformed float
    ])
    def test_malformed_raises(self, bad):
        with pytest.raises(ValueError):
            self.parser.parse(bad)

    def test_scientific_notation_clear_message(self):
        with pytest.raises(ValueError, match="scientific"):
            self.parser.parse("1e-3")

    def test_lone_sign_is_rejected(self):
        # A lone '+' or '-' has no term after it — that's a malformed
        # expression, not zero. Silently accepting it would hide a typo.
        with pytest.raises(ValueError):
            self.parser.parse("+")
        with pytest.raises(ValueError):
            self.parser.parse("-")

    def test_redundant_or_trailing_signs_are_rejected(self):
        with pytest.raises(ValueError):
            self.parser.parse("++e1")
        with pytest.raises(ValueError):
            self.parser.parse("e1++e2")
        with pytest.raises(ValueError):
            self.parser.parse("e1+")

    def test_huge_index_parses(self):
        assert self.parser.parse("e999999") == {999999: 1.0}

    def test_leading_zeros(self):
        assert self.parser.parse("e007") == {7: 1.0}

    def test_1e3_is_basis_not_scientific(self):
        # '1e3' means 1*e3 (basis element), NOT 1000.
        assert self.parser.parse("1e3") == {3: 1.0}

    def test_many_terms(self):
        expr = " + ".join(f"e{i}" for i in range(60))
        assert len(self.parser.parse(expr)) == 60

    def test_combines_duplicate_indices(self):
        assert self.parser.parse("e2 + e2 + e2") == {2: 3.0}


# ======================================================================
# 2. FORMATTER ATTACK
# ======================================================================

class TestFormatterAttack:
    def setup_method(self):
        self.formatter = ElementFormatter()

    def test_empty_is_zero(self):
        assert self.formatter.format({}) == "0"

    def test_all_near_zero_pruned(self):
        assert self.formatter.format({1: 1e-20, 2: -1e-20}) == "0"

    def test_negative_zero_pruned(self):
        assert self.formatter.format({0: -0.0}) == "0"

    def test_huge_index(self):
        assert self.formatter.format({999999: 1.0}) == "e999999"

    def test_unsorted_input_gets_sorted(self):
        out = self.formatter.format({5: 1.0, 1: 2.0, 3: -1.0})
        assert out == "2e1 - e3 + e5"


# ======================================================================
# 3. RESOLVER ATTACK
# ======================================================================

class TestResolverAttack:
    def test_standard_rejects_negative(self):
        r = StandardResolver()
        with pytest.raises(ValueError):
            r.resolve(-1, 5)
        with pytest.raises(ValueError):
            r.resolve(5, -1)

    def test_split_rejects_negative(self):
        r = SplitResolver(dim=3)
        with pytest.raises(ValueError):
            r.resolve(-1, 5)

    def test_split_rejects_out_of_range(self):
        r = SplitResolver(dim=3)  # limit 8
        with pytest.raises(ValueError):
            r.resolve(8, 1)
        with pytest.raises(ValueError):
            r.resolve(1, 8)

    def test_split_negative_dim_rejected(self):
        with pytest.raises(ValueError):
            SplitResolver(dim=-1)

    def test_standard_huge_index_works(self):
        # Standard is dimension-independent; huge indices are fine.
        r = StandardResolver()
        sign, idx = r.resolve(10**6, 10**6 + 1)
        assert sign in (1, -1)
        assert idx == (10**6) ^ (10**6 + 1)


# ======================================================================
# 4. SPARSE MULTIPLIER ATTACK
# ======================================================================

class TestSparseAttack:
    def setup_method(self):
        self.mult = SparseMultiplier(StandardResolver())

    def test_empty_inputs(self):
        assert self.mult.multiply({}, {1: 1.0}) == {}
        assert self.mult.multiply({1: 1.0}, {}) == {}
        assert self.mult.multiply({}, {}) == {}

    def test_near_zero_skipped(self):
        assert self.mult.multiply({1: 1e-20}, {2: 1.0}) == {}

    def test_single_element_product_property(self):
        # e_i * e_j ALWAYS yields exactly one basis element.
        for i in range(32):
            for j in range(32):
                assert len(self.mult.multiply({i: 1.0}, {j: 1.0})) == 1

    def test_result_term_count_bounded(self):
        # k1 * k2 is the ceiling before combining like indices.
        a = {i: 1.0 for i in range(10)}
        b = {i: 1.0 for i in range(10)}
        assert len(self.mult.multiply(a, b)) <= 100


# ======================================================================
# 5. FACADE ATTACK
# ======================================================================

class TestFacadeAttack:
    def test_empty_mm_raises(self):
        with pytest.raises(ValueError):
            multiply_many_expressions([])

    def test_split_dim_violation_raises(self):
        with pytest.raises(ValueError):
            multiply_split_expressions("e5", "e1", dim=2)
        with pytest.raises(ValueError):
            multiply_many_split_expressions(["e5"], dim=2)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            ExpressionMultiplier(kind="bogus")

    def test_split_requires_dim(self):
        with pytest.raises(ValueError):
            ExpressionMultiplier(kind="split")

    def test_zero_absorbing_in_mm(self):
        assert multiply_many_expressions(["e1", "0", "e2", "e3"]) == "0"

    def test_single_mm_identity(self):
        assert multiply_many_expressions(["2e1 + e2"]) == "2e1 + e2"

    def test_standard_and_split_isolated(self):
        # Same input, different algebra -> different result, no leakage.
        assert multiply_expressions("e1", "e1") == "-1"
        assert multiply_split_expressions("e1", "e1", dim=1) == "1"


# ======================================================================
# 6. PROPERTY-BASED (randomized, fixed seeds)
# ======================================================================

class TestProperties:
    def setup_method(self):
        self.rng = random.Random(12345)
        self.mult = SparseMultiplier(StandardResolver())
        self.parser = ElementParser()
        self.formatter = ElementFormatter()

    def test_multiplicative_identity(self):
        for _ in range(100):
            a = _random_sparse(self.rng)
            assert _dicts_close(self.mult.multiply(a, {0: 1.0}), a)
            assert _dicts_close(self.mult.multiply({0: 1.0}, a), a)

    def test_zero_absorbing(self):
        for _ in range(100):
            a = _random_sparse(self.rng)
            assert self.mult.multiply(a, {}) == {}
            assert self.mult.multiply({}, a) == {}

    def test_distributivity(self):
        for _ in range(100):
            a = _random_sparse(self.rng)
            b = _random_sparse(self.rng)
            c = _random_sparse(self.rng)
            left = self.mult.multiply(a, _add_dicts(b, c))
            right = _add_dicts(self.mult.multiply(a, b), self.mult.multiply(a, c))
            assert _dicts_close(left, right)

    def test_quaternion_associativity(self):
        # Indices 0..3 (quaternions) are associative.
        r = StandardResolver()
        for _ in range(200):
            i, j, k = (self.rng.randint(0, 3) for _ in range(3))
            s1, ij = r.resolve(i, j)
            s2, left_idx = r.resolve(ij, k)
            t1, jk = r.resolve(j, k)
            t2, right_idx = r.resolve(i, jk)
            assert (s1 * s2, left_idx) == (t1 * t2, right_idx)

    def test_parse_format_roundtrip(self):
        for _ in range(100):
            d = _random_sparse(self.rng)
            reparsed = self.parser.parse(self.formatter.format(d))
            assert _dicts_close(reparsed, d)

    def test_mm_left_fold_matches_nested_me(self):
        m = ExpressionMultiplier(kind="standard")
        for _ in range(50):
            exprs = [self.formatter.format(_random_sparse(self.rng, max_index=3))
                     for _ in range(4)]
            mm = m.multiply_many(exprs)
            nested = m.multiply(m.multiply(m.multiply(exprs[0], exprs[1]),
                                           exprs[2]), exprs[3])
            assert mm == nested


# ======================================================================
# 7. CROSS-VALIDATION STRESS vs build_table
# ======================================================================

class TestCrossValidationStress:
    def test_standard_random_pairs_match_table(self):
        rng = random.Random(999)
        resolver = StandardResolver()
        for n in range(6):
            signs, indices = build_table("standard", n)
            dim = 1 << n
            for _ in range(300):
                i, j = rng.randrange(dim), rng.randrange(dim)
                assert resolver.resolve(i, j) == (int(signs[i, j]), int(indices[i, j]))

    def test_split_random_pairs_match_table(self):
        rng = random.Random(888)
        for n in range(1, 5):
            signs, indices = build_table("split", n)
            dim = 1 << n
            resolver = SplitResolver(dim=n)
            for _ in range(300):
                i, j = rng.randrange(dim), rng.randrange(dim)
                assert resolver.resolve(i, j) == (int(signs[i, j]), int(indices[i, j]))


# ======================================================================
# 8. ENGINE ZERO REJECTION  — (0,0) input is meaningless
# ======================================================================

class TestEngineZeroRejection:
    def test_engine_rejects_zero_tuple_input(self):
        engine = FastStandard()
        with pytest.raises(Exception):
            engine.multiply((0, 0), (1, 1))
        with pytest.raises(Exception):
            engine.multiply((1, 1), (0, 0))
    def test_out_of_range_caught_even_after_a_zero_short_circuits_the_fold(self):
        # 'e99' is out of range for dim=2. Every expression is validated up
        # front, before any multiplication happens, so this is caught even
        # though a zero earlier in the list would otherwise short-circuit
        # the fold before 'e99' is ever multiplied.
        with pytest.raises(ValueError):
            multiply_many_split_expressions(["e1", "0", "e99"], dim=2)

    def test_out_of_range_error_message_is_specific_not_generic(self):
        # The validation error must carry the resolver's real diagnosis
        # (which dim, which bound) rather than a generic placeholder.
        try:
            multiply_many_split_expressions(["e1", "e99"], dim=2)
            assert False, "should have raised"
        except ValueError as e:
            assert "dim=2" in str(e)
            assert "0..3" in str(e)

    def test_wrapper_never_produces_zero_tuple(self):
        # Our wrapper uses multiply_indices (ints), never (sign,index) tuples,
        # so it can never feed a (0,0) tuple to the engine.
        r = StandardResolver()
        for i in range(16):
            for j in range(16):
                sign, idx = r.resolve(i, j)
                assert sign in (1, -1)  # never 0 for standard/split basis


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])