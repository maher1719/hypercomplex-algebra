# hypercomplex-algebra


**A mathematically rigorous, architecturally clean application layer for hypercomplex number systems.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+%20%7C%203.11%20%7C%203.13%20%7C%203.14-yellow.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-340+%20passed-brightgreen.svg)](https://github.com/maher1719/hypercomplex-algebra/actions)

> *"Juniors are your future seniors. I am a junior at abstract algebra and hypercomplex numbers; I had an idea, and I executed it. Just as you were taught by seniors yesterday, teach juniors today, and let the sacred message pass on."*

---

## Table of Contents

- [What This Is](#what-this-is)
- [The Math](#the-math)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Algebras](#supported-algebras)
- [Tensor Products](#tensor-products)
- [Unary Operations](#unary-operations)
  - [Conjugate](#conjugate)
  - [Norm](#norm)
  - [Inverse](#inverse)
- [Conjugate Types (Dual Algebras)](#conjugate-types-dual-algebras)
- [Changelog](#changelog)
- [Testing Philosophy](#testing-philosophy)
- [Licensing & Philosophy](#licensing--philosophy)
- [Author & References](#author--references)

---

## What This Is

`hypercomplex-algebra` is a **string-expression interface** for multiplying elements of hypercomplex number systems — standard Cayley-Dickson algebras, split algebras, dual algebras, dual-split algebras, and **tensor products of any combination** of these.

It is built on top of [`hypercomplex-engine`](https://github.com/maher1719/hypercomplex-engine) (the computational core), and provides:

- **A human-readable expression language** (`"3e1 + 2eps_e2"`, `"e[1,3,2]"`)
- **A clean parser/formatter pipeline** with zero silent faults
- **An adversarial test suite** (340+ tests) cross-validated against the engine's authoritative multiplication tables
- **A resolver/factory/facade architecture** that scales to arbitrary algebra combinations without code changes

This is not a toy. This is a **foundational computational package** designed to be relied upon, extended, and trusted.

---

## The Math

This library implements multiplication for the **Cayley-Dickson construction** and its variants:

| Algebra | Basis | Key Property |
|---|---|---|
| **Standard** (ℂ, ℍ, 𝕆, 𝕊…) | `e0, e1, e2, …` | `e_i² = -1`, anti-commutative |
| **Split** | `e0, e1, e2, …` | Some `e_i² = +1` (depends on dim) |
| **Dual** | `e0, e1, …, ε, ε·e1, …` | `ε² = 0` (nilpotent), ε commutes |
| **Dual-Split** | Same as dual | Dual over a split parent |
| **Tensor** | `e[i, j, k, …]` | Component-wise multiplication |

The sign laws for standard and split algebras are derived from the **OPMT (Ordered Pair Multiplication Theorem)**, which provides a closed-form O(1) computation for the sign of any basis product. See the [author's preprints](#author--references) for the full proof.

For dual algebras, the canonical form is `a + ε·b`, where `ε` is a nilpotent unit (`ε² = 0`) that commutes with all basis elements. The formatter outputs base terms first, then dual terms, matching the mathematical convention.

---

## Architecture

The package follows a **strict layered architecture** (onion/hexagonal principles) with clear separation of concerns:

```
hypercomplex_algebra/
├── facade.py                  ← Public API (multiply_expressions, etc.)
├── application/
│   └── multiplier.py          ← ExpressionMultiplier (orchestrator)
├── core/
│   ├── base/                  ← BasisProductResolver, SparseMultiplier
│   ├── dual/                  ← DualResolver, DualSparseMultiplier
│   └── tensor/                ← TensorResolver, TensorSparseMultiplier
├── adapters/
│   ├── factory.py             ← create_resolver(kind, dim)
│   ├── standard_resolver.py   ← wraps FastStandard
│   ├── split_resolver.py      ← wraps FastSplit
│   ├── dual_standard_resolver.py
│   └── dual_split_resolver.py
└── expression/
    ├── tokenizer.py           ← shared tokenization
    ├── base/                  ← ElementParser, ElementFormatter
    ├── dual/                  ← DualElementParser, DualElementFormatter
    └── tensor/                ← TensorElementParser, TensorElementFormatter
```

**Key design decisions:**

- **Resolver pattern**: All algebra-specific multiplication logic is encapsulated in resolvers. The multiplier, parser, and formatter are algebra-agnostic.
- **Factory pattern**: `create_resolver(kind, dim)` builds the correct resolver from a string, enabling runtime algebra selection.
- **Facade pattern**: Simple functions like `multiply_expressions(a, b)` hide all internal complexity.
- **No silent faults**: Every invalid index, malformed expression, or dimensional mismatch raises an explicit error. Zero tolerance for silent corruption.

---

## Installation

```bash
pip install hypercomplex-algebra
```

**Requirements:**
- Python ≥ 3.10
- `hypercomplex-engine` (installed automatically as a dependency)

---

## Quick Start

### Standard Algebras (Complex, Quaternions, Octonions…)

```python
from hypercomplex_algebra import multiply_expressions, multiply_many_expressions

# Quaternion multiplication
result = multiply_expressions("e1", "e2")
print(result)  # "e3"

# Octonion multiplication (non-associative, left-fold)
result = multiply_many_expressions(["e1", "e2", "e4"])
print(result)  # "e7" (or whatever the left-fold produces)

# With coefficients
result = multiply_expressions("3e1 + 2e2", "e1 - e2")
print(result)  # "-3 + e3"  (example; actual result depends on sign laws)
```

### Split Algebras

```python
from hypercomplex_algebra import multiply_split_expressions

# Split-complex: e1² = +1 (not -1)
result = multiply_split_expressions("e1", "e1", dim=1)
print(result)  # "1"

# Split-quaternions
result = multiply_split_expressions("e2", "e2", dim=2)
print(result)  # "1"  (e2 is a split element in dim=2)
```

### Dual Algebras

```python
from hypercomplex_algebra import multiply_dual_expressions

# Nilpotency: ε² = 0
result = multiply_dual_expressions("eps", "eps")
print(result)  # "0"

# ε commutes with basis elements
result = multiply_dual_expressions("e1", "eps")
print(result)  # "eps_e1"

# Binomial square: (1 + ε)² = 1 + 2ε
result = multiply_dual_expressions("1 + eps", "1 + eps")
print(result)  # "1 + 2eps"

# Canonical form: base terms first, then dual terms
result = multiply_dual_expressions("e1 + eps_e2", "e1 + eps_e2")
print(result)  # "-1"  (cross terms cancel due to anti-commutativity)
```

### Dual-Split Algebras

```python
from hypercomplex_algebra import multiply_dual_split_expressions

# Split base products with dual nilpotency
result = multiply_dual_split_expressions("e1", "e1", dim=1)
print(result)  # "1"  (split: e1² = +1)

result = multiply_dual_split_expressions("eps", "eps", dim=1)
print(result)  # "0"  (nilpotency unchanged)
```

### Tensor Products

```python
from hypercomplex_algebra import multiply_tensor_expressions

# C ⊗ H tensor product
slots = [("standard", None), ("standard", None)]
result = multiply_tensor_expressions("e[1,1]", "e[1,1]", slots)
print(result)  # "1"  (each slot: e1·e1 = -e0, signs multiply: (-1)(-1) = +1)

# Mixed algebra tensor: Split ⊗ Standard
slots = [("split", 2), ("standard", None)]
result = multiply_tensor_expressions("e[2,0]", "e[2,0]", slots)
print(result)  # "1"  (split e2² = +1, standard e0² = e0)

# 3-slot tensor
slots = [("standard", None), ("standard", None), ("standard", None)]
result = multiply_tensor_expressions("e[1,1,1]", "e[1,1,1]", slots)
print(result)  # "-1"  (three slots, each gives -1: (-1)³ = -1)
```

---

## Supported Algebras

| Kind | `kind` string | `dim` required? | Description |
|---|---|---|---|
| Standard | `"standard"` | No | ℂ, ℍ, 𝕆, 𝕊… (dimension-independent) |
| Split | `"split"` | **Yes** | Split-complex, split-quaternions, etc. |
| Dual | `"dual"` | No | Dual over standard parent |
| Dual-Split | `"dual_split"` | **Yes** | Dual over split parent |
| Tensor | `"tensor"` | Via `slots` | Tensor product of any combination |

---

## Tensor Products

The tensor product implementation uses **component-wise multiplication**: each slot resolves independently with its own algebra's rules, the signs multiply, and the result indices form a tuple.

```
e[1,3,2] · e[1,4,3]
  Slot 0: e1 · e1 = -e0  → sign -1, index 0
  Slot 1: e3 · e4 = ±e_j → sign s₁, index j
  Slot 2: e2 · e3 = ±e_k → sign s₂, index k

  Total sign = (-1) · s₁ · s₂
  Result key = (0, j, k)
```

The `e[i,j,k]` notation uses **0-indexed slots** (consistent with Python conventions). The coefficient is always **outside** the bracket: `-3e[1,2,0]`.

---


## Unary Operations

Beyond multiplication, the package provides **conjugate**, **norm**, and **inverse** operations
for all supported algebras. These are accessed as methods on the `ExpressionMultiplier` instance:

```python
m = ExpressionMultiplier(kind="dual")
m.conjugate("3 + 2e1 + eps")   # "3 - 2e1 + eps"
m.norm("3 + 2e1 + eps")        # "13 + 6eps"
m.inverse("3 + 2e1 + eps")     # inverse element as a string
```

### Conjugate

Negates basis elements according to the algebra's conjugation rules.

**Standard / Split algebras:**
```python
m = ExpressionMultiplier(kind="standard")
m.conjugate("1 + 2e1 + 3e2")   # "1 - 2e1 - 3e2"
```

**Dual algebras** support three conjugate modes (see table below):
```python
m = ExpressionMultiplier(kind="dual")
expr = "3 + 2e1 + eps + 4eps_e2"

m.conjugate(expr)                          # default: "hypercomplex"
m.conjugate(expr, mode="hypercomplex")     # "3 - 2e1 + eps - 4eps_e2"
m.conjugate(expr, mode="dual")             # "3 + 2e1 - eps - 4eps_e2"
m.conjugate(expr, mode="total")            # "3 - 2e1 - eps + 4eps_e2"
```

### Norm

Computes $N(x) = x \cdot x^*$ using the **Hypercomplex Conjugate**.

- **Standard / Split:** returns a real scalar string (e.g. `"14"`).
- **Dual:** returns a **dual scalar** string (e.g. `"13 + 6eps"`), since the norm of a
  dual number is itself a dual number.

```python
m = ExpressionMultiplier(kind="standard")
m.norm("1 + 2e1 + 3e2")    # "14"  (= 1² + 2² + 3²)

m = ExpressionMultiplier(kind="dual")
m.norm("3 + 2e1 + eps")    # "13 + 6eps"
```

> **Note:** The `"dual"` conjugate mode is **rejected** for `norm()` and `inverse()` because it
> does not collapse the product to a scalar. Only `"hypercomplex"` (default) and `"total"` are
> valid modes for these operations.

### Inverse

Computes $x^{-1} = x^* / N(x)$. The conjugate mode must match the norm's mode so that
$x \cdot x^{-1} = 1$ holds.

For dual algebras, division by the dual scalar norm $(A + B\varepsilon)$ uses the formula:

$$(A + B\varepsilon)^{-1} = A^{-1} - B A^{-2}\varepsilon \quad (A \neq 0)$$

```python
m = ExpressionMultiplier(kind="standard")
m.inverse("1 + 2e1 + 3e2")
# "0.071428571428571 - 0.142857142857143e1 - 0.214285714285714e2"

m = ExpressionMultiplier(kind="dual")
result = m.inverse("3 + 2e1 + eps")
m.multiply("3 + 2e1 + eps", result)   # "1"
```

Raises `ZeroDivisionError` if the norm is zero (element is not invertible).

### Numerical Precision Limitations

The library uses a pruning tolerance of `1e-12` to suppress floating-point
noise from inverse operations. As a consequence:

- Elements whose **norm falls below `1e-12`** cannot be inverted and will
  raise `ZeroDivisionError`, even if they are mathematically invertible.
- For standard algebras, this means elements with magnitude below `~1e-6`
  cannot be inverted.
- This is a deliberate trade-off: preserving clean output for typical
  use cases over supporting extremely small values.

If you need to invert very small values, consider scaling them up first,
inverting, then scaling back down.
---

## Conjugate Types (Dual Algebras)

Dual algebras admit **three distinct conjugations** in addition to the identity.
The sign applied to each component is summarized below:

| Number Type | Real ($a$) | Imaginary ($e_i,\ i>0$) | Pure Dual ($\varepsilon$) | Mixed ($\varepsilon e_i$) |
|---|---|---|---|---|
| **Original** ($W$) | + | + | + | + |
| **Dual Conjugate** ($\overline{W}$) | + | + | − | − |
| **Hypercomplex Conjugate** ($W^*$) | + | − | + | − |
| **Total Conjugate** ($W^\dagger$) | + | − | − | + |

**Key facts:**
- $\varepsilon$ is **not** an imaginary unit; the Hypercomplex Conjugate leaves pure $\varepsilon$ positive.
- The **Total Conjugate** negates both imaginary and pure-$\varepsilon$ parts; mixed terms stay positive
  because $(-e_i)(-\varepsilon) = +\varepsilon e_i$.
- Only the **Hypercomplex Conjugate** reliably collapses $x \cdot x^*$ to a pure dual scalar,
  making it the correct choice for `norm()` and `inverse()`.

---

## Changelog

### v0.5.0

**Engine alignment (`hypercomplex-engine` ≥ 0.5.0):**
- Removed the legacy dual 2-tuple `(sign, global_index)` "overflow" representation.
- Dual elements now use the strict 3-tuple `(sign, local_index, eps_flag)` throughout,
  reflecting the mathematical fact that $\varepsilon$ is a central nilpotent unit,
  not an imaginary Cayley-Dickson dimension.

**New unary operations** (`application/unary/`):
- `conjugate(expr, mode)` — supports `"hypercomplex"`, `"dual"`, and `"total"` modes for duals.
- `norm(expr, mode)` — computes $x \cdot x^*$; returns a dual scalar string for dual algebras.
- `inverse(expr, mode)` — computes $x^* / N(x)$ using dual-scalar division.

**Fixes:**
- `DualElementParser`: fixed parsing of `{coeff}eps` tokens (e.g. `0.5eps`) that were
  previously mis-parsed as basis elements.
- `DualElementFormatter` / `ElementFormatter`: coefficients are now formatted in fixed-point
  decimal notation; scientific notation (e.g. `2e-10`) is never emitted, preventing
  parse round-trip failures.
- Default pruning tolerance raised to `1e-9` to suppress floating-point residuals from
  inverse calculations.

---

## Testing Philosophy

This package is tested with an **adversarial, cross-validation approach**:

- **340+ tests** covering standard, split, dual, dual-split, and tensor algebras
- **Cross-validation against `build_table`**: Every resolver is verified against the engine's authoritative multiplication tables for dimensions 1 through 4+
- **Mega-tests**: Exhaustive Cartesian product validation for tensor products (hundreds of thousands of individual multiplications)
- **Property-based checks**: Single-element-or-zero property, nilpotency, ε-commutativity, dimensional boundary enforcement
- **Round-trip tests**: `parse(format(d)) == d` for all algebra families

**No silent faults. No untested paths. No "it works on my machine."**

Run the full suite:
```bash
pytest tests/ -v
```

---

## Licensing & Philosophy

### Engine: Apache 2.0

`hypercomplex-engine` is released under the **Apache 2.0 License**. It is a direct implementation of mathematical facts. While it has elementary proofs, it is not yet peer-reviewed as of this writing. I do not consider this math to belong to me; it belongs to the people to use, and potentially to save lives.

### This Package: Apache 2.0

`hypercomplex-algebra` is also released under the **Apache 2.0 License**.

### A Note on the Software Industry

The reason this package is permissively licensed is because of a belief in open knowledge and the future of the next generation of engineers. It will definitively remain permissive as long as the industry supports junior hiring, fresh graduates, and maintains a healthy, organic ecosystem.

> *"Juniors are your future seniors. I was a junior at abstract algebra and hypercomplex numbers; I had an idea, and I executed it. Juniors bring innovation and new perspectives — they are not a burden to be managed. Remember that your future and well-being will depend on the young of today. You may own the present, but tomorrow belongs to the young. Just as you were taught by seniors yesterday, teach juniors today, and let the sacred message pass on."*


---

## Author & References

**Maher Ben Abdessalem**
Mobile & Web Software Engineer | Experimental & Computational Mathematician

- **ORCID:** [0000-0001-5948-9718](https://orcid.org/0000-0001-5948-9718)
- **Figshare:** [maher_ben_abdessalem](https://figshare.com/authors/maher_ben_abdessalem/13274766)
- **GitHub:** [maher1719](https://github.com/maher1719)
- **LinkedIn:** [maher-ben-abdessalam](https://tn.linkedin.com/in/maher-ben-abdessalam/en)

### Preprints

- *"An O(1) Bitwise Evaluator for Cayley--Dickson Sign Structure: Ordinary, Split, Dual, and Tensor Constructions"* — [Figshare](https://figshare.com/articles/preprint/A_Proven_Sign_Law_for_Cayley-Dickson_Algebras_Ordinary_and_Split_Constructions/33705022)
- *"Unveiling the Structure of Cayley-Dickson Algebras: Zero Divisor Counting, Alternative Constructions, and a Novel Sign Compression Scheme"* — [OSF Preprints](https://osf.io/preprints/osf/byqfw_v3)
- *"Octonionic Associator Interactions"* — [Figshare](https://figshare.com/articles/preprint/Octonionic_Associator_Interactions_/33867982)

### Acknowledgments

The author gratefully acknowledges **Greg Wilmot** for his work on the structure of Cayley-Dickson algebras and for acknowledging the author's contribution to his paper *"Structure of the Cayley-Dickson algebras"* ([arXiv:2505.11747](https://arxiv.org/abs/2505.11747)).

---

## Disclaimer

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

---

*Built with focus, rigor, and the belief that tomorrow belongs to the young.*

***

Apache 2.0 License.

See [`LICENSE`](LICENSE) for details.

Copyright (c) 2026 Maher Ben Abdessalem