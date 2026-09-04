---
sidebar_label: Index Arithmetic
---

# Index Arithmetic

Tensor compilers spend most of their optimization budget on index arithmetic. A `tensor[i, j]` access with shape `[H, W]` becomes `i * W + j`. After tiling, vectorization, and loop transformations, these expressions accumulate nested divisions and modulos. Simplifying them is critical — a single unnecessary `idiv` costs 20-40 cycles vs 1 cycle for the equivalent shift (approximate, modern x86-64).

This page documents the patterns that simplify index expressions. These are NOT optimizations in the traditional sense — they are the algebra that makes tensor indexing work efficiently.

**Key concept — value range analysis**: Each UOp tracks the minimum (`vmin`) and maximum (`vmax`) values it can take at runtime, computed eagerly during node construction from its inputs' bounds. Many index patterns use these bounds to prove simplifications at compile time (e.g., "`x` is always in `[0, N)`" enables `x % N` → `x`).

These patterns run at multiple stages of the [codegen pipeline](../codegen/overview.md):
- **Stage 4** (initial symbolic, during rangeify)
- **Stage 8** (post-optimization symbolic)
- **Stage 15** (index dtype lowering via `pm_lower_index_dtype`)
- **Stage 16** (post-index symbolic)

Svod source: `schedule/src/symbolic/patterns.rs`, `schedule/src/symbolic/index_lowering.rs`

Tinygrad source: `tinygrad/uop/divandmod.py`, `tinygrad/uop/symbolic.py`

---

## 1. The Div-Mod Identity

The fundamental theorem of integer division:

$$
x = \lfloor x / n \rfloor \cdot n + (x \bmod n)
$$

Five variants exploit this identity in the pattern set:

| # | Pattern | Condition | Name |
|---|---------|-----------|------|
| 1 | `x%n + (x//n)*n` -> `x` | -- | Core identity |
| 2 | `((x//a) % c) + (x//b)*c` -> `x//a` | `a*c == b` | Composed divisor |
| 3 | `(x%c1)*c2 + (x//c1)*c3` -> `x*c2` | `c1*c2 == c3` | Scaled |
| 4 | `y + (x%n) + (x//n)*n` -> `y + x` | -- | Three-term |
| 5 | `(a//c1 + c2) // c3` -> `(a + c1*c2) // (c1*c3)` | `c1>0, c3>0` | Nested division |

**Proof of #1.** By the division algorithm, for integers `x` and `n > 0`, there exist unique integers `q` and `r` such that `x = q*n + r` where `0 <= r < n`. By definition, `q = x // n` and `r = x % n`. Substituting: `(x % n) + (x // n) * n = r + q*n = x`. QED.

**Why #2-#5 are corollaries.**

Variant #2 composes two levels of division. Since `b = a*c`, we have `x // b = (x // a) // c`. Applying the core identity at the inner level: `((x//a) % c) + ((x//a) // c) * c = x // a`. But `(x//a) // c = x // (a*c) = x // b`, giving the pattern.

Variant #3 scales both sides of the core identity by `c2`. From `x = (x % c1) + (x // c1) * c1`, multiplying by `c2`: `x * c2 = (x % c1) * c2 + (x // c1) * c1 * c2`. Since `c1 * c2 = c3`, this becomes `(x % c1) * c2 + (x // c1) * c3 = x * c2`.

Variant #4 adds an independent term `y` to both sides of #1.

Variant #5 flattens nested floor division. Given `(a // c1 + c2) // c3`, multiply `c2` by the inner divisor to get an equivalent single-level division: `(a + c1*c2) // (c1*c3)`. This holds when `a >= 0` and `c2 >= 0` (or both non-positive), ensuring floor division semantics are preserved.

All five patterns use `Arc::ptr_eq` checks on duplicate variable names (e.g., `x` appearing twice means both must be the same hash-consed node).

### Implementation

```rust
// From schedule/src/symbolic/patterns.rs — div_mod_recombine_dsl_patterns()

// #1: x%n + (x//n)*n -> x
Add[Mod(x, n), Mul[Idiv(x, n), n]] => x,

// #2: ((x//a) % c) + (x // b) * c -> x // a  when a*c == b
Add[Mod(Idiv(x, a @const(a_val)), c @const(c_val)), Mul[Idiv(x, _b @const(b_val)), c]]
    => { /* guard: a_val * c_val == b_val */ },

// #5: (a//c1 + c2) // c3 -> (a + c1*c2) // (c1*c3)
Idiv(Add[Idiv(a, c1 @const(c1_val)), _c2 @const(c2_val)], _c3 @const(c3_val))
    => { /* guard: c1>0, c3>0, same-sign */ },
```

---

## 2. Range-Based Mod/Div

Value range analysis (`vmin`/`vmax`) enables simplifications that are invisible to purely syntactic pattern matching. Each UOp carries cached bounds computed during construction.

| Pattern | Guard | Example |
|---------|-------|---------|
| `x % n` -> `x` | `0 <= vmin(x)` and `vmax(x) < n` | `RANGE(3) % 3` -> `RANGE(3)` |
| `(a*m + b) % n` -> `b % n` | `m == n` | `(row*512 + col) % 512` -> `col % 512` |
| `(a*m + b) / n` -> `a + b/n` | `m == n` and `0 <= b < n` | `(row*512 + col) / 512` -> `row` |
| `x / n` -> `k` | all values in bucket `[k*n, (k+1)*n)` | `RANGE(3) / 3` -> `0` |
| `(x + c) // d` -> `x // d` | `max_remainder + c < d` | `(R*4 + 1) // 8` -> `R*4 // 8` |
| `(x + c) // d` -> `(x + c%d) // d + c//d` | `c >= d` | `(x + 70) // 8` -> `(x + 6) // 8 + 8` |

The first pattern is the workhorse. After range splitting, `RANGE(n)` produces values in `[0, n)`, so `RANGE(n) % n` trivially simplifies to `RANGE(n)`. This single rule eliminates most modulos created by tiling.

The fifth pattern (small constant) uses a tight bound on the maximum remainder within the range `[vmin, vmax]`. If the range spans fewer than `d` values and adding `c` never crosses a bucket boundary, the constant is dead weight.

The sixth pattern (large offset split) canonicalizes offsets larger than the divisor. This exposes the small-constant pattern for the next rewrite iteration.

:::caution
The `(a*m + b) / n` -> `a + b/n` pattern requires `0 <= b < n`. Without the range check, negative remainders produce incorrect quotients due to truncation-toward-zero semantics. The implementation explicitly checks `vmin(b) >= 0 && vmax(b) < n`.
:::

---

## 3. The `fold_divmod_general` Algorithm

The catch-all for Index-dtype `Idiv` and `Mod`. Implements all 8 rules from Tinygrad's `divandmod.py:8-93` in priority order, including the recursive `nest_div_by_smallest_factor`. Each rule is tried in sequence; the first match wins.

Entry point: when `Idiv(x, y)` or `Mod(x, y)` has `dtype == Index`, the pattern delegates to `fold_divmod_general(op, x, y)`.

### Rule 1 -- cancel_divmod

If the entire range `[x_min, x_max]` maps to a single quotient across all corner combinations of `(x, y)`, the result is that constant.

**Guard**: `y_min * y_max > 0` (denominator never crosses zero), and all four corner quotients `x_min/y_min`, `x_min/y_max`, `x_max/y_min`, `x_max/y_max` are equal.

**What it does**: For `Idiv`, returns the constant quotient. For `Mod`, returns `x - q*y`.

**Example**: `RANGE(3) // 3` -> `0`. Values 0, 1, 2 all divide to 0.

### Rule 2 -- remove_nested_mod

`(a%4 + b) % 2` -> `(a + b) % 2` when `2 | 4`. The outer modulus divides the inner, so the inner modulus is redundant.

**Guard**: `op == Mod`, `x_min >= 0`, and for each term that is a `Mod(inner_x, inner_y)`, the denominator `y` divides `inner_y`.

**What it does**: Strips inner `Mod` operations whose modulus is a multiple of the outer modulus, then re-applies `Mod`.

**Example**: `(RANGE(8) % 4 + RANGE(2)) % 2` -> `(RANGE(8) + RANGE(2)) % 2`

### Rule 3 -- fold_binary_numerator

When a single non-constant term has exactly 2 values (`vmax - vmin == 1`), the result is a linear interpolation: `(y2 - y1) * (v - v_min) + y1`.

**Guard**: Exactly one non-constant term after decomposition, and that term's range spans exactly 2 values.

**What it does**: Evaluates the div/mod at both endpoints and constructs a linear map between them. This avoids the division entirely.

**Example**: For `(v * 3 + 2) % 5` where `v` is in `{0, 1}`:
- `v=0`: `(0 + 2) % 5 = 2`
- `v=1`: `(3 + 2) % 5 = 0`
- Result: `(0 - 2) * (v - 0) + 2 = -2*v + 2`

### Rule 4 -- fold_divmod_congruence

For each term `f_i * v_i`, compute the closest residue `r_i = min(f_i % c, f_i % c - c)` by absolute value. If the residue sum stays within one floor-division bucket of `c`, the mod/div simplifies. This is modular arithmetic optimization.

**Guard**: `x_min >= 0`, constant denominator `c > 0`, and `rem_min // c == rem_max // c` (all residue-sum values land in the same bucket).

**What it does**: Replaces each factor with its residue mod `c`. For `Mod`, returns the residue sum (adjusted by bucket offset). For `Idiv`, returns the quotient-coefficient sum.

**Example**: `(r*8 + v) % 7` -> `(r + v) % 7` because `8 = 1 (mod 7)`, so the residue of `8` is `1`.

### Rule 5 -- gcd_with_remainder

Compute the symbolic GCD of all additive terms and the denominator. If GCD > 1, factor it out: `(g*a + g*b) // (g*c)` -> `(a + b) // c` (with the remainder scaled back for `Mod`).

**Guard**: `x_min >= 0`, constant denominator, GCD > 1, and the reduced numerator has `vmin >= 0`.

**What it does**: Divides both numerator terms and denominator by their GCD, recursively enabling simpler patterns to fire.

**Example**: `(6*a + 4*b) // 8` with `GCD(6, 4, 8) = 2` -> `(3*a + 2*b) // 4`

### Rule 6 -- divide_by_gcd

Variable denominator version of Rule 5. Computes `GCD(all_terms..., y)` including both numerator and denominator, then divides both sides. Unlike Rule 5, this works when the denominator is not a constant.

**Guard**: GCD is non-trivial (not 1), and both `x` and `y` are exactly divisible by the GCD.

**Example**: `(4*a) // (2*b)` -> `(2*a) // b`

### Rule 7 -- factor_remainder

Last resort. Partition terms into exactly-divisible (quotient) and remainder.

**Guard**: `x_min >= 0` and `y_min >= 0`, and at least one term divides `y` exactly.

**What it does**: For `Idiv`: `quo_sum + rem // y`. For `Mod`: `rem % y` (with coefficient reduction for constant `y`).

**Example**: `(8*a + 3*b) // 8` -> `a + (3*b) // 8`

### Rule 8 -- nest_div_by_smallest_factor

Recursive decomposition for constant divisors. Finds the smallest factor shared between the divisor and any term's coefficient, divides both by it, then recurses.

**Guard**: `x_min >= 0`, constant `y > 1`, and at least one non-constant term has a factor `f > 1` where `y % f == 0`.

**What it does**: Picks `div = min(|f|)` among qualifying factors, rewrites `x // y` as `(x // div) // (y / div)`. Each step reduces `y`, converging to rules 1-7.

**Example**: `(6*a + 4*b) // 12` → `((6*a + 4*b) // 2) // 6` → `(3*a + 2*b) // 6` → `(3*a + 2*b) // 6` (then rule 7 finishes).

Tinygrad: `divandmod.py:62-67`. Svod: `nest_div_by_smallest_factor` in `fold_divmod_general`.

:::caution
Rules 5-8 require non-negative numerators (`x_min >= 0`). Floor division with negative operands has different rounding semantics (toward negative infinity in Python/Tinygrad, toward zero in hardware). The implementation returns `None` for negative ranges, falling through to let later passes handle the expression.
:::

---

## 4. Advanced Division Patterns

Standalone patterns outside `fold_divmod_general` that handle additional cases:

| Pattern | Guard | Source |
|---------|-------|--------|
| `(a // b) // c` -> `a // (b*c)` | `b != 0, c != 0` | `advanced_division_dsl_patterns` |
| `expr // divisor` -> exact quotient | `expr` is exactly divisible | `advanced_division_dsl_patterns` |
| `(a + b) % c` coefficient reduction | `a` or `b` has factor divisible by `c` | `advanced_division_dsl_patterns` |
| `(a + b) // c` -> `a//c + b//c` | both divide evenly | `advanced_division_dsl_patterns` |
| `(a - b) // c` -> `a//c - b//c` | both divide evenly | `advanced_division_dsl_patterns` |
| `c * (a + b)` -> `c*a + c*b` | `c` is constant | `advanced_division_dsl_patterns` |

The nested division collapse `(a // b) // c` -> `a // (b*c)` is particularly important after tiling, where splitting a range into outer/inner components creates two levels of division that should collapse to one.

The exact-division pattern uses `divides()` which checks if every additive term's constant factor is divisible by the divisor. When it succeeds, the `Idiv` is eliminated entirely -- no division instruction emitted.

The coefficient reduction pattern converts `(r*8 + v) % 7` -> `(r*1 + v) % 7 = (r + v) % 7` by reducing each factor modulo the divisor. This fires when no factor is an exact multiple of the modulus but the residues are smaller.

---

## 5. Index Dtype Lowering (3-Phase Cascade)

Tinygrad: `ops.py:1291-1313`. Svod: `schedule/src/symbolic/index_lowering.rs`.

The abstract `Index` type carries no width -- it represents "whatever integer width is needed for this index." The lowering pass converts `Index` to concrete `i32` or `i64` based on value bounds.

### Phase 1 -- Create wrappers (leaf nodes)

Leaf nodes with `Index` dtype get replaced by their concrete equivalent wrapped in a cast back to `Index`:

| Input | Output |
|-------|--------|
| `CONST(Index)` | `CONST(concrete).cast(Index)` |
| `DEFINE_VAR(Index)` | `DEFINE_VAR(concrete).cast(Index)` |
| `VCONST(Vector<Index, N>)` | `VCONST(Vector<concrete, N>).cast(Vector<Index, N>)` |

### Phase 2 -- Process wrapped values upward

Binary operations, control flow, and structural nodes propagate the concrete type through `.cast(Index)` wrappers:

| Input | Output |
|-------|--------|
| `Binary(x.cast(Index), y.cast(Index))` | `Binary(x.cast(dt), y.cast(dt)).cast(result_dtype)` |
| `WHERE(cond, x.cast(Index), y.cast(Index))` | `WHERE(cond, x.cast(dt), y.cast(dt)).cast(Index)` |
| `RANGE(end.cast(Index))` | `RANGE(end, end.dtype).cast(Index)` |
| `SPECIAL(end.cast(Index))` | `SPECIAL(end, i32).cast(Index)` |
| `VECTORIZE(e0.cast(Index), ...)` | `VECTORIZE(e0.cast(dt), ...).cast(Vector<Index, N>)` |
| `BIND(var.cast(Index), val.cast(Index))` | `var.cast(dt).bind(val.cast(dt)).cast(Index)` |

The `dt` is computed as `least_upper_dtype(select_dtype(result), x.dtype, y.dtype)` -- the widest type needed by any operand or the result.

### Phase 3 -- Strip wrappers at terminals

Terminal nodes consume the index and discard the `Index` wrapper:

| Input | Output |
|-------|--------|
| `INDEX(buf, idx.cast(Index))` | `INDEX(buf, idx)` |
| `INDEX(buf, WHERE(cond, idx, Invalid))` | `INDEX(buf, idx, gate=cond)` |
| `SINK(sources with .cast(Index))` | `SINK(unwrapped sources)` |
| `END(computation.cast(Index))` | `END(unwrapped computation)` |

The `WHERE(cond, idx, Invalid)` -> `gate=cond` transformation is significant: it extracts validity conditions from the index expression into the `INDEX` node's gate field, which codegen backends use to emit predicated loads.

### `select_dtype()`

Returns `i32` if the UOp's value bounds fit `[-2^31, 2^31 - 1]`, otherwise `i64`. Most tensor indices fit in `i32` -- even a 2B-element tensor's flat index fits. The `i64` path exists for very large tensors or accumulated offsets.

---

## 6. Commutative Canonicalization

```rust
// For Index dtype ONLY:
op(a, b) -> op(b, a)   when b.id < a.id
```

This ensures commutative operations have a deterministic operand order based on the UOp's unique ID. Applied to: `Add`, `Mul`, `Max`, `Eq`, `Ne`, `And`, `Or`, `Xor`.

**Why Index-only**: Without canonicalization, `R1*8000 + R2*16` and `R2*16 + R1*8000` are distinct nodes after hash-consing, breaking grouping in `expand_vector_index`. The expander needs to identify identical index patterns across vector lanes, and non-canonical ordering defeats this.

**Why NOT applied to non-Index types**: Applying canonicalization to float/int arithmetic would reorder VECTORIZE elements and break vector math merging in later passes. Tinygrad makes the same choice (`symbolic.py:178-182`).

:::caution
Canonicalization interacts with the rewrite engine's fixed-point iteration. If two patterns disagree on operand order (one canonicalizes, another produces non-canonical output), the engine can oscillate. All index-producing patterns must respect canonical order, or the 1000-iteration safety limit will trigger.
:::

---

## Worked Example

Consider `tensor[i, j]` with shape `[4, 8]`, accessed as a flat iteration over 32 elements.

### Initial state

Range `R0` iterates `0..32` (flat index). The access pattern decomposes into:

```text
row = R0 // 8       (which of the 4 rows)
col = R0 % 8        (which of the 8 columns)
addr = row * 8 + col = (R0 // 8) * 8 + (R0 % 8)
```

By the div-mod identity (#1), `(R0 // 8) * 8 + (R0 % 8) = R0`. The address is just the flat index -- no division needed.

### After tiling (UPCAST by 4)

Range splitting decomposes `R0` into `R1 * 4 + R2` where `R1` is in `[0, 8)` and `R2` is in `[0, 4)`:

```text
row = (R1*4 + R2) // 8
col = (R1*4 + R2) % 8
```

**Simplifying `row`**: The expression `(R1*4 + R2) // 8` enters `fold_divmod_general`.

Rule 4 (congruence) fires: factor `4` has residue `4 % 8 = 4`, and `R2` has residue `1 % 8 = 1`. The residue sum is `4*R1 + R2` with range `[0, 31]`. Since `0 // 8 != 31 // 8`, Rule 4 does not collapse it to a constant. Rule 7 (factor remainder) fires instead: `4` does not divide `8` exactly, but the expression can be decomposed. Since no term divides 8 exactly, we fall through to the range-based pattern `(a*m + b) / n` with `m = 4, n = 8` -- this does not match (`m != n`).

The expression stays as `(R1*4 + R2) // 8`. In the generated code, if `R2` is vectorized (UPCAST), the backend emits this as a single division of a 4-wide vector.

However, if we further split `R1` into `R3 * 2 + R4` (where `R3` is in `[0, 4)`, `R4` in `[0, 2)`):

```text
row = (R3*2*4 + R4*4 + R2) // 8
    = (R3*8 + R4*4 + R2) // 8
```

Now the range-based pattern `(a*m + b) / n` fires with `m = n = 8`:
- `a = R3`, `b = R4*4 + R2`
- `vmin(b) = 0`, `vmax(b) = 1*4 + 3 = 7 < 8`
- Result: `R3 + (R4*4 + R2) // 8`

And `(R4*4 + R2) // 8`: `vmax = 1*4 + 3 = 7`, `vmin = 0`, so `0 // 8 = 7 // 8 = 0`. The cancel_divmod rule fires:
- Result: `R3 + 0 = R3`

**Simplifying `col`**: `(R3*8 + R4*4 + R2) % 8`

The range-based pattern `(a*m + b) % n` fires with `m = n = 8`:
- `(R3*8 + R4*4 + R2) % 8` -> `(R4*4 + R2) % 8`

Then `vmin(R4*4 + R2) = 0`, `vmax(R4*4 + R2) = 7 < 8`, so `x % n` -> `x`:
- Result: `R4*4 + R2`

### Final tree

```text
Before (after tiling, before simplification):
  STORE(
    INDEX(buf, (R3*8 + R4*4 + R2) // 8 * 8 + (R3*8 + R4*4 + R2) % 8),
    value)

After index arithmetic:
  STORE(
    INDEX(buf, R3*8 + R4*4 + R2),
    value)
```

The entire address calculation collapses back to a linear expression -- zero divisions, zero modulos. The patterns have proven that the tiled index is equivalent to the flat index, purely through algebraic rewriting.
