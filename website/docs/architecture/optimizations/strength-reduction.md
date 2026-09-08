---
sidebar_label: Strength Reduction
---

# Strength Reduction and Late Rewrite Patterns

Strength reduction replaces expensive operations with cheaper equivalents. These patterns run late in the pipeline (Stages 18-19) because earlier passes need to see the original operation structure. For example, `Add(Mul(a, b), c)` must remain visible for algebraic simplification before being fused into `MULACC(a, b, c)`.

Tinygrad source: `tinygrad/uop/decompositions.py:438-480` (`get_late_rewrite_patterns`).
Svod source: `schedule/src/rangeify/patterns.rs` (late decomposition group) + `schedule/src/symbolic/fast_div.rs`.

*Cycle estimates throughout this page are approximate for modern x86-64. Actual latencies vary by microarchitecture and pipeline state.*

All patterns are combined into a single fixed-point rewrite pass (`PM_FINAL`) together with `symbolic_simple()` (algebraic cleanup).

---

## 1. Power-of-Two Optimization

The most impactful strength reduction. Integer division and modulo by constants are extremely common in tensor indexing -- stride calculations, tiling, and coordinate recovery from flat indices all produce them.

| Pattern | Before | After | Cycle savings |
|---------|--------|-------|---------------|
| `x % 2^n` | `idiv` + `imul` + `isub` (~25 cycles) | `and` (1 cycle) | ~24x |
| `x * 2^n` | `imul` (~3-4 cycles) | `shl` (1 cycle) | ~3x |
| `x // 2^n` (unsigned) | `idiv` (~20-40 cycles) | `shr` (1 cycle) | ~20-40x |

The modulo optimization works because `2^n - 1` is a bitmask of the lower n bits. Example: `x % 8` = `x & 0b111`.

Tinygrad: `decompositions.py:448-454`. Svod: `pm_mod_to_and`, `pm_mul_to_shl`, `pm_div_to_shr` in `rangeify/patterns.rs`.

:::caution[Signed Division]
For signed integers, `x // 2^n` is NOT simply `x >> n`. Arithmetic right shift rounds toward negative infinity, but integer division rounds toward zero.

Fix: `(x + (x < 0 ? 2^n - 1 : 0)) >> n`

The bias `2^n - 1` added for negative values corrects the rounding direction. This matches the identity:

```
floor(x / 2^n) = (x + 2^n - 1) >> n    when x < 0
                  x >> n                  when x >= 0
```

Svod checks `vmin >= 0` via range analysis (`VminVmaxProperty`) to skip the bias when the dividend is provably non-negative. Tinygrad uses dtype membership (`dtypes.uints`) for the same purpose.

Tinygrad: `decompositions.py:452-454`. Svod: `pm_div_to_shr` in `rangeify/patterns.rs`.
:::

Generated C output for signed power-of-two division:

```c
// Before: x / 8
int result = x / 8;

// After: strength reduction (signed path)
int result = (x + ((x >> 31) & 7)) >> 3;
//           bias for negatives ^^^   ^shift
```

When `x` is provably non-negative (common in index calculations), the signed path is eliminated entirely:

```c
// After: strength reduction (unsigned path, vmin >= 0)
int result = x >> 3;
```

---

## 2. Fast Integer Division (Hacker's Delight)

For non-power-of-2 constants, replace `x / d` with multiply-and-shift: `(x * M) >> S`.

### The math

For a positive constant `d` and value range `[0, max_val]`, find magic number `M` and shift `S` such that:

```
(x * M) >> S == x / d    for all 0 <= x <= max_val
```

**Why this works**: Division by `d` is equivalent to multiplication by `1/d`. We approximate `1/d` as `M / 2^S` where `M` and `S` are chosen so the approximation is exact over the value range. The key insight is that integer truncation makes exact representation possible -- we only need `floor(x * M / 2^S) == floor(x / d)`, not real-valued equality.

### The algorithm

From Hacker's Delight Chapter 10 (Tinygrad's `magicgu`, `decompositions.py:272-280`):

1. Compute `nc = floor((max_val + 1) / d) * d - 1` (the critical threshold)
2. Compute `nbits = bit_length(max_val)`
3. For `s` from 0 to `2 * nbits`:
   - If `2^s > nc * (d - 1 - (2^s - 1) mod d)`: found valid shift
   - Compute `M = ceil((2^s + d - 1 - (2^s - 1) mod d) / d)`
4. Return `(M, s)` -- the smallest valid `(multiplier, shift)` pair

The loop finds the smallest `s` that produces a valid magic number. Smaller `s` means smaller `M`, which is critical for fitting the intermediate product `x * M` in narrow integer types.

Svod implementation: `magic_unsigned()` in `schedule/src/symbolic/fast_div.rs`.

### Three-stage strategy

Matching Tinygrad `decompositions.py:282-300` (`fast_idiv`):

| Stage | Condition | Transform | Example |
|-------|-----------|-----------|---------|
| 1. Same-dtype | `M * vmax` fits in dtype range | `(x * M) >> S` | `x / 3` with `x` in i32 |
| 2. Factor pow2 | `d = 2^k * d'` where `d' > 1` | `(x >> k) / d'` then magic on `d'` | `x / 6` becomes `(x >> 1) / 3` |
| 3. Widen to i64 | Int32 overflow in `x * M` | cast to i64, multiply, shift, cast back | Fallback for large `M` |

The factorization stage (2) is important: dividing by 12 (`= 4 * 3`) becomes a shift-right by 2 followed by magic division by 3, which often fits in the original dtype where direct magic division by 12 would overflow.

For signed values, add correction: `((x * M) >> S) + (x < 0 ? 1 : 0)`. This accounts for truncation-toward-zero semantics -- without it, negative dividends round in the wrong direction.

### Concrete example

```
x / 7 where x in [0, 255]:
  magic_unsigned(255, 7) → M = 293, S = 11

  Verify: (100 * 293) >> 11 = 29300 >> 11 = 14 = floor(100 / 7)
  Verify: (  7 * 293) >> 11 =  2051 >> 11 =  1 = floor(  7 / 7)
  Verify: (255 * 293) >> 11 = 74715 >> 11 = 36 = floor(255 / 7)

  Generated: (x * 293) >> 11  instead of  x / 7
  Cost: 1 imul + 1 shr (~4-5 cycles) vs 1 idiv (~20-40 cycles)
```

### Generated LLVM IR

```llvm
; Before: x / 7
%result = sdiv i32 %x, 7

; After: fast integer division (unsigned path)
%mul = mul i32 %x, 293
%result = lshr i32 %mul, 11
```

---

## 3. Float Division to Multiply

`x / c` becomes `x * (1/c)` for float constant `c`.

Float multiply is 1-2 cycles (fully pipelined), while float divide is 10-20 cycles (not pipelined on most hardware). This is a straightforward 5-10x speedup for a common pattern.

**Guards**:
- Skip if `c == 0.0` -- division by zero must remain to preserve IEEE 754 semantics (`x / 0.0` produces `+/-inf` or `NaN`)
- Skip if `1/c` is not finite (overflow to `inf` means `c` is too small)
- Only for float types

Tinygrad: `decompositions.py:477-479` (FDIV-based backends emit `RECIP` as `1/x`). Svod: `pm_fdiv_to_mul` in `rangeify/patterns.rs`.

```c
// Before
float result = x / 3.14159f;

// After
float result = x * 0.31831f;  // 1/pi
```

---

## 4. FMA Fusion (Fused Multiply-Add)

`a * b + c` becomes `MULACC(a, b, c)`.

This maps to hardware FMA instructions (`vfmadd` on x86 AVX, `fmadd` on ARM NEON, `fma.rn` on CUDA). A single instruction replaces two, with a single rounding step instead of two -- making FMA both faster and more precise than separate multiply + add.

**Why applied late**: Earlier passes need to see `Add(Mul(a, b), c)` structure for algebraic simplification. If fused early, patterns like `(x*2 + x*3)` could not simplify to `x*5` because the `Mul` nodes would be buried inside MULACC.

**Shift-add fusion**: `(x << n) + c` is also fused to `MULACC(x, 2^n, c)`, catching cases where MUL-to-SHL ran first in the same fixed-point pass. Svod's `pm_shl_add_to_mulacc` is added alongside `pm_fma_decomposition` whenever the renderer supports both `MulAcc` and `Shl`.

**Guards**: Only matches when all three operands (`a`, `b`, `c`) share the same float dtype. Integer FMA is not fused because hardware FMA instructions are float-only.

Tinygrad: `decompositions.py:472-475`. Svod: `pm_fma_decomposition` in `rangeify/patterns.rs`.

---

## 5. Negation Extraction

`x * -1` becomes `NEG(x)`.

NEG is a single instruction (flip sign bit for float via `xorps`, negate for int via `neg`). Multiplication by -1 unnecessarily occupies the multiplier pipeline for 3-4 cycles.

Only fires when the backend supports `NEG` as a native op. Tinygrad: `decompositions.py:458-459`. Svod: `pm_neg_from_mul`.

---

## 6. Comparison Negations

Late rewrites for negated and compound comparisons on integers. These patterns simplify instruction sequences that arise from boolean logic optimizations in earlier passes.

| Pattern | Before | After | Savings |
|---------|--------|-------|---------|
| `!(x < c)` | NOT + CMP | `(c-1) < x` | Eliminate NOT |
| `!(c < x)` | NOT + CMP | `x < (c+1)` | Eliminate NOT |
| `(c1 < x) & (x < c2)` where `c2 == c1+2` | 2 CMPs + AND | `x == (c1+1)` | 2 ops eliminated |
| `x * -1 < c` | MUL + CMP | `-c < x` | Eliminate MUL |
| `x * -1 < y * c` | 2 MULs + CMP | `y * (-c) < x` | Eliminate 1 MUL |

The range compression (row 3) is particularly valuable. When the open interval `(c1, c2)` contains exactly one integer value, two comparisons and a logical AND collapse to a single equality check. This arises naturally in tiled index calculations where a range variable selects exactly one tile.

:::caution[Integer Overflow in Constants]
The negation patterns guard against overflow: `!(x < c)` becomes `(c-1) < x` only if `c-1` does not underflow, and `!(c < x)` becomes `x < (c+1)` only if `c+1` does not overflow. Both use `checked_sub` / `checked_add` and return `None` (no transformation) on overflow.
:::

Tinygrad: `decompositions.py:461-470`. Svod: `pm_comparison_negations` in `rangeify/patterns.rs`.

---

## 7. De Morgan's Laws (Late)

```
!a & !b  -->  !(a | b)
!a | !b  -->  !(a & b)
```

These appear in *two* places in the pipeline:

1. **Early** (Stage 4-5): `boolean_dsl_patterns()` in `schedule/src/symbolic/patterns.rs`, part of the full `symbolic()` matcher. Catches De Morgan opportunities in the original expression structure.

2. **Late** (Stage 18-19): `symbolic_simple()` includes boolean patterns and runs alongside the strength reduction patterns in `PM_FINAL`. This catches new De Morgan opportunities created by comparison negation patterns -- for example, after `!(x < 3)` and `!(x < 7)` are rewritten to `2 < x` and `6 < x`, any AND/OR combining them may now have new NOT-elimination opportunities.

Svod: `boolean_dsl_patterns()` in `schedule/src/symbolic/patterns.rs`.

---

## 8. ERF Decomposition

`erf(x)` is replaced with a polynomial approximation (Abramowitz & Stegun 7.1.26):

```
erf(x) = sign(x) * (1 - t * P(t) * exp(-x^2))
where t = 1 / (1 + 0.3275911 * |x|)
      P(t) = Horner(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592])
```

**Why**: `@llvm.erf` is a libcall intrinsic (requires libm linkage), not a native hardware instruction. The LLVM JIT backend does not link libm, so `erf` must be decomposed before codegen. Tinygrad decomposes `erf` at the tensor level (`elementwise.py`), so it never reaches the renderer; Svod keeps `Erf` as a UOp until this late pass.

Maximum error: ~1.5e-7 (sufficient for float32 ML workloads).

Svod: `pm_erf_decomposition` in `rangeify/patterns.rs`.

---

## Pattern Composition: When Each Pattern Runs

All strength reduction patterns are composed into a single `PM_FINAL` matcher that runs as a fixed-point graph rewrite:

```
PM_FINAL = pm_commit_weak() + pm_cast_weak() + pm_decomp
         + renderer.extra_matcher()   -- optional, per-backend
         + pm_split_ends()
```

Where `pm_decomp` chains the early decompositions (which start from `symbolic_simple()`), the transcendental decompositions, and `get_late_rewrite_patterns()`:

```
Stage 18-19 (PM_FINAL fixed-point rewrite):
  symbolic_simple()              -- algebraic cleanup (identities, constant folding)
  + pm_erf_decomposition         -- erf(x) -> polynomial approx
  + pm_mod_to_and                -- x % 2^n -> x & (2^n-1)
  + pm_demorgan                  -- !a & !b -> !(a | b)
  + pm_mul_to_shl                -- x * 2^n -> x << n
  + pm_div_to_shr                -- x // 2^n -> x >> n
  + fast_division_patterns       -- x // d -> (x * M) >> S
  + pm_neg_from_mul              -- x * -1 -> NEG(x)
  + pm_comparison_negations      -- !(x<c) -> (c-1)<x, etc.
  + pm_fma_decomposition         -- a*b+c -> MULACC(a,b,c)
  + pm_shl_add_to_mulacc         -- (x<<n)+c -> MULACC(x, 2^n, c)
  + pm_fdiv_to_mul               -- x / c -> x * (1/c)
```

Because the rewriter runs to a fixed point, patterns can feed into each other. For example:

1. `pm_mul_to_shl` converts `x * 4` to `x << 2`
2. On the next iteration, `pm_fma_decomposition` fuses `(x << 2) + c` into `MULACC(x, 4, c)`
3. `symbolic_simple()` cleans up any identities created by the transformations

Every entry below `pm_mod_to_and` is conditional: `get_late_rewrite_patterns()` adds it only when the renderer declares support for the ops it produces (`Shl` for the shift rewrites, `MulAcc` for the FMA fusions, `Fdiv` for the reciprocal rewrite). `pm_split_ends` runs inside the same fixed point, splitting multi-range ENDs into nested single-range ENDs for the linearizer; a final `pm_remove_invalid()` rewrite then clears any leftover Invalid markers.

Cross-reference: [Codegen Pipeline Overview](../codegen/overview.md) for the full stage listing.
