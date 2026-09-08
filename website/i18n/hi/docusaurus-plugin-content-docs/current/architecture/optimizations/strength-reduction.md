---
sidebar_label: Strength Reduction
---

# Strength Reduction और Late Rewrite Patterns

Strength reduction expensive operations को cheaper equivalents से replace करता है। ये पैटर्न pipeline में late (Stages 18-19) चलते हैं क्योंकि earlier passes को original operation structure देखना ज़रूरी है। जैसे, `Add(Mul(a, b), c)` algebraic simplification के लिए visible रहना चाहिए `MULACC(a, b, c)` में fuse होने से पहले।

Tinygrad source: `tinygrad/uop/decompositions.py:438-480` (`get_late_rewrite_patterns`)।
Svod source: `schedule/src/rangeify/patterns.rs` (late decomposition group) + `schedule/src/symbolic/fast_div.rs`।

*इस पेज में cycle estimates approximate हैं modern x86-64 के लिए। Actual latencies microarchitecture और pipeline state से vary करती हैं।*

सभी पैटर्न `symbolic_simple()` (algebraic cleanup) के साथ single fixed-point rewrite pass (`PM_FINAL`) में combine होते हैं।

---

## 1. Power-of-Two ऑप्टिमाइज़ेशन

सबसे impactful strength reduction। Integer division और modulo by constants tensor indexing में बेहद common हैं — stride calculations, tiling, और flat indices से coordinate recovery सब इन्हें produce करते हैं।

| पैटर्न | Before | After | Cycle savings |
|--------|--------|-------|---------------|
| `x % 2^n` | `idiv` + `imul` + `isub` (~25 cycles) | `and` (1 cycle) | ~24x |
| `x * 2^n` | `imul` (~3-4 cycles) | `shl` (1 cycle) | ~3x |
| `x // 2^n` (unsigned) | `idiv` (~20-40 cycles) | `shr` (1 cycle) | ~20-40x |

Modulo optimization काम करता है क्योंकि `2^n - 1` lower n bits का bitmask है। Example: `x % 8` = `x & 0b111`।

Tinygrad: `decompositions.py:448-454`। Svod: `pm_mod_to_and`, `pm_mul_to_shl`, `pm_div_to_shr` in `rangeify/patterns.rs`।

:::caution[Signed Division]
Signed integers के लिए, `x // 2^n` simply `x >> n` **नहीं** है। Arithmetic right shift negative infinity की तरफ़ round करता है, लेकिन integer division zero की तरफ़।

Fix: `(x + (x < 0 ? 2^n - 1 : 0)) >> n`

Negative values के लिए add किया गया bias `2^n - 1` rounding direction correct करता है। यह इस identity से match करता है:

```
floor(x / 2^n) = (x + 2^n - 1) >> n    when x < 0
                  x >> n                  when x >= 0
```

Svod range analysis (`VminVmaxProperty`) से `vmin >= 0` चेक करता है ताकि dividend provably non-negative हो तो bias skip करे। Tinygrad same purpose के लिए dtype membership (`dtypes.uints`) इस्तेमाल करता है।

Tinygrad: `decompositions.py:452-454`। Svod: `pm_div_to_shr` in `rangeify/patterns.rs`।
:::

Signed power-of-two division के लिए generated C output:

```c
// Before: x / 8
int result = x / 8;

// After: strength reduction (signed path)
int result = (x + ((x >> 31) & 7)) >> 3;
//           bias for negatives ^^^   ^shift
```

जब `x` provably non-negative हो (index calculations में common), signed path पूरी तरह eliminate:

```c
// After: strength reduction (unsigned path, vmin >= 0)
int result = x >> 3;
```

---

## 2. Fast Integer Division (Hacker's Delight)

Non-power-of-2 constants के लिए, `x / d` को multiply-and-shift से replace: `(x * M) >> S`।

### Math

Positive constant `d` और value range `[0, max_val]` के लिए, magic number `M` और shift `S` ढूँढो ताकि:

```
(x * M) >> S == x / d    for all 0 <= x <= max_val
```

**यह क्यों काम करता है**: `d` से division `1/d` से multiplication के equivalent है। हम `1/d` को `M / 2^S` के रूप में approximate करते हैं जहाँ `M` और `S` ऐसे चुने जाते हैं कि approximation value range पर exact हो। Key insight यह है कि integer truncation exact representation possible बनाता है — हमें सिर्फ़ `floor(x * M / 2^S) == floor(x / d)` चाहिए, real-valued equality नहीं।

### एल्गोरिदम

Hacker's Delight Chapter 10 से (Tinygrad का `magicgu`, `decompositions.py:272-280`):

1. `nc = floor((max_val + 1) / d) * d - 1` compute करो (critical threshold)
2. `nbits = bit_length(max_val)` compute करो
3. `s` के लिए 0 से `2 * nbits` तक:
   - अगर `2^s > nc * (d - 1 - (2^s - 1) mod d)`: valid shift मिल गया
   - `M = ceil((2^s + d - 1 - (2^s - 1) mod d) / d)` compute करो
4. `(M, s)` return करो — smallest valid `(multiplier, shift)` pair

Loop smallest `s` ढूँढता है जो valid magic number produce करे। Smaller `s` मतलब smaller `M`, जो intermediate product `x * M` को narrow integer types में fit करने के लिए critical है।

Svod implementation: `magic_unsigned()` in `schedule/src/symbolic/fast_div.rs`।

### Three-stage strategy

Tinygrad `decompositions.py:282-300` (`fast_idiv`) से match करते हुए:

| Stage | Condition | Transform | Example |
|-------|-----------|-----------|---------|
| 1. Same-dtype | `M * vmax` dtype range में fit | `(x * M) >> S` | `x / 3` जहाँ `x` i32 में |
| 2. Factor pow2 | `d = 2^k * d'` जहाँ `d' > 1` | `(x >> k) / d'` फिर `d'` पर magic | `x / 6` बनता है `(x >> 1) / 3` |
| 3. Widen to i64 | `x * M` में Int32 overflow | i64 में cast, multiply, shift, वापस cast | Large `M` के लिए fallback |

Factorization stage (2) important है: 12 (`= 4 * 3`) से divide करना shift-right by 2 बन जाता है followed by magic division by 3, जो अक्सर original dtype में fit हो जाता है जहाँ direct magic division by 12 overflow करता।

Signed values के लिए, correction add करो: `((x * M) >> S) + (x < 0 ? 1 : 0)`। यह truncation-toward-zero semantics account करता है — इसके बिना, negative dividends wrong direction में round करते।

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

`x / c` बनता है `x * (1/c)` float constant `c` के लिए।

Float multiply 1-2 cycles है (fully pipelined), जबकि float divide 10-20 cycles है (ज़्यादातर hardware पर pipelined नहीं)। यह common pattern के लिए straightforward 5-10x speedup है।

**Guards**:
- `c == 0.0` हो तो skip — division by zero IEEE 754 semantics preserve करने के लिए रहना चाहिए (`x / 0.0` `+/-inf` या `NaN` produce करता है)
- `1/c` finite नहीं हो तो skip (overflow to `inf` मतलब `c` बहुत छोटा है)
- सिर्फ़ float types

Tinygrad: `decompositions.py:477-479` (FDIV-based backends `RECIP` को `1/x` emit करते हैं)। Svod: `pm_fdiv_to_mul` in `rangeify/patterns.rs`।

```c
// Before
float result = x / 3.14159f;

// After
float result = x * 0.31831f;  // 1/pi
```

---

## 4. FMA Fusion (Fused Multiply-Add)

`a * b + c` बनता है `MULACC(a, b, c)`।

यह hardware FMA instructions में map होता है (x86 AVX पर `vfmadd`, ARM NEON पर `fmadd`, CUDA पर `fma.rn`)। Single instruction दो replace करता है, दो rounding steps की जगह एक — जो FMA को separate multiply + add से तेज़ और ज़्यादा precise दोनों बनाता है।

**Late apply क्यों**: Earlier passes को `Add(Mul(a, b), c)` structure algebraic simplification के लिए देखना चाहिए। अगर early fuse हो तो `(x*2 + x*3)` जैसे पैटर्न `x*5` में simplify नहीं कर पाते क्योंकि `Mul` nodes MULACC के अंदर buried हो जाते।

**Shift-add fusion**: `(x << n) + c` भी `MULACC(x, 2^n, c)` में fuse होता है, ऐसे cases पकड़ता है जहाँ MUL-to-SHL same fixed-point pass में पहले चल गया। जब renderer `MulAcc` और `Shl` दोनों सपोर्ट करता है, Svod `pm_shl_add_to_mulacc` को `pm_fma_decomposition` के साथ जोड़ देता है।

**Guards**: सिर्फ़ तभी match जब तीनों operands (`a`, `b`, `c`) same float dtype share करें। Integer FMA fuse नहीं होता क्योंकि hardware FMA instructions सिर्फ़ float हैं।

Tinygrad: `decompositions.py:472-475`। Svod: `pm_fma_decomposition` in `rangeify/patterns.rs`।

---

## 5. Negation Extraction

`x * -1` बनता है `NEG(x)`।

NEG single instruction है (float के लिए `xorps` से sign bit flip, int के लिए `neg`)। -1 से multiplication unnecessarily multiplier pipeline को 3-4 cycles occupy करता है।

सिर्फ़ तब fire जब backend `NEG` native op support करता हो। Tinygrad: `decompositions.py:458-459`। Svod: `pm_neg_from_mul`।

---

## 6. Comparison Negations

Integers पर negated और compound comparisons के late rewrites। ये पैटर्न ऐसी instruction sequences simplify करते हैं जो earlier passes में boolean logic optimizations से arise होती हैं।

| पैटर्न | Before | After | Savings |
|--------|--------|-------|---------|
| `!(x < c)` | NOT + CMP | `(c-1) < x` | NOT eliminate |
| `!(c < x)` | NOT + CMP | `x < (c+1)` | NOT eliminate |
| `(c1 < x) & (x < c2)` जहाँ `c2 == c1+2` | 2 CMPs + AND | `x == (c1+1)` | 2 ops eliminate |
| `x * -1 < c` | MUL + CMP | `-c < x` | MUL eliminate |
| `x * -1 < y * c` | 2 MULs + CMP | `y * (-c) < x` | 1 MUL eliminate |

Range compression (row 3) particularly valuable है। जब open interval `(c1, c2)` exactly एक integer value contain करे, दो comparisons और logical AND single equality check में collapse हों। यह tiled index calculations में naturally arise होता है जहाँ range variable exactly एक tile select करता है।

:::caution[Constants में Integer Overflow]
Negation patterns overflow guard करते हैं: `!(x < c)` बनता है `(c-1) < x` सिर्फ़ अगर `c-1` underflow न करे, और `!(c < x)` बनता है `x < (c+1)` सिर्फ़ अगर `c+1` overflow न करे। दोनों `checked_sub` / `checked_add` इस्तेमाल करते हैं और overflow पर `None` (कोई transformation नहीं) return करते हैं।
:::

Tinygrad: `decompositions.py:461-470`। Svod: `pm_comparison_negations` in `rangeify/patterns.rs`।

---

## 7. De Morgan's Laws (Late)

```
!a & !b  -->  !(a | b)
!a | !b  -->  !(a & b)
```

ये pipeline में *दो* जगह दिखते हैं:

1. **Early** (Stage 4-5): `boolean_dsl_patterns()` in `schedule/src/symbolic/patterns.rs`, full `symbolic()` matcher का हिस्सा। Original expression structure में De Morgan opportunities पकड़ता है।

2. **Late** (Stage 18-19): `symbolic_simple()` boolean patterns include करता है और `PM_FINAL` में strength reduction patterns के साथ चलता है। Comparison negation patterns से बनी नई De Morgan opportunities पकड़ता है — जैसे, `!(x < 3)` और `!(x < 7)` के `2 < x` और `6 < x` में rewrite होने के बाद, उन्हें combine करने वाले AND/OR में अब NOT-elimination के नए मौके हो सकते हैं।

Svod: `boolean_dsl_patterns()` in `schedule/src/symbolic/patterns.rs`।

---

## 8. ERF Decomposition

`erf(x)` polynomial approximation से replace होता है (Abramowitz & Stegun 7.1.26):

```
erf(x) = sign(x) * (1 - t * P(t) * exp(-x^2))
where t = 1 / (1 + 0.3275911 * |x|)
      P(t) = Horner(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592])
```

**क्यों**: `@llvm.erf` एक libcall intrinsic है (libm linkage ज़रूरी), native hardware instruction नहीं। LLVM JIT backend libm link नहीं करता, इसलिए `erf` codegen से पहले decompose होना चाहिए। Tinygrad `erf` tensor level पर decompose करता है (`elementwise.py`), तो renderer तक कभी पहुँचता ही नहीं; Svod `Erf` को UOp के रूप में इस late pass तक रखता है।

Maximum error: ~1.5e-7 (float32 ML workloads के लिए sufficient)।

Svod: `pm_erf_decomposition` in `rangeify/patterns.rs`।

---

## Pattern Composition: हर पैटर्न कब चलता है

सभी strength reduction patterns single `PM_FINAL` matcher में compose होते हैं जो fixed-point graph rewrite के रूप में चलता है:

```
PM_FINAL = pm_commit_weak() + pm_cast_weak() + pm_decomp
         + renderer.extra_matcher()   -- optional, per-backend
         + pm_split_ends()
```

जहाँ `pm_decomp` early decompositions (जो खुद `symbolic_simple()` से शुरू होती हैं), transcendental decompositions और `get_late_rewrite_patterns()` को जोड़ता है:

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
  + pm_comparison_negations      -- !(x<c) -> (c-1)<x, आदि
  + pm_fma_decomposition         -- a*b+c -> MULACC(a,b,c)
  + pm_shl_add_to_mulacc         -- (x<<n)+c -> MULACC(x, 2^n, c)
  + pm_fdiv_to_mul               -- x / c -> x * (1/c)
```

क्योंकि rewriter fixed point तक चलता है, पैटर्न एक दूसरे को feed कर सकते हैं। जैसे:

1. `pm_mul_to_shl` `x * 4` को `x << 2` convert करता है
2. अगली iteration में, `pm_fma_decomposition` `(x << 2) + c` को `MULACC(x, 4, c)` में fuse करता है
3. `symbolic_simple()` transformations से बनी identities clean up करता है

`pm_mod_to_and` के नीचे की हर entry conditional है: `get_late_rewrite_patterns()` उसे तभी जोड़ता है जब renderer उन ops का सपोर्ट declare करे जो वह पैटर्न पैदा करता है (shift rewrites के लिए `Shl`, FMA fusions के लिए `MulAcc`, reciprocal rewrite के लिए `Fdiv`)। `pm_split_ends` उसी fixed point के अंदर चलता है और linearizer के लिए मल्टी-रेंज END को nested सिंगल-रेंज END में बाँटता है; इसके बाद आख़िरी `pm_remove_invalid()` rewrite बचे हुए Invalid markers हटा देता है।

Cross-reference: [Codegen Pipeline Overview](../codegen/overview.md) full stage listing के लिए।
