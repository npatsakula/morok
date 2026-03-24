---
sidebar_label: इंडेक्स अरिथमेटिक
---

# इंडेक्स अरिथमेटिक

Tensor compilers अपने ऑप्टिमाइज़ेशन बजट का ज़्यादातर हिस्सा index arithmetic पर खर्च करते हैं। `tensor[i, j]` एक्सेस shape `[H, W]` के साथ `i * W + j` बन जाता है। Tiling, vectorization, और loop transformations के बाद, ये expressions nested divisions और modulos जमा करती हैं। इन्हें simplify करना critical है — एक unnecessary `idiv` की cost ~20-40 cycles है बनाम equivalent shift के लिए 1 cycle (approximate, modern x86-64)।

यह पेज index expressions simplify करने वाले पैटर्न document करता है। ये traditional sense में "optimizations" नहीं हैं — ये वो algebra है जो tensor indexing को efficiently काम कराता है।

**Key concept — value range analysis**: हर UOp अपनी runtime minimum (`vmin`) और maximum (`vmax`) values ट्रैक करता है, जो नोड construction के दौरान inputs के bounds से eagerly compute होती हैं। कई index पैटर्न इन bounds का इस्तेमाल compile time पर simplifications prove करने के लिए करते हैं (जैसे, "`x` हमेशा `[0, N)` में है" तो `x % N` → `x` enable होता है)।

ये पैटर्न [codegen pipeline](../codegen/overview.md) के कई stages पर चलते हैं:
- **Stage 4** (initial symbolic, rangeify के दौरान)
- **Stage 8** (post-optimization symbolic)
- **Stage 15** (index dtype lowering via `pm_lower_index_dtype`)
- **Stage 16** (post-index symbolic)

Morok source: `schedule/src/symbolic/patterns.rs`, `schedule/src/symbolic/index_lowering.rs`

Tinygrad source: `tinygrad/uop/divandmod.py`, `tinygrad/uop/symbolic.py`

---

## 1. Div-Mod Identity

Integer division का fundamental theorem:

$$
x = \lfloor x / n \rfloor \cdot n + (x \bmod n)
$$

पाँच variants इस identity को pattern set में exploit करते हैं:

| # | पैटर्न | Condition | नाम |
|---|--------|-----------|-----|
| 1 | `x%n + (x//n)*n` -> `x` | -- | Core identity |
| 2 | `((x//a) % c) + (x//b)*c` -> `x//a` | `a*c == b` | Composed divisor |
| 3 | `(x%c1)*c2 + (x//c1)*c3` -> `x*c2` | `c1*c2 == c3` | Scaled |
| 4 | `y + (x%n) + (x//n)*n` -> `y + x` | -- | Three-term |
| 5 | `(a//c1 + c2) // c3` -> `(a + c1*c2) // (c1*c3)` | `c1>0, c3>0` | Nested division |

**#1 का Proof।** Division algorithm से, integers `x` और `n > 0` के लिए, unique integers `q` और `r` exist करते हैं जहाँ `x = q*n + r` जहाँ `0 <= r < n`। Definition से, `q = x // n` और `r = x % n`। Substitute करने पर: `(x % n) + (x // n) * n = r + q*n = x`। QED.

**#2-#5 corollaries क्यों हैं।**

Variant #2 division के दो levels compose करता है। चूँकि `b = a*c`, हमें `x // b = (x // a) // c` मिलता है। Inner level पर core identity apply करने से: `((x//a) % c) + ((x//a) // c) * c = x // a`। लेकिन `(x//a) // c = x // (a*c) = x // b`, जो पैटर्न देता है।

Variant #3 core identity के दोनों sides को `c2` से scale करता है। `x = (x % c1) + (x // c1) * c1` से, `c2` से multiply करने पर: `x * c2 = (x % c1) * c2 + (x // c1) * c1 * c2`। चूँकि `c1 * c2 = c3`, यह बनता है `(x % c1) * c2 + (x // c1) * c3 = x * c2`।

Variant #4 एक independent term `y` को #1 के दोनों sides में add करता है।

Variant #5 nested floor division flatten करता है। `(a // c1 + c2) // c3` दिया हो तो, `c2` को inner divisor से multiply करके equivalent single-level division मिलता है: `(a + c1*c2) // (c1*c3)`। यह तब hold करता है जब `a >= 0` और `c2 >= 0` (या दोनों non-positive), ताकि floor division semantics preserve हों।

पाँचों पैटर्न duplicate variable names पर `Arc::ptr_eq` चेक इस्तेमाल करते हैं (जैसे, `x` दो बार दिखने का मतलब दोनों एक ही hash-consed node होने चाहिए)।

### Implementation

```rust
// From schedule/src/symbolic/patterns.rs — div_mod_recombine_dsl_patterns()

// #1: x%n + (x//n)*n -> x
Add[Mod(x, n), Mul[Idiv(x, n), n]] ~> |x| Arc::clone(x),

// #2: ((x//a) % c) + (x // b) * c -> x // a  when a*c == b
Add[Mod(Idiv(x, a), c), Mul[Idiv(x, _b), c]]
    => |x, a, a_val, c_val, b_val| { /* guard: a_int * c_int == b_int */ },

// #5: (a//c1 + c2) // c3 -> (a + c1*c2) // (c1*c3)
Idiv(Add[Idiv(a, c1), _c2], _c3)
    => |a, c1, c1_val, c2_val, c3_val| { /* guard: c1>0, c3>0, same-sign */ },
```

---

## 2. Range-Based Mod/Div

Value range analysis (`vmin`/`vmax`) ऐसी simplifications enable करता है जो purely syntactic pattern matching को दिखाई नहीं देतीं। हर UOp construction के दौरान compute किए गए cached bounds रखता है।

| पैटर्न | गार्ड | Example |
|--------|------|---------|
| `x % n` -> `x` | `0 <= vmin(x)` और `vmax(x) < n` | `RANGE(3) % 3` -> `RANGE(3)` |
| `(a*m + b) % n` -> `b % n` | `m == n` | `(row*512 + col) % 512` -> `col % 512` |
| `(a*m + b) / n` -> `a + b/n` | `m == n` और `0 <= b < n` | `(row*512 + col) / 512` -> `row` |
| `x / n` -> `k` | सभी values bucket `[k*n, (k+1)*n)` में | `RANGE(3) / 3` -> `0` |
| `(x + c) // d` -> `x // d` | `max_remainder + c < d` | `(R*4 + 1) // 8` -> `R*4 // 8` |
| `(x + c) // d` -> `(x + c%d) // d + c//d` | `c >= d` | `(x + 70) // 8` -> `(x + 6) // 8 + 8` |

पहला पैटर्न workhorse है। Range splitting के बाद, `RANGE(n)` values `[0, n)` में produce करता है, तो `RANGE(n) % n` trivially `RANGE(n)` में simplify होता है। यह single rule tiling से बनी ज़्यादातर modulos eliminate कर देता है।

पाँचवाँ पैटर्न (small constant) range `[vmin, vmax]` के अंदर maximum remainder पर tight bound इस्तेमाल करता है। अगर range `d` values से कम span करती है और `c` add करने से कभी bucket boundary cross नहीं होती, तो constant dead weight है।

छठा पैटर्न (large offset split) divisor से बड़े offsets canonicalize करता है। यह अगली rewrite iteration के लिए small-constant पैटर्न expose करता है।

:::caution
`(a*m + b) / n` -> `a + b/n` पैटर्न को `0 <= b < n` चाहिए। Range check के बिना, negative remainders truncation-toward-zero semantics से incorrect quotients produce करते हैं। Implementation explicitly `vmin(b) >= 0 && vmax(b) < n` चेक करता है।
:::

---

## 3. `fold_divmod_general` एल्गोरिदम

Index-dtype `Idiv` और `Mod` के लिए catch-all। Tinygrad के `divandmod.py:8-93` के सभी 8 rules priority order में implement करता है, recursive `nest_div_by_smallest_factor` सहित। हर rule sequence में try होता है; पहला match जीतता है।

Entry point: जब `Idiv(x, y)` या `Mod(x, y)` का `dtype == Index` हो, पैटर्न `fold_divmod_general(op, x, y)` को delegate करता है।

### Rule 1 -- cancel_divmod

अगर पूरी range `[x_min, x_max]` `(x, y)` के सभी corner combinations में single quotient को map करती है, तो result वो constant है।

**गार्ड**: `y_min * y_max > 0` (denominator कभी zero cross नहीं करता), और सभी four corner quotients `x_min/y_min`, `x_min/y_max`, `x_max/y_min`, `x_max/y_max` equal हैं।

**क्या करता है**: `Idiv` के लिए, constant quotient return करता है। `Mod` के लिए, `x - q*y` return करता है।

**Example**: `RANGE(3) // 3` -> `0`। Values 0, 1, 2 सभी 0 को divide करती हैं।

### Rule 2 -- remove_nested_mod

`(a%4 + b) % 2` -> `(a + b) % 2` जब `2 | 4`। Outer modulus inner को divide करता है, तो inner modulus redundant है।

**गार्ड**: `op == Mod`, `x_min >= 0`, और हर term जो `Mod(inner_x, inner_y)` है, denominator `y` `inner_y` को divide करता है।

**क्या करता है**: Inner `Mod` operations strip करता है जिनका modulus outer modulus का multiple है, फिर `Mod` re-apply करता है।

**Example**: `(RANGE(8) % 4 + RANGE(2)) % 2` -> `(RANGE(8) + RANGE(2)) % 2`

### Rule 3 -- fold_binary_numerator

जब single non-constant term के exactly 2 values हों (`vmax - vmin == 1`), result एक linear interpolation है: `(y2 - y1) * (v - v_min) + y1`।

**गार्ड**: Decomposition के बाद exactly एक non-constant term, और उस term की range exactly 2 values span करती है।

**क्या करता है**: Div/mod को दोनों endpoints पर evaluate करता है और उनके बीच linear map construct करता है। यह division पूरी तरह avoid करता है।

**Example**: `(v * 3 + 2) % 5` जहाँ `v` `{0, 1}` में है:
- `v=0`: `(0 + 2) % 5 = 2`
- `v=1`: `(3 + 2) % 5 = 0`
- Result: `(0 - 2) * (v - 0) + 2 = -2*v + 2`

### Rule 4 -- fold_divmod_congruence

हर term `f_i * v_i` के लिए, closest residue `r_i = min(f_i % c, f_i % c - c)` absolute value से compute करता है। अगर residue sum `c` के एक floor-division bucket में रहता है, तो mod/div simplify होता है। यह modular arithmetic ऑप्टिमाइज़ेशन है।

**गार्ड**: `x_min >= 0`, constant denominator `c > 0`, और `rem_min // c == rem_max // c` (सभी residue-sum values एक ही bucket में)।

**क्या करता है**: हर factor को उसके residue mod `c` से replace करता है। `Mod` के लिए, residue sum return करता है (bucket offset से adjusted)। `Idiv` के लिए, quotient-coefficient sum return करता है।

**Example**: `(r*8 + v) % 7` -> `(r + v) % 7` क्योंकि `8 = 1 (mod 7)`, तो `8` का residue `1` है।

### Rule 5 -- gcd_with_remainder

सभी additive terms और denominator का symbolic GCD compute करता है। अगर GCD > 1, तो factor out: `(g*a + g*b) // (g*c)` -> `(a + b) // c` (`Mod` के लिए remainder scale back)।

**गार्ड**: `x_min >= 0`, constant denominator, GCD > 1, और reduced numerator का `vmin >= 0`।

**क्या करता है**: Numerator terms और denominator दोनों को उनके GCD से divide करता है, recursively simpler पैटर्न fire होने देता है।

**Example**: `(6*a + 4*b) // 8` जहाँ `GCD(6, 4, 8) = 2` -> `(3*a + 2*b) // 4`

### Rule 6 -- divide_by_gcd

Rule 5 का variable denominator version। `GCD(all_terms..., y)` compute करता है numerator और denominator दोनों include करके, फिर दोनों sides divide करता है। Rule 5 से अलग, यह तब काम करता है जब denominator constant नहीं है।

**गार्ड**: GCD non-trivial है (1 नहीं), और `x` और `y` दोनों GCD से exactly divisible हैं।

**Example**: `(4*a) // (2*b)` -> `(2*a) // b`

### Rule 7 -- factor_remainder

Last resort। Terms को exactly-divisible (quotient) और remainder में partition करता है।

**गार्ड**: `x_min >= 0` और `y_min >= 0`, और कम से कम एक term `y` को exactly divide करता है।

**क्या करता है**: `Idiv` के लिए: `quo_sum + rem // y`। `Mod` के लिए: `rem % y` (constant `y` के लिए coefficient reduction)।

**Example**: `(8*a + 3*b) // 8` -> `a + (3*b) // 8`

### Rule 8 -- nest_div_by_smallest_factor

Constant divisors के लिए recursive decomposition। Divisor और किसी term के coefficient के बीच shared smallest factor ढूँढता है, दोनों को उससे divide करता है, फिर recurse करता है।

**गार्ड**: `x_min >= 0`, constant `y > 1`, और कम से कम एक non-constant term में factor `f > 1` है जहाँ `y % f == 0`।

**क्या करता है**: Qualifying factors में `div = min(|f|)` pick करता है, `x // y` को `(x // div) // (y / div)` में rewrite करता है। हर step `y` reduce करता है, rules 1-7 की तरफ़ converge करता है।

**Example**: `(6*a + 4*b) // 12` → `((6*a + 4*b) // 2) // 6` → `(3*a + 2*b) // 6` → `(3*a + 2*b) // 6` (फिर rule 7 finish करता है)।

Tinygrad: `divandmod.py:62-67`। Morok: `nest_div_by_smallest_factor` in `fold_divmod_general`।

:::caution
Rules 5-8 को non-negative numerators चाहिए (`x_min >= 0`)। Negative operands के साथ floor division की rounding semantics अलग होती है (Python/Tinygrad में negative infinity की तरफ़, hardware में zero की तरफ़)। Implementation negative ranges के लिए `None` return करता है, बाद के passes को expression handle करने देता है।
:::

---

## 4. Advanced Division पैटर्न

`fold_divmod_general` के बाहर standalone पैटर्न जो additional cases handle करते हैं:

| पैटर्न | गार्ड | Source |
|--------|------|--------|
| `(a // b) // c` -> `a // (b*c)` | `b != 0, c != 0` | `advanced_division_dsl_patterns` |
| `expr // divisor` -> exact quotient | `expr` exactly divisible है | `advanced_division_dsl_patterns` |
| `(a + b) % c` coefficient reduction | `a` या `b` में `c` से divisible factor | `advanced_division_dsl_patterns` |
| `(a + b) // c` -> `a//c + b//c` | दोनों evenly divide हों | `advanced_division_dsl_patterns` |
| `(a - b) // c` -> `a//c - b//c` | दोनों evenly divide हों | `advanced_division_dsl_patterns` |
| `c * (a + b)` -> `c*a + c*b` | `c` constant है | `advanced_division_dsl_patterns` |

Nested division collapse `(a // b) // c` -> `a // (b*c)` tiling के बाद particularly important है, जहाँ range को outer/inner components में split करने से division के दो levels बनते हैं जो एक में collapse होने चाहिए।

Exact-division पैटर्न `divides()` इस्तेमाल करता है जो चेक करता है कि हर additive term का constant factor divisor से divisible है या नहीं। सफ़ल होने पर, `Idiv` पूरी तरह eliminate — कोई division instruction emit नहीं।

Coefficient reduction पैटर्न `(r*8 + v) % 7` -> `(r*1 + v) % 7 = (r + v) % 7` convert करता है हर factor को modulus से reduce करके। यह तब fire करता है जब कोई factor modulus का exact multiple नहीं लेकिन residues छोटे हैं।

---

## 5. Index Dtype Lowering (3-Phase Cascade)

Tinygrad: `ops.py:1291-1313`। Morok: `schedule/src/symbolic/index_lowering.rs`।

Abstract `Index` type कोई width नहीं रखता — यह represent करता है "इस index के लिए जो भी integer width चाहिए।" Lowering pass `Index` को concrete `i32` या `i64` में convert करता है value bounds के basis पर।

### Phase 1 -- Wrappers बनाना (leaf nodes)

`Index` dtype वाले leaf nodes अपने concrete equivalent से replace होते हैं `Index` में cast back के साथ wrap:

| Input | Output |
|-------|--------|
| `CONST(Index)` | `CONST(concrete).cast(Index)` |
| `DEFINE_VAR(Index)` | `DEFINE_VAR(concrete).cast(Index)` |
| `VCONST(Vector<Index, N>)` | `VCONST(Vector<concrete, N>).cast(Vector<Index, N>)` |

### Phase 2 -- Wrapped values ऊपर process करना

Binary operations, control flow, और structural nodes `.cast(Index)` wrappers से concrete type propagate करते हैं:

| Input | Output |
|-------|--------|
| `Binary(x.cast(Index), y.cast(Index))` | `Binary(x.cast(dt), y.cast(dt)).cast(result_dtype)` |
| `WHERE(cond, x.cast(Index), y.cast(Index))` | `WHERE(cond, x.cast(dt), y.cast(dt)).cast(Index)` |
| `RANGE(end.cast(Index))` | `RANGE(end, end.dtype).cast(Index)` |
| `SPECIAL(end.cast(Index))` | `SPECIAL(end, i32).cast(Index)` |
| `VECTORIZE(e0.cast(Index), ...)` | `VECTORIZE(e0.cast(dt), ...).cast(Vector<Index, N>)` |
| `BIND(var.cast(Index), val.cast(Index))` | `var.cast(dt).bind(val.cast(dt)).cast(Index)` |

`dt` `least_upper_dtype(select_dtype(result), x.dtype, y.dtype)` से compute होता है — सबसे wide type जो किसी भी operand या result को चाहिए।

### Phase 3 -- Terminals पर wrappers strip करना

Terminal nodes index consume करते हैं और `Index` wrapper discard:

| Input | Output |
|-------|--------|
| `INDEX(buf, idx.cast(Index))` | `INDEX(buf, idx)` |
| `INDEX(buf, WHERE(cond, idx, Invalid))` | `INDEX(buf, idx, gate=cond)` |
| `SINK(sources with .cast(Index))` | `SINK(unwrapped sources)` |
| `END(computation.cast(Index))` | `END(unwrapped computation)` |

`WHERE(cond, idx, Invalid)` -> `gate=cond` transformation significant है: यह validity conditions को index expression से `INDEX` node के gate field में extract करता है, जिसे codegen backends predicated loads emit करने के लिए इस्तेमाल करते हैं।

### `select_dtype()`

`i32` return करता है अगर UOp के value bounds `[-2^31, 2^31 - 1]` में fit हों, otherwise `i64`। ज़्यादातर tensor indices `i32` में fit होती हैं — 2B-element tensor का flat index भी fit होता है। `i64` path बहुत बड़े tensors या accumulated offsets के लिए है।

---

## 6. Commutative Canonicalization

```rust
// For Index dtype ONLY:
op(a, b) -> op(b, a)   when b.id < a.id
```

यह ensure करता है कि commutative operations में deterministic operand order हो UOp के unique ID के basis पर। Apply होता है: `Add`, `Mul`, `Max`, `Eq`, `Ne`, `And`, `Or`, `Xor`।

**Index-only क्यों**: Canonicalization के बिना, `R1*8000 + R2*16` और `R2*16 + R1*8000` hash-consing के बाद distinct nodes हैं, जो `expand_vector_index` में grouping तोड़ता है। Expander को vector lanes में identical index patterns identify करने चाहिए, और non-canonical ordering इसे defeat करती है।

**Non-Index types पर apply क्यों नहीं**: Float/int arithmetic पर canonicalization apply करने से VECTORIZE elements reorder होते और बाद के passes में vector math merging टूट जाती। Tinygrad भी यही choice करता है (`symbolic.py:178-182`)।

:::caution
Canonicalization rewrite engine के fixed-point iteration से interact करता है। अगर दो पैटर्न operand order पर disagree करें (एक canonicalize करे, दूसरा non-canonical output produce करे), तो engine oscillate कर सकता है। सभी index-producing पैटर्न canonical order respect करने चाहिए, वरना 1000-iteration safety limit trigger होगी।
:::

---

## Worked Example

`tensor[i, j]` पर विचार करें shape `[4, 8]` के साथ, flat iteration से 32 elements पर access।

### Initial state

Range `R0` `0..32` iterate करता है (flat index)। Access pattern decompose होता है:

```text
row = R0 // 8       (which of the 4 rows)
col = R0 % 8        (which of the 8 columns)
addr = row * 8 + col = (R0 // 8) * 8 + (R0 % 8)
```

Div-mod identity (#1) से, `(R0 // 8) * 8 + (R0 % 8) = R0`। Address बस flat index है — कोई division ज़रूरत नहीं।

### Tiling के बाद (UPCAST by 4)

Range splitting `R0` को `R1 * 4 + R2` में decompose करता है जहाँ `R1` `[0, 8)` में और `R2` `[0, 4)` में:

```text
row = (R1*4 + R2) // 8
col = (R1*4 + R2) % 8
```

**`row` simplify करना**: Expression `(R1*4 + R2) // 8` `fold_divmod_general` में enter करता है।

Rule 4 (congruence) fire होता है: factor `4` का residue `4 % 8 = 4`, और `R2` का residue `1 % 8 = 1`। Residue sum `4*R1 + R2` range `[0, 31]` के साथ। चूँकि `0 // 8 != 31 // 8`, Rule 4 इसे constant में collapse नहीं करता। Rule 7 (factor remainder) fire होता है: `4` `8` को exactly divide नहीं करता, लेकिन expression decompose हो सकता है। चूँकि कोई term 8 exactly divide नहीं करता, हम range-based pattern `(a*m + b) / n` पर fall through करते हैं `m = 4, n = 8` के साथ — यह match नहीं (`m != n`)।

Expression `(R1*4 + R2) // 8` ही रहता है। Generated code में, अगर `R2` vectorized है (UPCAST), backend इसे 4-wide vector की single division emit करता है।

लेकिन, अगर हम `R1` को आगे `R3 * 2 + R4` में split करें (जहाँ `R3` `[0, 4)` में, `R4` `[0, 2)` में):

```text
row = (R3*2*4 + R4*4 + R2) // 8
    = (R3*8 + R4*4 + R2) // 8
```

अब range-based pattern `(a*m + b) / n` fire करता है `m = n = 8` के साथ:
- `a = R3`, `b = R4*4 + R2`
- `vmin(b) = 0`, `vmax(b) = 1*4 + 3 = 7 < 8`
- Result: `R3 + (R4*4 + R2) // 8`

और `(R4*4 + R2) // 8`: `vmax = 1*4 + 3 = 7`, `vmin = 0`, तो `0 // 8 = 7 // 8 = 0`। cancel_divmod rule fire करता है:
- Result: `R3 + 0 = R3`

**`col` simplify करना**: `(R3*8 + R4*4 + R2) % 8`

Range-based pattern `(a*m + b) % n` fire करता है `m = n = 8` के साथ:
- `(R3*8 + R4*4 + R2) % 8` -> `(R4*4 + R2) % 8`

फिर `vmin(R4*4 + R2) = 0`, `vmax(R4*4 + R2) = 7 < 8`, तो `x % n` -> `x`:
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

पूरी address calculation एक linear expression में collapse हो जाती है — zero divisions, zero modulos। पैटर्न ने prove कर दिया कि tiled index flat index के equivalent है, purely algebraic rewriting से।
