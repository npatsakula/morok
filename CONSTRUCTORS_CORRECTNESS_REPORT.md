# UOp Constructors Correctness Report

**Analysis Date:** 2025-02-05
**Scope:** Morok's UOp constructors vs Tinygrad's implementation
**Files Compared:** 8 modules in `ir/src/uop/constructors/`

---

## Executive Summary

After spawning 8 focused agents to thoroughly compare each constructor module with Tinygrad's reference implementation, we found **9 correctness bugs** across 3 modules:

| Module | Bugs Found | Severity |
|--------|-----------|----------|
| `compute.rs` | 4 | 1 Critical, 2 High, 1 Medium |
| `shape.rs` | 2 | 2 Medium |
| `hardware.rs` | 2 | 1 High, 1 Medium |
| `control.rs` | 1 | **CRITICAL** |
| `data.rs` | 0 | - |
| `memory.rs` | 0 | - |
| `reduce.rs` | 0 | - |
| `graph.rs` | 0 | - |

**Total:** 9 correctness bugs (1 Critical, 3 High, 5 Medium)

---

## Critical Bugs (Must Fix Before Any Release)

### Bug #1: IF operation linearization produces incorrect instruction order

**Module:** `control.rs`
**Root Cause:** `/home/mrpink/projects/morok/ir/src/op.rs:366-370`

**Tinygrad's correct behavior:**
- IF operation has `src=(condition, first_body_element)` where `first_body_element` is only ONE element
- Body elements beyond the first are NOT in IF's `src`
- They appear in linear sequence BETWEEN IF and ENDIF
- Linearization produces correct order: **IF → body_elements → ENDIF**

**Morok's incorrect behavior:**
- `If { condition, body }` stores ALL body elements inline
- The `children()` method returns: `[condition, ...body_elements...]`
- This makes ALL body elements dependencies of IF
- Linearizer decrements `out_degree` for all children when scheduling a node
- Body elements are scheduled BEFORE IF because `in_degree[IF]` is high and `out_degree[body_elements]` becomes 0 early
- Linearization produces incorrect order: **body_elements → IF → ENDIF**

**Impact:** **CRITICAL** - The body of conditional blocks executes BEFORE the IF check, making conditional code completely unconditional. Operations that should execute only when `gate=true` will execute unconditionally.

**Fix required:**
The `If` op should only store the first body element in `src`, with additional body elements appearing in sequence between IF and ENDIF.

---

### Bug #2: Binary operations fail for Index operands

**Module:** `compute.rs`
**File:** `ir/src/uop/constructors/compute.rs:34-36`

**Tinygrad's correct behavior:**
Binary operations use the **last operand's dtype**:
```python
Int32(5) + Index(3) → Index
Index(3) + Int32(5) → Int32
```

**Morok's incorrect behavior:**
Uses `promote_and_cast` which calls `least_upper_dtype`. Index has empty promotion lattice, causing `TypePromotionFailed`.

**Impact:** **CRITICAL** - Loop index arithmetic and operations involving Index types will fail completely.

**Fix required:**
Binary operations should use the last operand's dtype instead of type promotion, matching Tinygrad's `alu` method behavior.

---

## High Priority Bugs

### Bug #3: Shift operations use LHS dtype instead of last operand dtype

**Module:** `compute.rs`
**File:** `ir/src/uop/constructors/compute.rs:76`

**Tinygrad's correct behavior:**
Uses last operand's dtype:
```python
Int32(5) << Index(3) → Index
```

**Morok's incorrect behavior:**
`let dtype = self.dtype()` uses LHS dtype → returns `Int32`

**Impact:** **HIGH** - Shift operations produce wrong result dtype when LHS and RHS have different types.

---

### Bug #4: Where operation uses true_val dtype instead of last operand dtype

**Module:** `compute.rs`
**File:** `ir/src/uop/constructors/compute.rs:346`

**Tinygrad's correct behavior:**
Uses last operand's dtype (false value):
```python
cond.where(Int32, Float32) → Float32
```

**Morok's incorrect behavior:**
`let dtype = true_val.dtype()` uses true value's dtype → returns `Int32`

**Impact:** **HIGH** - Conditional selection produces wrong result dtype.

---

### Bug #5: Unchecked broadcast allows non-scalar sources

**Module:** `hardware.rs`
**File:** `ir/src/uop/constructors/hardware.rs:104-110`

**Tinygrad's correct behavior:**
```python
def broadcast(self, count:int):
  assert self.dtype.vcount == 1  # Enforces scalar-only
```

**Morok's incorrect behavior:**
No validation that `self.dtype().vcount() == 1`, creating semantically incorrect "vector of vectors".

**Impact:** **HIGH** - Broadcasting a vector creates incorrect semantics where flattening or proper casting should occur.

---

## Medium Priority Bugs

### Bug #6: MulAcc operation uses first operand dtype instead of last operand dtype

**Module:** `compute.rs`
**File:** `ir/src/uop/constructors/compute.rs:353`

**Tinygrad's correct behavior:**
Uses last operand's dtype (accumulator):
```python
MulAcc(Float32(2), Float32(3), Int32(4)) → Int32
```

**Morok's incorrect behavior:**
`let dtype = a.dtype()` uses first operand's dtype → returns `Float32`

**Impact:** **MEDIUM** - Multiply-accumulate produces wrong result dtype when accumulator differs from other operands.

---

### Bug #7: WMMA dtype calculation from upcast_axes causes panic

**Module:** `hardware.rs`
**File:** `ir/src/uop/constructors/hardware.rs:39`

**Tinygrad's correct behavior:**
WMMA dtype is explicitly specified and passed directly to the UOp constructor.

**Morok's incorrect behavior:**
Incorrectly calculates `vec_size` by flattening all upcast_axes, then calls `.vec(vec_size)` on `metadata.dtype_out`, which **panics** if dtype_out is already a vector type.

**Impact:** **MEDIUM** - When `metadata.dtype_out` is already a vector type, calling `.vec()` causes a panic.

**Fix required:**
```rust
pub fn wmma(a: Arc<Self>, b: Arc<Self>, c: Arc<Self>, metadata: WmmaMetadata) -> Arc<Self> {
    let dtype = metadata.dtype_out.clone();
    Self::new(Op::Wmma { a, b, c, metadata }, dtype)
}
```

---

### Bug #8: Expand validation rejects ns=0

**Module:** `shape.rs`
**File:** `ir/src/uop/constructors/shape.rs:128-129`

**Tinygrad's correct behavior:**
```python
s==ns or (s==1 and ns>=0)
```
Allows expanding from dimension 1 to any size ≥ 0, including 0.

**Morok's incorrect behavior:**
```rust
ensure!(
    s == ns || (s == 1 && ns >= 1),
    ExpandInvalidDimensionSnafu { dim: dim_idx, input: s, output: ns }
);
```
Rejects expanding when `ns == 0`.

**Impact:** **MEDIUM** - Prevents valid operations like expanding `(1, 1)` → `(1, 0)`. Zero-sized dimensions are valid in NumPy, PyTorch, and Tinygrad.

---

### Bug #9: Reshape validation rejects zero dimensions

**Module:** `shape.rs`
**File:** `ir/src/uop/constructors/shape.rs:84`

**Tinygrad's correct behavior:**
```python
if not all(x >= 0 for x in self.marg): raise ValueError(...)
```
Allows zero dimensions (checks `>= 0`, not `> 0`).

**Morok's incorrect behavior:**
```rust
ensure!(val > 0, ReshapeNegativeDimensionSnafu { shape: vec![val as isize] });
```
Rejects zero dimensions with `val > 0`.

**Impact:** **MEDIUM** - Prevents valid reshapes with zero dimensions.

---

## Modules with No Correctness Issues Found

### data.rs ✅
- Const creation, const hashing, buffer creation, buffer views, device specifications, const_like, and VCONST dtype inference all match Tinygrad's behavior correctly.

### memory.rs ✅
- INDEX dtype behavior, index gating, load/store dtype handling, bufferize options, copy semantics, memory definitions, GEP, and address spaces are all correct.

### reduce.rs ✅
- ReduceAxis early-return, ReduceAxis vs Reduce distinction, AllReduce device handling, reduce operations, axis validation, empty reductions, and shape propagation are all correct.

### graph.rs ✅
- SINK, GROUP, ASSIGN, AFTER, CONTIGUOUS, DETACH, CUSTOM/CUSTOMI, and PreCast all have semantically equivalent implementations. Differences are design choices, not bugs.

---

## Root Cause Analysis

### Primary Issue: Dtype Selection Strategy

The most significant correctness issue (affecting bugs #2, #3, #4, #6) stems from a fundamental difference in dtype selection:

- **Morok's approach:** Uses `promote_and_cast` for all binary operations, which calls `least_upper_dtype`
- **Tinygrad's approach:** Uses the **last operand's dtype** in `alu` method
- **Tinygrad's type promotion:** Only applies `least_upper_dtype` in specific lowering patterns (`pm_lower_index_dtype`), not in the default `alu` behavior

**Impact:** This affects ALL binary operations (arithmetic, bitwise, shift, comparison, ternary), making operations involving Index types fail completely.

### Secondary Issue: Control Flow Linearization

The IF operation linearization bug (#1) is caused by Morok storing ALL body elements inline vs Tinygrad storing only the first body element in IF's `src`. This affects ALL conditional code, making it execute unconditionally.

### Tertiary Issues: Validation Completeness

Several bugs (#5, #7, #8, #9) stem from missing or overly strict validation that doesn't match Tinygrad's behavior:
- Broadcast missing scalar-only validation
- WMMA incorrectly calculating vector size from upcast_axes
- Expand/reshape rejecting zero dimensions

---

## Recommended Fix Priority

### Phase 1: Critical Fixes (Do Immediately)
1. Fix IF operation linearization (#1) - Affects ALL conditional code
2. Fix binary operations dtype selection (#2) - Affects ALL operations involving Index types

### Phase 2: High Priority Fixes
3. Fix shift operations dtype (#3)
4. Fix where operation dtype (#4)
5. Add broadcast scalar-only validation (#5)

### Phase 3: Medium Priority Fixes
6. Fix MulAcc dtype (#6)
7. Fix WMMA dtype calculation (#7)
8. Allow expand with ns=0 (#8)
9. Allow reshape with zero dimensions (#9)

---

## Verification Plan

After fixes, create tests for:

1. **Conditional execution order** - Validates IF linearization fix
2. **Index type operations** - Validates binary operations with Index operands
3. **Shift result dtype** - Validates shift dtype selection
4. **Where result dtype** - Validates where dtype selection
5. **Broadcast scalar-only** - Validates broadcast validation
6. **MulAcc result dtype** - Validates MulAcc dtype selection
7. **WMMA with vector dtype_out** - Validates WMMA dtype handling
8. **Expand to zero dimensions** - Validates expand fix
9. **Reshape with zero dimensions** - Validates reshape fix

---

## Conclusion

The analysis found **9 correctness bugs** across 3 modules. The most critical issues are:

1. **IF linearization bug** - Makes ALL conditional code execute unconditionally
2. **Binary operations bug** - Makes ALL operations involving Index types fail

Both of these are fundamental issues that would cause widespread incorrect results and must be fixed before any release.

The other bugs are significant but more localized in their impact. The fact that 5 out of 8 modules have **zero correctness issues** shows that the overall constructor implementation is sound and follows Tinygrad's design closely.
