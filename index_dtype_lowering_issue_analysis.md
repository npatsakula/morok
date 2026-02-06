# Index Dtype Lowering Issue Analysis

## Problem Summary

Test `test_matmul_validated_3x3` fails with LLVM JIT compilation error:
```
error: %r0' defined with type 'i64' but expected 'i32'
  %v16 = mul i32 3, %r0
```

The error occurs because `%r0` (a RANGE) has type `i64` but the multiplication uses `i32 3 * %r0`.

---

## Root Cause

After Stage 15 (Index dtype lowering), the AST contains:
```
Mul : Vector { scalar: Index, count: 3 }
```

This Vector operation has an Index scalar that should have been lowered to i32/i64 but wasn't.

### Why It Wasn't Lowered

Looking at `schedule/src/symbolic/index_lowering.rs:192-203`, Pattern 7 for VECTORIZE of Index elements:

```rust
vec @ Vectorize { elements } if vec.dtype().base() == ScalarDType::Index => |vec, elements| {
    let target_dtype = select_concrete_dtype(vec);
    let lowered_elements: Vec<_> = elements
        .iter()
        .map(|e| e.cast(target_dtype.clone()))
        .collect();
    Some(UOp::vectorize(lowered_elements.into()))
},
```

The pattern only matches when `vec.dtype().base() == ScalarDType::Index`. However, the Vector dtype is:
- `Vector { scalar: Float32, count: 3 }` → base is `Float32`, pattern DOESN'T match
- `Vector { scalar: Int64, count: 3 }` → base is `Int64`, pattern DOESN'T match

The pattern fails to match because the vector's scalar is NOT `Index` type after some transformation (possibly GEP extraction or CAST).

---

## What Tinygrad Does Differently

Tinygrad runs `pm_lower_index_dtype + load_store_indexing` as a single phase (codegen/__init__.py:84):

```python
sink = graph_rewrite(sink, pm_lower_index_dtype+load_store_indexing, ctx=ren.device, name="lower all index dtypes")
```

The `load_store_indexing` pattern matcher (devectorizer.py:227-246) includes:

```python
load_store_indexing = PatternMatcher([
    # simplify away long after index has been lowered
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("x", dtypes.long), UPat.var("c", dtypes.bool))), lambda buf,x,c: simplify_valid_load(buf, x, c)),
    # drop true gate
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("x"), UPat.const(dtypes.bool, True))), lambda buf,x: buf.index(x, ptr=True)),
    # + expand_index, GEP movement, PTRCAT distribution patterns
])
```

Key observation: The pattern uses `dtypes.long` (i64) for the index and calls `simplify_valid_load(buf, x, c)`.

The `simplify_valid_load` function (devectorizer.py:227-246) creates:
```python
def simplify_valid_load(buf:UOp, start_idx:UOp, valid:UOp) -> UOp|None:
    idx = uop_given_valid(valid, start_idx)
    if not isinstance(buf.dtype, ImageDType): return None if idx is start_idx else buf.index(idx.valid(valid), ptr=True)
    # ...
```

The critical line is `buf.index(idx.valid(valid), ptr=True)` - this creates a new INDEX with the dtype derived from `buf.dtype` and `idx.dtype`.

When `idx` has dtype `dtypes.long` (i64), and `buf.dtype` is something (e.g., Float32), the resulting INDEX gets a dtype that's the "widening" of the two.

This means that in Tinygrad:
1. Vector INDEX with Index/Int64 indices gets transformed
2. The `simplify_valid_load` pattern creates a new INDEX where the dtype is correctly inferred
3. This new INDEX gets properly lowered by `pm_lower_index_dtype`

---

## Why Morok's Current Approach Fails

Morok's `expand_index` function (devectorize.rs:536-574) calls:

```rust
index.expand(0..cnt as i64)
```

This creates individual INDEX operations with dtype `Index`. It doesn't:
1. Match vectorized INDEX with Index/Int64 indices
2. Apply type conversion like Tinygrad's `simplify_valid_load`

The issue is that `expand_index` is designed for the devectorization phase, but it doesn't include the dtype conversion logic that's needed for vectors with Index scalars.

---

## Stage Ordering Analysis

### Tinygrad Pipeline
```
sink = graph_rewrite(sink, pm_lower_index_dtype + load_store_indexing, ...)
```

Both patterns run in the same graph rewrite pass. When `load_store_indexing` transforms a vector INDEX, the result can immediately be matched by `pm_lower_index_dtype` patterns within the same pass.

### Morok Pipeline
```rust
// Stage 15: Index dtype lowering
let with_lowered_idx = graph_rewrite(&crate::symbolic::pm_lower_index_dtype(), with_gpudims, &mut ());

// Stage 16: Post-index symbolic
let with_lowered_idx = graph_rewrite(&symbolic_simple(), with_lowered_idx, &mut ());

// Stage 18: Devectorization (includes expand_index)
let with_lowered_idx = graph_rewrite(&devectorize(), with_lowered_idx, &mut ());
```

The problem is:
1. Stage 15 runs BEFORE devectorization
2. Stage 18 (`devectorize()`) runs AFTER Stage 15
3. The `expand_index` pattern in `devectorize()` only expands vectors, doesn't do type conversion
4. Vectors with Index scalars that were created BEFORE Stage 15 are never caught by Stage 15
5. Later operations like `Mul : Vector { scalar: Index }` use unlowered Index values

---

## The Actual Fix

Morok needs to add a pattern that matches vectorized INDEX with Index/Int64 indices and simplifies them, similar to Tinygrad's `simplify_valid_load`.

### Proposed Solution

Add a new pattern in `schedule/src/devectorize.rs` or `schedule/src/symbolic/index_lowering.rs`:

**Option 1: Add to `devectorize.rs` (before expand_index_patterns)**

```rust
pub fn simplify_index_dtype_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // Match vectorized INDEX with Index scalar indices
        // Similar to Tinygrad's (UPat(Ops.INDEX, src=(buf, x:dtypes.long, ...))
        index @ Index { buffer, indices }
            if indices.len() == 1 => {
                let idx = &indices[0];
                if idx.dtype() != DType::Index {
                    return None;
                }

                // Check if index is a computation that could be simplified
                // If it's a CAST with Int64 source, create new index with Int64
                if let Op::Cast { src: cast_src, dtype } = idx.op() {
                    if matches!(cast_src.op(), Op::Const(_)) {
                        // Replace CAST(Int64) with CONST(Int64)
                        return Some(UOp::index()
                            .buffer(buffer.clone())
                            .indices(smallvec::smallvec![cast_src.clone()]));
                    }
                }

                // Create new INDEX with Int64 dtype for proper lowering
                // This ensures the index will be caught by pm_lower_index_dtype
                let new_idx = UOp::index()
                    .buffer(buffer.clone())
                    .indices(smallvec::smallvec![idx.clone()])
                    .call();

                Some(UOp::index()
                    .buffer(buffer.clone())
                    .indices(smallvec::smallvec![new_idx]))
            },
    }
}
```

**Option 2: Add to `pm_lower_index_dtype`**

Add a pattern for vectorized INDEX that needs simplification:

```rust
// Pattern for INDEX inside vectorized context that has Int64 index
// This catches: INDEX(buf, idx=CAST(Int64)) and simplifies to INDEX(buf, idx=CONST(Int64))
index @ Index { buffer, indices }
    if indices.len() == 1 => {
        let idx = &indices[0];
        if matches!(idx.op(), Op::Cast { src: _, dtype: DType::Scalar(ScalarDType::Int64) }) {
            return Some(UOp::index()
                .buffer(buffer.clone())
                .indices(smallvec::smallvec![idx.src().clone()])
                .call());
        }
        None
    },
```

Then update the pipeline:

```rust
// In schedule/src/optimizer/mod.rs, add to Stage 15:
let with_simplified_idx = graph_rewrite(
    &pm_lower_index_dtype().with(&simplify_index_dtype_patterns()),
    with_lowered_idx,
    &mut ()
);
```

**Option 3: Fix `expand_index` to include type conversion**

Update `expand_index_patterns` in devectorize.rs:

```rust
pub fn expand_index_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // First, simplify Index dtypes (add this!)
        index if needs_index_dtype_simplification(index) => simplify_index_dtype(index),

        // Then expand
        index if is_vector_index(index) => expand_vector_index(index),
    }
}

fn needs_index_dtype_simplification(index: &Arc<UOp>) -> bool {
    matches!(index.op(), Op::Index { indices }) && {
        indices.iter().any(|idx| {
            matches!(idx.dtype(), DType::Index) ||
            matches!(idx.op(), Op::Cast { src: _, dtype: DType::Scalar(ScalarDType::Int64) })
        })
    }
}

fn simplify_index_dtype(index: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Index { buffer, indices } = index.op() else { return None };
    if indices.len() != 1 { return None };

    let idx = &indices[0];
    match idx.op() {
        Op::Cast { src, dtype: DType::Scalar(ScalarDType::Int64) } => {
            // Simplify: INDEX(buf, CAST(Int64)) → INDEX(buf, CONST(Int64))
            if let Op::Const(cv) = src.op() {
                let new_idx = UOp::const_(DType::Scalar(ScalarDType::Int64), cv);
                return Some(UOp::index().buffer(buffer.clone()).indices(smallvec::smallvec![new_idx]).call());
            }
        }
        _ => None,
    }
}
```

---

## Why Tinygrad's Approach Works

Tinygrad's `load_store_indexing` pattern is applied **together with** `pm_lower_index_dtype`. This means:

1. Vector INDEX with Int64 indices gets transformed by `simplify_valid_load`
2. The result (new INDEX with Int64) is immediately available to be matched
3. `pm_lower_index_dtype` runs and can match the new pattern
4. Both patterns reach fixpoint in the same graph rewrite pass

Morok's approach separates these concerns across different stages, which breaks the fixpoint behavior.

---

## Summary

| Aspect | Tinygrad | Morok (Current) | Issue |
|---------|-----------|------------------|--------|
| **Pattern scope** | `pm_lower_index_dtype + load_store_indexing` | `pm_lower_index_dtype` alone | Morok doesn't catch vectorized INDEX with Int64 |
| **Stage where fix is applied** | Same pass as lowering | Different stage (devectorize, not Stage 15) | Can't reach fixpoint |
| **Functionality** | `simplify_valid_load` creates new INDEX with correct dtype | `expand_index` just expands vectors, no type conversion | Unlowered Index remains |

**Recommendation:** Add a pattern to simplify Index dtypes in vectorized INDEX operations, and run it together with index lowering in Stage 15 (or ensure it's included in devectorization patterns).
