# Index Dtype Lowering - Final Design

## Problem Statement

Stage 15 (Lower Index Dtype) converts abstract `Index` dtype to concrete integer types (i32 or i64). The test case `test_matmul_validated_3x3` fails because:
- Vectorized INDEX operations with Index/Int64 scalars aren't being lowered to concrete types
- LLVM JIT error: `%r0' defined with type 'i64' but expected 'i32'`

## Root Cause Analysis

### Tinygrad's Approach

Tinygrad's `pm_lower_index_dtype` works because:

1. **All patterns require `.cast(Index)` wrappers**
   ```python
   (UPat(Ops.VECTORIZE, src=UPat().cast(dtypes.index), name="v"),
    lambda v: v.replace(dtype=(dt:=select_dtype(v)), 
                      src=tuple(s.src[0].cast(dt.scalar()) for s in v.src))
                 .cast(dtypes.index))
   ```

2. **CONST pattern creates the wrappers**
   ```python
   (UPat((Ops.CONST, Ops.VCONST), dtype=dtypes.index, name="u"), 
    lambda u: u.replace(dtype=select_dtype(u)).cast(u.dtype))
   ```
   - `CONST(5): Index` → `CONST(5): Int32.cast(Index)`
   - The `.cast(Index)` wrapper is required for other patterns to match

3. **Patterns run together in fixpoint**
   ```python
   sink = graph_rewrite(sink, pm_lower_index_dtype + load_store_indexing, ...)
   ```
   - `pm_lower_index_dtype` creates `.cast(Index)` wrappers
   - `load_store_indexing` simplifies `INDEX(buf, idx.cast(Index))` → `INDEX(buf, idx)`
   - Running together ensures newly created INDEX with proper dtype is immediately simplified

### Morok's Current Issues

1. **Patterns too permissive** - accept direct Index dtype, not requiring `.cast(Index)` wrappers
2. **Patterns separated** - `pm_lower_index_dtype()` and `load_store_indexing()` run separately, breaking fixpoint
3. **No eager `.cast(Index)` creation** - CONST pattern wraps in `.cast(Index)`, but other patterns reject without wrapper

## Final Design

### Key Principle

**All Index dtype lowering patterns MUST create `.cast(Index)` wrappers as intermediate values.**

The cascade works as:
1. CONST(Index) → `CONST(concrete).cast(Index)`  [Pattern 2 creates wrapper]
2. VECTORIZE of `.cast(Index)` → `VECTORIZE(concrete).cast(Index)`  [Pattern 7 consumes wrapper]
3. INDEX with `.cast(Index)` → `INDEX(buf, concrete)`  [cleanup pattern strips wrapper]

### Pattern Structure

#### Pattern 1: Binary ops with Index operands

```rust
Binary { binary_op, lhs, rhs } 
    if node.dtype() == DType::Index 
        && lhs.dtype() == DType::Index 
        && rhs.dtype() == DType::Index 
=> |node, binary_op, lhs, rhs| {
    // Unwrap .cast(Index) wrappers to get concrete types
    // (Both must be wrapped in .cast(Index))
    let lhs_inner = unwrap_cast_index(lhs)?;
    let rhs_inner = unwrap_cast_index(rhs)?;

    // Determine concrete type from bounds and operand types
    let binary_bounds = select_concrete_dtype(node);
    let common_dtype = least_upper_int_dtype(&binary_bounds, &lhs_inner.dtype());

    // Compute in concrete type, then wrap in .cast(Index)
    let result = UOp::new(Op::Binary(*binary_op, lhs_inner, rhs_inner), common_dtype);
    Some(result.cast(DType::Index))
}
```

#### Pattern 2: CONST with Index dtype

```rust
c @const(cv) if c.dtype() == DType::Index => |c, cv| {
    // Select concrete type based on bounds
    let target_dtype = select_concrete_dtype(c);
    
    // Create CONST with concrete type, wrap in .cast(Index)
    let lowered = UOp::const_(target_dtype, cv);
    Some(lowered.cast(DType::Index))
}
```

#### Pattern 3: WHERE with Index branches

```rust
Where(cond, x, y) 
    if x.dtype() == DType::Index && y.dtype() == DType::Index 
=> |cond, x, y| {
    // Unwrap .cast(Index) wrappers from both branches
    let x_inner = unwrap_cast_index(x)?;
    let y_inner = unwrap_cast_index(y)?;

    // Cast branches to common concrete type
    let common_dtype = least_upper_int_dtype(&x_inner.dtype(), &y_inner.dtype());
    let x_cast = x_inner.cast(common_dtype.clone());
    let y_cast = y_inner.cast(common_dtype);
    let result = UOp::try_where(cond, x_cast, y_cast)?;
    
    // Wrap result in .cast(Index)
    Some(result.cast(DType::Index))
}
```

#### Pattern 4: SPECIAL with Index end

```rust
Special { name, end } if special.dtype() == DType::Index => |name, end| {
    // Unwrap .cast(Index) wrapper from end
    let end_inner = unwrap_cast_index(end)?;
    
    // SPECIAL always produces i32 indices
    let lowered = UOp::new(Op::Special { end: end_inner, name: name.clone() }, DType::Scalar(ScalarDType::Int32));
    Some(lowered.cast(DType::Index))
}
```

#### Pattern 5: DEFINE_VAR with Index dtype

```rust
DefineVar { name, min_val, max_val } if dv.dtype() == DType::Index => |name, min_val, max_val| {
    // DEFINE_VAR always uses i32 (bounds checked via min_val/max_val)
    let lowered = UOp::new(Op::DefineVar { name: name.clone(), min_val: *min_val, max_val: *max_val }, DType::Scalar(ScalarDType::Int32));
    Some(lowered.cast(DType::Index))
}
```

#### Pattern 6: RANGE with Index dtype

```rust
Range { end, axis_id, axis_type } if range.dtype() == DType::Index => |end, axis_id, axis_type| {
    // Unwrap .cast(Index) wrapper from end
    let end_inner = unwrap_cast_index(end)?;
    
    // Use end.dtype (i32 or i64) after being lowered elsewhere
    let lowered = UOp::new(Op::Range { end: end_inner, axis_id: *axis_id, axis_type: *axis_type }, end.dtype());
    Some(lowered.cast(DType::Index))
}
```

#### Pattern 7: VECTORIZE of Index elements

```rust
vec @ Vectorize { elements } if vec.dtype().base() == ScalarDType::Index => |vec, elements| {
    // Extract elements, requiring .cast(Index) wrapper
    let inner_elements: Vec<_> = elements.iter().map(|e| {
        match e.op() {
            Op::Cast { src, dtype } if *dtype == DType::Index => {
                Some((src.clone(), src.dtype()))
            }
            _ => None,  // Reject if not wrapped in .cast(Index)
        }
    }).collect::<Option<_>>()?;

    // Determine common scalar type from element dtypes
    let target_scalar = inner_elements.iter().fold(
        DType::Scalar(ScalarDType::Int32),
        |acc, (e, dt)| {
            if *dt == DType::Scalar(ScalarDType::Int64) {
                DType::Scalar(ScalarDType::Int64)
            } else {
                acc.clone()
            }
        }
    );

    // Cast each element to target scalar
    let lowered_elements: Vec<_> = inner_elements.iter().map(|(e, _dt)| e.cast(target_scalar.clone())).collect();
    let result = UOp::vectorize(lowered_elements.into());
    
    // Wrap result in .cast(Index)
    Some(result.cast(DType::Index))
}
```

#### Pattern 8: BIND with Index dtypes

```rust
Bind { var, value } if var.dtype() == DType::Index && value.dtype() == DType::Index 
=> |var, value| {
    // Unwrap .cast(Index) wrappers from both operands
    let var_inner = unwrap_cast_index(var)?;
    let val_inner = unwrap_cast_index(value)?;
    
    // Bind concrete values, wrap in .cast(Index)
    let bound = var_inner.bind(val_inner);
    Some(bound.cast(DType::Index))
}
```

#### Pattern 9: Cleanup - INDEX with hanging cast

```rust
Index { buffer, indices, gate: None } if indices.len() == 1 => |buffer, indices| {
    let idx = &indices[0];
    
    // Check if idx is .cast(Index) with concrete inner type
    if let Op::Cast { src: idx_inner, dtype: cast_dtype } = idx.op() 
        && *cast_dtype == DType::Index 
        && idx_inner.dtype().is_int() {
        // Strip the .cast(Index) wrapper, use concrete int directly
        return Some(UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![idx_inner.clone()], gate: None }, buffer.dtype().base().clone()));
    }
    None
}
```

### Helper Function

```rust
fn unwrap_cast_index(operand: &Arc<UOp>) -> Option<Arc<UOp>> {
    match operand.op() {
        Op::Cast { src, dtype } if *dtype == DType::Index => Some(src.clone()),
        _ => None,
    }
}
```

### Type Selection Functions

```rust
fn select_concrete_dtype(uop: &Arc<UOp>) -> DType {
    let (vmin, vmax) = VminVmaxProperty::get(uop);
    let fits_i32 = match (vmin, vmax) {
        (ConstValue::Int(min), ConstValue::Int(max)) => *min >= i32::MIN as i64 && *max <= i32::MAX as i64,
        (ConstValue::UInt(min), ConstValue::UInt(max)) => *min <= i32::MAX as u64 && *max <= i32::MAX as u64,
        _ => false,
    };
    if fits_i32 {
        DType::Scalar(ScalarDType::Int32)
    } else {
        DType::Scalar(ScalarDType::Int64)
    }
}

fn least_upper_int_dtype(a: &DType, b: &DType) -> DType {
    match (a, b) {
        (DType::Scalar(ScalarDType::Int64), _) | (_, DType::Scalar(ScalarDType::Int64)) => {
            DType::Scalar(ScalarDType::Int64)
        }
        _ => DType::Scalar(ScalarDType::Int32),
    }
}
```

## Fixpoint Execution

Stage 15 MUST run combined patterns:

```rust
pub fn pm_lower_index_dtype_full() -> TypedPatternMatcher {
    pm_lower_index_dtype() + load_store_indexing()
}
```

This ensures:
1. `pm_lower_index_dtype` creates `.cast(Index)` wrappers
2. `load_store_indexing` strips redundant `.cast(Index)` from INDEX
3. Patterns reach fixpoint with concrete types

## Why Current Implementation Fails

| Issue | Why |
|--------|-------|
| Patterns too permissive | Accept direct Index dtype, breaking `.cast(Index)` requirement |
| Separated pattern matchers | `pm_lower_index_dtype` and `load_store_indexing` run separately, no fixpoint |
| Vectorize creates i64 vectors | Uses element dtypes directly without common type selection |
| Cast cleanup pattern eager | Strips `.cast(Index)` before VECTORIZE can consume it |

## Implementation Notes

1. **All patterns return `Option<Arc<UOp>>`** - fallible rewrites
2. **All patterns wrap results in `.cast(DType::Index)`** - maintains interface
3. **Cleanup patterns only strip when beneficial** - when concrete type is already known
4. **`unwrap_cast_index` returns None** if not `.cast(Index)` - ensures cascade works correctly

## Testing

Test `test_matmul_validated_3x3` should pass because:
1. `CONST(Int(3)): Index` → `CONST(Int(3)): i32.cast(Index)`  [Pattern 2]
2. VECTORIZE of `.cast(Index)` elements → VECTORIZE(i32 elements).cast(Index)  [Pattern 7]
3. INDEX with `.cast(Index)` idx → INDEX with i32 idx  [cleanup pattern]
4. LLVM receives concrete i32 types only

## Comparison to Tinygrad

| Aspect | Tinygrad | Morok (Final Design) |
|---------|-----------|----------------------|
| Pattern requirements | `.cast(dtypes.index)` | `.cast(DType::Index)` |
| Wrapper creation | `u.replace(dtype=...).cast(u.dtype)` | `lowered.cast(DType::Index)` |
| Cleanup | `buf.index(idx, ptr=True)` | `INDEX(buf, idx)` |
| Fixpoint | Combined `pm + load_store_indexing` | Combined `pm_lower_index_dtype_full()` |
| Type selection | `select_dtype(u)` | `select_concrete_dtype(node)` |
