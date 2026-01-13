---
name: patterns
description: Reference for Morok's patterns! DSL and rewrite engine API. Use when writing optimization patterns, debugging pattern matching, or understanding rewrite limitations.
---

# Morok Patterns! DSL and Rewrite Engine

## Pattern Syntax Quick Reference

### Basic Pattern Structure

```rust
use morok_schedule::patterns;

let matcher = patterns! {
    // Infallible rewrite (always returns Arc<UOp>)
    Pattern ~> rewrite_expr,

    // Fallible rewrite (returns Option<Arc<UOp>>)
    Pattern => rewrite_expr,
};
```

### Variable Binding

| Syntax | Description | Example |
|--------|-------------|---------|
| `x` | Bind any UOp to variable `x` | `Add(x, y)` |
| `_` | Wildcard (ignore) | `Add(_, y)` |
| `name @ pattern` | Bind entire match to `name` | `result @ Add(x, y)` |
| `c @const(cv)` | Bind UOp to `c`, ConstValue to `cv` | `Add(x, c@const(cv))` |

### Constant Patterns

```rust
// Specific constants
Const(0)          // Integer 0
Const(1.0)        // Float 1.0

// Special constants
@zero             // Matches 0 or 0.0
@one              // Matches 1 or 1.0
@const            // Any constant
```

### Operation Patterns

**Tuple style (positional):**
```rust
Neg(x)                    // Unary
Add(x, y)                 // Binary
Where(cond, t, f)         // Ternary
```

**Struct style (named fields):**
```rust
Cast { src: x, dtype }
Load { buffer: b, index: i }
Bufferize { compute: x, ranges, .. }  // .. ignores rest
```

### Guards and Conditions

```rust
// Guard before arrow
Pattern if condition ~> rewrite_expr

// Examples
Add(x, y) if !x.dtype().is_float() => transform(x, y)
Cast { src: x, dtype } if x.dtype() == *dtype ~> x
Lt(x, x) if !x.dtype().is_float() ~> false.into_uop(DType::Bool)
```

### Rewrite Expressions

```rust
// Simple variable return
Add(x, @zero) ~> x

// Closure (infallible)
Mul(x, y) ~> |x| x.clone()

// Closure (fallible)
Add(x, @zero) => |x| Some(x.clone())

// Multi-parameter
Mul(x, y) => |x, y| x.try_mul(y).ok()

// Block expression
Mul(x, y) => {
    let result = x.try_mul(y).ok()?;
    Some(result)
}

// General expression
Neg(x) => UOp::neg(x)
```

### Context-Aware Patterns

```rust
// Declare context type
let matcher = patterns! {
    @context MyContext;

    Add(x, @zero) => {
        ctx.track_rewrite();
        Some(Arc::clone(x))
    }
};

// Use with context
let mut ctx = MyContext::new();
let result = graph_rewrite(&matcher, root, &mut ctx);
```

## Advanced Pattern Features

### Commutative Patterns

Use `[]` instead of `()` to try all orderings:

```rust
// Matches Add(x, zero) OR Add(zero, x)
Add[x, @zero] ~> |x| x.clone()

// Matches Mul(x, one) OR Mul(one, x)
Mul[x, @one] ~> |x| x.clone()

// Non-commutative - only one ordering
Sub(x, @zero) ~> |x| x.clone()
```

### Alternative Patterns

Match if ANY pattern matches:

```rust
(Add | Mul)(x, y)       // Matches Add(x, y) OR Mul(x, y)
(Neg | Not)(x)          // Matches Neg(x) OR Not(x)
```

### Nested Patterns

```rust
Add(Neg(x), y)          // Nested operations
Mul(Add(a, b), c)       // Complex nesting
Index { buffer: Bufferize { compute: Cast { src: x }, .. }, .. }
```

### For-Blocks (Iteration)

Generate multiple patterns at once:

```rust
// Iterate over specific operations
for op in unary [Neg, Sqrt, Exp] {
    op(c @const(cv)) => eval_unary(op, cv),
}

for op in binary [Add, Mul, Sub] {
    op(x, @zero) ~> |x| x.clone()
}

for op in ternary [Where, MulAcc] {
    op(a, b, c) => fold_ternary(op, a, b, c)
}

// Wildcard - ALL operations
for op in unary [*] {
    op(c @const(cv)) => eval_unary(op, cv),
}

for op in binary [*] {
    op(x, @zero) ~> x
}
```

### Duplicate Detection

Same variable name generates `Arc::ptr_eq` check:

```rust
// x + x → 2*x (only when operands are identical)
Add(x, x) => |x| x.try_mul(&UOp::const_(x.dtype(), 2)).ok()

// x // x → 1
Idiv(x, x) => |x| Some(UOp::const_(x.dtype(), 1))

// x & x → x
And(x, x) ~> |x| x.clone()
```

### Option Patterns

```rust
Some(pattern)           // Matches Option::Some
None                    // Matches Option::None

// Example
Index { buffer: b, gate: None } ~> b
Index { buffer: b, gate: Some(g) } => Some(g.clone())
```

## Supported Operations

### Unary Operations (20)

```
Neg, Not, Abs, Sqrt, Rsqrt, Exp, Exp2, Log, Log2, Sin, Cos, Tan,
Reciprocal, Trunc, Floor, Ceil, Round, Sign, Erf, Square
```

### Binary Operations (20)

```
Add, Mul, Sub, Mod, Max, Pow, Idiv, Fdiv,
Lt, Le, Eq, Ne, Gt, Ge,
And, Or, Xor, Shl, Shr, Threefry
```

### Ternary Operations (2)

```
Where, MulAcc
```

### Single-Source Operations

```
Cast, Reshape, Permute, Expand, Pad, Shrink, Flip, Contract,
Unroll, Contiguous, Detach, BitCast, ReduceAxis, Multi, Bufferize,
```

### Special Operations

```
Range, End, If, Buffer, BufferView, MSelect, Index, PointerIndex,
Copy, Load, LoadGated, Store, StoreGated, Bind, Assign,
Reduce, AllReduce, Wmma, Kernel, Sink, Vectorize, GEP
```

## Rewrite Engine API

### Creating Pattern Matchers

```rust
use morok_schedule::patterns;

// Simple matcher
let matcher = patterns! {
    Add(x, @zero) ~> |x| x.clone(),
    Mul(x, @one) ~> |x| x.clone(),
};

// Context-aware matcher
let matcher = patterns! {
    @context MyContext;
    Pattern => |x, ctx| ctx.transform(x)
};

// Combining matchers
let combined = identity_patterns() + constant_folding_patterns();
```

### Graph Rewrite Functions

#### `graph_rewrite()`

Default top-down rewrite:

```rust
use morok_schedule::rewrite::graph_rewrite;

let result = graph_rewrite(&matcher, root, &mut ());

// With context
let mut ctx = MyContext::new();
let result = graph_rewrite(&matcher, root, &mut ctx);
```

#### `graph_rewrite_bottom_up()`

Always processes children before parents:

```rust
use morok_schedule::rewrite::graph_rewrite_bottom_up;

let result = graph_rewrite_bottom_up(&matcher, root, &mut ctx);

// Use when patterns need to transform deep nodes first
```

#### `graph_rewrite_with_map()`

Returns both result and transformation map:

```rust
use morok_schedule::rewrite::graph_rewrite_with_map;

let output = graph_rewrite_with_map(&matcher, root, &mut ctx);
// output.root - the rewritten root
// output.becomes_map - HashMap<UOpKey, Arc<UOp>> of transformations
```

### Running Optimization Passes

```rust
use morok_schedule::symbolic::patterns::symbolic;

// Symbolic simplification (17+ pattern categories)
let optimized = graph_rewrite(&symbolic(), graph, &mut ());

// Rangeify transformations
use morok_schedule::rangeify::patterns::{
    apply_rangeify_patterns, buffer_folding, dead_axis_removal,
    movement_op_patterns
};

// Combine multiple passes
let full_pipeline = apply_rangeify_patterns()
    + buffer_folding()
    + dead_axis_removal()
    + movement_op_patterns();

let result = graph_rewrite(&full_pipeline, graph, &mut ctx);
```

## Limitations and Constraints

### Patterns That CANNOT Be Expressed

| Limitation | Workaround |
|------------|------------|
| **No negative matching** `Add(!Const(_), y)` | Use guards: `Add(x, y) if !matches!(x.op(), Op::Const(_))` |
| **No backtracking** once committed to branch | Use explicit alternatives: `(Op1 \| Op2)(x, y)` |
| **No cross-traversal context** "if Y seen earlier" | Use `@context` parameter with manual tracking |
| **No graph topology queries** (consumers, cycles) | Pre-analysis passes or manual traversal |
| **Fixed-point limit** MAX_DEPTH = 100 | Deep/circular chains truncated |
| **No higher-order patterns** "any commutative op" | Use `for op in binary [Add, Mul, ...]` |
| **No variable-arity chains** | Explicit enumeration: `Add(Add(x, y), z)` |

### Performance Considerations

1. **Wildcard patterns expensive** - `x if cond` checked for EVERY op. Use specific OpKey instead.
2. **Deep nesting slow** - Triple nested patterns like `Index { buffer: Bufferize { compute: Cast { src: x } } }` should use guards or intermediate patterns.
3. **Large alternative lists** - Keep under ~10 ops. Prefer `for op in binary [...]` over `(Add | Mul | Sub)(x, y)`.
4. **Permutation overhead** - `Add[x, y]` tries both orderings. Use `Add(x, @zero)` when order doesn't matter.
5. **Ensure progress** - `Neg(x) => x.try_neg()` may loop. `Neg(Neg(x)) => x` makes structural progress.

### Known Issues

1. **Bool vectors (LLVM bug)** - `<N x i1>` broken. Use `pm_bool_devectorize` to convert to scalar.
2. **Reduce context inlining** - Unary ops NOT inlined in reduce to avoid N recomputations.
3. **Float self-comparison** - `Lt(x, x)` NOT folded for floats due to NaN semantics (NaN < NaN is false).
4. **Division distribution** - `(a+b)//c → a//c + b//c` only when values in same bucket.
5. **GEP pattern ordering** - BROADCAST GEP must come BEFORE general VECTORIZE GEP.

### Common Pitfalls

1. **`~>` vs `=>`**: `~>` is infallible (returns `Arc<UOp>`), `=>` is fallible (returns `Option`)
2. **Wildcard performance**: `x if condition` checked for EVERY op - use specific OpKey patterns
3. **Commutative**: `Add[x, y]` tries both orderings - use `Add(x, y)` when ordering matters
4. **Duplicate detection**: `Add(x, x)` auto-generates `Arc::ptr_eq` - only identical variable names
5. **Guard placement**: Guard goes AFTER pattern, BEFORE arrow: `Pattern if cond ~> rewrite`

### Debugging

```bash
RUST_LOG=morok_ir::pattern=debug cargo test test_name        # Pattern matching details
RUST_LOG=morok_ir::pattern::simplified=trace cargo test      # Which patterns are tried
```

## Key Files

| File | Purpose |
|------|---------|
| `macros/src/patterns/parser.rs` | DSL parser |
| `macros/src/patterns/codegen.rs` | Code generator |
| `ir/src/pattern/simplified.rs` | SimplifiedPatternMatcher |
| `ir/src/rewrite/engine.rs` | Rewrite engine |
| `schedule/src/symbolic/patterns.rs` | Symbolic patterns |
| `schedule/src/rangeify/patterns.rs` | Rangeify patterns |
