---
name: patterns
description: Reference for Svod's patterns! DSL and rewrite engine API. Use when writing optimization patterns, debugging pattern matching, or understanding rewrite limitations.
---

# Svod Patterns! DSL and Rewrite Engine

## Pattern Syntax Quick Reference

The left-hand side follows Rust pattern syntax, extended only with what a Rust
pattern cannot say across an `Arc<UOp>` edge: nested op patterns, commutative
`Op[a, b]`, `@zero`/`@one`, and `x @const(v)` value extraction. Everything else in
a struct field is a plain Rust pattern with rustc's own diagnostics.

### Basic Pattern Structure

```rust
use svod_schedule::patterns;

let matcher = patterns! {
    Pattern => rewrite_expr,              // rewrite: Arc<UOp>, Option<Arc<UOp>> or RewriteResult
    Pattern if guard => rewrite_expr,     // guard before the arrow
};
```

The right-hand side may evaluate to `Arc<UOp>`, `Option<Arc<UOp>>` (`None` declines),
or a `RewriteResult` (so `Gate` is expressible); `?` works inside a block body.

### Variable Binding

| Syntax | Description | Example |
|--------|-------------|---------|
| `x` | Bind any UOp to variable `x` (snake_case = binding) | `Add(x, y)` |
| `_` | Wildcard (ignore) | `Add(_, y)` |
| `name @ pattern` | Bind the whole sub-match to `name` | `result @ Add(x, y)` |
| `c @const(cv)` | Bind UOp to `c`, its `ConstValue` to `cv` | `Add(x, c @const(cv))` |
| `c @vconst(vs)` / `c @anyconst(vs)` | VCONST lanes / CONST-or-VCONST as `Vec<ConstValue>` | `Mul(x, c @anyconst(vs))` |

A name used twice must be the same node: `Add(x, x)` generates `Arc::ptr_eq`. A value
name used twice (`Add[_a @const(v), _b @const(v)]`) must extract equal values.

### Constant Patterns

```rust
Const(_)                    // any CONST
Const(v)                    // bind the ConstValue
Const(ConstValue::Int(0))   // a Rust pattern over the ConstValue
@zero / @one                // zero / one of any numeric dtype
```

### Operation Patterns

**ALU ops (Unary/Binary/Ternary) by kind, positional:**
```rust
Neg(x)                    // Unary
Add(x, y)                 // Binary
Where(cond, t, f)         // Ternary
Add[x, @zero]             // commutative: also tries the swapped order
```
Names resolve against `svod_ir::op::alu`, so a typo or wrong arity is a normal
rustc error at the op's span.

**Struct ops by field:**
```rust
Cast { src: x, dtype }                       // child fields nest; `dtype` binds the field
Range { axis_type: AxisType::Upcast, .. }    // non-child fields are verbatim Rust patterns
Reduce { reduce_op: op @ (ReduceOp::Add | ReduceOp::Max), src, .. }
GetTuple { index: 2, .. }
Load { index, gate: Some(g), .. }            // Option<Arc<UOp>> children: Some(pat) / None
Reduce { .. }                                // any REDUCE
Noop                                         // unit variant
```
A field is a child pattern when it is `_`, `@..`, a snake_case binding, or an
identifier applied to `(..)`/`[..]`/`{..}`/`@`; anything else is passed through
as a Rust pattern, where a bare capitalized identifier (`axis_type: Upcast`) must
resolve to a variant or const — an unresolved one is a compile error, never a
binding. There is no positional form for struct ops.

### Rewrite Expressions

```rust
Add(x, @zero) => x                       // bare binding: clone of it
Neg(x) => UOp::neg(x)                    // expression
Mul(x, y) => x.try_mul(y).ok()           // Option declines with None
Mul(x, y) => { let r = x.try_mul(y).ok()?; Some(r) }
```

### Context-Aware Patterns

```rust
let matcher = patterns! {
    @context MyContext;
    Add(x, @zero) => { ctx.track_rewrite(); Some(Arc::clone(x)) }
};
let mut ctx = MyContext::new();
let result = graph_rewrite(&matcher, root, &mut ctx);
```

## Advanced Pattern Features

### Commutative Patterns

`[]` instead of `()` tries all orderings of that level; the shared prefix and the
rewrite body are generated once and retried per ordering (Tinygrad semantics).

```rust
Add[x, @zero] => x                       // Add(x, 0) or Add(0, x)
Add[Mul[x, c1 @const(a)], Mul[x, c2 @const(b)]] => ...   // 2 × 2 × 2 orderings
Sub(x, @zero) => x                       // non-commutative: one ordering
```

### Nested Patterns

```rust
Add(Neg(x), y)
Index { buffer: Stage { compute: Cast { src: x, .. }, .. }, .. }
```

### For-Blocks (Iteration)

```rust
for op in unary [Neg, Sqrt, Exp] {
    op(c @const(cv)) => eval_unary(op, cv),
}
for op in binary [*] {                   // every op of the kind
    op(x, @zero) => x,
}
```
Inside the block `op` is the ALU op value (`BinaryOp` etc.) and is usable in the
guard and body. Op sets over different kinds are written as separate rules or
for-blocks; there is no `(Add | Mul)(x, y)` form.

### Removed forms (do not use)

`~>`, `=> |x, y| body` closure parameter lists, `Const(0)` literal shorthand, bare
`@const`, `(A | B)(..)` alternatives, tuple positional form for struct ops
(`Store(idx, val)`, `Range(_)`, `Noop()`), and `reduce_op: Add | Max` bare-name
alternatives. Use the Rust-pattern forms above.

## Rewrite Engine API

### 3-Stage Algorithm (Tinygrad-aligned)

The rewrite engine uses a 3-stage algorithm matching Tinygrad's `unified_rewrite`:

| Stage | Name | What Happens | Patterns See |
|-------|------|--------------|--------------|
| 0 | PushChildren | Apply `bpm` patterns (if any), push children | ORIGINAL children |
| 1 | ApplyPatterns | Reconstruct with optimized children, apply `pm` patterns | OPTIMIZED children |
| 2 | Link | Link original node to final result | N/A |

**Key insight**: The semantic difference between `graph_rewrite()` and `graph_rewrite_bottom_up()` is WHEN patterns are applied:

- `graph_rewrite()`: Patterns applied in **Stage 1** (after children processed) → see OPTIMIZED children
- `graph_rewrite_bottom_up()`: Patterns applied in **Stage 0** (before children processed) → see ORIGINAL children

### Creating Pattern Matchers

```rust
use svod_schedule::patterns;

// Simple matcher
let matcher = patterns! {
    Add(x, @zero) => x,
    Mul(x, @one) => x,
};

// Context-aware matcher
let matcher = patterns! {
    @context MyContext;
    Add(x, y) => ctx.transform(x, y)
};

// Combining matchers (same context type)
let combined = identity_patterns() + constant_folding_patterns();
```

### Context Lifting with `with_context()`

When combining matchers that use **different context types**, use `.with_context::<D>()` to
lift a context-free matcher (`TypedPatternMatcher`, i.e. `SimplifiedPatternMatcher<()>`) into
a matcher with context type `D`. The lifted patterns simply ignore `&mut D` and pass `&mut ()`
to the original closures.

```rust
// Problem: symbolic() returns TypedPatternMatcher (ctx = ())
//          buffer_removal() returns TypedPatternMatcher<PcontigConfig>
//          Can't combine with + because context types differ!

// Solution: lift context-free matcher into the target context type
let mega_pass = symbolic().with_context::<PcontigConfig>()
    + reduction_simplify_patterns().with_context()  // type inferred from context
    + buffer_removal_with_pcontig();                // TypedPatternMatcher<PcontigConfig>

let mut ctx = PcontigConfig::default();
let result = graph_rewrite(&mega_pass, root, &mut ctx);
```

**Rules:**
- Only `SimplifiedPatternMatcher<()>` (context-free) has `.with_context()` — you cannot
  lift a matcher that already uses a non-`()` context into a different context type.
- The target type `D` can be specified explicitly (`.with_context::<MyCtx>()`) or inferred
  from the `+` combination (`.with_context()`).
- The lifted matcher consumes `self` (moves ownership). If you need the original matcher
  elsewhere, call the constructor again (e.g., `early_rewrites().with_context()`).

**Common pattern — mega-pass with shared context:**
```rust
// Multiple context-free matchers + one context-dependent matcher
let pass = matcher_a().with_context::<SharedCtx>()
    + matcher_b().with_context()
    + matcher_c().with_context()
    + context_dependent_matcher();  // TypedPatternMatcher<SharedCtx>
```

### Graph Rewrite Functions

#### `graph_rewrite()` - Default (patterns see OPTIMIZED children)

Patterns are applied in Stage 1, after children have been processed. Use this when patterns need to see the already-optimized children.

```rust
use svod_schedule::rewrite::graph_rewrite;

let result = graph_rewrite(&matcher, root, &mut ());

// With context
let mut ctx = MyContext::new();
let result = graph_rewrite(&matcher, root, &mut ctx);
```

**Example**: For `Add(Add(UNROLL_a, UNROLL_b), UNROLL_c)`, the `do_expand` pattern sees:
1. Inner `Add` already transformed to `UNROLL_ab`
2. Outer `Add` sees `Add(UNROLL_ab, UNROLL_c)` → correctly expands all 3 axes

#### `graph_rewrite_bottom_up()` - Patterns see ORIGINAL children

Patterns are applied in Stage 0, before children are processed. Use this when patterns need to see the original graph structure.

```rust
use svod_schedule::rewrite::graph_rewrite_bottom_up;

let result = graph_rewrite_bottom_up(&matcher, root, &mut ctx);
```

**Use cases**:
- Patterns that match nested structures like `Index { buffer: Bufferize { ... } ... }`
- Patterns that need to see the original child structure before optimization
- Dead axis removal, buffer removal heuristics

#### `graph_rewrite_with_bpm()` - Both stages

Use both `pm` (Stage 1) and `bpm` (Stage 0) patterns:

```rust
use svod_schedule::rewrite::graph_rewrite_with_bpm;

// bpm patterns see ORIGINAL children (Stage 0)
// pm patterns see OPTIMIZED children (Stage 1)
let result = graph_rewrite_with_bpm(&pm, &bpm, root, &mut ctx);
```

#### `graph_rewrite_with_map()`

Returns both result and transformation map:

```rust
use svod_schedule::rewrite::graph_rewrite_with_map;

let output = graph_rewrite_with_map(&matcher, root, &mut ctx);
// output.root - the rewritten root
// output.becomes_map - HashMap<UOpKey, Arc<UOp>> of transformations
```

### Choosing the Right Rewrite Function

| Scenario | Function | Reason |
|----------|----------|--------|
| Algebraic simplification | `graph_rewrite()` | Patterns like `x + 0 → x` work on any children |
| Expansion (UNROLL propagation) | `graph_rewrite()` | Need to see already-expanded children |
| Nested structure matching | `graph_rewrite_bottom_up()` | Need original `Index { buffer: Bufferize { ... } }` |
| Dead axis removal | `graph_rewrite_bottom_up()` | Need original BUFFERIZE ranges |
| Buffer removal heuristics | `graph_rewrite_bottom_up()` | Need to count original buffers |

### Running Optimization Passes

```rust
use svod_schedule::symbolic::patterns::symbolic;

// Symbolic simplification (17+ pattern categories)
let optimized = graph_rewrite(&symbolic(), graph, &mut ());

// Rangeify transformations
use svod_schedule::rangeify::patterns::{
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
| **No backtracking** once committed to branch | Split into separate rules |
| **No cross-traversal context** "if Y seen earlier" | Use `@context` parameter with manual tracking |
| **No graph topology queries** (consumers, cycles) | Pre-analysis passes or manual traversal |
| **No fixed-point limit** `REWRITE_STACK_LIMIT = 500_000` | A runaway rewrite panics; ensure every rule makes progress |
| **No higher-order patterns** "any commutative op" | Use `for op in binary [Add, Mul, ...]` |
| **No variable-arity chains** | Explicit enumeration: `Add(Add(x, y), z)` |

### Performance Considerations

1. **Wildcard patterns expensive** - `x if cond` checked for EVERY op. Use specific OpKey instead.
2. **Deep nesting slow** - Triple nested patterns like `Index { buffer: Bufferize { compute: Cast { src: x } } }` should use guards or intermediate patterns.
3. **Op sets** - Use `for op in binary [...]` for the same rule over several ops.
4. **Permutation overhead** - `Add[x, y]` tries both orderings. Use `Add(x, @zero)` when order doesn't matter.
5. **Ensure progress** - `Neg(x) => x.try_neg()` may loop. `Neg(Neg(x)) => x` makes structural progress.

### Known Issues

1. **Bool vectors (LLVM bug)** - `<N x i1>` broken. Use `pm_bool_devectorize` to convert to scalar.
2. **Reduce context inlining** - Unary ops NOT inlined in reduce to avoid N recomputations.
3. **Float self-comparison** - `Lt(x, x)` NOT folded for floats due to NaN semantics (NaN < NaN is false).
4. **Division distribution** - `(a+b)//c → a//c + b//c` only when values in same bucket.
5. **GEP pattern ordering** - BROADCAST GEP must come BEFORE general VECTORIZE GEP.

### Tinygrad Semantic Alignment

The rewrite engine semantics match Tinygrad's `unified_rewrite` (ops.py:1177-1234):

| Tinygrad | Svod | Patterns See |
|----------|-------|--------------|
| `graph_rewrite(pm, bottom_up=False)` | `graph_rewrite(pm)` | OPTIMIZED children |
| `graph_rewrite(pm, bottom_up=True)` | `graph_rewrite_bottom_up(bpm)` | ORIGINAL children |
| `RewriteContext(pm, bpm, ctx)` | `graph_rewrite_with_bpm(pm, bpm)` | Both stages |

**Migration note**: If patterns stop matching after this change, check if they need to see ORIGINAL children (use `graph_rewrite_bottom_up`) or OPTIMIZED children (use `graph_rewrite`).

### Common Pitfalls

1. **Arrow**: `=>` accepts `Arc<UOp>`, `Option<Arc<UOp>>` or `RewriteResult`; there is no `~>`
2. **Wildcard performance**: `x if condition` checked for EVERY op - use specific OpKey patterns
3. **Commutative**: `Add[x, y]` tries both orderings - use `Add(x, y)` when ordering matters
4. **Duplicate detection**: `Add(x, x)` auto-generates `Arc::ptr_eq` - only identical variable names
5. **Guard placement**: Guard goes AFTER pattern, BEFORE arrow: `Pattern if cond => rewrite`
6. **Rewrite function semantics**: `graph_rewrite()` patterns see OPTIMIZED children; use `graph_rewrite_bottom_up()` for patterns that need ORIGINAL structure (e.g., nested `Index { buffer: Bufferize { ... } }`)

### Debugging

```bash
RUST_LOG=svod_ir::pattern=debug cargo test test_name        # Pattern matching details
RUST_LOG=svod_ir::pattern::simplified=trace cargo test      # Which patterns are tried
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
