# morok-schedule-macros

Proc-macro crate providing the `patterns!` DSL for declarative pattern matching and rewriting.

## Example

```rust
use morok_schedule::patterns;

let matcher = patterns! {
    // Commutative identity folding
    Add[x, @zero] ~> x,
    Mul[x, @one] ~> x,

    // Constant extraction and folding
    Add(a @const(a_val), b @const(b_val))
      => eval_add(a_val, b_val).map(|r| UOp::const_(a.dtype(), r)),

    // Self-patterns (auto ptr_eq check)
    And(x, x) ~> Rc::clone(x),

    // Struct patterns with field extraction
    Cast { src: c @const(cv), dtype }
      => cv.cast(&dtype).map(|v| UOp::const_(dtype, v)),
};
```

## Pattern Syntax

| Feature | Syntax | Description |
|---------|--------|-------------|
| Tuple pattern | `Add(x, y)` | Match op with positional args |
| Struct pattern | `Cast { src, dtype }` | Match op with named fields |
| Rest pattern | `End(comp, ..)` | Variable argument count |
| Commutative | `Add[x, y]` | Try all orderings (binary) |
| Alternatives | `(Add \| Mul)(x, y)` | Match any listed op |
| Wildcard | `_` | Match without binding |
| Named binding | `name @ pattern` | Rename nested pattern |

## Constant Matchers

| Matcher | Matches |
|---------|---------|
| `@zero` | `0` or `0.0` |
| `@one` | `1` or `1.0` |
| `@const` | Any constant |
| `c @const(val)` | Constant with value extraction |
| `Const(42)` | Specific literal |

## Arrows

| Arrow | RHS Type | Use Case |
|-------|----------|----------|
| `~>` | `Rc<UOp>` | Infallible rewrite |
| `=>` | `Option<Rc<UOp>>` | Fallible rewrite |

## Guards

```rust
Add(x, y) if x.dtype() == y.dtype() ~> x,
Lt(x, x) if !x.dtype().is_float() ~> false.into_uop(DType::Bool),
```

## For-Loops

Generate patterns for multiple operations:

```rust
for op in unary [Neg, Sqrt, Exp2, Log2, Sin] {
    op(c @const(cv)) => eval_unary_op(op, cv).map(|r| UOp::const_(c.dtype(), r)),
},

for op in binary [Add, Mul, Sub, Mod, Max] {
    op(a @const(av), b @const(bv)) => eval_binary_op(op, av, bv).map(|r| UOp::const_(a.dtype(), r)),
},

for op in ternary [Where, MulAcc] {
    op(a @const(av), b @const(bv), c @const(cv)) => eval_ternary_op(op, av, bv, cv),
},
```

## Context Parameter

Pass mutable context to all rewrite closures:

```rust
let matcher = patterns! {
    @context KernelContext;

    buf if matches!(buf.op(), Op::Buffer { .. }) => debuf(buf, ctx),
    r if matches!(r.op(), Op::Range { .. }) => renumber_range(r, ctx),
};
```

## Supported Operations

| Category | Operations |
|----------|------------|
| Binary (17) | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Max`, `Lt`, `Eq`, `Ne`, `And`, `Or`, `Xor`, `Shl`, `Shr`, `Idiv`, `Fdiv`, `Pow` |
| Unary (8) | `Neg`, `Not`, `Abs`, `Sqrt`, `Exp`, `Log`, `Sin`, `Cos` |
| Ternary (2) | `Where`, `MulAcc` |
| Movement (6) | `Reshape`, `Permute`, `Expand`, `Pad`, `Shrink`, `Flip` |
| Memory (4) | `Load`, `Store`, `Index`, `Bufferize` |
| Control (4) | `Range`, `End`, `Reduce`, `Bind` |
| Other (4) | `Cast`, `Const`, `After`, `Detach` |

## Advanced Features

| Feature | Example | Description |
|---------|---------|-------------|
| Duplicate detection | `Add(x, x)` | Auto `Rc::ptr_eq` check |
| Nested struct | `Index { buffer: Bufferize { compute, .. }, .. }` | Multi-level extraction |
| Block RHS | `=> { let v = compute(); Some(v) }` | Multi-statement logic |

## Testing

```bash
cargo test -p morok-schedule-macros
```
