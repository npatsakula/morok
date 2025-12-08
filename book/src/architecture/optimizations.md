# Optimization System

Morok's optimizer is built on pattern matching and graph rewriting.

## UPat: Universal Pattern

```rust
pub enum UPat {
    Match {
        op: Option<Vec<OpFilter>>,      // Operations to match
        dtype: Option<Vec<DType>>,      // Types to match
        src: Option<SrcPattern>,        // Child patterns
        arg: Option<ArgPattern>,        // Argument constraints
        name: Option<String>,           // Binding name
    },
    Any(Vec<UPat>),  // OR-pattern
}
```

### Source Pattern Variants

```rust
pub enum SrcPattern {
    Tuple(Vec<UPat>),      // Fixed arity: Add(x, y)
    Repeat(Box<UPat>),     // All match: Sink(stores..)
    Fork(Vec<Vec<UPat>>),  // OR over arities
    Permute(Vec<UPat>),    // Commutative: Add[x, y]
}
```

## Pattern Matching Algorithm

1. **Check operation type** against `OpFilter`
2. **Check dtype** against allowed list
3. **Check argument** via `ArgPredicate` (IsZero, IsOne, etc.)
4. **Bind or verify** named variables (pointer equality)
5. **Match sources** recursively based on `SrcPattern`

### Commutative Matching

For `Add[x, @zero]`, both orderings are tried:

```rust
// Fast path for binary (n=2)
if patterns[0].matches(children[0]) && patterns[1].matches(children[1]) { return true; }
if patterns[0].matches(children[1]) && patterns[1].matches(children[0]) { return true; }
```

## PatternMatcher Indexing

Patterns indexed by operation for O(1) lookup:

```rust
struct PatternMatcher<C> {
    patterns: Vec<(UPat, VarIntern, RewriteFn<C>)>,
    pdict: HashMap<OpKey, Vec<usize>>,  // op -> pattern indices
    wildcard_indices: Vec<usize>,       // patterns matching any op
}
```

## Rewrite Engine

Fixed-point iteration with 2-stage algorithm:

```rust
enum Stage { BottomUp, SourceReconstruction }

fn rewrite(&mut self, root: Rc<UOp>) -> Rc<UOp> {
    let mut stack = vec![(root, Stage::BottomUp)];
    while let Some((uop, stage)) = stack.pop() {
        match stage {
            Stage::BottomUp => {
                // Apply patterns, push children
            }
            Stage::SourceReconstruction => {
                // Rebuild with rewritten children
                // Apply patterns again (enables multi-stage opts)
            }
        }
    }
}
```

### Multi-Stage Example

```
WHERE(Lt(3, 5), t, f)
  → [constant fold Lt] → WHERE(true, t, f)
  → [DCE] → t
```

The reconstruction stage re-applies patterns, enabling cascading optimizations.

## The `patterns!` DSL

Proc-macro generates efficient Rust code:

```rust
let matcher = patterns! {
    // Commutative identity
    Add[x, @zero] ~> x,

    // Constant folding with for-loop
    for op in binary [Add, Mul, Sub] {
        op(a @const(av), b @const(bv))
          => eval_binary_op(op, av, bv).map(|r| UOp::const_(a.dtype(), r)),
    },

    // Self-pattern (auto ptr_eq)
    And(x, x) ~> Rc::clone(x),
};
```

### Compile-Time Optimizations

1. **Variable Index Resolution:** Names → u8 indices at macro expansion
2. **Duplicate Detection:** `Add(x, x)` generates `Rc::ptr_eq` check
3. **Binding Storage:** `SmallVec<[(u8, Rc<UOp>); 4]>` (stack for ≤4 bindings)

## Optimization Categories

| Category | Patterns | Examples |
|----------|----------|----------|
| **Constant Folding** | 22 | `Add(3, 5) → 8` |
| **Identity** | 8 | `x + 0 → x`, `x * 1 → x` |
| **Zero Propagation** | 4 | `x * 0 → 0` |
| **Self-Folding** | 6 | `x / x → 1`, `x & x → x` |
| **ALU Folding** | 4 | `(x + c1) + c2 → x + (c1+c2)` |
| **Division** | 5 | `(a*b)/b → a` |
| **DCE** | 6 | `WHERE(true, t, f) → t` |
| **Tensor Core** | 3 | TC matching, swizzle, apply |
| **Vectorization** | - | Upcasting to float4, etc. |
| **Loop Unrolling** | - | Reductions ≤ 32 |
