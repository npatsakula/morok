# Pattern-Based Optimization

Open any production ML compiler and you'll find dozens of optimization passes: constant folding, dead code elimination, operator fusion, loop tiling, vectorization, memory layout optimization. Each pass has its own data structures, its own traversal logic, its own bugs.

Morok takes a different approach: **one mechanism for everything**.

```text
Traditional Compiler:              Morok:
┌─────────────────────────┐       ┌─────────────────────────┐
│  Constant Folding       │       │                         │
│  Dead Code Elimination  │       │   patterns! {           │
│  Loop Unrolling         │       │       Add[x, @zero] ~> x│
│  Operator Fusion        │       │       Mul[x, @zero] ~> 0│
│  Vectorization          │       │       // ...more        │
│  Memory Planning        │       │   }                     │
│  ...20 more passes      │       │                         │
└─────────────────────────┘       │   graph_rewrite(...)    │
     Custom logic each            └─────────────────────────┘
                                       One mechanism
```

Every optimization in Morok is expressed as a **pattern**: "when you see this structure, replace it with that structure." The same `graph_rewrite()` function applies constant folding, converts movement ops to loops, optimizes memory access patterns, and lowers to hardware primitives.

This chapter explains how pattern-based optimization works and why it's powerful.

---

## The `patterns!` DSL

Morok provides a domain-specific language for writing optimization patterns. Here's what it looks like:

```rust
patterns! {
    // Identity folding: x + 0 → x
    Add[x, @zero] ~> |x| x.clone(),

    // Constant folding: 3 + 4 → 7
    Add(a @const(a_val), b @const(b_val))
        => |a, a_val, b_val| eval_add(a_val, b_val).map(|r| UOp::const_(a.dtype(), r)),

    // Self-folding: x / x → 1
    Idiv(x, x) ~> |x| UOp::one(x.dtype()),

    // Dead code elimination: if(true) { t } else { f } → t
    Where(@true, t, _f) ~> |t| t.clone(),
}
```

The macro compiles these patterns into efficient Rust code. Let's break down the syntax:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `(x, y)` | **Ordered.** Match in exact order. | `Sub(x, @zero) ~> x` |
| `[x, y]` | **Commutative.** Try both orderings. | `Add[x, @zero] ~> x` |
| `@zero` | **Zero constant.** Matches 0 or 0.0. | `Mul[_, z @ @zero] ~> z` |
| `@one` | **One constant.** Matches 1 or 1.0. | `Mul[x, @one] ~> x` |
| `@const(val)` | **Extract constant.** Binds the value. | `Add(@const(a), @const(b))` |
| `x, x` | **Same operand.** Auto-generates ptr_eq check. | `Idiv(x, x) ~> UOp::one(...)` |
| `~>` | **Infallible.** Always succeeds, returns `Arc<UOp>`. | `Add[x, @zero] ~> x` |
| `=>` | **Fallible.** May fail, returns `Option<Arc<UOp>>`. | `=> eval(...).map(...)` |
| `for op in binary [...]` | **Template.** Generate patterns for multiple ops. | See below |
| `@context Type` | **Stateful.** Access mutable context in patterns. | See below |

### Template Expansion

Instead of writing the same pattern for every binary operation, use a for-loop:

```rust
patterns! {
    for op in binary [Add, Mul, Sub, Idiv, Fdiv, Max] {
        op(a @const(a_val), b @const(b_val))
            => |a, a_val, b_val| eval_binary(op, a_val, b_val)
                .map(|r| UOp::const_(a.dtype(), r))
    }
}
```

This expands to six separate patterns at compile time—one for each operation.

### Stateful Patterns

Some optimizations need context (e.g., which kernel we're in, what ranges are active). Declare a context type:

```rust
patterns! {
    @context KernelContext;

    ReduceAxis { src } => |reduce, src, ctx| {
        ctx.record_reduction(reduce);
        transform_reduce(reduce, src, ctx)
    }
}
```

The context is passed as the last argument to pattern closures.

---

## How Pattern Matching Works

The `patterns!` macro generates a `SimplifiedPatternMatcher` that dispatches patterns in **O(1)** time.

### The OpKey Index

Every UOp has an operation type (Add, Mul, Load, etc.). The `#[derive(PatternEnum)]` macro generates an `OpKey` enum that maps operations to hashable keys:

```rust
pub enum OpKey {
    Binary(BinaryOp),    // Add, Mul, Sub, ...
    Unary(UnaryOp),      // Neg, Sqrt, Exp, ...
    Ternary(TernaryOp),  // Where, MulAcc
    Const,
    Load,
    Store,
    // ... one variant per operation category
}
```

### The Matcher Structure

```rust
pub struct SimplifiedPatternMatcher<C = ()> {
    indexed: HashMap<OpKey, Vec<PatternClosure<C>>>,  // O(1) lookup
    wildcards: Vec<PatternClosure<C>>,                 // patterns matching any op
}
```

When matching a UOp:

1. **Extract OpKey** from the UOp's operation
2. **Lookup** in the HashMap—O(1)
3. **Try each closure** until one matches
4. **Fall back** to wildcards if no indexed pattern matches

This is 5-10x faster than scanning all patterns linearly.

### Commutative Handling

For patterns like `Add[x, @zero]`, the macro generates code that tries both orderings:

```rust
// Try (x, @zero)
if let Some(result) = try_match_ordered(&children[0], &children[1]) {
    return result;
}
// Try (@zero, x)
if let Some(result) = try_match_ordered(&children[1], &children[0]) {
    return result;
}
```

### Duplicate Detection

When you write `Idiv(x, x)`, the pattern should only match if both operands are the *same* UOp (pointer equality, not structural equality). The macro automatically generates this check:

```rust
// Generated code for Idiv(x, x)
let x = &children[0];
let x_dup = &children[1];
if !Arc::ptr_eq(x, x_dup) {
    return NoMatch;
}
// ... rest of pattern
```

This leverages hash consing—identical subexpressions share the same pointer.

---

## The Rewrite Engine: Two-Stage Algorithm

Pattern matching alone isn't enough. Consider this expression:

```text
WHERE(Lt(3, 5), t, f)
```

To simplify it, we need two steps:
1. `Lt(3, 5)` → `true` (constant folding)
2. `WHERE(true, t, f)` → `t` (dead code elimination)

But the `WHERE` pattern won't match until its child is simplified. The rewrite engine solves this with a **two-stage algorithm**.

### Stage 0: Pattern Application

```rust
fn rewrite_stage0(&mut self, uop: &Arc<UOp>) -> RewriteResult {
    match self.matcher.try_match(uop) {
        Some(replacement) => RewriteResult::Rewritten(replacement),
        None => RewriteResult::Gate(uop.clone()),  // process children
    }
}
```

If no pattern matches, return `Gate`—a signal to process children first.

### Stage 1: Source Reconstruction

After children are rewritten, rebuild the node with new children and try patterns again:

```rust
fn rewrite_stage1(&mut self, uop: &Arc<UOp>, new_children: Vec<Arc<UOp>>) {
    // Rebuild with optimized children
    let rebuilt = uop.with_sources(new_children);

    // Try patterns again—might match now!
    match self.matcher.try_match(&rebuilt) {
        Some(replacement) => replacement,
        None => rebuilt,
    }
}
```

### The Magic: Cascading Optimizations

```text
Stage 0: WHERE(Lt(3, 5), t, f)     → Gate (no match, process children)
         └── Lt(3, 5)              → true (constant folding matches!)

Stage 1: WHERE(true, t, f)         → t (dead code elimination matches!)
```

The reconstruction stage re-applies patterns, enabling multi-step optimizations in a single traversal.

### Safety Limits

To prevent infinite loops, the engine has limits:
- **1000 iterations** per node maximum
- **100,000 iterations** total maximum
- Panics with diagnostic info if limits exceeded

In practice, well-formed patterns converge quickly.

---

## The Full Optimization Pipeline

Pattern matching is one part of a larger pipeline. When you call `tensor.realize()`, here's what happens:

```text
Tensor.realize()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RANGEIFY                                               │
│  Convert movement ops (RESHAPE, PERMUTE, EXPAND)        │
│  into explicit RANGE loops with INDEX operations        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  KERNEL SPLITTING                                       │
│  Split computation graph at STORE boundaries            │
│  Each STORE becomes a separate kernel                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FOR EACH KERNEL:                                       │
│                                                         │
│  1. Symbolic Simplification (algebraic patterns)        │
│                                                         │
│  2. Scheduler Creation                                  │
│     └── Convert LOOP → GLOBAL for GPU parallelization   │
│                                                         │
│  3. Kernel Optimization (heuristic OR beam search)      │
│     ├── Tensor Cores (WMMA) for matmul                  │
│     ├── Vectorization (UPCAST)                          │
│     ├── Loop Unrolling (UNROLL)                         │
│     ├── GPU Local Memory (LOCAL)                        │
│     ├── Grouped Reductions (GROUP)                      │
│     └── Threading (THREAD) for CPU                      │
│                                                         │
│  4. Post-Optimization Passes                            │
│     ├── Devectorize (memory coalescing)                 │
│     ├── Expand (UNROLL → vector operations)             │
│     ├── FMA Decomposition (a*b+c → MulAcc)              │
│     └── Bool Storage (cast bool↔uint8 for memory)       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  CODE GENERATION                                        │
│  Render optimized AST to LLVM IR, compile, execute      │
└─────────────────────────────────────────────────────────┘
```

Each box uses pattern-based rewriting. The difference is which patterns are applied:

- **Rangeify**: Movement op → BUFFERIZE + INDEX patterns
- **Symbolic**: Algebraic simplification patterns
- **Post-opt**: Memory access optimization patterns

---

## Kernel Optimization: Heuristics vs Beam Search

After symbolic simplification, each kernel needs *scheduling decisions*: how to tile loops, where to parallelize, whether to use tensor cores. Morok offers two strategies.

### Heuristics (Default)

The heuristic optimizer applies optimizations in a fixed order:

```rust
pub fn hand_coded_optimizations(scheduler: &mut Scheduler) {
    // 1. Tensor cores (if matmul pattern detected)
    if let Some(tc) = detect_tensor_core_pattern(scheduler) {
        apply_tensor_core(scheduler, tc);
        return;  // TC handles everything
    }

    // 2. Grouped reductions (two-stage for large reductions)
    apply_grouped_reduction_if_needed(scheduler);

    // 3. Vectorization (UPCAST output dimensions)
    apply_upcast(scheduler, 4);

    // 4. GPU local memory (workgroup dimensions)
    apply_local_dims(scheduler);

    // 5. CPU threading
    apply_threading(scheduler);
}
```

**Pros**: Fast (~50ms per kernel), predictable, no hardware measurement needed.

**Cons**: May miss optimization opportunities, fixed heuristics don't adapt to workload.

### Beam Search (Optional)

For production workloads, beam search finds better schedules:

```rust
pub fn beam_search(scheduler: Scheduler, config: BeamConfig) -> Scheduler {
    let mut beam = vec![scheduler];

    for iteration in 0..config.max_iterations {
        let mut candidates = vec![];

        for state in &beam {
            // Generate all valid next actions
            for action in generate_actions(state) {
                if let Ok(next) = state.apply(action) {
                    candidates.push(next);
                }
            }
        }

        // Compile and time each candidate
        let timed: Vec<_> = candidates.par_iter()
            .map(|c| (c, measure_kernel_time(c)))
            .collect();

        // Keep top K by execution time
        beam = timed.into_iter()
            .sorted_by_key(|(_, time)| *time)
            .take(config.beam_width)
            .map(|(c, _)| c)
            .collect();
    }

    beam.into_iter().next().unwrap()
}
```

The action space includes ~500 predefined actions:
- `UPCAST(axis, amount)` — vectorize output dimension
- `UNROLL(axis, amount)` — unroll reduction loop
- `LOCAL(axis, amount)` — use GPU shared memory
- `GROUP(axis, amount)` — two-stage reduction
- `THREAD(axis, amount)` — CPU parallelization
- `SWAP(axis1, axis2)` — reorder global dimensions

**Pros**: Finds near-optimal schedules, adapts to hardware.

**Cons**: Minutes per kernel (but results are cached by AST hash).

### Configuration

```bash
# Disable optimization (debugging)
MOROK_NOOPT=1 cargo run

# Enable beam search with width 8
MOROK_BEAM=8 cargo run
```

Or programmatically:

```rust
let config = OptimizerConfig::builder()
    .strategy(OptStrategy::Beam { width: 8 })
    .build();

tensor.realize_with(config)?;
```

---

## Comparison: How Other Compilers Optimize

Different ML compilers take different approaches to optimization:

| Aspect | XLA | TVM/Ansor | Triton | **Morok** |
|--------|-----|-----------|--------|-----------|
| **Philosophy** | Fixed heuristics | Search-based | Programmer-guided | Pattern-based |
| **Fusion** | Conservative rules | Tile-and-fuse | Block-level | Graph rewriting |
| **Auto-tuning** | None | Evolutionary + cost model | Grid search | Beam search |
| **Tuning cost** | 0 | Hours | Minutes | Minutes (cached) |
| **Flexibility** | Low | High | Medium | High |
| **Transparency** | Low (C++ passes) | Medium (Python) | Medium (DSL) | High (patterns!) |

### XLA — Production Conservative

XLA uses fixed heuristics for fusion decisions. Safe and predictable, but leaves performance on the table. The fusion rules are hard-coded in C++—extending them requires deep compiler knowledge.

### TVM/Ansor — Maximum Auto-Tuning

TVM separates *what* to compute from *how* to compute it. Ansor uses evolutionary search with a learned cost model to explore the schedule space. Can achieve best-in-class performance, but tuning takes hours per model.

### Triton — Programmer-Guided

Triton exposes a Python-like DSL where you write blocked algorithms explicitly. The compiler handles register allocation and memory management. Good balance of control and automation, but requires GPU programming expertise.

### Morok — Pattern Composition

Morok's insight: express optimizations as composable patterns. Each pattern is local and verifiable. Complex optimizations emerge from composition. Beam search adds auto-tuning when needed, with results cached for reuse.

---

## Why This Matters: Practical Benefits

Pattern-based optimization has concrete advantages for developers:

**Debugging is direct.** Patterns are readable code. Add a `println!` to any pattern to trace when it fires:

```rust
patterns! {
    Add[x, @zero] ~> |x| {
        println!("Folding add-zero: {:?}", x);
        x.clone()
    }
}
```

**Extensibility is easy.** Adding a custom optimization is two lines:

```rust
patterns! {
    // Your domain-specific optimization
    MyOp(x, y) if is_special_case(x, y) ~> transform(x, y)
}
```

No need to understand compiler internals, write visitors, or modify pass managers.

**Correctness is local.** Each pattern is a small theorem: "if this structure appears, replacing it with that structure preserves semantics." Verify each pattern independently. Composition of correct patterns yields correct programs.

**Performance is tunable.** O(1) pattern dispatch is fast by default. Enable beam search for production workloads. Cache results by AST hash—tune once, benefit forever.

---

## The Deeper Insight

Pattern matching trades generality for composability.

A general-purpose optimization pass can do anything—but that's exactly the problem. It's hard to verify, hard to extend, hard to compose with other passes. Ordering matters. Interactions are subtle.

A pattern is constrained: it matches a specific structure and produces a specific replacement. But constraints enable composition. Run patterns in any order—the result converges to the same fixed point. Add new patterns without breaking existing ones. Delete patterns without cascading failures.

Each pattern is a theorem about semantic equivalence. The rewrite engine is a theorem prover, finding derivations from input to optimized output. Correctness follows from the correctness of individual steps.

This is the Unix philosophy applied to compilers: small, focused tools that compose. Pattern-based optimization won't solve every problem—but for the problems it solves, it solves them elegantly.
