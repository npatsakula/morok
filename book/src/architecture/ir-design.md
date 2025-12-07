# IR Design Philosophy

Morok uses a single unified IR (UOp) for all compilation stages, following Tinygrad's design philosophy.

## The UOp Structure

```rust
pub struct UOp {
    pub id: u64,                    // Unique stable ID
    pub(crate) op: Op,              // Operation (Rust tagged union)
    pub(crate) dtype: DType,        // Data type
    // Cached properties (computed lazily)
    pub(crate) shape_cache: OnceCell<...>,
    pub(crate) vmin_vmax_cache: OnceCell<...>,
}
```

**Key fields:**

- `op`: An `Op` enum with 80+ operations spanning all abstraction levels
- `dtype`: Type information (scalars, vectors, pointers)
- `id`: Unique identifier for caching and provenance tracking

## Why One IR?

Traditional ML compilers use multiple IRs:

- **TensorFlow:** Graph → XLA HLO → MLIR → LLVM IR
- **PyTorch:** Python AST → TorchScript → FX Graph → Inductor IR → Triton
- **JAX:** Python → Jaxpr → StableHLO → MHLO → platform IR

Morok/Tinygrad uses **one IR** that represents all abstraction levels:

### Same UOp at Different Stages

```
# High-level (after tensor ops):
BUFFER.reshape([2,3]).reduce(ADD, axis=0)

# Mid-level (after rangeify):
RANGE(2) -> RANGE(3) -> LOAD -> REDUCE -> STORE -> END

# Low-level (after codegen passes):
DEFINE_GLOBAL -> INDEX -> LOAD -> ADD -> STORE
```

### Enabling Factors

1. **Graph Rewriting as Universal Mechanism**

   Pattern matching + rewriting handles all transformations:

   ```rust
   let optimized = graph_rewrite(&patterns, uop_graph, &mut ctx);
   ```

2. **Hash Consing (Structural Sharing)**

   Identical subgraphs share memory via a thread-local cache:

   ```rust
   thread_local! {
       static CACHE: RefCell<HashMap<UOpKey, Weak<UOp>>> = ...;
   }
   ```

   Benefits:

   - O(1) equality checking (pointer comparison)
   - No duplicate subgraphs in memory
   - Pattern matching can use pointer identity

3. **Lazy Property Computation**

   Expensive analyses computed once and cached:

   ```rust
   pub(crate) shape_cache: OnceCell<Result<Option<Shape>>>,
   pub(crate) vmin_vmax_cache: OnceCell<(ConstValue, ConstValue)>,
   ```

4. **Operation Hierarchy**

   Ops organized by level to support progressive lowering:

   ```rust
   impl Op {
       pub fn is_movement(&self) -> bool { ... }
       pub fn is_buffer(&self) -> bool { ... }
       pub fn is_alu(&self) -> bool { ... }
   }
   ```

## Trade-offs

| Aspect | Single IR | Multi-IR |
|--------|-----------|----------|
| Complexity | Lower | Higher |
| Translation bugs | None | Possible |
| Cross-level optimization | Natural | Requires bridging |
| Compile-time safety | Runtime checks | Per-IR guarantees |
| Codebase size | ~15k lines | 100k+ lines |

## Morok vs Tinygrad UOp

| Aspect | Tinygrad | Morok |
|--------|----------|-------|
| Language | Python dataclass | Rust struct |
| Children | `src: tuple[UOp, ...]` | Encoded in `Op` variants |
| Type safety | Runtime | Compile-time |
| Extra data | `arg: Any` (untyped) | Typed per variant |
| Memory | Weakref + GC | `Rc<UOp>` + explicit cleanup |

Morok's Rust implementation adds compile-time guarantees:

```rust
// Each Op variant encodes its exact structure
Op::Binary(BinaryOp::Add, lhs, rhs)  // vs Tinygrad's (Ops.ADD, (lhs, rhs))
Op::Reduce { src, ranges, reduce_op } // Named fields, typed
```
