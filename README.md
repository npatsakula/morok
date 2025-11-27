# Morok

> ⚠️ **Pre-alpha software.** APIs are unstable and may change without notice. Not recommended for production use. 🚧💀

A Rust-based ML compiler inspired by [Tinygrad](https://github.com/tinygrad/tinygrad). Lazy tensor evaluation with UOp-based IR, pattern-driven optimization, and multi-backend code generation.

## Highlights

| Feature | Description |
|---------|-------------|
| **Declarative Optimization** | `patterns!` DSL for graph rewrites with Z3-verified correctness |
| **Lazy Evaluation** | Tensors build computation graphs, compiled only at `realize()` |
| **CUDA Support** | Unified memory, D2D copy, LRU buffer caching |
| **Provenance Tracking** | `#[track_caller]` traces every UOp to source location |
| **80+ IR Operations** | Arithmetic, memory, control flow, WMMA tensor cores |
| **20+ Optimizations** | Constant folding, tensor cores, vectorization, loop unrolling |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        morok-tensor                         │
│              High-level API: Tensor, lazy ops               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         morok-ir                            │
│         UOp graph, 80+ ops, symbolic integers, WMMA         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              morok-schedule + morok-schedule-macros         │
│    patterns! DSL, RANGEIFY, kernel splitting, Z3 proofs     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       morok-codegen                         │
│                    LLVM IR generation                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            morok-runtime + morok-device                     │
│     JIT execution, CPU/CUDA buffers, unified memory         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       morok-dtype                           │
│       Scalars, vectors, pointers, address spaces            │
└─────────────────────────────────────────────────────────────┘
```

## Workspace

| Crate | Description | Highlights |
|-------|-------------|------------|
| [dtype](dtype/) | Type system | 14 scalar types, vectors, pointers, images |
| [device](device/) | Buffer management | Lazy alloc, zero-copy views, CUDA unified/D2D |
| [ir](ir/) | Core IR | 80+ ops, provenance tracking, WMMA |
| [schedule](schedule/) | Optimization engine | 20+ passes, RANGEIFY, Z3 verification |
| [schedule-macros](schedule-macros/) | Pattern DSL | `patterns!` macro with 45 op types |
| [codegen](codegen/) | Code generation | LLVM IR backend |
| [runtime](runtime/) | Kernel execution | JIT compilation |
| [tensor](tensor/) | High-level API | Lazy tensors, arithmetic, reductions |

## Quick Example

```rust
use morok_tensor::Tensor;

// Build lazy computation graph
let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3])?;
let c = (a + b).sum();

// Compile and execute
let result = c.realize()?;
```

## Pattern DSL Example

```rust
use morok_schedule::patterns;

let optimizer = patterns! {
    // Identity folding (commutative)
    Add[x, @zero] ~> x,
    Mul[x, @one] ~> x,

    // Constant folding
    for op in binary [Add, Mul, Sub] {
        op(a @const(av), b @const(bv))
          => eval_binary_op(op, av, bv).map(|r| UOp::const_(a.dtype(), r)),
    },
};
```

## Build

```bash
cargo build --release
cargo build --release --features cuda  # with CUDA support
```

## Test

```bash
cargo test
cargo test --features z3     # with Z3 verification
cargo test --features cuda   # with CUDA tests
```

## License

MIT
