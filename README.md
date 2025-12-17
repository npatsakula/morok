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
let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
let c = (&a + &b)?.sum(morok_tensor::AxisSpec::All)?;

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

## Development

### Environment setup

#### Nix

This project contains a pre-defined Nix development environment
with all dependencies and compilers included. The same infrastructure
is used for CI/CD, so it's the preferred way to develop and test.

```bash
nix develop # Open development shell
nix flake check # Run CI tests
```

#### Bare metal

| Dependency | Version | Description |
|------------|---------|-------------|
| Rust toolchain | 1.75+ | Required for building and testing |
| LLVM | >21.x | CPU code generation backend |
| NVCC | >13.x | CUDA code generation backend (optional) |
| MESA NAK | >=25.x | CUDA code generation backend (optional) |
| Z3 | >=4.15 | SMT solver for optimization verification (optional) |
| zlib | >=1.3 | Compression library |
| libffi | >=3.4 | Foreign function interface library |
| libxml | >=2.13 | XML parsing library |

## Test

```bash
cargo test
cargo test --features z3,proptest  # With Z3 verification and PB generated tests
cargo test --features cuda   # With CUDA tests
```
