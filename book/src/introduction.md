# Morok

> ⚠️ **Pre-alpha software.** APIs are unstable and may change without notice. Not recommended for production use. 🚧💀

Morok is a Rust-based ML compiler inspired by [Tinygrad](https://github.com/tinygrad/tinygrad). It features lazy tensor evaluation with UOp-based IR, pattern-driven optimization, and multi-backend code generation.

## Highlights

| Feature | Description |
|---------|-------------|
| **Declarative Optimization** | `patterns!` DSL for graph rewrites with Z3-verified correctness |
| **Lazy Evaluation** | Tensors build computation graphs, compiled only at `realize()` |
| **CUDA Support** | Unified memory, D2D copy, LRU buffer caching |
| **Provenance Tracking** | `#[track_caller]` traces every UOp to source location |
| **80+ IR Operations** | Arithmetic, memory, control flow, WMMA tensor cores |
| **20+ Optimizations** | Constant folding, tensor cores, vectorization, loop unrolling |

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

## License

MIT
