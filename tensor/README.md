# morok-tensor

High-level tensor API with lazy evaluation.

## Basic Example

```rust
use morok_tensor::Tensor;

let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
let c = a.try_add(&b)?;
let result = c.realize()?;
```

## Prepare/Execute Infrastructure

For repeated kernel executions (e.g., benchmarks, inference loops), separate preparation from execution:

```rust
use morok_tensor::Tensor;
use morok_runtime::global_executor;

let a = Tensor::from_slice(&data_a);
let b = Tensor::from_slice(&data_b);
let result = a.matmul(&b)?;

// One-time preparation (compiles kernels, allocates buffers)
let plan = result.prepare()?;

// Fast repeated execution
let mut executor = global_executor();
for _ in 0..1000 {
    plan.execute(&mut executor)?;
}
```

## Zero-Copy Buffer Access

Access underlying device buffer without copying data:

```rust
let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).realize()?;

// Direct buffer access (no copy)
if let Some(buffer) = tensor.buffer() {
    // Use buffer directly
}
```

## Features

**Supported:**
- Lazy tensor construction
- Arithmetic: add, sub, mul, div, pow
- Math: sqrt, exp, log, sin, cos
- Reduction: sum, mean, max, min, argmax, argmin
- Shape: reshape, transpose, permute, expand, squeeze
- Activation: relu, sigmoid, tanh, softmax, gelu
- Matrix: matmul, dot, linear

## Testing

```bash
cargo test -p morok-tensor
```
