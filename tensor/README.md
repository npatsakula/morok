# morok-tensor

High-level tensor API with lazy evaluation and automatic differentiation.

## Example

```rust
use morok_tensor::Tensor;

let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3])?;
let c = a.try_add(&b)?;
let result = c.realize()?;
```

## Features

**Supported:**
- Lazy tensor construction
- Arithmetic: add, sub, mul, div, pow
- Math: sqrt, exp, log, sin
- Reduction: sum, mean, max, min
- Shape: reshape, transpose, broadcast, pad

**Planned:**
- Convolution ops
- Pooling ops
- Automatic differentiation (backward)
- JIT compilation
- Multi-device tensors

## Testing

```bash
cargo test -p morok-tensor
```
