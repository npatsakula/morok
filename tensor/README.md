# morok-tensor

High-level tensor API with lazy evaluation.

## Basic Example

```rust
use morok_tensor::Tensor;
use ndarray::array;

let a = Tensor::from_ndarray(&array![1.0f32, 2.0, 3.0]);
let b = Tensor::from_ndarray(&array![4.0f32, 5.0, 6.0]);
let mut c = &a + &b;
c.realize()?;

let view = c.array_view::<f32>()?;
assert_eq!(view.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
```

## ndarray Interop

```rust
use morok_tensor::Tensor;
use ndarray::array;

// Zero-copy from ndarray (fast path for C-contiguous arrays)
let input = array![[1.0f32, 2.0], [3.0, 4.0]];
let t = Tensor::from_ndarray(&input);

// Compute and extract back as ndarray
let result = (t * 2.0).as_ndarray::<f32>()?;
assert_eq!(result, array![[2.0, 4.0], [6.0, 8.0]].into_dyn());
```

### Zero-Copy View

For realized tensors on CPU, `array_view` returns a borrowed ndarray view
without copying data:

```rust
let mut t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
t.realize()?;
let view = t.array_view::<f32>()?;  // no copy, lifetime tied to tensor
assert_eq!(view.len(), 3);
```

## Prepare/Execute Infrastructure

For repeated kernel executions (e.g., benchmarks, inference loops), separate preparation from execution:

```rust
use morok_tensor::Tensor;

let a = Tensor::from_slice(&data_a);
let b = Tensor::from_slice(&data_b);
let mut result = a.matmul(&b)?;

// One-time preparation (compiles kernels, allocates buffers)
let plan = result.prepare()?;  // prepare() takes &mut self

// Fast repeated execution
for _ in 0..1000 {
    plan.execute()?;
}
```

## Batch Execution

Realize multiple tensors together — shares compilation and avoids redundant work.
Tested in `tensor/src/test/unit/batch.rs`:

```rust
// test_batch_shared_input
let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
let ten = Tensor::full(&[3], 10.0f32, DType::Float32)?;
let two = Tensor::full(&[3], 2.0f32, DType::Float32)?;
let mut a = &x + &ten;
let mut b = &x * &two;
Tensor::realize_batch([&mut a, &mut b])?;
// a = [11, 12, 13], b = [2, 4, 6]
```

Prepare once, execute many times (`test_prepare_batch_execute`):

```rust
let mut a = &Tensor::from_slice([1.0f32, 2.0]) + &Tensor::from_slice([3.0f32, 4.0]);
let mut b = &Tensor::from_slice([10.0f32, 20.0]) * &Tensor::from_slice([2.0f32, 3.0]);
let plan = Tensor::prepare_batch([&mut a, &mut b])?;
plan.execute()?;
assert_eq!(plan.num_outputs(), 2);
// a = [4, 6], b = [20, 60]
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

## Dynamic Shapes (Variable API)

Create tensors with symbolic dimensions for dynamic batching.
Tested in `tensor/src/test/unit/variable.rs`:

```rust
// test_variable_create, test_full_dynamic_symbolic_shape
let batch = Variable::new("batch", 1, 32);
let bound = batch.bind(16)?;
let t = Tensor::full_dynamic(&[bound.as_sint(), SInt::from(4)], 0.0f32, DType::Float32)?;
```

Compile once, execute with different variable values (`test_prepare_execute_loop`):

```rust
let batch = Variable::new("N", 1, 16);
let input = Tensor::empty_dynamic(&[batch.bind(4)?.as_sint()], DType::Float32);

// Assign initial data (lazy — no allocation yet)
input.assign(&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]));

// Compile plan (resolves assigns, allocates buffers)
let mut sum = input.sum(())?;
let mut plan = Tensor::prepare_batch([&mut sum])?;
plan.execute()?;  // first run

// Fast loop: write new data via array_view_mut, rebind N
input.array_view_mut::<f32>()?[..3].copy_from_slice(&[10.0, 20.0, 30.0]);
let bound = batch.bind(3)?;
plan.execute_with_vars(&[bound.as_var_val()])?;
```

## Testing

```bash
cargo test -p morok-tensor
```
