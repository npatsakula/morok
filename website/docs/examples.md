---
sidebar_label: Hands-On Examples
---

# Hands-On: From Tensors to Models

This chapter teaches Morok through progressive examples. You'll start with basic tensor operations and build up to a working neural network classifier.

**What you'll learn:**
- Creating and manipulating tensors
- Shape operations (reshape, transpose, broadcast)
- Matrix multiplication
- Building reusable layers
- Composing a complete model

**Prerequisites:**
- Basic Rust knowledge
- Add `morok_tensor` to your `Cargo.toml`

**Key pattern:** Morok uses *lazy evaluation*. Operations build a computation graph without executing. Call `realize()` to compile and run everything at once.

---

## Example 1: Hello Tensor

Let's create tensors, perform operations, and get results.

```rust
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors from slices
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    // Lazy operations (no execution yet)
    let sum = &a + &b;
    let scaled = &sum * &Tensor::from_slice([0.1f32]);

    // Execute and get results
    let result = scaled.realize()?;
    let data = result.to_ndarray::<f32>()?;
    println!("Result: {:?}", data);
    // Output: [1.1, 2.2, 3.3, 4.4]

    Ok(())
}
```

**What's happening:**

1. `Tensor::from_slice()` creates a 1D tensor from array data. The `f32` suffix tells Rust the element type.

2. `&a + &b` doesn't compute anything yet. It returns a new `Tensor` that *represents* the addition. The `&` borrows the tensors so we can reuse them.

3. `realize()` is where the magic happens. Morok:
   - Analyzes the computation graph
   - Fuses operations where possible
   - Generates optimized code
   - Executes on the target device

4. `to_ndarray()` extracts the result as an `ndarray::ArrayD` for inspection.

**Try this:** Remove the `realize()` call. The code still runs, but `data` would be empty—nothing was computed.

---

## Example 2: Shape Gymnastics

Neural networks constantly reshape data. Let's master the basics.

```rust
use ndarray::array;

fn shape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1D tensor with 6 elements
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Original shape: {:?}", data.shape()?);  // [6]

    // Reshape to a 2x3 matrix (or create directly with from_ndarray)
    let matrix = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("Matrix shape: {:?}", matrix.shape());  // [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Transpose to 3x2
    let transposed = matrix.try_transpose(0, 1)?;
    println!("Transposed shape: {:?}", transposed.shape()?);  // [3, 2]
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]

    // Broadcasting: add a row vector to every row
    // [3, 2] + [1, 2] → [3, 2]
    let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
    let biased = &transposed + &bias;

    let result = biased.realize()?;
    println!("{:?}", result.to_ndarray::<f32>()?);
    // [[101, 204],
    //  [102, 205],
    //  [103, 206]]

    Ok(())
}
```

**Key operations:**

| Operation | What it does |
|-----------|--------------|
| `try_reshape(&[2, 3])` | Change shape (same total elements) |
| `try_reshape(&[-1, 3])` | Infer dimension from total size |
| `try_transpose(0, 1)` | Swap dimensions 0 and 1 |
| `try_squeeze(dim)` | Remove dimension of size 1 |
| `try_unsqueeze(dim)` | Add dimension of size 1 |

**Broadcasting rules** (same as NumPy/PyTorch):
- Shapes align from the right
- Each dimension must match or be 1
- Dimensions of size 1 are "stretched" to match

```text
[3, 2] + [1, 2] → [3, 2]  ✓ (1 broadcasts to 3)
[3, 2] + [2]    → [3, 2]  ✓ (implicit [1, 2])
[3, 2] + [3]    → error   ✗ (2 ≠ 3)
```

---

## Example 3: Matrix Multiply

Matrix multiplication is the workhorse of neural networks. Every layer uses it.

```rust
use ndarray::array;

fn matmul_example() -> Result<(), Box<dyn std::error::Error>> {
    // Input: 4 samples, 3 features each → shape [4, 3]
    let input = Tensor::from_ndarray(&array![
        [1.0f32, 2.0, 3.0],    // sample 0
        [4.0, 5.0, 6.0],       // sample 1
        [7.0, 8.0, 9.0],       // sample 2
        [10.0, 11.0, 12.0],    // sample 3
    ]);

    // Weights: 3 inputs → 2 outputs → shape [3, 2]
    let weights = Tensor::from_ndarray(&array![
        [0.1f32, 0.2],  // feature 0 → outputs
        [0.3, 0.4],     // feature 1 → outputs
        [0.5, 0.6],     // feature 2 → outputs
    ]);

    // Matrix multiply: [4, 3] @ [3, 2] → [4, 2]
    let output = input.dot(&weights)?;

    let result = output.realize()?;
    println!("Output shape: {:?}", result.shape());  // [4, 2]
    println!("{:?}", result.to_ndarray::<f32>()?);
    // Each row: weighted sum of that sample's features

    Ok(())
}
```

**Shape rules for `dot()`:**

| Left | Right | Result |
|------|-------|--------|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]` (vector-matrix) |
| `[M, K]` | `[K]` | `[M]` (matrix-vector) |
| `[B, M, K]` | `[B, K, N]` | `[B, M, N]` (batched) |

The inner dimensions must match (the `K`). Think of it as: "for each row of left, dot product with each column of right."

---

## Example 4: Building a Linear Layer

A linear layer computes `y = x @ W.T + b`. Morok provides `nn::Linear` out of the box.

```rust
use morok_tensor::{Tensor, nn::{Linear, Layer}};

fn linear_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a layer: 4 inputs → 2 outputs
    let layer = Linear::with_dims(4, 2, morok_dtype::DType::Float32);

    // Single sample with 4 features
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Forward pass
    let output = layer.forward(&input)?;

    let result = output.realize()?;
    println!("Output: {:?}", result.to_ndarray::<f32>()?);

    Ok(())
}
```

**Why transpose the weights?**

PyTorch convention stores weights as `[out_features, in_features]`. For a layer mapping 4 → 2:
- Weight shape: `[2, 4]`
- Input shape: `[4]` or `[batch, 4]`
- We need: `input @ weight.T` = `[batch, 4] @ [4, 2]` = `[batch, 2]`

This convention makes it easy to read the weight matrix: row `i` contains all weights feeding into output `i`.

---

## Example 5: MNIST Classifier

Let's build a complete neural network using `sequential()` to chain layers.

```rust
use morok_tensor::{Tensor, nn::{Linear, Relu, Layer}};

fn mnist_example() -> Result<(), Box<dyn std::error::Error>> {
    // Architecture: 784 (28×28 pixels) → 128 (hidden) → 10 (digits)
    let fc1 = Linear::with_dims(784, 128, morok_dtype::DType::Float32);
    let fc2 = Linear::with_dims(128, 10, morok_dtype::DType::Float32);

    // Simulate a 28×28 grayscale image (flattened to 784)
    let fake_image: Vec<f32> = (0..784)
        .map(|i| (i as f32) / 784.0)
        .collect();
    let input = Tensor::from_slice(fake_image)
        .try_reshape(&[1, 784])?;  // batch size 1

    // Forward pass: linear → ReLU → linear
    let logits = input.sequential(&[&fc1, &Relu, &fc2])?;
    let probs = logits.softmax(-1)?;

    // Get results
    let probs_result = probs.realize()?;
    println!("Probabilities: {:?}", probs_result.to_ndarray::<f32>()?);

    // Get predicted class
    let prediction = logits.argmax(Some(-1))?;
    let pred_result = prediction.realize()?;
    println!("Predicted digit: {:?}", pred_result.to_ndarray::<i32>()?);

    Ok(())
}
```

**Key concepts:**

1. **`sequential()`** chains layers together: each layer's output feeds into the next. No manual wiring needed.

2. **ReLU activation:** `Relu` is a zero-size layer that applies `max(0, x)`. It introduces non-linearity—without it, stacking linear layers would just be one big linear layer.

3. **Logits vs probabilities:** The raw output of the last layer (logits) can be any real number. `softmax()` converts them to probabilities that sum to 1.

4. **argmax:** Returns the index of the maximum value—the predicted class.

5. **Batch dimension:** We use shape `[1, 784]` for a single image. For 32 images, use `[32, 784]`. The model handles batches automatically.

---

## Example 6: Under the Hood

Want to see what Morok generates? Here's how to inspect the IR and generated code.

```rust
fn inspect_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = &a + &b;

    // Print the computation graph (before compilation)
    println!("=== IR Graph ===");
    println!("{}", c.uop().tree());

    // Compile and execute
    let result = c.realize()?;

    // Inspect generated kernels
    // Note: kernel tracking via kernels() is a placeholder — use prepare() for
    // full kernel inspection (see matmul IR test).
    println!("\n=== Generated Kernels ===");
    for (i, kernel) in result.kernels().iter().enumerate() {
        println!("Kernel {}: {}", i, kernel.name);
        println!("Backend: {}", kernel.backend);
        println!("Code:\n{}\n", kernel.code);
    }

    Ok(())
}
```

**What you'll see:**

1. **IR Graph:** The UOp tree shows operations like `BUFFER`, `LOAD`, `ADD`, `STORE`. This is Morok's intermediate representation before optimization.

2. **Generated Code:** The actual LLVM IR or GPU code that runs. Notice how Morok fuses the loads and add into a single kernel—no intermediate buffers needed.

**Debugging tip:** If something seems slow or wrong, print the IR tree. Look for:
- Unexpected operations (redundant reshapes, extra copies)
- Missing fusion (separate kernels where one would do)
- Shape mismatches (often the root cause of errors)

---

## Summary

You've learned the core patterns for using Morok:

| Task | Code |
|------|------|
| Create tensor | `Tensor::from_slice([1.0f32, 2.0])` |
| Arithmetic | `&a + &b`, `&a * &b`, `-&a` |
| Reshape | `t.try_reshape(&[2, 3])?` |
| Transpose | `t.try_transpose(0, 1)?` |
| Matrix multiply | `a.dot(&b)?` |
| Linear layer | `Linear::with_dims(in, out, dtype)` |
| Chain layers | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| Activation | `t.relu()?`, `t.softmax(-1)?` |
| Execute | `t.realize()?` |
| Extract data | `result.to_ndarray::<f32>()?` |

**The lazy evaluation pattern:**

1. Build your computation graph with operations
2. Call `realize()` once at the end
3. Morok optimizes and executes everything together

**Next steps:**

- [Op Bestiary](./architecture/op-bestiary) — Reference for IR operations
- [Execution Pipeline](./architecture/pipeline) — How compilation works
- [Pattern Engine](./architecture/optimizations/pattern-system) — Pattern-based rewrites
