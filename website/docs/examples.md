---
sidebar_label: Hands-On Examples
---

# Hands-On: From Tensors to Models

This chapter teaches Svod through progressive examples. You'll start with basic tensor operations and build up to a working neural network classifier.

**What you'll learn:**
- Creating and manipulating tensors
- Shape operations (reshape, transpose, broadcast)
- Matrix multiplication
- Building reusable layers
- Composing a complete model

**Prerequisites:**
- Basic Rust knowledge
- Add `svod_tensor` to your `Cargo.toml`

**Key pattern:** Svod uses *lazy evaluation*. Operations build a computation graph without executing. Call `realize()` to compile and run everything at once.

---

## Example 1: Hello Tensor

Let's create tensors, perform operations, and get results.

```rust
use svod_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors from slices
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    // Lazy operations (no execution yet); a scalar is a valid right-hand side
    let sum = (&a + &b)?;
    let scaled = (&sum * 0.1)?;

    // Execute and get results
    scaled.realize()?;
    let data = scaled.as_ndarray::<f32>()?;
    println!("Result: {:?}", data);
    // Output: [1.1, 2.2, 3.3, 4.4]

    Ok(())
}
```

**What's happening:**

1. `Tensor::from_slice()` creates a 1D tensor from array data. The `f32` suffix tells Rust the element type.

2. `&a + &b` doesn't compute anything yet. It returns `Result<Tensor>` — a shape or dtype mismatch is a recoverable error, hence the `?` — wrapping a tensor that *represents* the addition. The `&` borrows the tensors so we can reuse them. `2.0 * &a` works too: scalars are accepted on either side and are materialized in the tensor's dtype.

3. `realize()` is where the magic happens. It takes `&self`, so a realized tensor can stay behind a shared borrow. Svod:
   - Analyzes the computation graph
   - Fuses operations where possible
   - Generates optimized code
   - Executes on the target device

4. `as_ndarray()` extracts the already-computed result as an `ndarray::ArrayD` for inspection.

**Try this:** Remove the `realize()` call. `as_ndarray()` then fails with a "no buffer" error—nothing was computed, so there is no result to read. `to_ndarray()`, `to_vec()` and `item()` realize on demand instead of failing; `as_ndarray()` / `as_vec()` never realize, so they stay usable where realization would be a bug.

---

## Example 2: Shape Gymnastics

Neural networks constantly reshape data. Let's master the basics.

```rust
use svod_tensor::Tensor;
use ndarray::array;

fn shape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1D tensor with 6 elements
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Original shape: {:?}", data.dims()?);  // [6]

    // Reshape to a 2x3 matrix (or create directly with from_ndarray)
    let matrix = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("Matrix shape: {:?}", matrix.dims()?);  // [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Transpose to 3x2
    let transposed = matrix.try_transpose(0, 1)?;
    println!("Transposed shape: {:?}", transposed.dims()?);  // [3, 2]
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]

    // Broadcasting: add a row vector to every row
    // [3, 2] + [1, 2] → [3, 2]
    let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
    let biased = (&transposed + &bias)?;

    biased.realize()?;
    println!("{:?}", biased.as_ndarray::<f32>()?);
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

**Reading a shape:** `dims()` gives a `Vec<usize>` and errors if any axis is symbolic; `dim(axis)` returns that axis as an `SInt` (symbolic or constant) and `dim_const(axis)` as a `usize`, failing with `NonConstDim` when it is not constant; `shape()` returns the whole `Shape` of `SInt`s. `dtype()` is infallible, and `Tensor` implements `Debug` — shape, dtype, device and whether it is realized, never the data, which would force a device read. Negative axes count from the end everywhere.

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
use svod_tensor::Tensor;
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

    output.realize()?;
    println!("Output shape: {:?}", output.dims()?);  // [4, 2]
    println!("{:?}", output.as_ndarray::<f32>()?);
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

A linear layer computes `y = x @ W.T + b`. Svod provides `nn::Linear` out of the box.

```rust
use svod_tensor::{Tensor, nn::{Linear, Layer}};

fn linear_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a layer: 4 inputs → 2 outputs, with a bias
    let layer = Linear::with_dims(4, 2, true, svod_dtype::DType::Float32);

    // Single sample with 4 features
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Forward pass
    let output = layer.forward(&input)?;

    output.realize()?;
    println!("Output: {:?}", output.as_ndarray::<f32>()?);

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
use svod_tensor::{Tensor, nn::{Linear, Relu, Layer}};

fn mnist_example() -> Result<(), Box<dyn std::error::Error>> {
    // Architecture: 784 (28×28 pixels) → 128 (hidden) → 10 (digits)
    let fc1 = Linear::with_dims(784, 128, true, svod_dtype::DType::Float32);
    let fc2 = Linear::with_dims(128, 10, true, svod_dtype::DType::Float32);

    // Simulate a 28×28 grayscale image (flattened to 784)
    let fake_image: Vec<f32> = (0..784)
        .map(|i| (i as f32) / 784.0)
        .collect();
    let input = Tensor::from_slice(fake_image)
        .try_reshape(&[1, 784])?;  // batch size 1

    // Forward pass: linear → ReLU → linear
    let logits = input.sequential(&[&fc1, &Relu, &fc2])?;
    let probs = logits.softmax(-1)?;

    // Get predicted class; realize both results in one compilation
    let prediction = logits.argmax(Some(-1))?;
    Tensor::realize_batch([&probs, &prediction])?;

    println!("Probabilities: {:?}", probs.as_ndarray::<f32>()?);
    println!("Predicted digit: {:?}", prediction.as_ndarray::<i32>()?);

    Ok(())
}
```

**Key concepts:**

1. **`sequential()`** chains layers together: each layer's output feeds into the next. No manual wiring needed.

2. **ReLU activation:** `Relu` is a zero-size layer that applies `max(0, x)`. It introduces non-linearity—without it, stacking linear layers would just be one big linear layer.

3. **Logits vs probabilities:** The raw output of the last layer (logits) can be any real number. `softmax()` converts them to probabilities that sum to 1.

4. **argmax:** Returns the index of the maximum value—the predicted class.

5. **Batch dimension:** We use shape `[1, 784]` for a single image. For 32 images, use `[32, 784]`. The model handles batches automatically.

6. **`realize_batch`:** Two results that share a subgraph (here the logits) compile and run together, so the shared part is computed once. It takes shared references — `[&a, &b]` — because realization is recorded in the tensor registry, not in the handle.

---

## Example 6: Under the Hood

Want to see what Svod generates? Here's how to inspect the IR and the compiled kernels.

```rust
use svod_tensor::Tensor;

fn inspect_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    let c = (&a + &b)?;

    // Print the computation graph (before compilation)
    println!("=== IR Graph ===");
    println!("{}", c.uop().tree());

    // Compile and inspect the execution plan
    let plan = c.prepare()?;  // prepare() takes &self
    println!("\nKernels: {}", plan.kernels().count());

    // Execute
    plan.execute()?;

    Ok(())
}
```

**What you'll see:**

1. **IR Graph:** The UOp tree shows operations like `BUFFER`, `LOAD`, `ADD`, `STORE`. This is Svod's intermediate representation before optimization.

2. **Execution plan:** `prepare()` returns the compiled kernels. Notice how Svod fuses the two loads and the add into a single kernel—no intermediate buffers needed.

**Debugging tip:** If something seems slow or wrong, print the IR tree. Look for:
- Unexpected operations (redundant reshapes, extra copies)
- Missing fusion (separate kernels where one would do)
- Shape mismatches (often the root cause of errors)

---

## Example 7: Layers, modules and state dicts

A layer struct owns its parameters plus the hyper-parameters its forward needs.
`#[derive(Module)]` turns those fields into a flat `StateDict`
(`HashMap<String, Tensor>`) keyed exactly as PyTorch names them, so a
checkpoint loads without a hand-written mapping:

```rust
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Module, StateDict};

#[derive(Clone, Module)]
struct Block {
    intermediate: usize,            // primitives are skipped automatically
    #[module(skip)]                 // a non-primitive that carries no weights
    dtype: DType,
    norm: LayerNorm,                // child module: "norm.weight", "norm.bias"
    #[module(key = "Wi.weight")]    // checkpoint name, dots allowed
    wi: Tensor,
    #[module(key = "Wo.weight")]
    wo: Tensor,
    #[module(optional)]             // written when Some, absent-tolerant on load
    out_bias: Option<Tensor>,
}

fn load(checkpoint: &StateDict) -> Result<Block, Box<dyn std::error::Error>> {
    let mut block = Block {
        intermediate: 3072,
        norm: LayerNorm::with_dims(768, true, 1e-5, DType::Float32),
        wi: Tensor::zeros(&[3072, 768], DType::Float32),
        wo: Tensor::zeros(&[768, 3072], DType::Float32),
        out_bias: None,
    };
    // Reads "layers.0.norm.weight", "layers.0.Wi.weight", ...
    block.load_state_dict(checkpoint, "layers.0")?;
    // ...and writes them back out under any prefix
    let _round_trip: StateDict = block.state_dict("layers.0");
    Ok(block)
}
```

| Attribute | Effect |
|-----------|--------|
| `#[module(key = "Wi.weight")]` | Replace the field-name key segment (may contain dots and digits) |
| `#[module(key = "")]` | Flatten: the field's keys use the parent prefix unchanged |
| `#[module(skip)]` | Ignore a non-primitive field (config, dtype, mode) |
| `#[module(optional)]` | Required on `Option<Tensor>`: saved when `Some`, load tolerates an absent key |
| `#[module(optional = "self.has_bias")]` | The key is required when the predicate holds and skipped otherwise |

Children compose through blanket impls: `Vec<Block>` keys its elements `0.`,
`1.`, …, and arrays, `Option`, tuples and `Box` delegate the same way. Enums
derive too. The forward pass stays out of `Module`: it lives in the `Layer`
trait (`fn forward(&self, x: &Tensor) -> Result<Tensor>`) when the signature
allows, and in inherent methods otherwise.

The built-in layers implement both traits, with `new` for loaded tensors and
`with_dims` for a fresh Kaiming-uniform (convolutions) or identity-affine
(normalizations) initialization:

| Layer | `with_dims` | State-dict keys |
|-------|-------------|-----------------|
| `Linear` | `(in, out, bias, dtype)` | `weight`, `bias` (when present) |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`, `bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`, `bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`, `bias`, `running_mean`, `running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`, `bias` (when present) |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

Hyper-parameters are set with builder-style `with_*` methods on the struct —
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`,
`LayerNorm::with_dims(..).with_axis(-2)`.

---

## Example 8: Recurrent layers

`rnn()`, `gru()` and `lstm()` are builders on `Tensor`. They accept either the
PyTorch weight names (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`, `h0`,
`c0`) or the ONNX ones (`w`, `r`/`r_weights`, `bias`, `initial_h`,
`initial_c`), and reorder the gate blocks for you:

```rust
use svod_tensor::Tensor;
use ndarray::Array3;

// seq=2, batch=1, input=3, hidden=4
let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
let w = Tensor::from_ndarray(&Array3::from_elem((1, 12, 3), 0.1f32));
let r = Tensor::from_ndarray(&Array3::from_elem((1, 12, 4), 0.1f32));

let out = x.gru().w(&w).r_weights(&r).hidden_size(4).call()?;
// ONNX-shaped: y [seq, num_directions, batch, hidden], y_h [num_directions, batch, hidden]
// PyTorch-shaped: output [seq, batch, D*hidden], h_n [num_directions, batch, hidden]
assert_eq!(out.y.dims()?, vec![2, 1, 1, 4]);
assert_eq!(out.output.dims()?, vec![2, 1, 4]);
```

`layout` picks `RnnLayout::SeqFirst` (`[seq, batch, input]`, the default) or
`BatchFirst`; `direction` takes `RnnDirection::{Forward, Backward,
Bidirectional}`, and a bidirectional pass concatenates the two directions on
the feature axis. The GRU's `linear_before_reset` defaults to PyTorch's
placement with PyTorch weights and to ONNX's with ONNX weights. `LstmOutput`
adds `y_c` / `c_n` for the cell state.

The time axis must be concrete, but the batch axis may be a symbolic
`Variable`. For a hand-rolled loop — a decoder stepping one token at a time —
use the cells directly: `GruCell`/`LstmCell`/`RnnCell` expose
`step(&x, &h) -> Result<..>`, and `RnnStack::new(cells)` steps a whole stack
at once.

---

## Example 9: Spectrograms

`stft()` is one `conv1d` against a windowed DFT kernel, so the whole transform
stays in the graph (and the batch axis may stay symbolic). The result is
`[B, F, T, 2]` — or `[F, T, 2]` for an unbatched `[L]` signal — with
`(real, imag)` on the trailing axis, matching
`torch.stft(..., return_complex=false)`:

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

Defaults follow torch: `hop = n_fft / 4`, `win_length = n_fft`, a periodic Hann
window, `center`, `onesided`, no normalization — and `istft` must be given the
same ones. `Window` is `Hann`, `Hamming`, `Rectangular` or `Custom(tensor)`,
and `Tensor::window(&Window::Hann, n, periodic, dtype)` materializes one.
Alongside `magnitude`, the trailing-2 axis has `power`, `complex_abs`,
`complex_mul` and `Tensor::complex_from_polar(&mag, &phase)`.

---

## Errors

Every fallible tensor method returns `svod_tensor::error::Result<T>`, whose
error is a pointer-sized `Error(Box<ErrorKind>)`; match on the cause through
`err.kind()` (or `into_kind()` to take it by value). Downstream crates convert
it with snafu's `context(false)`, so a model's own error enum absorbs it with a
plain `?` — no `.context(TensorSnafu)` at every call site.

Not everything is fallible. `cast`, `neg`, `abs`, `floor`, `ceil`, `round`,
`trunc`, `square`, `sign` and the `Tensor::full` / `zeros` / `ones`
constructors cannot fail and return a plain `Tensor`; `-&a` is likewise plain,
while the binary operators return `Result<Tensor>`.

---

## Summary

You've learned the core patterns for using Svod:

| Task | Code |
|------|------|
| Create tensor | `Tensor::from_slice([1.0f32, 2.0])` |
| Arithmetic | `(&a + &b)?`, `(&a * 2.0)?`, `(2.0 * &a)?`, `-&a` |
| Reshape | `t.try_reshape(&[2, 3])?` |
| Transpose | `t.try_transpose(0, 1)?` |
| Matrix multiply | `a.dot(&b)?` |
| Inspect | `t.dims()?`, `t.dim_const(-1)?`, `t.dtype()` |
| Linear layer | `Linear::with_dims(in, out, bias, dtype)` |
| Chain layers | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| Activation | `t.relu()?`, `t.softmax(-1)?` |
| Load weights | `model.load_state_dict(&sd, "")?` |
| Spectrogram | `x.stft().n_fft(512).hop(160).call()?` |
| Recurrent layer | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| Execute | `t.realize()?` |
| Batch realize | `Tensor::realize_batch([&a, &b])?` |
| Extract data | `t.to_vec::<f32>()?`, `t.to_ndarray::<f32>()?`, `t.item::<f32>()?` |

**The lazy evaluation pattern:**

1. Build your computation graph with operations
2. Call `realize()` once at the end
3. Svod optimizes and executes everything together

**Next steps:**

- [Op Bestiary](./architecture/op-bestiary) — Reference for IR operations
- [Execution Pipeline](./architecture/pipeline) — How compilation works
- [Pattern Engine](./architecture/optimizations/pattern-system) — Pattern-based rewrites
