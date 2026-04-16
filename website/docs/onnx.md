---
sidebar_label: ONNX Inference
---

# ONNX Model Inference

Morok's ONNX importer is the recommended way to run model inference. It loads standard `.onnx` files, decomposes operators into Morok's lazy tensor operations, and compiles them through the full optimization pipeline — no C++ runtime required.

**Current status:**

| Capability | Status |
|------------|--------|
| Forward inference | Supported |
| 162 / 200 ONNX operators | [Parity details](https://github.com/patsak/morok/blob/main/onnx/PARITY.md) |
| CNN architectures (ResNet, DenseNet, VGG, ...) | 9 models validated |
| Microsoft extensions (Attention, RotaryEmbedding) | Supported |
| Dynamic batch size | Supported (Variable API) |
| Training / backward pass | Not supported |

**How does Morok compare to other Rust ML frameworks?**

Among pure-Rust frameworks, Morok offers the broadest ONNX operator coverage — 162 operators with 1361 passing conformance tests across dual backends (Clang + LLVM). `candle` and `burn` each support fewer operators and lack conformance test suites of comparable scope. That said, if you need maximum compatibility with production ONNX models, use `ort` — a Rust wrapper around the C++ ONNX Runtime — which covers the full ONNX spec.

---

## Quick Start

Add `morok-onnx` and `morok-tensor` to your `Cargo.toml`:

```toml
[dependencies]
morok-onnx = { git = "https://github.com/patsak/morok" }
morok-tensor = { git = "https://github.com/patsak/morok" }
```

### Simple: All-Initializer Models

For models where all inputs are baked into the file (no runtime inputs):

```rust
use morok_onnx::{OnnxImporter, OnnxModel};
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let OnnxModel { mut outputs, .. } = importer.import("model.onnx", &[])?;

    // Schedule all outputs together, execute once
    let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
    Tensor::realize_batch(&mut outs)?;

    for (name, tensor) in &outputs {
        println!("{name}: {:?}", tensor.as_ndarray::<f32>()?);
    }
    Ok(())
}
```

### Models with Runtime Inputs

Most models need runtime data (images, tokens, audio). Destructure the `OnnxModel` and use `remove()` to take ownership of input tensors:

```rust
use morok_onnx::{OnnxImporter, OnnxModel};
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let OnnxModel { mut inputs, mut outputs, .. } = importer.import("model.onnx", &[])?;

    // Assign input data (lazy — no allocation yet)
    let input = inputs.remove("input").unwrap();
    input.assign(&Tensor::from_slice(&my_data));

    // Schedule all outputs together, execute once
    // (resolves input assigns internally — no separate realize needed)
    let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
    Tensor::realize_batch(&mut outs)?;
    Ok(())
}
```

---

## Architecture

### Two-Phase Design

The importer processes ONNX models in two distinct phases:

**`import(path, dim_bindings)`** does both phases in a single call: parses the protobuf, extracts initializers and input specs, walks the graph in topological order dispatching each ONNX node to its Tensor implementation, and returns an `OnnxModel { inputs, outputs, variables }`. No execution happens — the result is a set of lazy `Tensor` handles that compile and execute when `realize()`'d.

```text
model.onnx → import(path, dims) → OnnxModel { inputs, outputs, variables } → realize() → results
```

For advanced use cases (inspecting graph structure before import), `import_model()` accepts a pre-parsed `ModelProto`.

### Operator Decomposition

Every ONNX operator is decomposed into Morok Tensor operations. The complexity varies:

**Direct mappings** — about 60 operators map 1:1 to a tensor method:

```rust
// In the registry:
"Add" => x.try_add(y)?
"Relu" => x.relu()?
"Sigmoid" => x.sigmoid()?
"Equal" => x.try_eq(y)?
```

**Builder patterns** — complex operators with many optional parameters use fluent APIs:

```rust
// Conv with optional bias, padding, dilation, groups
x.conv()
    .weight(w)
    .maybe_bias(bias)
    .auto_pad(AutoPad::SameLower)
    .group(32)
    .maybe_dilations(Some(&[2, 2]))
    .call()?
```

**Multi-step decompositions** — operators like BatchNormalization, Attention, and Mod require intermediate computations. For example, Python-style integer `Mod` decomposes into truncation mod + sign adjustment:

```rust
let trunc_mod = x.try_mod(y)?;
let signs_differ = trunc_mod.bitwise_xor(y)?.try_lt(&zero)?;
let needs_adj = mod_ne_zero.bitwise_and(&signs_differ)?;
trunc_mod.try_add(&y.where_(&needs_adj, &zero)?)?
```

### Attribute Validation

The `Attrs` helper uses pop-based extraction — each call to `attrs.int("axis", -1)` or `attrs.float("epsilon", 1e-5)` removes the attribute from the map. After the operator finishes, `attrs.done()` asserts the map is empty. Any leftover attributes trigger an error, catching incomplete operator implementations at trace time rather than producing silent wrong results.

### Opset Versioning

ONNX models declare opset imports per domain. The importer tracks these and passes the version to each operator handler. Operators switch behavior based on version — for example, `Softmax`'s default axis changed from `1` (opset < 13) to `-1` (opset >= 13), and `ReduceSum` moved its axes from an attribute to an input tensor at opset 13.

---

## Working with Models

### Dynamic Dimensions

ONNX inputs can have symbolic dimensions like `"batch_size"` or `"sequence_length"`. Bind them at import time via the `dim_bindings` parameter:

```rust
let model = importer.import("model.onnx", &[
    ("batch_size", 1),
    ("sequence_length", 512),
])?;

// Variables are auto-extracted from dim_param annotations
for (name, var) in &model.variables {
    println!("{name}: bounds {:?}", var.bounds());
}
```

Unbound dynamic dimensions cause a clear error at import time. You can inspect which dimensions are dynamic via `InputSpec::shape`:

```rust
for (name, spec) in &graph.inputs {
    for dim in &spec.shape {
        match dim {
            DimValue::Static(n) => print!("{n} "),
            DimValue::Dynamic(name) => print!("{name}? "),
        }
    }
}
```

### External Weights

For models with weights stored outside the `.onnx` file, use `import_model_with_inputs()` with a pre-parsed `ModelProto`:

```rust
let model_proto = ModelProto::decode(bytes)?;
let model = importer.import_model_with_inputs(
    model_proto,
    external_weights,  // HashMap<String, Tensor>
    &[],
)?;
```

### Microsoft Extensions

The importer supports several `com.microsoft` contrib operators commonly found in transformer models exported from ONNX Runtime:

| Extension | What it does |
|-----------|-------------|
| `Attention` | Packed QKV projection with masking, past KV cache |
| `RotaryEmbedding` | Rotary positional embeddings (interleaved/non-interleaved) |
| `SkipLayerNormalization` | Fused residual + LayerNorm + scale |
| `EmbedLayerNormalization` | Token + position + segment embeddings → LayerNorm |

Standard ONNX transformer operators (`Attention` from the ai.onnx domain) are also supported with grouped query attention (GQA), causal masking, past KV caching, and softcap.

---

## Control Flow and Limitations

### Semantic If: Both Branches Always Execute

ONNX's `If` operator has data-dependent control flow — the condition determines which branch runs. Morok's lazy evaluation model is fundamentally incompatible with this: since nothing executes at trace time, the condition value is unknown.

**Morok's solution:** Trace *both* branches, then merge results with `Tensor::where_()`:

```text
ONNX:    if condition { then_branch } else { else_branch }
Morok:   then_result.where_(&condition, &else_result)
```

This enables **trace-once, run-many** — the compiled graph handles any condition value at runtime. But it has a hard constraint: **both branches must produce identical output shapes and dtypes.** Models with shape-polymorphic branches (where the then-branch produces `[3, 4]` and the else-branch produces `[5, 6]`) cannot be traced.

In practice, most ONNX models with `If` nodes satisfy this constraint because they use conditional logic for value selection, not shape-changing control flow.

### No Loop or Scan

Iterative control flow (`Loop`, `Scan`) is not implemented. These operators require repeated tracing or unrolling, which conflicts with the single-trace architecture. Models using recurrent patterns typically work via unrolled operators (LSTM, GRU, RNN are implemented as native ops).

### Batch Execution

Multiple tensors can be realized together, sharing computation across outputs
(tested in `tensor/src/test/unit/batch.rs`):

```rust
// Realize all outputs at once (shares compilation and execution)
let mut outputs: Vec<&mut Tensor> = model.outputs.values_mut().collect();
Tensor::realize_batch(&mut outputs)?;
```

For repeated inference, use the prepare/execute pattern (tested in
`tensor/src/test/unit/variable.rs::test_prepare_execute_loop`):

```rust
let OnnxModel { mut inputs, mut outputs, variables } =
    importer.import("model.onnx", &[("batch", 1)])?;

// 1. Assign initial data (lazy — no allocation yet)
let input = inputs.remove("audio").unwrap();
input.assign(&Tensor::from_slice(&first_frame));

// 2. Compile the execution plan (resolves assigns, allocates buffers)
let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
let mut plan = Tensor::prepare_batch(&mut outs)?;
plan.execute()?;  // first run

// 3. Fast loop: zero-copy writes via array_view_mut, no recompilation
for frame in audio_frames {
    input.array_view_mut::<f32>()?[..frame.len()].copy_from_slice(&frame);
    plan.execute()?;
}

// Re-execute with different variable bindings
let bound = variables["batch"].bind(8)?;
plan.execute_with_vars(&[bound.as_var_val()])?;
```

### No Training

The importer is inference-only. There is no backward pass, gradient computation, or optimizer support.

### Missing Operator Categories

| Category | Examples | Why |
|----------|----------|-----|
| Quantization | DequantizeLinear, QuantizeLinear | Requires quantized dtype support in IR |
| Sequence ops | SequenceConstruct, SequenceAt | Non-tensor types not in Morok's type system |
| Random | RandomNormal, RandomUniform | Stateful RNG not yet implemented |
| Signal processing | DFT, STFT, MelWeightMatrix | Low priority; niche use cases |
| Text | StringNormalizer, TfIdfVectorizer | String types not supported |

For models using these operators, consider `ort` (ONNX Runtime wrapper) which covers the full spec.

---

## Debugging

### Per-Node Output Tracing

Set the trace log level to dump intermediate outputs:

```bash
RUST_LOG=morok_onnx::importer=trace cargo run
```

This realizes each node's output individually and prints the first 5 values — useful for numerical bisection when a model produces wrong results. Note that this breaks kernel fusion (each node runs separately), so it's purely a debugging tool.

### Inspecting the Graph

Use the `OnnxModel` structure to understand what a model needs:

```rust
let model = importer.import("model.onnx", &[])?;

println!("Inputs:");
for (name, tensor) in &model.inputs {
    println!("  {name}: {:?}", tensor.shape());
}

println!("Outputs: {:?}", model.outputs.keys().collect::<Vec<_>>());
println!("Variables: {:?}", model.variables.keys().collect::<Vec<_>>());
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Entry point** | `OnnxImporter::new()` |
| **Simple import** | `importer.import("model.onnx", &[])?` |
| **Dynamic dims** | `importer.import(path, &[("batch", 4)])?` |
| **Operators** | 162 / 200 ([full parity table](https://github.com/patsak/morok/blob/main/onnx/PARITY.md)) |
| **Validated models** | ResNet50, DenseNet121, VGG19, Inception, AlexNet, ShuffleNet, SqueezeNet, ZFNet |
| **Backends** | Clang + LLVM (identical results) |
| **Extensions** | com.microsoft Attention, RotaryEmbedding, SkipLayerNorm, EmbedLayerNorm |
| **Limitations** | No training, no Loop/Scan, shape-polymorphic If |

**Next:** [Hands-On Examples](./examples) for tensor basics, or [Execution Pipeline](./architecture/pipeline) for how compilation works under the hood.
