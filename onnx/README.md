# morok-onnx

![ONNX coverage](assets/coverage.svg)

ONNX model frontend for Morok. Parses `.onnx` files and builds lazy Morok
tensor graphs that can be compiled once and executed repeatedly.

## Quick Start

```rust
let mut importer = OnnxImporter::new();
let OnnxModel { mut outputs, .. } = importer.import("model.onnx", &[])?;

let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
Tensor::realize_batch(&mut outs)?;
```

## Import with Dynamic Dimensions

When the model has symbolic dimensions (e.g., `batch_size`), bind them at
import time:

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

## Prepare / Execute — Compile Once, Run Many

For repeated inference (tested in `tensor/src/test/unit/variable.rs::test_prepare_execute_loop`):

```rust
let OnnxModel { mut inputs, mut outputs, variables } =
    importer.import("model.onnx", &[("batch", 1)])?;

// 1. Assign initial data (lazy — no allocation yet)
let input = inputs.remove("input").unwrap();
input.assign(&Tensor::from_slice(&initial_data));

// 2. Compile the execution plan (resolves assigns, allocates buffers)
let mut outs: Vec<&mut Tensor> = outputs.values_mut().collect();
let mut plan = Tensor::prepare_batch(&mut outs)?;
plan.execute()?;  // first run

// 3. Fast loop: zero-copy writes via array_view_mut
input.array_view_mut::<f32>()?[..new_data.len()].copy_from_slice(&new_data);
plan.execute()?;
```

## Control Flow — If via Where

ONNX `If` nodes execute both branches and merge results with
`Tensor::where_()`. The condition selects elements lazily at runtime,
enabling the compile-once / run-many pattern for models with data-dependent
branching (e.g., Silero VAD).

Both branches must produce outputs with identical shapes and dtypes.
Models with incompatible branches (e.g., expanded AffineGrid) are
rejected at import time.

## Operator Support

See [PARITY.md](PARITY.md) for the full operator support table with per-operator
test results from the ONNX backend conformance suite.

To regenerate (runs tests automatically, nightly toolchain required):

```bash
uv run --with='onnx' python onnx/scripts/parity.py
```
