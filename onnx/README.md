# morok-onnx

![ONNX coverage](assets/coverage.svg)

ONNX model frontend for Morok. Parses `.onnx` files and builds lazy Morok
tensor graphs that can be compiled once and executed repeatedly.

## Quick Start — All-Initializer Model

Models where every input has an initializer (weights baked into the file):

```rust
let mut importer = OnnxImporter::new();
let outputs = importer.import_path("model.onnx")?;
let result = outputs["output"].realize()?;
```

## Trace API — Compile Once, Run Many

For models with runtime inputs (e.g., audio frames, token embeddings), the
trace API separates graph construction from execution:

```rust
use morok_onnx::OnnxImporter;
use prost::Message;

// 1. Parse and prepare the graph structure
let importer = OnnxImporter::new();
let model = ModelProto::decode(bytes)?;
let graph = importer.prepare(model)?;

// 2. Trace — builds the lazy computation graph, allocates input buffers
let (inputs, outputs) = importer.trace(&graph)?;

// 3. Compile into an execution plan (kernels compiled, buffers allocated)
let mut plan = outputs["output"].prepare()?;
let mut executor = morok_runtime::global_executor();

// 4. Feed data and run
let buf_id = inputs["audio_frame"].uop().base().id;
plan.buffer_mut_by_id(buf_id).unwrap().copyin(&frame_bytes);
plan.execute(&mut executor)?;

// 5. Re-run with new data — no recompilation
plan.buffer_mut_by_id(buf_id).unwrap().copyin(&next_frame_bytes);
plan.execute(&mut executor)?;

// 6. Read output
let output_data = plan.output_buffer();
```

### Dynamic Dimensions

When the model has symbolic dimensions (e.g., `batch_size`), bind them to
concrete values at trace time:

```rust
let (inputs, outputs) = importer.trace_with_dims(&graph, &[
    ("batch_size", 1),
    ("sequence_length", 512),
])?;
```

Unbound dynamic dimensions produce an error.

### External Weights

For models with weights stored outside the `.onnx` file:

```rust
let weights: HashMap<String, Tensor> = load_weights("weights.bin")?;
let (inputs, outputs) = importer.trace_external(&graph, weights)?;
```

Provided tensors override auto-resolved placeholders — if you supply a
runtime input in `weights`, no zero-filled buffer is created for it.

Both variants can be combined:

```rust
let (inputs, outputs) = importer.trace_external_with_dims(
    &graph,
    weights,
    &[("batch_size", 4)],
)?;
```

## Control Flow — If via Where

ONNX `If` nodes execute both branches and merge results with
`Tensor::where_()`. The condition selects elements lazily at runtime,
enabling the trace-once / run-many pattern for models with data-dependent
branching (e.g., Silero VAD).

Both branches must produce outputs with identical shapes and dtypes.
Models with incompatible branches (e.g., expanded AffineGrid) are
rejected at trace time.

## Operator Support

See [PARITY.md](PARITY.md) for the full operator support table with per-operator
test results from the ONNX backend conformance suite.

To regenerate (runs tests automatically, nightly toolchain required):

```bash
uv run --with='onnx' python onnx/scripts/parity.py
```
