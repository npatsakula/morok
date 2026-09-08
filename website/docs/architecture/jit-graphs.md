---
sidebar_label: JIT Graphs
---

# JIT Graphs

A streaming ASR pipeline calls the same encoder hundreds of times. Building
the tensor graph, optimizing it, generating kernel source, compiling it through
the backend's [JIT loader](../backends/jit-loader.md), and allocating device buffers on
every call wastes work that does not depend on the input.

The `jit_wrapper!` macro and the `model::jit` runtime layer turn that
build-once / run-many pattern into **a typed Rust struct**. You declare the
inputs and the graph; the macro generates a wrapper that compiles the graph
once during `prepare()` and replays it on every `execute()` with the device
buffers held in place.

```mermaid
flowchart TD
  subgraph WO["Without the wrapper (every call)"]
    WO1["build graph"] --> WO2["optimize patterns"]
    WO2 --> WO3["generate kernels"]
    WO3 --> WO4["compile kernels"]
    WO4 --> WO5["alloc buffers"]
    WO5 --> WO6["execute"]
  end
  subgraph WP["With the wrapper (prepare() once)"]
    WP1["build graph"] --> WP2["optimize patterns"]
    WP2 --> WP3["generate kernels"]
    WP3 --> WP4["compile kernels"]
    WP4 --> WP5["alloc buffers"]
  end
  subgraph WS["Every step"]
    WS1["write input buffers"] --> WS2["execute"]
    WS2 --> WS3["read output buffer"]
  end
  WP --> WS
```

The wrapper composes with the [pattern engine](./optimizations/pattern-system.md)
(which runs at `prepare()` time) and the [JIT loader](../backends/jit-loader.md) (which
turns the optimized kernels into in-memory machine code). This page covers the
wrapper layer that sits above both.

---

## The `jit_wrapper!` DSL

A wrapper declaration names the struct, the model type the build closure
receives, the inputs the wrapper exposes, optional symbolic shape variables,
and a `build` block that constructs the graph:

```rust
jit_wrapper! {
    MyModelJit(MyModel) {
        input1: Tensor,
        input2: Tensor,

        vars {
            b: (1, max_batch),
            t: (1, max_time),
        }

        build(input1, input2, b, t) {
            model.forward(input1, input2, &b, &t)
        }
    }
}
```

| Section | Meaning | Required |
|---|---|---|
| `WrapperName(ModelType) { ... }` | name of the generated struct and the type of the model the build closure receives | yes |
| `input_name: Tensor` lines | one per input the wrapper exposes; the `: Tensor` annotation is informational | optional (usually one or more) |
| `inputs { ... }` | the same slots inside a block, where `#[unbatched]` and `[Tensor; N]` are also allowed | optional |
| `vars { name: (min, max), ... }` | symbolic shape variables with compile-time bounds | optional |
| `batch_var name: (min, max)` | a var that also shrinks every batched input's dim 0 to it | optional |
| `state { name, ... }` | inputs the plan also writes, recycled in place between calls | optional |
| `outputs { name, ... }` | one named buffer accessor per output; the `build` closure then returns a tuple of that many tensors, in this order | optional |
| `build(args...) { ... }` | closure that builds the output tensor from inputs and vars; `model` is in scope | yes |

The `build` arguments must each name either an input or a declared var (the
macro rejects names that don't match at expansion time). Inside the block,
each input is a `&Tensor` — or a `[&Tensor; N]` for an array slot — (the macro
allocates a zero-initialized placeholder per buffer when `prepare()` runs),
each var is a `svod_tensor::BoundVariable` already
bound to its upper bound — pass it on as `&name` — and `model` is a shared
reference to the wrapper's owned model value. The closure returns
`Result<Tensor, E>` for any `E: std::error::Error + Send + Sync + 'static`;
failures surface as `JitError::Build`.

Without an `outputs` block the closure returns a single `Tensor`, reachable
through `output()`. With one, it returns a tuple of exactly that many tensors
and each gets its own named `&Buffer` accessor, positioned by declaration
order. If the scheduler fused or elided one of them the positional accessors
would silently misalign, so `prepare()` fails with
`JitError::OutputCountMismatch` instead.

---

## Array slots, batch variables and state

The block forms of the declaration add three things a streaming model needs.
All of them are optional; a wrapper written against the older flat form keeps
working unchanged.

```rust
jit_wrapper! {
    StepJit(StepModel) {
        inputs {
            x: Tensor,
            #[unbatched] bias: Tensor,
            taps: [Tensor; 3],
        }
        batch_var b: (1, 4),
        state { h: Tensor, tail: [Tensor; 2] }
        outputs { emitted }

        // returns (emitted, h, tail): declared outputs first, then state
        build(x, bias, taps, h, tail) {
            model.step(x, bias, taps, h, tail)
        }
    }
}
```

**`[Tensor; N]` slots** put N buffers behind one name: `prepare` takes
`[InputSpec; N]`, the build closure receives `[&Tensor; N]`, and the generated
accessors take a leaf index — `jit.taps_view_mut::<f32>(1)?`. Outputs may be
arrays too.

**`batch_var b: (min, max)`** declares a symbolic variable *and* shrinks every
batched input's dim 0 to it once the placeholders are realized, so one plan
serves a range of batch sizes. `#[unbatched]` opts an input out — a shared bias
or a table whose leading axis is not the batch. Bind it per call with the
generated `execute_bound(4)`.

**`state { ... }`** slots are inputs the plan also writes. The build tuple
carries a new value for each, the macro assigns it straight back into that
slot's own device-local buffer, and the next `execute()` reads it there — a
recurrence that never round-trips through the host. State slots are not
exposed as outputs; `reset()` zeros all of them for a fresh sequence.

The build tuple has one element per declared output slot plus one per state
slot — and no tuple at all when there is exactly one of them.

---

## Symbolic variables

A `vars { ... }` block declares values that participate in the graph as shape
or index expressions but whose exact value is supplied at execute time. They
let one prepared plan serve a range of input shapes without recompiling.

Each entry `name: (min, max)` generates three configuration setters on the
wrapper:

| Setter | Effect |
|---|---|
| `with_<name>_bound(max)` | override only the upper bound; panics if `max < min` |
| `with_<name>_min_bound(min)` | override only the lower bound; panics if `min > max` |
| `with_<name>_fixed(value)` | pin both bounds to `value`, turning the var into a JIT-time constant; panics on `value == 0` |

All three return `Self` (builder style) and must be called before `prepare()`
because the build closure captures the bounds when it runs.

A wider range generates a more general kernel that has to handle every shape
in the range; a tighter range lets the optimizer specialize. Pin a var with
`with_<name>_fixed` when the value never changes, and shrink the upper bound
when an outer caller advertises a smaller maximum than the model's hard
ceiling.

At execute time, pass actual values through `execute_with_vars`, or through
`execute_bound`, which takes one `i64` per declared variable in declaration
order and forwards to it:

```rust
jit.execute_with_vars(&[("b", batch as i64), ("t", time as i64)])?;
jit.execute_bound(batch as i64, time as i64)?;   // same thing, positionally
```

Each pair binds one var; vars not listed keep whatever they hold — their
`prepare()`-time upper bound, or the value a previous `execute_with_vars` left
them at. Bindings are sticky, not per-call. A value outside the var's declared
`[min, max]` is an out-of-bounds access rather than an error: buffers are
allocated to `max`.

---

## Generated runtime API

The macro emits one method group per phase of the wrapper's life cycle:

| Method | Phase | Notes |
|---|---|---|
| `new(model)` | construction | takes the model by value; no kernels compiled yet |
| `with_<var>_bound` / `with_<var>_min_bound` / `with_<var>_fixed` | between `new` and `prepare` | configure shape envelope |
| `prepare(input1: InputSpec, ...)` | one-time | build graph, run patterns, compile kernels, allocate buffers; reads `PrepareConfig::from_env()` |
| `prepare_with_config(..., &PrepareConfig)` | one-time | same as `prepare` with an explicit config |
| `<input>_mut() -> Result<&mut Buffer>` | per step | raw buffer for each declared input |
| `<input>_view_mut::<T>() -> Result<ArrayViewMutD<T>>` | per step | typed write view over that buffer, dtype-checked |
| `output() -> Result<&Buffer>` | per step | output of the prepared graph |
| `<output>_shape() / _view::<T>() / _to_vec::<T>()` | per step | live output shape and reads, resolved against the current variable bindings |
| `reset() -> Result<()>` | per step | zero every `state` slot |
| `execute() -> Result<()>` | per step | replay with current input buffers |
| `execute_bound(v1, v2, ...) -> Result<()>` | per step | replay, binding each declared variable positionally |
| `execute_with_vars(&[(name, value)]) -> Result<()>` | per step | replay and rebind one or more symbolic variables |
| `execute_profiled` / `execute_with_vars_profiled` | optional | same as the non-profiled variants but return `Vec<KernelProfile>` |
| `execute_profiled_static()` | optional | one profiled run through `ExecutionPlan::profile`, returning the last stage's kernels |
| `copy_output_to_<input>(out_pos, dst_off, src_off, len)` | per step | on-device copy of an output region back into an input buffer; no host round-trip |
| `replicate() -> Result<Self>` | optional | deep-copy a prepared JIT for concurrent execution: forked buffers, shared model and kernels, its own queue |

Four lower-level accessors expose plan details for tooling:

| Accessor | Returns |
|---|---|
| `buffers()` | every buffer the plan owns |
| `output_buffers()` | the plan's declared output buffers |
| `input_buffer_ids()` | device buffer ids the wrapper writes to |
| `prepared_kernels()` | the compiled kernels |

Most callers do not need these. Calling any per-step method before `prepare()`
returns `JitError::NotPrepared`.

---

## `InputSpec`

`InputSpec`, `JitError` and the buffer helpers the macro expands to live in
`svod_tensor::jit`, so a crate hosting a `jit_wrapper!` needs only that
dependency (`svod_model::jit` re-exports them for the historical paths).

`prepare()` takes one `InputSpec` per declared input — or one `[InputSpec; N]`
per array slot:

```rust
pub struct InputSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Allocate the input device-local (no host mapping).
    pub device_local: bool,
}

impl InputSpec {
    pub fn new(shape: &[usize], dtype: DType) -> Self { ... }
    pub fn f32(shape: &[usize]) -> Self { ... }
    pub fn i32(shape: &[usize]) -> Self { ... }
    pub fn i64(shape: &[usize]) -> Self { ... }
    pub fn device_local(mut self) -> Self { ... }
}
```

The macro uses the shape and dtype to allocate a zero-initialized placeholder
tensor before invoking the build closure. Callers do not construct
`Tensor::zeros(...).realize()` placeholders themselves. The shape becomes the
maximum input size; symbolic variables shrink it at execute time through
operations like `try_shrink` — a coding pattern, not a runtime contract
enforced by the wrapper. `InputSpec::device_local()` drops the host mapping for inputs the host only
writes through `copyin` or refills on-device; `state` slots are allocated that
way automatically. On the output side, `PrepareConfig::device_local()` is the
same idea for the plan's outputs — it is `from_env()` with
`device_local_outputs` set.

---

## Recurrent execution

A recurrent model's state stays on the device: declare it in `state { ... }`
and every step is one `execute()`, with no host round trip and no packing
helper.

```rust
jit.reset()?;                                    // zero the state, new sequence
for chunk in chunks {
    for (slot, v) in jit.x_view_mut::<f32>()?.iter_mut().zip(chunk) {
        *slot = v;                               // per-step input, written in place
    }
    jit.execute()?;                              // reads state, writes it back
    let frame = jit.emitted_to_vec::<f32>()?;    // only the emitted head crosses
}
```

:::tip[Read-before-write ordering]
Each state buffer is recycled in place, so a slot must not depend on another
slot's *new* value inside one `build`: the per-buffer ordering is only
unambiguous when every slot advances from the values the step was entered
with. Derive the new values from the inputs and the old state, then return
them all in the build tuple.
:::

The state buffers are allocated device-local, so nothing maps them to the
host. Read back only what the caller actually needs — the declared outputs —
through `<output>_to_vec` or `<output>_view`.

---

## Example: GigaAM encoder

The GigaAM Conformer encoder is prepared at constant shape. The batch and
mel-frame bounds are computed once at construction and baked into the plan;
shorter chunks are zero-padded into the same buffers:

```rust
jit_wrapper! {
    GigaAmEncoderJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        build(mel, lengths) {
            let out = model.encoder.forward_batch(mel, lengths)?;
            // Permute [B, d_model, T_sub] → [B, T_sub, d_model] on-device: the
            // RN-T decoder consumes frame-major rows, and doing it here turns
            // a host-side strided transpose into one contiguous copyout.
            Ok::<_, super::error::Error>(
                out.cast(svod_dtype::DType::Float32).try_permute(&[0, 2, 1])?
            )
        }
    }
}
```

The wrapper takes a mel-spectrogram input and a per-batch length vector and
produces `[B, T_sub, d_model]`. `GigaAmTranscriber` sizes the plan once: the
mel length is rounded up to the next power of two so codegen sees a clean
factorisation and clamped to `config.max_mel_frames`, and the batch is capped so
the live SDPA score tiles stay inside `max_scores_mib`. Every chunk then
replays the same plan through `execute()`.

`cast` is infallible, so it needs no `?`, and the model's error type absorbs
the tensor error with a plain `?` — the build closure returns
`Result<_, E>` for any `E: std::error::Error + Send + Sync + 'static`.

The `out.cast(DType::Float32)` is the fp32 boundary between the
encoder and any downstream head. The encoder may run in fp16 or bf16 for
speed, but every consumer (CTC log-softmax, RN-T predictor and joint) sees a
uniform fp32 input. Placing the cast inside the JIT lets it fuse into the
encoder's tail kernels.

---

## Example: Silero VAD

Silero V5 is a recurrent network, but its recurrence is far too small to pay
for a launch per window. The JIT therefore covers only the batched conv
front-end plus the LSTM input projection; the scan itself stays on the host:

```rust
jit_wrapper! {
    SileroVadFeatureJit(SileroVad) {
        chunks: Tensor,

        build(chunks) {
            // [FEATURE_BATCH, CHUNK_LEN] -> [FEATURE_BATCH, 4*HIDDEN] LSTM gate
            // pre-activations (conv features + input projection, biases folded).
            model.forward_gates(chunks)
        }
    }
}
```

The leading dimension is a fixed `FEATURE_BATCH` (4096) rather than a var: the
front-end is row-independent, so a partial batch simply fills fewer rows, and a
symbolic leading dim trips the reflect-pad lowering. Preparation asks for a
device-local output, because the 8 MiB gate readback belongs on the copy engine
rather than the host mapping:

```rust
let mut jit = SileroVadFeatureJit::new(vad);
jit.prepare_with_config(
    InputSpec::f32(&[FEATURE_BATCH, CHUNK_LEN]),
    &svod_tensor::PrepareConfig::device_local(),
)?;
```

`VadInference::probs` then walks the waveform in `FEATURE_BATCH`-sized
dispatches — pack `chunks_mut()`, `execute()`, `copyout_prefix` the valid rows
— and hands the gates to `VadHead::scan`, an 8-lane `f32x8` LSTM plus sigmoid
head on the host. That split replaced a one-tiny-dispatch-per-window path whose
round-trip latency dominated the whole model.

---

## Data-independence contract

The wrapper compiles the graph once and replays it many times. That only
works if the graph topology is fixed at `prepare()` time. Anything that can
change at execute time has to flow through input buffers (via `*_mut`) or
symbolic vars (via `execute_with_vars`). A branch on a tensor value inside
the build closure specializes the graph to that branch; this is a build-time
decision, not a runtime one.

:::note[Pitfalls]
- A `Tensor::full(value).realize()` inside the build closure bakes that value
  into the single prepared plan. Any per-call variation requires re-running
  `prepare()` from scratch — full graph build plus kernel compile. Host-side
  scratch buffers (for example `ndarray::Array3`) are the right choice for
  per-step setup that the JIT does not need to see.
- The idiomatic way to handle a dynamic batch is `batch_var`, which shrinks
  dim 0 of every batched input for you; bind it per call with
  `execute_bound`. ResNet and YOLO are both one `images` input, one
  `batch_var b: (1, max_batch_size)` and one output. For any other dynamic
  axis, `try_shrink` on a maximum-sized input with a var-bound length plus
  `execute_with_vars` at the call site is the manual equivalent.
:::

Violating the contract produces one of two failure modes: wrong results,
because the cached plan replays with a stale assumption about a value that
turned out to vary; or silent slowness, because every call ends up in a
recompile path. Diagnose these by re-reading the build closure; kernel output
rarely helps.

---

## Errors

`JitError` covers the runtime failures the wrapper can raise. Most are
unrecoverable and indicate a usage bug rather than a transient condition.

| Variant | Triggered by |
|---|---|
| `NotPrepared` | per-step method called before `prepare`, or output buffer unavailable |
| `InputBufferNotFound` | input index resolution failed inside the prepared plan |
| `DuplicateInputBuffer` | two declared inputs map to the same device buffer at `prepare` time |
| `InputAliased` | an input resolved to a foreign plan buffer — a concurrent `prepare` corrupted its graph identity |
| `Build` | the build closure returned `Err`; the inner error is preserved as `Box<dyn Error + Send + Sync>` |
| `Tensor` | tensor op failed during `prepare` or in the build closure |
| `Device` | a device or buffer operation failed |
| `OutputCountMismatch` | a wrapper declared N output plus state slots but the compiled plan kept a different number |
| `DtypeMismatch` | a typed view or read asked for a dtype the buffer does not hold |
| `ViewOutOfBounds` | a live output shape needs more elements than its buffer holds — the bound variables exceed what the plan was compiled for |
| `InferredOutputDim` | an output shape carried a `-1` dimension, which has no live value to substitute |
| `Runtime` | kernel execution failed |

Configuration mistakes on the symbolic-variable setters (`with_<var>_*`)
panic at the call site instead of returning an error, since they happen
before any plan exists.

---

## Why this matters

**Lifecycle is explicit.** `prepare` is the only way into the prepared state,
and every per-step accessor goes through it. The wrapper holds the plan behind
an `Option`, so calling out of order fails immediately with
`JitError::NotPrepared` rather than reading a half-built plan.

**Replay is cheap.** One graph build, one kernel compile, one set of
allocations — paid once. Every subsequent call is buffer writes plus an
`execute`.

**Contract is local.** The data-independence rule is the single invariant
that lets the wrapper skip the per-call dance safely. Every other guarantee
follows from it.

**Errors are explicit.** Runtime failures surface as `JitError` variants;
only configuration-time misuse on the variable setters still panics.

The wrapper does not invent new primitives. It takes the build / prepare /
execute cycle and gives it a shape that the type system can hold, so
streaming inference runs at the speed of one-shot evaluation without the
per-call overhead.
