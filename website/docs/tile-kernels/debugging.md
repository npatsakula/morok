---
sidebar_label: Debugging
---

# Debugging and Verifying Kernels

A hand-written kernel is only as trustworthy as your ability to check it. The
[Flash Attention](./flash-attention) walkthrough showed the kind of kernel worth writing by
hand; this chapter is how you come to trust it. The USE face hands you a lazy `Tensor` that
fuses into a big graph — convenient, but a bad place to ask "is this one kernel correct, and how
fast is it?" `tk`'s **DEBUG face** exists for exactly that: run a single kernel against concrete
buffers, read the result back, time it, and prove that a refactor didn't change its behavior.

---

## Direct dispatch: run one kernel, see the bytes

The direct-launch API (`tk/src/launch.rs`) bypasses the tensor scheduler entirely. You give it
a finished `Kernel` and real input buffers; it renders, compiles, and dispatches, writing the
result into an output buffer you can read back:

```rust
// The DEBUG face from tk/src/lib.rs. `outs` are written in place.
run_kernel("tile_add", [1, 1, 1], block, &mut [&mut out], &[&input_a, &input_b], build)?;
let values = out.as_vec::<f32>()?;   // read the GPU result straight back
assert_eq!(values, expected);
```

Because this skips scheduling, fusion, and dependency tracking, what you measure is *just your
kernel* — not a graph that happens to contain it. That isolation is the point: when a number is
wrong, you want to know it's wrong *here*, not somewhere in a fused pipeline.

A note on the path: skipping the *scheduler* is not skipping the *optimizer*. `compile` still runs
the production `optimize_kernel_with_config` over your `SINK` — which applies zero schedule opts to a
hand-lowered body (that's what the `opts_to_apply: Some(vec![])` marker buys) but still performs the
shared rewrites every kernel needs before rendering, index-dtype lowering among them. You get
correct code without the scheduler.

---

## Timing on real hardware

For performance work, `CompiledLaunch` (from `compile` / `compile_kernel`) exposes hardware
timestamps rather than wall-clock guesses:

```rust
// Render + compile once …
let launch = compile_kernel("matmul", grid, block, &mut [&mut c], &[&a, &b], build)?;
// … then dispatch in a loop, outside the timed region.
// SAFETY: the bound buffers stay allocated for `launch`'s lifetime.
unsafe { launch.dispatch(true) }?;
let ns = launch.dispatch_gpu_ns()?;   // Option<u64>: device-measured dispatch time
```

`dispatch_gpu_ns()` reads the GPU's own timestamp counters around the dispatch, so you're
measuring time on the device, not the round-trip latency of launching it. The criterion benches
reach the same device-time stamps one layer up, through `plan.profile`, to compare a `tk` kernel
against the graph-native baseline. Those same
benches do more under `cargo bench --profile-time`: each benchmarked plan is fed through the full
layered profiler — device time, roofline, occupancy, and hardware counters — accumulated by
per-kernel minimum and written to a table. See [Profiling & Benchmarking](./profiling) for the
tiers, the env vars, and the criterion wiring.

:::tip[For GPU experts]
`KernelFingerprint` is a *structural* hash of the `SINK`'s UOp graph — it captures the shape (ops, dtypes, edges) independent of instance IDs, so it is stable across runs and processes. That is what makes it a golden-test key: a behavior-preserving refactor reproduces the fingerprint, while any change to the emitted IR moves it. `dispatch_gpu_ns` reads the device's own timestamp counters around the dispatch, so it measures on-device time, not launch latency.
:::

---

## Fingerprints: proving a refactor is behavior-preserving

The subtle risk with hand-written kernels: you "clean up" the builder code, the kernel still
compiles and still produces plausible numbers, but the *generated IR* changed in a way that
only shows up on some shape or some architecture later.

`KernelFingerprint` (`tk/src/fingerprint.rs`) guards against this. It computes a deterministic,
structural hash of a kernel's UOp graph — the shape of the SINK, not the pointer identities. You
snapshot the fingerprint as a golden value, and a refactor that's meant to be purely cosmetic
must reproduce it:

```rust
let fp = kernel_fingerprint(&sink);
assert_eq!(fp.digest, GOLDEN_MATMUL_DIGEST);  // structure unchanged ⇒ behavior unchanged
```

If the fingerprint moves, you changed the emitted IR — intentionally or not — and the golden
test makes you look. The unit tests in `tk/src/test/unit/golden.rs` use exactly this to lock
the matmul and Flash Attention graphs (digest *and* node count).

---

## Which tool for which question

| You're asking… | Use |
|----------------|-----|
| "Does this kernel produce the right numbers?" | `run_kernel` + `as_vec`, compare against a reference |
| "How fast is it on this GPU?" | `compile_kernel` + `dispatch_gpu_ns` |
| "Did my refactor change the emitted IR?" | `KernelFingerprint` golden test |
| "Is the *device/driver layer* misbehaving?" | [AMD Backend → Debugging](../backends/amd/debugging), [CUDA Backend → Debugging](../backends/cuda/debugging) |

That last row matters: this chapter is about debugging *kernels* — the IR you authored and the
numbers it produces. When the problem is below that — queue dispatch, memory faults, the driver, the
PTX JIT — the per-backend chapters are the right place:
[AMD](../backends/amd/debugging) and [CUDA](../backends/cuda/debugging).

---

## Why this matters

Hand-authoring trades the optimizer's safety net for control. The DEBUG face is how you make
that trade safely: isolation to localize correctness bugs, hardware timestamps to make
performance claims you can defend, and structural fingerprints so that "I just tidied the code"
can't silently become "I changed the kernel." With those three, a hand-written kernel is as
verifiable as an autotuned one.
