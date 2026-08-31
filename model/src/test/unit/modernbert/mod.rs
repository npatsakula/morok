mod attention;
mod classifier;
mod config;
mod embedder;
mod head;
mod jit;
mod model;
mod parity;
mod pooling;
mod rotary;
mod token_classifier;

// Every JIT-compile test in this tree is gated behind `#[ignore = "heavy"]`
// because compiling even a 2-layer transformer graph through the CPU backend
// takes ~1 min/plan, and the codegen backend deadlocks when two plans compile
// (or compile-while-another-executes) concurrently. The embedder/classifier/
// token_classifier tests below follow that convention.
//
// Fast default suite: `cargo test -p svod-model` (these are skipped).
// Full JIT coverage (shared fixtures → only 4 plans compile, not one per test):
//   cargo test -p svod-model -- --ignored test::unit::modernbert:: --test-threads=1
// The `--test-threads=1` is required: the backend is not concurrency-safe.
//
// Parity goldens (`golden*.safetensors`) are generated locally and intentionally
// not committed — see `parity`'s module doc for the generate commands. Model
// weights / configs are fetched from HF Hub on first run.
