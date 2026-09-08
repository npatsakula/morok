//! JIT support types. They live in `svod-tensor` so that any crate hosting a
//! `jit_wrapper!` invocation needs only that dependency; this module keeps the
//! historical `crate::jit::…` paths working.

pub use svod_tensor::jit::*;
