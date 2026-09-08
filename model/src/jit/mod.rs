//! JIT support types. They live in `svod-tensor` so that any crate hosting a
//! `jit_wrapper!` invocation needs only that dependency; this module re-exports
//! them under the `crate::jit::…` paths this crate's wrappers name.

pub use svod_tensor::jit::*;
