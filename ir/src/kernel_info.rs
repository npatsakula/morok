//! Kernel metadata for optimized ASTs.
//!
//! This module is a placeholder. The actual KernelInfo implementation
//! lives in the `schedule` crate to avoid circular dependencies
//! (KernelInfo references Opt which is a schedule-layer type).
//!
//! UOp's metadata field uses `Arc<dyn Any + Send + Sync>` to allow
//! any metadata type to be attached, including `schedule::KernelInfo`.
