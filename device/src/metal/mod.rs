//! Metal (Apple GPU) support.
//!
//! Compiled on every host: all Apple frameworks are bound at runtime with
//! `libloading`, so on Linux the dlopen fails, [`has_devices`] is `false` and
//! the runtime never registers the `METAL` factory — the same "always
//! compiled, hardware detected at runtime" contract as [`crate::amd`].
//!
//! Model (v1): one `MTLDevice` + `MTLCommandQueue` per process; shared-storage
//! `MTLBuffer`s whose host mapping doubles as the kernel argument handle
//! (resolved back to `(buffer, offset)` through a pointer registry); one
//! command buffer per dispatch, tracked in an in-flight list that is drained
//! before any host access; all-static plans replay as one indirect command
//! buffer ([`graph`]). Kernels are compiled to metallibs by the private
//! `MTLCodeGenService` (falling back to `newLibraryWithSource:`).

pub mod allocator;
pub mod compile;
pub mod device;
pub mod graph;
pub mod mtl4;
#[doc(hidden)]
pub mod objc;
pub mod program;

pub use allocator::MetalAllocator;
pub use device::{MetalDevice, has_devices};
pub use graph::MetalGraph;
pub use mtl4::Mtl4Profiler;
pub use program::{MetalDispatchTimestamps, MetalProgram};
