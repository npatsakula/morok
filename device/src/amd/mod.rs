//! AMD GPU support for the device crate (KFD-direct, no HIP runtime).
//!
//! Compiled on all Unix hosts (the FFI is `cfg(unix)` — `nix`/KFD ioctls).
//! Availability is decided at runtime, not compile time: the device registry
//! registers the AMD factory only when [`topology::has_devices`] finds a
//! supported GPU, so a host with no `/dev/kfd` simply never reaches this code.

pub mod sys;
pub mod topology;

/// Userspace AMD ("AM") driver — a second [`iface::AmdIface`] backend that
/// drives the GPU's PCI BARs directly, bypassing the kernel amdgpu/KFD driver
/// (selected at runtime via `SVOD_AMD_BACKEND=am`). The memory-manager
/// submodules (`am::mm`) are pure logic — page-table / PTE encoding + the TLSF
/// allocator — and unit-test without a GPU. The privileged bring-up
/// (PCI/PSP/dispatch) is added incrementally. Compiled on all Unix hosts (no
/// extra deps) so it's always type-checked, linted, and tested.
#[cfg(unix)]
pub mod am;

#[cfg(unix)]
pub mod allocator;
#[cfg(unix)]
pub mod connector;
#[cfg(unix)]
pub mod device;
#[cfg(unix)]
pub mod graph;
#[cfg(unix)]
pub mod iface;
#[cfg(unix)]
pub mod kernarg;
pub mod metadata;
pub mod occupancy;
#[cfg(unix)]
pub mod pmc;
#[cfg(unix)]
pub mod program;
#[cfg(unix)]
pub mod queue;
#[cfg(unix)]
pub mod signal;
#[cfg(unix)]
pub mod va_registry;

#[cfg(unix)]
pub use allocator::AmdAllocator;
#[cfg(unix)]
pub use connector::{OwnerCtx, PoolQueue};
#[cfg(unix)]
pub use device::{AmdDevice, AmdDeviceCore};
#[cfg(unix)]
pub use graph::AmdGraph;
#[cfg(unix)]
pub use iface::{AllocKind, AllocResult, AmdIface, KfdIface, QueueHandle, RingDesc};
#[cfg(unix)]
pub use kernarg::KernargArena;
pub use metadata::{KernelArg, KernelMeta, MetadataError, ValueKind, parse_amdgpu_metadata};
#[cfg(unix)]
pub use program::AmdProgram;
#[cfg(unix)]
pub use queue::{AmdComputeQueue, AmdCopyQueue};
#[cfg(unix)]
pub use signal::{AmdSignal, SignalPool, Timeline};
pub use topology::{AmdNode, enumerate, has_devices};
#[cfg(unix)]
pub use va_registry::AllocTag;
