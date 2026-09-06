//! Target-capability gate for hand-built tile kernels.
//!
//! A tile kernel is built for a specific GPU arch (its WMMA descriptor, wave width,
//! and lane distribution are arch-specific) and compiles via `clang -x ir`. This
//! gate validates the kernel inputs' [`DeviceSpec`] against the **arch(es) the
//! kernel declares it supports** and that the AMD LLVM toolchain is present —
//! failing fast with a clear message instead of mis-rendering or failing deep in
//! compile.
//!
//! The gate is generic over the supported-arch set: a kernel passes its own
//! `&[AmdArch]` (flash-attention declares `[Gfx942]` today). Adding a GPU is
//! "declare its arch here (and supply its arch-specific kernel bits)", not "rewrite
//! this"; the generic launch infra (`compile`/`run_kernel`/`graph_launch`) stays
//! arch-agnostic — only the per-kernel launcher invokes this.
//!
//! It validates **from the `DeviceSpec`** (no full-`Device` open): `DeviceSpec::Amd`
//! deliberately omits the arch (it's a hardware property — baking it into the spec
//! invites the "two specs, one physical device" trap; see `svod_dtype::DeviceSpec`),
//! so the arch is resolved from the spec's `device_id` via the KFD topology.

use svod_dtype::{AmdArch, DeviceSpec};

use crate::launch::{Result, ToolchainUnavailableSnafu, UnsupportedArchSnafu};

/// Resolve the concrete AMD [`AmdArch`] backing a [`DeviceSpec`] from the KFD
/// topology (a non-AMD or unreadable device → `None`). The arch is deliberately
/// not in the spec (a hardware property), so it is looked up by `device_id`.
/// [`resolve_supported_arch`] gates on it and returns it; [`check_target`] is the
/// `()`-returning wrapper for callers that only need the gate.
pub fn resolve_arch(spec: &DeviceSpec) -> Option<AmdArch> {
    match spec {
        DeviceSpec::Amd { device_id } => svod_device::registry::resolve_amd_arch_from_topology(*device_id).ok(),
        // Tile kernels are AMD-only today: a CUDA or Metal GPU has an arch of
        // its own but no `AmdArch`, so the gate reports `UnsupportedArch`.
        DeviceSpec::Cuda { .. }
        | DeviceSpec::Metal { .. }
        | DeviceSpec::Cpu
        | DeviceSpec::WebGpu
        | DeviceSpec::Disk { .. } => None,
    }
}

/// Gate the kernel inputs' device `spec` to one of the kernel's `supported` AMD
/// arches **and** verify the AMD LLVM (`clang` amdgcn) toolchain — returning the
/// resolved arch so the launcher can build [`crate::ArchCaps::for_arch`] from it
/// **without a second topology probe**. A non-AMD spec, an unsupported/unreadable
/// device, or a missing toolchain fails. This is the single arch resolution per
/// launch; call it from a kernel launcher with `Tensor::device()`.
pub fn resolve_supported_arch(spec: &DeviceSpec, supported: &'static [AmdArch]) -> Result<AmdArch> {
    let resolved = resolve_arch(spec);
    let Some(arch) = resolved.filter(|a| supported.contains(a)) else {
        return UnsupportedArchSnafu { supported, spec: spec.clone(), resolved }.fail();
    };
    if !svod_runtime::amd::has_amdgpu_target() {
        return ToolchainUnavailableSnafu.fail();
    }
    Ok(arch)
}

/// [`resolve_supported_arch`] discarding the arch — the gate-only wrapper for
/// launchers that don't need the resolved arch (the SDPA-fallback eligibility
/// check folds this into [`resolve_supported_arch`] directly instead).
pub fn check_target(spec: &DeviceSpec, supported: &'static [AmdArch]) -> Result<()> {
    resolve_supported_arch(spec, supported).map(|_| ())
}
