//! Metal unit + hardware tests. The host-only tests (ObjC/block layout, request
//! framing, reply parsing, pointer registry) run on every platform; the hardware
//! tests return early through `metal_device_or_skip()` — no `#[ignore]`, so the
//! Apple-silicon development host runs them by default and CI (no Mac) skips.

mod allocator;
mod compile;
mod graph;
mod objc;
mod program;
mod registry;

use std::sync::Arc;

use crate::metal::{MetalAllocator, MetalDevice};

pub(crate) fn metal_device_or_skip() -> Option<Arc<MetalDevice>> {
    let device = MetalDevice::open(0).ok();
    if device.is_none() {
        eprintln!("skipping Metal hardware test: no Metal device on this host");
    }
    device
}

pub(crate) fn metal_alloc_or_skip() -> Option<MetalAllocator> {
    metal_device_or_skip().map(|dev| MetalAllocator { dev, device_id: 0 })
}

/// Compile through whichever path the host offers.
pub(crate) fn compile_for_test(dev: &MetalDevice, source: &str) -> crate::Result<Vec<u8>> {
    use crate::metal::compile::{codegen_service_available, compile_msl, compile_msl_public, metal_std_flag};
    if codegen_service_available() {
        let params =
            format!("-fno-fast-math -std={} --driver-mode=metal -x metal -fno-caret-diagnostics", metal_std_flag());
        compile_msl(source, &params)
    } else {
        compile_msl_public(dev, source)
    }
}

/// `out[i] = a[i] + b[i]` over 32-thread groups.
pub(crate) const VADD_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
kernel void vadd(device float* out, device const float* a, device const float* b, \
uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n\
  uint i = gid.x * 32 + lid.x;\n  out[i] = a[i] + b[i];\n}\n";

/// `out[i] = a[i] * n` with a scalar argument bound through `setBytes`.
pub(crate) const SCALE_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
kernel void scale(device float* out, device const float* a, constant int& n, \
uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n\
  uint i = gid.x * 32 + lid.x;\n  out[i] = a[i] * float(n);\n}\n";
