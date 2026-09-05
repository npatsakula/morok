//! Device implementations for different backends.

pub mod amd;
pub mod cpu;
pub mod metal;

pub use amd::{create_amd_codegen, create_amd_device};
pub use cpu::{
    CpuBackend, cpu_device_with_backend, create_cpu_codegen, create_cpu_device, create_cpu_device_with_backend,
    ensure_thread_pool,
};
pub use metal::{create_metal_codegen, create_metal_device, create_metal_program};
