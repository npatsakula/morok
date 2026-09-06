//! CUDA unit + hardware tests. Host-only tests (symbol table, struct layout,
//! kernarg packing, PTX entry parsing, alias signatures) run everywhere; the
//! hardware tests return early through `cuda_device_or_skip()` — no
//! `#[ignore]`, so a CUDA host runs them by default and CI without one skips.

mod allocator;
pub(super) mod graph;
pub(super) mod program;
mod sync;
mod sys;

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};

use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::cuda::{CudaAllocator, CudaDevice, CudaProgram};
use crate::device::{AbiParamDescriptor, AbiParamKind};

pub(crate) fn cuda_device_or_skip() -> Option<Arc<CudaDevice>> {
    let device = CudaDevice::open(0).ok();
    if device.is_none() {
        eprintln!("skipping CUDA hardware test: no CUDA device on this host");
    }
    device
}

pub(crate) fn cuda_alloc_or_skip() -> Option<CudaAllocator> {
    cuda_device_or_skip().map(|dev| CudaAllocator { dev, device_id: 0 })
}

/// Three kernels over the PTX kernel ABI (buffers as `.param .u64`, scalars
/// as `.param .u32`), as `nvcc -arch=sm_75 -ptx` emits them with the ISA
/// version lowered to 7.0 so any driver since CUDA 11 JITs them:
/// `vadd`: `out[i] = a[i] + b[i]`; `scale`: `out[i] = a[i] * n`;
/// `tile`: reverses each block through 4 KiB of shared memory.
pub(crate) const KERNELS_PTX: &str = r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry vadd(
	.param .u64 vadd_param_0,
	.param .u64 vadd_param_1,
	.param .u64 vadd_param_2
)
{
	.reg .f32 	%f<4>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<11>;

	ld.param.u64 	%rd1, [vadd_param_0];
	ld.param.u64 	%rd2, [vadd_param_1];
	ld.param.u64 	%rd3, [vadd_param_2];
	cvta.to.global.u64 	%rd4, %rd1;
	cvta.to.global.u64 	%rd5, %rd3;
	cvta.to.global.u64 	%rd6, %rd2;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.u32 	%rd7, %r4, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.f32 	%f1, [%rd8];
	add.s64 	%rd9, %rd5, %rd7;
	ld.global.f32 	%f2, [%rd9];
	add.f32 	%f3, %f1, %f2;
	add.s64 	%rd10, %rd4, %rd7;
	st.global.f32 	[%rd10], %f3;
	ret;
}

.visible .entry scale(
	.param .u64 scale_param_0,
	.param .u64 scale_param_1,
	.param .u32 scale_param_2
)
{
	.reg .f32 	%f<4>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<8>;

	ld.param.u64 	%rd1, [scale_param_0];
	ld.param.u64 	%rd2, [scale_param_1];
	ld.param.u32 	%r1, [scale_param_2];
	cvta.to.global.u64 	%rd3, %rd1;
	cvta.to.global.u64 	%rd4, %rd2;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %tid.x;
	mad.lo.s32 	%r5, %r2, %r3, %r4;
	mul.wide.u32 	%rd5, %r5, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.f32 	%f1, [%rd6];
	cvt.rn.f32.s32 	%f2, %r1;
	mul.f32 	%f3, %f1, %f2;
	add.s64 	%rd7, %rd3, %rd5;
	st.global.f32 	[%rd7], %f3;
	ret;
}

.visible .entry tile(
	.param .u64 tile_param_0
)
{
	.reg .f32 	%f<3>;
	.reg .b32 	%r<12>;
	.reg .b64 	%rd<5>;
	.shared .align 4 .b8 tile_shared[4096];

	ld.param.u64 	%rd1, [tile_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %tid.x;
	cvt.rn.f32.u32 	%f1, %r1;
	shl.b32 	%r2, %r1, 2;
	mov.u32 	%r3, tile_shared;
	add.s32 	%r4, %r3, %r2;
	st.shared.f32 	[%r4], %f1;
	bar.sync 	0;
	mov.u32 	%r5, %ntid.x;
	not.b32 	%r6, %r1;
	add.s32 	%r7, %r5, %r6;
	shl.b32 	%r8, %r7, 2;
	add.s32 	%r9, %r3, %r8;
	ld.shared.f32 	%f2, [%r9];
	mov.u32 	%r10, %ctaid.x;
	mad.lo.s32 	%r11, %r10, %r5, %r1;
	mul.wide.u32 	%rd3, %r11, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.f32 	[%rd4], %f2;
	ret;
}
"#;

pub(crate) fn storage(slot: usize) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Float32, name: None }
}

pub(crate) fn scalar(slot: usize, name: &str) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some(name.to_string()) }
}

pub(crate) fn vadd_abi() -> Vec<AbiParamDescriptor> {
    vec![storage(0), storage(1), storage(2)]
}

pub(crate) fn scale_abi() -> Vec<AbiParamDescriptor> {
    vec![storage(0), storage(1), scalar(2, "n")]
}

pub(crate) fn load(dev: &Arc<CudaDevice>, entry: &str, abi: &[AbiParamDescriptor]) -> CudaProgram {
    CudaProgram::load_ptx(Arc::clone(dev), KERNELS_PTX.as_bytes(), entry, abi).expect("JIT of the test kernels")
}

pub(crate) fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is re-viewed byte-wise.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// A device-local buffer holding `values`.
pub(crate) fn upload(alloc: &CudaAllocator, values: &[f32]) -> RawBuffer {
    let spec = BufferSpec { cpu_access: false, ..BufferSpec::default() };
    let buffer = alloc._alloc(values.len() * 4, &spec, false).unwrap();
    alloc._copyin(&buffer, 0, f32_bytes(values)).unwrap();
    buffer
}

pub(crate) fn download(alloc: &CudaAllocator, buffer: &RawBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    alloc._copyout(&mut bytes, buffer, 0).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect()
}

/// The kernarg address of a buffer: its device pointer.
pub(crate) fn device_ptr(buffer: &RawBuffer) -> *mut u8 {
    let RawBuffer::Cuda { device_ptr, .. } = buffer else { unreachable!() };
    *device_ptr as usize as *mut u8
}
