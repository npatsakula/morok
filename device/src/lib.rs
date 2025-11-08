use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice};
use morok_dtype::DType;

enum Handle {
    CPU { buffer: Box<[u8]> },
    CUDA { buffer: CudaSlice<u8> },
}

pub struct Buffer {
    buffer: Handle,
    dtype: DType,
    shape: Vec<usize>,
}

#[enum_delegate::implement(DeviceExt)]
pub enum Device {
    CPU(CPU),
    CUDA(CUDA),
}

impl Device {
    pub const fn cpu() -> Self {
        Device::CPU(CPU)
    }

    pub fn cuda(id: usize) -> Self {
        let context = CudaContext::new(id).unwrap();
        Device::CUDA(CUDA { context })
    }
}

#[enum_delegate::register]
pub trait DeviceExt {
    fn allocate_dtype(&mut self, dtype: DType, shape: Vec<usize>) -> Buffer;
    fn allocate_zeroes_dtype(&mut self, dtype: DType, shape: Vec<usize>) -> Buffer {
        self.allocate_dtype(dtype, shape)
    }
}

pub struct CPU;

impl DeviceExt for CPU {
    fn allocate_dtype(&mut self, dtype: DType, shape: Vec<usize>) -> Buffer {
        let bytes = dtype.bytes() * shape.iter().product::<usize>();
        Buffer { buffer: Handle::CPU { buffer: vec![0; bytes].into_boxed_slice() }, dtype, shape }
    }
}

pub struct CUDA {
    context: Arc<CudaContext>,
}

impl DeviceExt for CUDA {
    fn allocate_dtype(&mut self, dtype: DType, shape: Vec<usize>) -> Buffer {
        let bytes = dtype.bytes() * shape.iter().product::<usize>();
        let buffer = unsafe { self.context.default_stream().alloc(bytes) }.expect("unable to allocate CUDA memory");
        Buffer { buffer: Handle::CUDA { buffer }, dtype, shape }
    }

    fn allocate_zeroes_dtype(&mut self, dtype: DType, shape: Vec<usize>) -> Buffer {
        let bytes = dtype.bytes() * shape.iter().product::<usize>();
        let buffer = self.context.default_stream().alloc_zeros(bytes).expect("unable to allocate CUDA memory");
        Buffer { buffer: Handle::CUDA { buffer }, dtype, shape }
    }
}
