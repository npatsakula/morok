//! A compute pipeline loaded from a metallib (or MSL source in fallback mode)
//! and dispatched one command buffer per launch.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

use super::compile::{new_library_from_source, validate_metallib};
use super::device::MetalDevice;
use super::objc::{
    AutoreleasePool, Id, MTL_PIPELINE_OPTION_NONE, MTLSize, NSUInteger, ObjcBool, ObjcId, ns_error_message, ns_string,
};
use crate::device::{AbiParamDescriptor, Program};
use crate::{Error, Result};

/// `[[buffer(n)]]` indices are positional in MSL, so the ABI slot is the bind
/// index; Metal exposes 31 of them.
const MAX_BUFFER_BINDINGS: usize = 31;

/// One resolved kernel argument, ready to bind.
enum Arg {
    Buffer(Id, NSUInteger),
    Scalar(i32),
}

pub struct MetalProgram {
    dev: Arc<MetalDevice>,
    name: String,
    label: ObjcId,
    _library: ObjcId,
    _function: ObjcId,
    pipeline: ObjcId,
    abi: Vec<AbiParamDescriptor>,
    buf_count: usize,
    var_count: usize,
    max_total_threads: usize,
    thread_execution_width: usize,
    static_threadgroup_memory: usize,
}

impl std::fmt::Debug for MetalProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalProgram")
            .field("name", &self.name)
            .field("buf_count", &self.buf_count)
            .field("var_count", &self.var_count)
            .field("max_total_threads", &self.max_total_threads)
            .finish_non_exhaustive()
    }
}

impl MetalProgram {
    pub fn load(dev: Arc<MetalDevice>, bytes: &[u8], name: &str, abi: &[AbiParamDescriptor]) -> Result<Self> {
        if abi.iter().enumerate().any(|(index, param)| param.slot != index) || abi.len() > MAX_BUFFER_BINDINGS {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "msl-kernel-abi-v1 binds arguments positionally: slots must be 0..{} (max {MAX_BUFFER_BINDINGS}), got {:?}",
                    abi.len(),
                    abi.iter().map(|param| param.slot).collect::<Vec<_>>()
                ),
            });
        }
        validate_metallib(bytes, name)?;

        let objc = dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        let runtime = |message: String| Error::Runtime { message };

        let library = if bytes.starts_with(b"MTLB") {
            // SAFETY: `dispatch_data_create` copies the bytes (no destructor); `newLibraryWithData:error:` returns +1 or nil.
            unsafe {
                let data = ObjcId::adopt((objc.dispatch_data_create)(
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    null_mut(),
                    null_mut(),
                ))
                .ok_or_else(|| runtime("dispatch_data_create returned nil".into()))?;
                let mut error: Id = null_mut();
                let library = objc.send2::<Id, *mut Id, Id>(
                    dev.mtl(),
                    sels.new_library_with_data_error,
                    data.as_raw(),
                    &mut error,
                );
                ObjcId::adopt(library).ok_or_else(|| {
                    runtime(format!(
                        "newLibraryWithData failed for kernel {name:?}: {}",
                        ns_error_message(objc, error).unwrap_or_default()
                    ))
                })?
            }
        } else {
            let source =
                std::str::from_utf8(bytes).map_err(|_| runtime("Metal fallback payload is not UTF-8 MSL".into()))?;
            new_library_from_source(&dev, source)?
        };

        let label = ns_string(objc, name)?;
        // SAFETY: `newFunctionWithName:` takes an NSString and returns +1 or nil.
        let function = unsafe {
            ObjcId::adopt(objc.send1::<Id, Id>(library.as_raw(), sels.new_function_with_name, label.as_raw()))
        }
        .ok_or_else(|| runtime(format!("Metal library has no kernel entry point {name:?}")))?;

        let descriptor = objc.new_object(objc.classes.compute_pipeline_descriptor)?;
        // SAFETY: setters with an object / BOOL argument.
        unsafe {
            objc.send1::<Id, ()>(descriptor.as_raw(), sels.set_compute_function, function.as_raw());
            objc.send1::<ObjcBool, ()>(descriptor.as_raw(), sels.set_support_indirect_command_buffers, 1);
        }
        let mut error: Id = null_mut();
        // SAFETY: (descriptor, options, reflection out-pointer, error out-pointer) → +1 pipeline or nil.
        let pipeline = unsafe {
            ObjcId::adopt(objc.send4::<Id, NSUInteger, *mut Id, *mut Id, Id>(
                dev.mtl(),
                sels.new_compute_pipeline_state,
                descriptor.as_raw(),
                MTL_PIPELINE_OPTION_NONE,
                null_mut(),
                &mut error,
            ))
        }
        .ok_or_else(|| {
            // SAFETY: autoreleased NSError or nil.
            runtime(format!(
                "compute pipeline creation failed for kernel {name:?}: {}",
                unsafe { ns_error_message(objc, error) }.unwrap_or_default()
            ))
        })?;
        // SAFETY: three NSUInteger accessors.
        let (max_total_threads, thread_execution_width, static_threadgroup_memory) = unsafe {
            (
                objc.send0::<NSUInteger>(pipeline.as_raw(), sels.max_total_threads_per_threadgroup) as usize,
                objc.send0::<NSUInteger>(pipeline.as_raw(), sels.thread_execution_width) as usize,
                objc.send0::<NSUInteger>(pipeline.as_raw(), sels.static_threadgroup_memory_length) as usize,
            )
        };

        Ok(Self {
            dev,
            name: name.to_string(),
            label,
            _library: library,
            _function: function,
            pipeline,
            abi: abi.to_vec(),
            buf_count: abi.iter().filter(|param| param.is_storage()).count(),
            var_count: abi.iter().filter(|param| !param.is_storage()).count(),
            max_total_threads,
            thread_execution_width,
            static_threadgroup_memory,
        })
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.max_total_threads
    }
}

impl Program for MetalProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        wait: bool,
    ) -> Result<()> {
        if buffers.len() != self.buf_count || vals.len() != self.var_count {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "kernel {} expects {} buffers and {} scalars, got {} and {}",
                    self.name,
                    self.buf_count,
                    self.var_count,
                    buffers.len(),
                    vals.len()
                ),
            });
        }
        // `global_size` is already the threadgroup count and `local_size` the
        // threads per group; direct-id kernels run one thread per group.
        let groups = global_size.unwrap_or([1, 1, 1]);
        let threads = local_size.unwrap_or([1, 1, 1]);
        let thread_count = threads.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim)).unwrap_or(usize::MAX);
        if thread_count > self.max_total_threads {
            return Err(Error::Runtime {
                message: format!(
                    "Metal kernel '{}' local size {threads:?} ({thread_count} threads) exceeds maxTotalThreadsPerThreadgroup {} \
                     (threadExecutionWidth {}, staticThreadgroupMemoryLength {} bytes)",
                    self.name, self.max_total_threads, self.thread_execution_width, self.static_threadgroup_memory
                ),
            });
        }

        // Resolve every argument before any Metal object exists: a command
        // encoder released without `endEncoding` is a fatal Metal assertion,
        // so the encoding loop below must be infallible.
        let mut args = Vec::with_capacity(self.abi.len());
        let (mut buffer_index, mut val_index) = (0usize, 0usize);
        for param in &self.abi {
            if param.is_storage() {
                let (buffer, offset) = self.dev.resolve(buffers[buffer_index])?;
                buffer_index += 1;
                args.push(Arg::Buffer(buffer, offset));
            } else {
                let value = i32::try_from(vals[val_index]).map_err(|_| Error::Runtime {
                    message: format!("Metal scalar argument {val_index} value {} does not fit i32", vals[val_index]),
                })?;
                val_index += 1;
                args.push(Arg::Scalar(value));
            }
        }

        let objc = self.dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        // SAFETY: `commandBuffer` / `computeCommandEncoder` return autoreleased objects.
        let command_buffer =
            unsafe { ObjcId::retain(objc, objc.send0::<Id>(self.dev.queue(), sels.command_buffer)) }
                .ok_or_else(|| Error::Runtime { message: "Metal command queue returned no command buffer".into() })?;
        let encoder =
            unsafe { ObjcId::retain(objc, objc.send0::<Id>(command_buffer.as_raw(), sels.compute_command_encoder)) }
                .ok_or_else(|| Error::Runtime { message: "Metal command buffer returned no compute encoder".into() })?;
        let enc = encoder.as_raw();

        // SAFETY: each selector is sent with its documented argument types.
        unsafe {
            objc.send1::<Id, ()>(enc, sels.set_compute_pipeline_state, self.pipeline.as_raw());
            for (index, arg) in args.iter().enumerate() {
                let index = index as NSUInteger;
                match arg {
                    Arg::Buffer(buffer, offset) => {
                        objc.send3::<Id, NSUInteger, NSUInteger, ()>(
                            enc,
                            sels.set_buffer_offset_at_index,
                            *buffer,
                            *offset,
                            index,
                        );
                    }
                    Arg::Scalar(value) => objc.send3::<*const c_void, NSUInteger, NSUInteger, ()>(
                        enc,
                        sels.set_bytes_length_at_index,
                        (&raw const *value).cast(),
                        std::mem::size_of::<i32>() as NSUInteger,
                        index,
                    ),
                }
            }
            objc.send2::<MTLSize, MTLSize, ()>(
                enc,
                sels.dispatch_threadgroups,
                MTLSize::from(groups),
                MTLSize::from(threads),
            );
            objc.send0::<()>(enc, sels.end_encoding);
            objc.send1::<Id, ()>(command_buffer.as_raw(), sels.set_label, self.label.as_raw());
            objc.send0::<()>(command_buffer.as_raw(), sels.commit);
        }

        if wait {
            // SAFETY: blocking wait, then an autoreleased NSError or nil.
            let message = unsafe {
                objc.send0::<()>(command_buffer.as_raw(), sels.wait_until_completed);
                ns_error_message(objc, objc.send0::<Id>(command_buffer.as_raw(), sels.error))
            };
            if let Some(message) = message {
                return Err(Error::Runtime { message: format!("Metal kernel '{}' failed: {message}", self.name) });
            }
        } else {
            self.dev.push_in_flight(command_buffer);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
