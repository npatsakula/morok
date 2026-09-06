//! Metal 4 per-dispatch GPU timestamps (macOS 26+).
//!
//! The legacy API stamps whole command buffers; Metal 4 command encoders write
//! `MTL4CounterHeap` entries between dispatches and can drive our legacy
//! pipelines, buffers and indirect command buffers, so a captured graph is
//! timed kernel by kernel in one submission. Absent on older systems, in which
//! case callers keep the one-command-buffer-per-kernel fallback.

use std::ptr::null_mut;

use parking_lot::Mutex;

use super::device::MetalDevice;
use super::objc::{
    AutoreleasePool, Id, MTL_STAGE_DISPATCH, MTL4_COUNTER_HEAP_TYPE_TIMESTAMP, MTL4_TIMESTAMP_GRANULARITY_PRECISE,
    MTL4_VISIBILITY_OPTION_DEVICE, MTLSize, NSInteger, NSRange, NSUInteger, ObjcBool, ObjcId, ns_data_to_vec,
    ns_error_message,
};
use crate::{Error, Result};

const WAIT_TIMEOUT_MS: u64 = 60_000;

/// A Metal 4 queue with a shared event, used only for profiled replays.
pub struct Mtl4Profiler {
    queue: ObjcId,
    allocator: ObjcId,
    event: ObjcId,
    next_value: Mutex<u64>,
    /// GPU timestamp ticks per second.
    frequency: u64,
}

/// One timed indirect command: its `(start, end)` in GPU nanoseconds.
pub type Stamp = (u64, u64);

fn runtime(message: String) -> Error {
    Error::Runtime { message }
}

impl Mtl4Profiler {
    /// `None` when the OS has no Metal 4 (`newMTL4CommandQueue`) or the device
    /// declines a queue.
    pub(crate) fn open(dev: &MetalDevice) -> Option<Self> {
        let objc = dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        if !objc.responds_to(dev.mtl(), sels.new_mtl4_command_queue)
            || objc.optional_class(c"MTL4CounterHeapDescriptor").is_none()
            || objc.optional_class(c"MTLResidencySetDescriptor").is_none()
        {
            return None;
        }
        // SAFETY: three `new*` constructors (+1 or nil) and two integer accessors.
        unsafe {
            let entry_size = objc.send1::<NSInteger, NSUInteger>(
                dev.mtl(),
                sels.size_of_counter_heap_entry,
                MTL4_COUNTER_HEAP_TYPE_TIMESTAMP,
            );
            if entry_size as usize != std::mem::size_of::<u64>() {
                tracing::warn!(
                    entry_size,
                    "Metal 4 timestamp entries are not 8 bytes; profiling stays on command buffers"
                );
                return None;
            }
            let queue = ObjcId::adopt(objc.send0::<Id>(dev.mtl(), sels.new_mtl4_command_queue))?;
            let allocator = ObjcId::adopt(objc.send0::<Id>(dev.mtl(), sels.new_command_allocator))?;
            let event = ObjcId::adopt(objc.send0::<Id>(dev.mtl(), sels.new_shared_event))?;
            let frequency = objc.send0::<u64>(dev.mtl(), sels.query_timestamp_frequency);
            (frequency > 0).then_some(Self { queue, allocator, event, next_value: Mutex::new(1), frequency })
        }
    }

    pub fn timestamp_frequency(&self) -> u64 {
        self.frequency
    }

    /// Execute `count` commands of a legacy indirect command buffer one at a
    /// time with a precise timestamp before and after each, and return the
    /// stamps once the GPU is done. `resources` must cover every buffer the
    /// commands touch and `pipelines` every pipeline they use: a Metal 4
    /// encoder silently skips the first execution of an indirect command
    /// whose pipeline it has not itself been given (observed on Apple9, macOS
    /// 26), so each pipeline is set on the encoder first; `zero_dispatch` adds
    /// the pre-Apple9 empty dispatch per pipeline on top.
    pub(crate) fn time_indirect_commands(
        &self,
        dev: &MetalDevice,
        icb: Id,
        count: usize,
        resources: &[ObjcId],
        pipelines: &[ObjcId],
        zero_dispatch: bool,
    ) -> Result<Vec<Stamp>> {
        let objc = dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        let mut error: Id = null_mut();

        let heap_descriptor = objc.new_object(objc.classes_optional_new(c"MTL4CounterHeapDescriptor")?)?;
        let residency_descriptor = objc.new_object(objc.classes_optional_new(c"MTLResidencySetDescriptor")?)?;
        // SAFETY: documented selectors; every object below is +1 (`new*`) or a
        // live autoreleased/owned reference for the duration of this call.
        let (heap, residency, command_buffer) = unsafe {
            objc.send1::<NSInteger, ()>(heap_descriptor.as_raw(), sels.set_type, MTL4_COUNTER_HEAP_TYPE_TIMESTAMP);
            objc.send1::<NSUInteger, ()>(heap_descriptor.as_raw(), sels.set_count, (2 * count) as NSUInteger);
            let heap = ObjcId::adopt(objc.send2::<Id, *mut Id, Id>(
                dev.mtl(),
                sels.new_counter_heap_with_descriptor_error,
                heap_descriptor.as_raw(),
                &mut error,
            ))
            .ok_or_else(|| {
                runtime(format!("Metal 4 counter heap failed: {}", ns_error_message(objc, error).unwrap_or_default()))
            })?;
            objc.send1::<NSUInteger, ()>(
                residency_descriptor.as_raw(),
                sels.set_initial_capacity,
                (resources.len() + 1) as NSUInteger,
            );
            let residency = ObjcId::adopt(objc.send2::<Id, *mut Id, Id>(
                dev.mtl(),
                sels.new_residency_set_with_descriptor_error,
                residency_descriptor.as_raw(),
                &mut error,
            ))
            .ok_or_else(|| {
                runtime(format!("Metal 4 residency set failed: {}", ns_error_message(objc, error).unwrap_or_default()))
            })?;
            for resource in resources {
                objc.send1::<Id, ()>(residency.as_raw(), sels.add_allocation, resource.as_raw());
            }
            objc.send1::<Id, ()>(residency.as_raw(), sels.add_allocation, icb);
            objc.send0::<()>(residency.as_raw(), sels.commit);
            let command_buffer = ObjcId::adopt(objc.send0::<Id>(dev.mtl(), sels.new_command_buffer))
                .ok_or_else(|| runtime("Metal 4 command buffer creation failed".into()))?;
            (heap, residency, command_buffer)
        };

        // The allocator is reused across calls; a previous submission has
        // always been waited on before we get here, so resetting is safe.
        // SAFETY: as above.
        unsafe {
            objc.send0::<()>(self.allocator.as_raw(), sels.reset);
            objc.send1::<Id, ()>(
                command_buffer.as_raw(),
                sels.begin_command_buffer_with_allocator,
                self.allocator.as_raw(),
            );
            objc.send1::<Id, ()>(command_buffer.as_raw(), sels.use_residency_set, residency.as_raw());
            let enc = objc.send0::<Id>(command_buffer.as_raw(), sels.compute_command_encoder);
            if enc.is_null() {
                return Err(runtime("Metal 4 command buffer returned no compute encoder".into()));
            }
            for pipeline in pipelines {
                objc.send1::<Id, ()>(enc, sels.set_compute_pipeline_state, pipeline.as_raw());
                if zero_dispatch {
                    objc.send2::<MTLSize, MTLSize, ()>(
                        enc,
                        sels.dispatch_threadgroups,
                        MTLSize::default(),
                        MTLSize::default(),
                    );
                }
            }
            for index in 0..count {
                let stamp = |slot: usize| {
                    objc.send3::<NSInteger, Id, NSUInteger, ()>(
                        enc,
                        sels.write_timestamp_into_heap_at_index,
                        MTL4_TIMESTAMP_GRANULARITY_PRECISE,
                        heap.as_raw(),
                        slot as NSUInteger,
                    )
                };
                stamp(2 * index);
                objc.send2::<Id, NSRange, ()>(
                    enc,
                    sels.execute_commands_in_buffer_with_range,
                    icb,
                    NSRange { location: index as NSUInteger, length: 1 },
                );
                objc.send3::<NSUInteger, NSUInteger, NSUInteger, ()>(
                    enc,
                    sels.barrier_after_stages,
                    MTL_STAGE_DISPATCH,
                    MTL_STAGE_DISPATCH,
                    MTL4_VISIBILITY_OPTION_DEVICE,
                );
                stamp(2 * index + 1);
            }
            objc.send0::<()>(enc, sels.end_encoding);
            objc.send0::<()>(command_buffer.as_raw(), sels.end_command_buffer);
        }

        let value = {
            let mut next = self.next_value.lock();
            let value = *next;
            *next += 1;
            value
        };
        // SAFETY: `commit:count:` reads one id from the array; the event wait
        // returns BOOL; `resolveCounterRange:` returns autoreleased NSData.
        let ticks = unsafe {
            let buffers = [command_buffer.as_raw()];
            objc.send2::<*const Id, NSUInteger, ()>(self.queue.as_raw(), sels.commit_count, buffers.as_ptr(), 1);
            objc.send2::<Id, u64, ()>(self.queue.as_raw(), sels.signal_event_value, self.event.as_raw(), value);
            let signaled = objc.send2::<u64, u64, ObjcBool>(
                self.event.as_raw(),
                sels.wait_until_signaled_value_timeout,
                value,
                WAIT_TIMEOUT_MS,
            );
            if signaled == 0 {
                return Err(runtime(format!("Metal 4 profiled replay did not complete within {WAIT_TIMEOUT_MS} ms")));
            }
            let data = objc.send1::<NSRange, Id>(
                heap.as_raw(),
                sels.resolve_counter_range,
                NSRange { location: 0, length: (2 * count) as NSUInteger },
            );
            ns_data_to_vec(objc, data)
        };
        let expected = 2 * count * std::mem::size_of::<u64>();
        if ticks.len() < expected {
            return Err(runtime(format!("Metal 4 counter heap resolved {} bytes, expected {expected}", ticks.len())));
        }
        Ok(ticks
            .as_chunks::<16>()
            .0
            .iter()
            .map(|pair| {
                let start = u64::from_le_bytes(pair[..8].try_into().expect("8 bytes"));
                let end = u64::from_le_bytes(pair[8..].try_into().expect("8 bytes"));
                (ticks_to_ns(start, self.frequency), ticks_to_ns(end, self.frequency))
            })
            .collect())
    }
}

/// GPU timestamp ticks → nanoseconds, exact for any tick count that fits u64.
pub(crate) fn ticks_to_ns(ticks: u64, frequency: u64) -> u64 {
    (u128::from(ticks) * 1_000_000_000 / u128::from(frequency)) as u64
}
