//! Batched replay of a captured kernel chain through one
//! `MTLIndirectCommandBuffer`: every replay costs one command buffer, one
//! encoder and one `executeCommandsInBuffer:` instead of a command buffer per
//! kernel. Commands keep their capture-order barriers, so a replay is
//! observably identical to the per-call sequence.
//!
//! Only all-static chains are captured (buffers only, no scalar arguments —
//! exactly what the execution plan offers a graph factory); anything else
//! declines with `Ok(None)` and stays on per-call dispatch.

use std::sync::Arc;

use parking_lot::Mutex;
use svod_dtype::MetalFamily;

use super::device::MetalDevice;
use super::objc::{
    AutoreleasePool, Id, MTL_INDIRECT_COMMAND_TYPE_CONCURRENT_DISPATCH, MTL_RESOURCE_OPTIONS_DEFAULT,
    MTL_RESOURCE_USAGE_READ_WRITE, MTLSize, NSRange, NSUInteger, ObjcBool, ObjcId, ns_string,
};
use super::program::{MAX_BUFFER_BINDINGS, MetalDispatchTimestamps, MetalProgram};
use crate::device::{Graph, GraphKernel, Program};
use crate::sync::DispatchTimestamps;
use crate::{Error, Result};

/// Indirect command buffers encode buffer offsets as 32-bit values.
const MAX_ICB_OFFSET: NSUInteger = u32::MAX as NSUInteger;

/// Before Apple9 (M3) the driver faults unless every pipeline the indirect
/// command buffer references was also used by the encoder itself; an empty
/// dispatch per pipeline is the known workaround (tinygrad's `FIX_METAL_ICB`).
pub(crate) fn needs_icb_fix(family: MetalFamily) -> bool {
    !matches!(family, MetalFamily::Apple(generation) if generation >= 9)
}

struct CapturedKernel {
    name: String,
    label: ObjcId,
    pipeline: ObjcId,
    buffer_count: usize,
    groups: MTLSize,
    threads: MTLSize,
}

/// The bindings a replay dispatches with; rewritten in place, so a replay
/// must retire the previous one first.
struct Bindings {
    /// Host address per flattened buffer slot, to skip unchanged slots.
    addresses: Vec<u64>,
    /// `(MTLBuffer, offset)` per slot, retained: the indirect command buffer
    /// does not own its resources, and a captured buffer may be freed by the
    /// allocator while the graph still binds it.
    bound: Vec<(ObjcId, NSUInteger)>,
    /// `bound` deduplicated, for `useResources:count:usage:`.
    resources: Vec<ObjcId>,
    last: Option<ObjcId>,
}

pub struct MetalGraph {
    dev: Arc<MetalDevice>,
    icb: ObjcId,
    kernels: Vec<CapturedKernel>,
    /// Distinct pipelines, for the pre-Apple9 workaround.
    pipelines: Vec<ObjcId>,
    /// `(command index, bind index)` per flattened buffer slot.
    slots: Vec<(usize, usize)>,
    needs_icb_fix: bool,
    label: ObjcId,
    state: Mutex<Bindings>,
}

impl std::fmt::Debug for MetalGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalGraph")
            .field("kernels", &self.kernels.iter().map(|kernel| kernel.name.as_str()).collect::<Vec<_>>())
            .field("slots", &self.slots.len())
            .field("needs_icb_fix", &self.needs_icb_fix)
            .finish_non_exhaustive()
    }
}

fn abi_mismatch(reason: String) -> Error {
    Error::ProgramAbiMismatch { reason }
}

impl MetalGraph {
    /// Capture `kernels` into an indirect command buffer. `Ok(None)` when the
    /// chain cannot be graphed on this device (not Metal programs, scalar
    /// arguments, a paravirtualized GPU, an offset beyond 32 bits, or no ICB
    /// support); the caller then dispatches per call.
    pub fn capture(dev: Arc<MetalDevice>, kernels: &[GraphKernel<'_>]) -> Result<Option<Box<dyn Graph>>> {
        if kernels.is_empty() || !dev.supports_graph() {
            return Ok(None);
        }
        let objc = dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;

        let mut captured = Vec::with_capacity(kernels.len());
        let mut pipelines: Vec<ObjcId> = Vec::new();
        let mut slots = Vec::new();
        let mut addresses = Vec::new();
        for (index, kernel) in kernels.iter().enumerate() {
            let Some(program) = kernel.program.as_any().downcast_ref::<MetalProgram>() else { return Ok(None) };
            // Scalars would need a graph-owned argument buffer; static plans have none.
            if program.var_count() != 0 || !kernel.vals.is_empty() {
                return Ok(None);
            }
            if kernel.buffers.len() != program.buf_count() {
                return Err(abi_mismatch(format!(
                    "kernel {} expects {} buffers, graph capture got {}",
                    program.name(),
                    program.buf_count(),
                    kernel.buffers.len()
                )));
            }
            let (groups, threads) = program.launch_sizes(kernel.global_size, kernel.local_size)?;
            // SAFETY: the program owns a reference to its live pipeline state.
            let pipeline = unsafe { ObjcId::retain(objc, program.pipeline()) }.expect("loaded pipeline");
            if !pipelines.iter().any(|known| known.as_raw() == pipeline.as_raw()) {
                pipelines.push(pipeline.clone());
            }
            slots.extend((0..kernel.buffers.len()).map(|bind| (index, bind)));
            addresses.extend(kernel.buffers.iter().map(|pointer| *pointer as usize as u64));
            captured.push(CapturedKernel {
                name: program.name().to_string(),
                label: ns_string(objc, program.name())?,
                pipeline,
                buffer_count: kernel.buffers.len(),
                groups,
                threads,
            });
        }
        let bound = dev.resolve_all(&addresses)?;
        if bound.iter().any(|(_, offset)| *offset > MAX_ICB_OFFSET) {
            return Ok(None);
        }

        let descriptor = objc.new_object(objc.classes.indirect_command_buffer_descriptor)?;
        // SAFETY: descriptor setters with NSUInteger / BOOL arguments, then the
        // `(descriptor, maxCommandCount, options)` constructor returning +1 or nil.
        let icb = unsafe {
            let desc = descriptor.as_raw();
            objc.send1::<NSUInteger, ()>(desc, sels.set_command_types, MTL_INDIRECT_COMMAND_TYPE_CONCURRENT_DISPATCH);
            objc.send1::<ObjcBool, ()>(desc, sels.set_inherit_buffers, 0);
            objc.send1::<ObjcBool, ()>(desc, sels.set_inherit_pipeline_state, 0);
            objc.send1::<NSUInteger, ()>(
                desc,
                sels.set_max_kernel_buffer_bind_count,
                MAX_BUFFER_BINDINGS as NSUInteger,
            );
            ObjcId::adopt(objc.send3::<Id, NSUInteger, NSUInteger, Id>(
                dev.mtl(),
                sels.new_indirect_command_buffer,
                desc,
                kernels.len() as NSUInteger,
                MTL_RESOURCE_OPTIONS_DEFAULT,
            ))
        };
        let Some(icb) = icb else {
            tracing::debug!(kernels = kernels.len(), "Metal declined indirect command buffer; per-call dispatch");
            return Ok(None);
        };

        let mut slot = 0;
        for (index, kernel) in captured.iter().enumerate() {
            // SAFETY: `indirectComputeCommandAtIndex:` returns a live (autoreleased)
            // command object; the setters take object / NSUInteger / MTLSize arguments.
            unsafe {
                let command = objc.send1::<NSUInteger, Id>(
                    icb.as_raw(),
                    sels.indirect_compute_command_at_index,
                    index as NSUInteger,
                );
                objc.send1::<Id, ()>(command, sels.set_compute_pipeline_state, kernel.pipeline.as_raw());
                for (bind, (buffer, offset)) in bound[slot..slot + kernel.buffer_count].iter().enumerate() {
                    objc.send3::<Id, NSUInteger, NSUInteger, ()>(
                        command,
                        sels.set_kernel_buffer_offset_at_index,
                        buffer.as_raw(),
                        *offset,
                        bind as NSUInteger,
                    );
                }
                objc.send2::<MTLSize, MTLSize, ()>(
                    command,
                    sels.concurrent_dispatch_threadgroups,
                    kernel.groups,
                    kernel.threads,
                );
                // Capture-order semantics: every command waits for the previous one.
                objc.send0::<()>(command, sels.set_barrier);
            }
            slot += kernel.buffer_count;
        }

        let resources = dedup_resources(&bound);
        let needs_icb_fix = needs_icb_fix(dev.family());
        tracing::debug!(kernels = captured.len(), slots = slots.len(), "captured Metal indirect command buffer");
        let label = ns_string(objc, &format!("batched {}", captured.len()))?;
        Ok(Some(Box::new(Self {
            dev,
            icb,
            kernels: captured,
            pipelines,
            slots,
            needs_icb_fix,
            label,
            state: Mutex::new(Bindings { addresses, bound, resources, last: None }),
        })))
    }

    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// The previous replay reads the bindings this one rewrites.
    fn retire_last(&self, state: &mut Bindings) -> Result<()> {
        match state.last.take() {
            Some(command_buffer) => self.dev.wait_command_buffer(command_buffer.as_raw(), "Metal graph replay"),
            None => Ok(()),
        }
    }

    /// Point the indirect commands at `buffers` (flattened in capture order);
    /// empty keeps the current bindings.
    fn rebind(&self, state: &mut Bindings, buffers: &[u64], vals: &[i64]) -> Result<()> {
        if !vals.is_empty() {
            return Err(abi_mismatch(format!("Metal graph captured no scalar arguments, replay got {}", vals.len())));
        }
        if buffers.is_empty() || buffers == state.addresses.as_slice() {
            return Ok(());
        }
        if buffers.len() != state.addresses.len() {
            return Err(abi_mismatch(format!(
                "Metal graph replay expected {} buffers, got {}",
                state.addresses.len(),
                buffers.len()
            )));
        }
        let objc = self.dev.objc();
        let sels = &objc.sels;
        for (slot, address) in buffers.iter().enumerate() {
            if *address == state.addresses[slot] {
                continue;
            }
            let (buffer, offset) = self.dev.resolve_retained(*address as usize as *mut u8)?;
            if offset > MAX_ICB_OFFSET {
                return Err(Error::Runtime {
                    message: format!(
                        "Metal graph cannot bind offset {offset} (indirect commands encode 32-bit offsets)"
                    ),
                });
            }
            let (command_index, bind) = self.slots[slot];
            // SAFETY: as in `capture`.
            unsafe {
                let command = objc.send1::<NSUInteger, Id>(
                    self.icb.as_raw(),
                    sels.indirect_compute_command_at_index,
                    command_index as NSUInteger,
                );
                objc.send3::<Id, NSUInteger, NSUInteger, ()>(
                    command,
                    sels.set_kernel_buffer_offset_at_index,
                    buffer.as_raw(),
                    offset,
                    bind as NSUInteger,
                );
            }
            state.bound[slot] = (buffer, offset);
            state.addresses[slot] = *address;
        }
        state.resources = dedup_resources(&state.bound);
        Ok(())
    }

    /// One command buffer executing the whole indirect command buffer.
    fn submit(&self, state: &Bindings) -> Result<ObjcId> {
        let objc = self.dev.objc();
        let sels = &objc.sels;
        let (command_buffer, encoder) = self.dev.begin_compute()?;
        let enc = encoder.as_raw();
        // SAFETY: `useResources:count:usage:` reads `count` object pointers from a
        // C array (`ObjcId` is a transparent `id`); the rest are documented
        // selectors with object / MTLSize / by-value NSRange arguments.
        unsafe {
            objc.send3::<*const Id, NSUInteger, NSUInteger, ()>(
                enc,
                sels.use_resources_count_usage,
                state.resources.as_ptr().cast(),
                state.resources.len() as NSUInteger,
                MTL_RESOURCE_USAGE_READ_WRITE,
            );
            if self.needs_icb_fix {
                for pipeline in &self.pipelines {
                    objc.send1::<Id, ()>(enc, sels.set_compute_pipeline_state, pipeline.as_raw());
                    objc.send2::<MTLSize, MTLSize, ()>(
                        enc,
                        sels.dispatch_threadgroups,
                        MTLSize::default(),
                        MTLSize::default(),
                    );
                }
            }
            objc.send2::<Id, NSRange, ()>(
                enc,
                sels.execute_commands_in_buffer_with_range,
                self.icb.as_raw(),
                NSRange { location: 0, length: self.kernels.len() as NSUInteger },
            );
            objc.send0::<()>(enc, sels.end_encoding);
            objc.send1::<Id, ()>(command_buffer.as_raw(), sels.set_label, self.label.as_raw());
            objc.send0::<()>(command_buffer.as_raw(), sels.commit);
        }
        Ok(command_buffer)
    }
}

fn dedup_resources(bound: &[(ObjcId, NSUInteger)]) -> Vec<ObjcId> {
    let mut resources: Vec<ObjcId> = Vec::with_capacity(bound.len());
    for (buffer, _) in bound {
        if !resources.iter().any(|known| known.as_raw() == buffer.as_raw()) {
            resources.push(buffer.clone());
        }
    }
    resources
}

impl Graph for MetalGraph {
    fn replay(&self, buffers: &[u64], vals: &[i64]) -> Result<()> {
        let mut state = self.state.lock();
        let _pool = AutoreleasePool::push(self.dev.objc());
        self.retire_last(&mut state)?;
        self.rebind(&mut state, buffers, vals)?;
        let command_buffer = self.submit(&state)?;
        state.last = Some(command_buffer.clone());
        self.dev.push_in_flight(command_buffer);
        Ok(())
    }

    /// Per-kernel GPU stamps. With Metal 4 the indirect commands run one at a
    /// time inside a single submission with precise timestamps between them;
    /// otherwise each kernel gets its own command buffer on the legacy queue.
    fn replay_profiled(&self, buffers: &[u64], vals: &[i64]) -> Result<Option<Vec<Arc<dyn DispatchTimestamps>>>> {
        let mut state = self.state.lock();
        let objc = self.dev.objc();
        let _pool = AutoreleasePool::push(objc);
        let sels = &objc.sels;
        self.retire_last(&mut state)?;
        self.rebind(&mut state, buffers, vals)?;

        if let Some(profiler) = self.dev.mtl4() {
            // The Metal 4 queue is not ordered against the legacy one: drain
            // everything that may still be writing this graph's inputs.
            self.dev.synchronize()?;
            let stamps = profiler.time_indirect_commands(
                &self.dev,
                self.icb.as_raw(),
                self.kernels.len(),
                &state.resources,
                &self.pipelines,
                self.needs_icb_fix,
            )?;
            return Ok(Some(
                stamps
                    .into_iter()
                    .map(|(start, end)| {
                        Arc::new(MetalDispatchTimestamps::from_ns(start, end)) as Arc<dyn DispatchTimestamps>
                    })
                    .collect(),
            ));
        }

        let mut command_buffers = Vec::with_capacity(self.kernels.len());
        let mut slot = 0;
        for kernel in &self.kernels {
            let (command_buffer, encoder) = self.dev.begin_compute()?;
            let enc = encoder.as_raw();
            // SAFETY: documented selectors; every bound slot was resolved by `rebind`/`capture`.
            unsafe {
                objc.send1::<Id, ()>(enc, sels.set_compute_pipeline_state, kernel.pipeline.as_raw());
                for (bind, (buffer, offset)) in state.bound[slot..slot + kernel.buffer_count].iter().enumerate() {
                    objc.send3::<Id, NSUInteger, NSUInteger, ()>(
                        enc,
                        sels.set_buffer_offset_at_index,
                        buffer.as_raw(),
                        *offset,
                        bind as NSUInteger,
                    );
                }
                objc.send2::<MTLSize, MTLSize, ()>(enc, sels.dispatch_threadgroups, kernel.groups, kernel.threads);
                objc.send0::<()>(enc, sels.end_encoding);
                objc.send1::<Id, ()>(command_buffer.as_raw(), sels.set_label, kernel.label.as_raw());
                objc.send0::<()>(command_buffer.as_raw(), sels.commit);
            }
            command_buffers.push(command_buffer);
            slot += kernel.buffer_count;
        }
        let mut handles: Vec<Arc<dyn DispatchTimestamps>> = Vec::with_capacity(self.kernels.len());
        for (kernel, command_buffer) in self.kernels.iter().zip(&command_buffers) {
            self.dev.wait_command_buffer(command_buffer.as_raw(), &format!("Metal kernel '{}'", kernel.name))?;
            handles.push(Arc::new(MetalDispatchTimestamps::read(&self.dev, command_buffer.as_raw())));
        }
        Ok(Some(handles))
    }
}
