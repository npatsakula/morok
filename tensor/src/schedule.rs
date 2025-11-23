//! Kernel scheduling types and execution.
//!
//! This module provides types and functions for managing the execution
//! schedule of tensor operations. After the rangeify pipeline transforms
//! the computation graph into KERNEL operations, we need to:
//!
//! 1. Extract kernel operations from the transformed graph
//! 2. Allocate buffers for intermediate results (DEFINE_GLOBAL/DEFINE_LOCAL)
//! 3. Execute kernels in dependency order
//!
//! The scheduling process converts from lazy tensor operations to
//! executable kernels with properly allocated device buffers.

use crate::{BUFFERS, Result};
use morok_device::Buffer;
use morok_ir::{Op, UOp};
use std::rc::Rc;

/// A single executable kernel with its buffers.
///
/// Each ScheduleItem represents one kernel that needs to be compiled
/// and executed. The kernel AST contains STORE operations that write
/// results to buffers.
#[derive(Clone)]
pub struct ScheduleItem {
    /// The kernel AST (Op::Kernel with SINK containing STORE ops)
    pub ast: Rc<UOp>,

    /// Device buffers for this kernel (in order expected by codegen)
    pub buffers: Vec<Buffer>,
}

/// Full execution schedule (list of kernels in dependency order).
pub type Schedule = Vec<ScheduleItem>;

/// Extract kernels from transformed graph and create schedule.
///
/// This function walks the transformed UOp graph (after rangeify and
/// kernel splitting) and extracts all KERNEL operations. For each kernel,
/// it identifies the buffers needed from the kernel's sources.
///
/// # Arguments
///
/// * `transformed` - The UOp graph after rangeify + kernel splitting
///
/// # Returns
///
/// A schedule of kernels to execute, with buffers attached.
///
/// # Errors
///
/// Returns error if:
/// - No kernels found after scheduling pipeline
/// - Buffer not found in registry for a kernel source
pub fn create_schedule(transformed: Rc<UOp>) -> Result<Schedule> {
    use crate::error::*;

    // Step 1: Find all KERNEL operations
    let mut kernels = Vec::new();
    for node in transformed.toposort() {
        if matches!(node.op(), Op::Kernel { .. }) {
            kernels.push(node);
        }
    }

    if kernels.is_empty() {
        return Err(Error::Runtime { message: "No kernels found after scheduling pipeline".to_string() });
    }

    // Step 2: For each kernel, collect buffers from sources
    let mut schedule = Vec::new();

    for kernel_uop in kernels {
        let (sources, _ast) = match kernel_uop.op() {
            Op::Kernel { sources, ast } => (sources, ast),
            _ => unreachable!("filtered to only kernels above"),
        };

        // Step 3: Map sources to actual Buffers
        // Sources can be:
        // - DEFINE_GLOBAL(id) - intermediate buffer to allocate
        // - DEFINE_LOCAL(id) - local/shared memory to allocate
        // - BUFFER - input buffer from registry
        // - BIND - variable binding (skip for buffer collection)

        // For MVP: We'll collect existing buffers and mark which need allocation
        // The actual allocation will happen in allocate_kernel_buffers()

        let buffers = collect_kernel_buffers(sources, &kernel_uop)?;

        schedule.push(ScheduleItem { ast: kernel_uop.clone(), buffers });
    }

    Ok(schedule)
}

/// Collect buffers for a kernel from its sources.
///
/// This walks the kernel sources and identifies:
/// - Input buffers (Op::Buffer) - get from BUFFERS registry
/// - Intermediate buffers (Op::DefineGlobal, Op::DefineLocal) - need allocation
///
/// For MVP, we only handle input buffers. Intermediate buffer allocation
/// is handled separately in allocate_kernel_buffers().
fn collect_kernel_buffers(sources: &[Rc<UOp>], _kernel: &Rc<UOp>) -> Result<Vec<Buffer>> {
    use crate::error::*;

    let mut buffers = Vec::new();

    for src in sources {
        match src.op() {
            Op::Buffer { .. } => {
                // Input buffer - get from registry
                if let Some(buffer) = BUFFERS.with(|b| b.borrow().get(&src.id).cloned()) {
                    buffers.push(buffer);
                } else {
                    return Err(Error::BufferNotFound { uop_id: src.id });
                }
            }
            Op::DefineGlobal(_id) | Op::DefineLocal(_id) => {
                // Intermediate buffer - needs allocation
                // For now, allocate in allocate_kernel_buffers()
                // Mark as needing allocation by skipping here
                continue;
            }
            Op::Bind { .. } => {
                // Variable binding - not a buffer
                continue;
            }
            _ => {
                // Unknown source type - might be OK, just skip
                continue;
            }
        }
    }

    Ok(buffers)
}

/// Allocate buffers for DEFINE_GLOBAL/DEFINE_LOCAL operations in a kernel.
///
/// This analyzes the kernel sources and creates device buffers for
/// any intermediate allocations. The buffers are allocated on the same
/// device as the kernel's input buffers.
///
/// # Arguments
///
/// * `kernel` - The KERNEL UOp whose buffers to allocate
///
/// # Returns
///
/// A vector of all buffers (inputs + intermediates) in the order expected
/// by codegen.
///
/// # Errors
///
/// Returns error if:
/// - Buffer size cannot be determined from kernel AST
/// - Device allocation fails
pub fn allocate_kernel_buffers(kernel: &Rc<UOp>) -> Result<Vec<Buffer>> {
    use crate::error::*;
    use morok_device::registry;

    let (sources, ast) = match kernel.op() {
        Op::Kernel { sources, ast } => (sources, ast),
        _ => {
            return Err(Error::Runtime { message: "Expected KERNEL operation".to_string() });
        }
    };

    let mut buffers = Vec::new();

    for src in sources {
        match src.op() {
            Op::DefineGlobal(id) => {
                // Allocate global buffer
                let dtype = src.dtype();
                let size = compute_buffer_size(ast, src)?;

                let device = registry::cpu()
                    .map_err(|e| Error::Device { message: format!("Failed to get CPU device: {}", e) })?;
                let buffer = Buffer::new(device, dtype, vec![size], Default::default());

                // Register in BUFFERS for future lookups
                BUFFERS.with(|b| b.borrow_mut().insert(*id as u64, buffer.clone()));

                buffers.push(buffer);
            }
            Op::DefineLocal(id) => {
                // Local/shared memory - allocate similarly
                let dtype = src.dtype();
                let size = compute_buffer_size(ast, src)?;

                let device = registry::cpu()
                    .map_err(|e| Error::Device { message: format!("Failed to get CPU device: {}", e) })?;
                let buffer = Buffer::new(device, dtype, vec![size], Default::default());

                BUFFERS.with(|b| b.borrow_mut().insert(*id as u64, buffer.clone()));

                buffers.push(buffer);
            }
            Op::Buffer { .. } => {
                // Input buffer - get from registry
                if let Some(buffer) = BUFFERS.with(|b| b.borrow().get(&src.id).cloned()) {
                    buffers.push(buffer);
                } else {
                    return Err(Error::BufferNotFound { uop_id: src.id });
                }
            }
            Op::Bind { .. } => {
                // Variable binding - not a buffer
                continue;
            }
            _ => {
                // Unknown source type
                return Err(Error::Runtime { message: format!("Unexpected kernel source: {:?}", src.op()) });
            }
        }
    }

    Ok(buffers)
}

/// Compute buffer size from kernel AST by finding STORE operations.
///
/// This walks the kernel AST to find STORE operations that write to the
/// given buffer definition, then extracts the size from the buffer's shape.
fn compute_buffer_size(ast: &Rc<UOp>, buffer_def: &Rc<UOp>) -> Result<usize> {
    use crate::error::*;
    use morok_ir::shape;

    // Find STORE operations that write to this buffer
    for node in ast.toposort() {
        if let Op::Store { buffer, .. } = node.op() {
            if Rc::ptr_eq(buffer, buffer_def) {
                // Get shape from buffer
                if let Ok(Some(shape_vec)) = buffer.shape() {
                    // Convert to static dimensions if possible
                    if let Some(static_shape) = shape::to_static(shape_vec) {
                        // Compute product of all dimensions
                        let numel: usize = static_shape.iter().product();
                        return Ok(numel);
                    } else {
                        return Err(Error::SymbolicShapeUnsupported {
                            operation: "buffer size computation".to_string(),
                        });
                    }
                }
            }
        }
    }

    // Couldn't determine size - this might be OK for local memory
    // which gets sized based on workgroup dimensions
    // For now, default to 1 element
    Err(Error::Runtime { message: "Could not determine buffer size from kernel AST".to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_ir::{DType, DeviceSpec};
    use smallvec::SmallVec;

    #[test]
    fn test_schedule_item_creation() {
        use morok_ir::ConstValue;

        // Create a simple kernel for testing
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, 10, DType::Float32);
        let index = UOp::const_(DType::Int32, ConstValue::Int(0));
        let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let store = UOp::store(buffer.clone(), index, value);
        let sink = UOp::sink(vec![store]);

        let mut sources = SmallVec::new();
        sources.push(buffer);
        let kernel = UOp::kernel(sources, sink);

        // ScheduleItem should be creatable
        let item = ScheduleItem { ast: kernel.clone(), buffers: vec![] };

        assert!(matches!(item.ast.op(), Op::Kernel { .. }));
    }
}
