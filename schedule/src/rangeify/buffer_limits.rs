//! Buffer limit enforcement for device-specific constraints.
//!
//! This module implements **active buffer limit enforcement** following Tinygrad's approach:
//! when a computation accesses more buffers than the device supports, force bufferization
//! of elementwise sources to reduce buffer count.
//!
//! # Device Limits
//!
//! Different hardware has different buffer/argument limits per kernel:
//! - **Metal**: 31 buffers (Apple Silicon hardware limit)
//! - **WebGPU**: 8 buffers (WebGPU specification limit)
//! - **CPU/CUDA**: No practical limit
//!
//! # Algorithm
//!
//! For each binary/ternary operation in the graph:
//! 1. Count accessed buffers (using `collect_accessed_buffers`)
//! 2. If count > max_buffers - 1 (accounting for output buffer):
//!    - Force bufferization of elementwise sources
//!    - This materializes intermediate results, reducing buffer count
//! 3. Otherwise: no transformation
//!
//! # Integration
//!
//! This runs at **Step 8.5** in the rangeify pipeline:
//! - After symbolic simplification
//! - Before bufferize_to_store
//!
//! This ensures buffer limits are enforced before kernel splitting.
//!
//! # Based on Tinygrad
//!
//! - File: `tinygrad/schedule/rangeify.py`
//! - Function: `limit_bufs()` (lines 268-290)
//! - Commit: 2c397eb2a (2025-09-30) - "rangeify implements input buffer limiting"

use std::collections::HashSet;
use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_ir::{AddrSpace, BufferizeOpts, Op, UOp, UOpKey};

use super::buffer_cost::collect_accessed_buffers;
use crate::pattern::UPat;
use crate::pattern::matcher::{PatternMatcher, RewriteResult};

/// Extract device specification from a UOp graph.
///
/// Walks the graph recursively looking for:
/// - Op::Device operations
/// - Op::Buffer operations (which contain a device)
///
/// Returns the first device found, or None if no device is present in the graph.
///
/// # Example
///
/// ```ignore
/// let device = extract_device_from_graph(&computation)?;
/// if let Some(limit) = device.max_buffers() {
///     // Enforce buffer limit
/// }
/// ```
#[allow(clippy::mutable_key_type)]
pub fn extract_device_from_graph(root: &Rc<UOp>) -> Option<DeviceSpec> {
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, visited: &mut HashSet<UOpKey>) -> Option<DeviceSpec> {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return None; // Already visited
        }

        match uop.op() {
            Op::Device(spec) => return Some(spec.clone()),
            Op::Buffer { device, .. } => {
                // Device is a UOp that should be Op::Device
                if let Op::Device(spec) = device.op() {
                    return Some(spec.clone());
                }
            }
            Op::Bufferize { opts, .. } => {
                // BufferizeOpts may contain optional device
                if let Some(device_spec) = &opts.device {
                    return Some(device_spec.clone());
                }
            }
            _ => {}
        }

        // Recursively visit children
        for child in uop.op().sources() {
            if let Some(device) = visit(&child, visited) {
                return Some(device);
            }
        }

        None
    }

    visit(root, &mut visited)
}

/// Check if a UOp is an elementwise operation.
///
/// Elementwise operations are those that can be computed independently for each element:
/// - Binary operations (ADD, MUL, SUB, DIV, etc.)
/// - Ternary operations (WHERE, etc.)
///
/// These are candidates for forced bufferization when buffer limits are exceeded.
///
/// # Returns
///
/// - `true` if the operation is Binary or Ternary
/// - `false` otherwise
///
/// # Example
///
/// ```ignore
/// if is_elementwise(&src) {
///     // Can force bufferize this source
/// }
/// ```
pub fn is_elementwise(uop: &Rc<UOp>) -> bool {
    matches!(uop.op(), Op::Binary(..) | Op::Ternary(..))
}

/// Create pattern matchers for buffer limit enforcement.
///
/// Returns a PatternMatcher that:
/// 1. Matches binary and ternary operations
/// 2. Counts accessed buffers in their sources
/// 3. If count > max_buffers - 1, forces bufferization of elementwise sources
///
/// The -1 accounts for the output buffer of the operation.
///
/// # Arguments
///
/// * `max_buffers` - Maximum buffer count allowed by the device
///
/// # Returns
///
/// A PatternMatcher that enforces buffer limits by forcing bufferization.
///
/// # Example
///
/// ```ignore
/// let limit = device.max_buffers().unwrap();
/// let matcher = buffer_limit_patterns(limit);
/// let result = graph_rewrite(&matcher, computation);
/// ```
///
/// # Based on Tinygrad
///
/// - Tinygrad's `limit_bufs()` function (rangeify.py:281-289)
/// - Counts buffers, forces bufferization when exceeded
pub fn buffer_limit_patterns(max_buffers: usize) -> PatternMatcher {
    use crate::pattern::matcher::RewriteFn;
    use std::collections::HashMap;

    let mut patterns = vec![];

    // Pattern: Binary/Ternary operations with too many buffers → Force bufferize elementwise sources
    let limit = max_buffers; // Copy for closure capture
    patterns.push((
        UPat::var("op"),
        Box::new(move |bindings: &HashMap<String, Rc<UOp>>| {
            let Some(op) = bindings.get("op") else {
                return RewriteResult::NoMatch;
            };

            // Only process Binary and Ternary operations (elementwise)
            let sources = match op.op() {
                Op::Binary(_, left, right) => vec![left.clone(), right.clone()],
                Op::Ternary(_, cond, true_val, false_val) => {
                    vec![cond.clone(), true_val.clone(), false_val.clone()]
                }
                _ => return RewriteResult::NoMatch,
            };

            // Count buffers accessed by all sources
            let mut all_buffers = Vec::new();
            for src in &sources {
                all_buffers.extend(collect_accessed_buffers(src));
            }

            // Deduplicate
            #[allow(clippy::mutable_key_type)]
            let mut seen = HashSet::new();
            all_buffers.retain(|b| seen.insert(UOpKey(Rc::clone(b))));

            // Check if exceeds limit (-1 for output buffer)
            if all_buffers.len() > limit.saturating_sub(1) {
                // Force bufferize elementwise sources
                let mut new_sources = Vec::new();
                let mut any_changed = false;

                for src in &sources {
                    let new_src = if is_elementwise(src) { force_bufferize(src) } else { Rc::clone(src) };

                    if !Rc::ptr_eq(&new_src, src) {
                        any_changed = true;
                    }
                    new_sources.push(new_src);
                }

                // If any source changed, reconstruct the operation
                if any_changed {
                    let dtype = op.dtype();
                    let rewritten = match op.op() {
                        Op::Binary(bin_op, _, _) => {
                            UOp::new(Op::Binary(*bin_op, new_sources[0].clone(), new_sources[1].clone()), dtype)
                        }
                        Op::Ternary(tern_op, _, _, _) => UOp::new(
                            Op::Ternary(
                                *tern_op,
                                new_sources[0].clone(),
                                new_sources[1].clone(),
                                new_sources[2].clone(),
                            ),
                            dtype,
                        ),
                        _ => unreachable!(),
                    };
                    return RewriteResult::Rewritten(rewritten);
                }
            }

            RewriteResult::NoMatch
        }) as RewriteFn,
    ));

    PatternMatcher::new(patterns)
}

/// Force bufferization of a computation to materialize intermediate result.
///
/// Creates a BUFFERIZE operation that:
/// - Materializes the computation in global memory
/// - Collects all ranges from the source
/// - Uses GLOBAL address space
///
/// This reduces buffer count by creating one intermediate buffer instead of
/// accessing multiple input buffers.
///
/// # Arguments
///
/// * `src` - The computation to bufferize
///
/// # Returns
///
/// A BUFFERIZE(src, ranges, GLOBAL) operation wrapped in INDEX to make it usable.
///
/// # Example
///
/// ```ignore
/// // Before: ADD accesses buf1 and buf2 (2 buffers)
/// let add = UOp::add(buf1_access, buf2_access);
///
/// // After: ADD result materialized (1 buffer)
/// let materialized = force_bufferize(add);
/// // Now: INDEX(BUFFERIZE(ADD(...)))
/// ```
///
/// # Based on Tinygrad
///
/// - Tinygrad's forced bufferization (rangeify.py:285-287)
/// - Uses `substitute` to change range types, then bufferize + index
fn force_bufferize(src: &Rc<UOp>) -> Rc<UOp> {
    // Collect all ranges from the source computation
    let ranges = collect_ranges(src);

    if ranges.is_empty() {
        // No ranges to bufferize, return original
        return Rc::clone(src);
    }

    // Create BUFFERIZE with GLOBAL address space
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let bufferized = UOp::bufferize(Rc::clone(src), ranges.clone(), opts);

    // Wrap in INDEX to make it usable
    UOp::index(bufferized, ranges).unwrap_or_else(|_| Rc::clone(src))
}

/// Collect all RANGE operations from a computation tree.
///
/// Recursively walks the UOp graph and collects all Op::Range operations.
/// These ranges are used when forcing bufferization.
#[allow(clippy::mutable_key_type)]
fn collect_ranges(src: &Rc<UOp>) -> Vec<Rc<UOp>> {
    let mut ranges = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, ranges: &mut Vec<Rc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return; // Already visited
        }

        if matches!(uop.op(), Op::Range { .. }) {
            ranges.push(Rc::clone(uop));
        }

        for child in uop.op().sources() {
            visit(&child, ranges, visited);
        }
    }

    visit(src, &mut ranges, &mut visited);

    // Deduplicate while preserving order
    let mut seen = HashSet::new();
    ranges.retain(|r| seen.insert(UOpKey(Rc::clone(r))));

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_elementwise() {
        use morok_dtype::DType;
        use morok_ir::ConstValue;

        // Binary operations are elementwise
        let left = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let right = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let add = left.try_add_op(&right).unwrap();
        assert!(is_elementwise(&add), "Binary ADD should be elementwise");

        // Ternary operations are elementwise
        let cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
        let true_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let false_val = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let where_op = UOp::where_op(cond, true_val, false_val).unwrap();
        assert!(is_elementwise(&where_op), "Ternary WHERE should be elementwise");

        // Constants are not elementwise
        let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        assert!(!is_elementwise(&const_op), "CONST should not be elementwise");
    }

    #[test]
    fn test_extract_device_no_device() {
        use morok_dtype::DType;
        use morok_ir::ConstValue;

        // Graph with no device info
        let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        assert_eq!(extract_device_from_graph(&const_op), None, "Should return None when no device");
    }

    #[test]
    fn test_extract_device_from_device_op() {
        use morok_device::DeviceSpec;

        // Graph with Op::Device
        let device_op = UOp::device(DeviceSpec::Cpu);
        assert_eq!(extract_device_from_graph(&device_op), Some(DeviceSpec::Cpu), "Should extract CPU device");
    }
}
