//! Two-stage reduction splitting for large tensors.
//!
//! This module implements the `split_reduceop` optimization, which transforms
//! large single-stage reductions into two-stage reductions for better parallelism
//! and memory locality.
//!
//! ## Algorithm Overview
//!
//! For reductions where `prod(input_shape) / prod(output_shape) >= threshold`:
//!
//! 1. **Detect expanded dimensions** - Find which dimensions are broadcast (expanded)
//! 2. **Find split candidates** - Search for valid (dimension, divisor) pairs
//! 3. **Apply transformation** - Split into two stages:
//!    - Reshape input to split one reduction dimension
//!    - Permute to move split dim to end (memory locality)
//!    - Reduce original axes (first stage - parallel)
//!    - Add CONTIGUOUS barrier (materialize intermediate)
//!    - Reduce the split dimension (second stage)
//!    - Reshape back to original output shape
//!
//! ## Example Transformation
//!
//! ```ignore
//! // Before: sum((1_000_000,)) - poor parallelism, 1 kernel
//! REDUCE_AXIS(tensor, op=ADD, axes=[0])
//!
//! // After: reshape + two-stage reduce - 8000 parallel kernels
//! tensor
//!   .reshape([125, 8000])
//!   .permute([1, 0])
//!   .reduce_axis(ADD, [1])    // 8000 parallel reductions
//!   .contiguous()             // Materialize intermediate [8000]
//!   .reduce_axis(ADD, [0])    // Final reduction
//!   .reshape([])              // Back to scalar
//! ```
//!
//! ## Performance Model
//!
//! - **Threshold:** `prod(input_shape) / prod(output_shape) >= 32768`
//! - **Output Cap:** `prod(output_shape) * divisor <= 2^22` (~4M elements)
//! - **Divisor Range:** 8 to 256
//! - **Speedup:** 2-10x for large tensor reductions (>32K elements)
//!
//! Based on Tinygrad's split_reduceop (tinygrad/schedule/rangeify.py:37-58).

use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use morok_dtype::DType;
use morok_ir::{ConstValue, Op, SInt, UOp, UOpKey};
use smallvec::SmallVec;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for split_reduceop optimization.
///
/// Controls when and how large reductions are split into two-stage operations
/// for better parallelism and memory locality.
///
/// # Examples
///
/// ```ignore
/// // Use default configuration
/// let config = SplitReduceOpConfig::default();
///
/// // Custom configuration
/// let config = SplitReduceOpConfig {
///     split_threshold: 65536,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitReduceOpConfig {
    /// Minimum reduction size threshold to trigger splitting.
    ///
    /// Only reductions where `prod(input_shape) / prod(output_shape) >= threshold`
    /// will be split. This ensures overhead of two kernels is justified.
    ///
    /// Default: 32768 (from Tinygrad REDUCEOP_SPLIT_THRESHOLD)
    /// Rationale: Below this, single-kernel overhead < two-kernel overhead
    pub split_threshold: usize,

    /// Maximum output buffer size (2^N elements) for intermediate results.
    ///
    /// Caps the size of the intermediate buffer created by the first reduction stage.
    /// Prevents excessive memory usage while maintaining parallelism.
    ///
    /// Default: 22 (4,194,304 elements = ~16MB for float32)
    /// Rationale: Achieves max occupancy with enough locals+upcasts for GEMM
    ///            ~2^10 should be enough if GROUP is used
    pub output_size_bits: u32,

    /// Maximum split divisor (how many chunks to split into).
    ///
    /// Limits the divisor used to split reduction dimensions.
    /// Higher values = more parallelism but more overhead.
    ///
    /// Default: 256
    /// Rationale: "negligible reduce" for low prod(reduce.shape)
    pub max_divisor: usize,

    /// Minimum split divisor (lower bound on chunking).
    ///
    /// Ensures meaningful parallelism from splitting.
    ///
    /// Default: 8
    /// Rationale: Minimum benefit threshold
    pub min_divisor: usize,

    /// Enable/disable the optimization.
    ///
    /// Default: true
    pub enabled: bool,
}

impl Default for SplitReduceOpConfig {
    fn default() -> Self {
        Self { split_threshold: 32768, output_size_bits: 22, max_divisor: 256, min_divisor: 8, enabled: true }
    }
}

impl SplitReduceOpConfig {
    /// Calculate maximum output buffer size in elements.
    pub fn max_output_size(&self) -> usize {
        2_usize.pow(self.output_size_bits)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract all RANGE axis IDs from an indexed UOp.
///
/// This walks the UOp tree and collects the `axis_id` field from all RANGE
/// operations. Used to determine which input dimensions are actually accessed
/// (vs. dimensions that are broadcast/expanded).
///
/// # Algorithm
/// 1. Perform topological sort of the UOp graph
/// 2. Filter for RANGE operations
/// 3. Extract axis_id from each RANGE
/// 4. Return as sorted Vec for stable comparison
///
/// # Arguments
/// - `indexed`: The UOp after indexing (result of `base.index(ranges)`)
///
/// # Returns
/// Sorted vector of axis IDs that appear in RANGE operations
///
/// # Examples
/// ```ignore
/// let range0 = UOp::range_const(10, 0);  // axis_id = 0
/// let range1 = UOp::range_const(5, 1);   // axis_id = 1
/// let indexed = buffer.index(vec![range0, range1]).unwrap();
/// let range_ids = collect_range_ids(&indexed);
/// assert_eq!(range_ids, vec![0, 1]);
/// ```
pub fn collect_range_ids(indexed: &Rc<UOp>) -> Vec<usize> {
    let mut range_ids: Vec<usize> = indexed
        .toposort()
        .into_iter()
        .filter_map(|node| if let Op::Range { axis_id, .. } = node.op() { Some(*axis_id) } else { None })
        .collect();

    // Sort for stable comparison
    range_ids.sort_unstable();
    range_ids.dedup();
    range_ids
}

// ============================================================================
// Data Structures
// ============================================================================

/// A candidate dimension and divisor for splitting.
#[derive(Debug, Clone)]
struct SplitCandidate {
    /// Index of dimension to split in original shape.
    dimension: usize,

    /// Divisor to use for splitting (dimension % divisor == 0).
    divisor: usize,

    /// Output buffer size after this split (for cap validation).
    #[allow(dead_code)]
    output_size: usize,
}

// ============================================================================
// Core Algorithm Functions
// ============================================================================

/// Detect which dimensions are expanded (broadcast).
///
/// A dimension is "expanded" if it's broadcast from size 1 to a larger size.
/// These dimensions don't actually consume memory and cannot be split.
///
/// # Algorithm
///
/// 1. Create INDEX with fresh RANGE for each dimension
/// 2. Substitute source buffer with NOOP (breaks dependency cycles)
/// 3. Apply movement patterns to transform movement ops into index arithmetic
/// 4. Collect range IDs that appear in the transformed expression
/// 5. Dimensions whose ranges disappeared are expanded
///
/// # Returns
///
/// Vec<bool> where true = dimension is expanded (can't split)
fn detect_expanded_dimensions(source: &Rc<UOp>, input_shape: &[SInt]) -> Vec<bool> {
    // Step 1: Create fresh RANGEs for each dimension
    let ranges: Vec<Rc<UOp>> = input_shape
        .iter()
        .enumerate()
        .map(|(axis_id, dim)| {
            match dim {
                SInt::Const(n) if *n > 1 => {
                    // Create RANGE(0, n) with unique axis_id
                    let end = UOp::const_(DType::Index, ConstValue::Int(*n as i64));
                    UOp::range_axis(end, axis_id, morok_ir::AxisType::Loop)
                }
                _ => {
                    // Size 0 or 1: use constant 0 (no iteration needed)
                    UOp::const_(DType::Index, ConstValue::Int(0))
                }
            }
        })
        .collect();

    // Step 2: Create INDEX operation
    let indexed = match UOp::index(Rc::clone(source), ranges) {
        Ok(idx) => idx,
        Err(_) => {
            // If indexing fails, assume all dimensions are not expanded
            return vec![false; input_shape.len()];
        }
    };

    // Step 3: Substitute source base with NOOP to break cycles
    let base = source.base();
    let noop = UOp::noop();
    #[allow(clippy::mutable_key_type)] // UOpKey contains Rc with interior mutability
    let mut substitutions = HashMap::new();
    substitutions.insert(UOpKey(base), noop);

    // Apply substitution
    let substituted = indexed.substitute(&substitutions);

    // Step 4: Apply movement patterns to transform movement ops
    // This pushes movement operations (RESHAPE, EXPAND, etc.) through INDEX,
    // converting them to index arithmetic. Critical for correct expanded detection.
    use crate::rangeify::movement_patterns::movement_op_patterns;
    use crate::rewrite::graph_rewrite;

    let pm_mops = movement_op_patterns();
    let transformed = graph_rewrite(&pm_mops, substituted);

    // Step 5: Collect range IDs that appear in transformed graph
    let surviving_range_ids = collect_range_ids(&transformed);
    let surviving_set: HashSet<usize> = surviving_range_ids.into_iter().collect();

    // Step 6: Mark dimensions as expanded if their range disappeared
    input_shape.iter().enumerate().map(|(axis_id, _)| !surviving_set.contains(&axis_id)).collect()
}

/// Find valid split candidates ranked by preference.
///
/// # Algorithm
///
/// 1. For each reduce axis:
///    - Check if dimension is expanded (skip if yes)
///    - For divisors in range [max_divisor, min_divisor] (descending):
///      - Check if dimension size is divisible by divisor
///      - Calculate output buffer size
///      - Check if size is within cap
///      - Add to candidates if valid
///
/// # Returns
///
/// Vec of SplitCandidate, sorted by preference (larger divisor = better parallelism).
fn find_split_candidates(
    reduce: &Rc<UOp>,
    input_shape: &[SInt],
    is_expanded: &[bool],
    config: &SplitReduceOpConfig,
) -> Vec<SplitCandidate> {
    let Op::ReduceAxis { axes: reduce_axes, .. } = reduce.op() else {
        return vec![];
    };

    // Get output shape (after reduction)
    let output_shape = match reduce.shape() {
        Ok(Some(shape)) => shape,
        _ => return vec![],
    };

    let output_size: usize = output_shape.iter().filter_map(|s| s.as_const()).product();

    let mut candidates = Vec::new();

    // Iterate reduce axes
    for &axis in reduce_axes {
        // Skip if dimension is expanded (can't split broadcast dims)
        if axis >= is_expanded.len() || is_expanded[axis] {
            continue;
        }

        // Get dimension size (only handle constant sizes)
        let dim_size = match &input_shape[axis] {
            SInt::Const(n) => *n,
            SInt::Symbolic(_) => continue, // Can't split symbolic dimensions
        };

        // Try divisors from large to small (prefer more parallelism)
        for divisor in (config.min_divisor..=config.max_divisor).rev() {
            // Check divisibility
            if dim_size % divisor != 0 {
                continue;
            }

            // Calculate output buffer size after split
            // output_size increases by `divisor` (one dim splits into two)
            let new_output_size = output_size * divisor;

            // Check against cap
            if new_output_size > config.max_output_size() {
                continue;
            }

            // Valid candidate!
            candidates.push(SplitCandidate { dimension: axis, divisor, output_size: new_output_size });
        }
    }

    candidates
}

/// Apply the two-stage reduction transformation.
///
/// # Transformation Steps
///
/// 1. **Reshape:** Split dimension into (divisor, remainder)
/// 2. **Permute:** Move split dimension to end for memory locality
/// 3. **First Reduce:** Reduce original axes (now shifted due to reshape)
/// 4. **Contiguous:** Materialize intermediate results
/// 5. **Second Reduce:** Reduce the split dimension
/// 6. **Reshape:** Back to original output shape
fn apply_split_transformation(
    source: &Rc<UOp>,
    reduce: &Rc<UOp>,
    candidate: &SplitCandidate,
    input_shape: &[SInt],
) -> Option<Rc<UOp>> {
    let Op::ReduceAxis { reduce_op, axes: reduce_axes, .. } = reduce.op() else {
        return None;
    };

    let dim_to_split = candidate.dimension;
    let divisor = candidate.divisor;
    let dim_size = input_shape[dim_to_split].as_const()?;
    let remainder = dim_size / divisor;

    // Step 1: Compute new shape after split
    // [d0, ..., di, ..., dn] -> [d0, ..., divisor, remainder, ..., dn]
    let mut splitted_shape: SmallVec<[SInt; 4]> = SmallVec::new();
    for (i, dim) in input_shape.iter().enumerate() {
        if i == dim_to_split {
            // Split this dimension
            splitted_shape.push(SInt::Const(divisor));
            splitted_shape.push(SInt::Const(remainder));
        } else {
            splitted_shape.push(dim.clone());
        }
    }

    // Step 2: Reshape to splitted shape
    let reshaped = source.try_reshape(&splitted_shape).ok()?;

    // Step 3: Generate permutation to move split dimension to end
    // After split, splitted_shape has one more dimension
    // We want to move dim_to_split to the end
    let mut permutation: Vec<usize> = (0..splitted_shape.len()).filter(|&i| i != dim_to_split).collect();
    permutation.push(dim_to_split);

    // Step 4: Apply permutation
    let permuted = reshaped.try_permute(permutation.clone()).ok()?;

    // Step 5: First stage reduction
    // Map reduce axes through the split transformation
    // Axes before dim_to_split stay the same
    // dim_to_split becomes (dim_to_split, dim_to_split+1) - we reduce the second part
    // Axes after dim_to_split shift by +1
    let adjusted_axes: Vec<usize> = reduce_axes
        .iter()
        .map(|&axis| {
            if axis < dim_to_split {
                axis
            } else if axis == dim_to_split {
                // Reduce the remainder part (shifted by +1 due to split)
                dim_to_split + 1
            } else {
                // Axes after split shift by +1
                axis + 1
            }
        })
        .collect();

    // Map axes through permutation
    let permuted_axes: Vec<usize> = adjusted_axes
        .iter()
        .map(|&old_axis| {
            // Find where old_axis ended up after permutation
            permutation.iter().position(|&p| p == old_axis).unwrap()
        })
        .collect();

    let first_reduce = permuted.try_reduce_axis(*reduce_op, permuted_axes).ok()?;

    // Step 6: Materialize with CONTIGUOUS
    let contiguous = UOp::contiguous(first_reduce);

    // Step 7: Second stage reduction
    // Reduce the split dimension (now at end, axis = output.ndim - 1)
    let output_shape = contiguous.shape().ok()??;
    let split_axis = output_shape.len() - 1;

    let second_reduce = contiguous.try_reduce_axis(*reduce_op, vec![split_axis]).ok()?;

    // Step 8: Reshape back to original output shape
    let final_shape = reduce.shape().ok()??;

    let final_result = second_reduce.try_reshape(final_shape).ok()?;

    Some(final_result)
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Split large REDUCE_AXIS operations into two-stage reductions.
///
/// # Algorithm
/// For reductions where `prod(input_shape) / prod(output_shape) >= threshold`:
/// 1. Detect expanded dimensions (via rangeify + range ID collection)
/// 2. Find best split candidate (divisor that fits output cap)
/// 3. Apply transformation:
///    - Reshape input to split one reduction dimension
///    - Permute to move split dim to end
///    - Reduce original axes
///    - Add CONTIGUOUS barrier
///    - Reduce the split dimension
///    - Reshape back to original output shape
///
/// # Arguments
/// - `reduce`: The REDUCE_AXIS operation to potentially split
/// - `config`: Configuration for split thresholds and caps
///
/// # Returns
/// - `Some(UOp)`: Transformed two-stage reduction
/// - `None`: No transformation (doesn't meet criteria or not beneficial)
///
/// # Example Transformation
/// ```ignore
/// // Before: sum((1000000,)) - poor parallelism
/// REDUCE_AXIS(src: (1000000,), op: ADD, axes: [0])
///
/// // After: reshape(1000, 1000).sum(1).sum(0) - 1000 parallel kernels
/// let reshaped = src.reshape((1000, 1000));
/// let permuted = reshaped.permute((1, 0));
/// let first_reduce = permuted.reduce_axis(ADD, [0]);
/// let contiguous = first_reduce.contiguous();
/// let second_reduce = contiguous.reduce_axis(ADD, [1]);
/// second_reduce.reshape(reduce.shape())
/// ```
pub fn split_reduceop(reduce: &Rc<UOp>, config: &SplitReduceOpConfig) -> Option<Rc<UOp>> {
    // Check if enabled
    if !config.enabled {
        return None;
    }

    // Must be REDUCE_AXIS
    let Op::ReduceAxis { src: source, .. } = reduce.op() else {
        return None;
    };

    // Get shapes
    let input_shape = source.shape().ok()??;
    let output_shape = reduce.shape().ok()??;

    // Only handle constant (non-symbolic) shapes
    // Check early to avoid unnecessary computation
    if !input_shape.iter().all(|s| s.is_const()) {
        return None;
    }

    // Calculate reduction ratio (safe to unwrap after const check)
    let input_size: usize = input_shape.iter().map(|s| s.as_const().unwrap()).product();
    let output_size: usize = output_shape.iter().map(|s| s.as_const().unwrap()).product();

    // Early exit: empty reduction
    if output_size == 0 {
        return None;
    }

    // Check threshold
    let ratio = input_size / output_size;
    if ratio < config.split_threshold {
        return None;
    }

    // Detect expanded dimensions
    let is_expanded = detect_expanded_dimensions(source, input_shape);

    // Find split candidates
    let candidates = find_split_candidates(reduce, input_shape, &is_expanded, config);

    // No valid candidates
    if candidates.is_empty() {
        return None;
    }

    // Apply transformation with best candidate (first in list)
    let best_candidate = &candidates[0];
    apply_split_transformation(source, reduce, best_candidate, input_shape)
}
