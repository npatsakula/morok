//! Cost-based buffer optimization with partial contiguous support.
//!
//! This module implements Tinygrad's partial contiguous (PCONTIG) optimization,
//! which selectively materializes buffer dimensions based on cost heuristics.
//!
//! # Algorithm
//!
//! The optimization operates in three stages:
//!
//! 1. **Cost Analysis**: Evaluate whether buffering is beneficial
//!    - `accessed_buffers`: Count unique buffers accessed
//!    - `out_in_ratio`: Compare output size to input sizes
//!    - `buffer_in_reduce`: Check if buffer is reused in reductions
//!
//! 2. **Range Partitioning**: Classify ranges as "materialize" or "substitute"
//!    - Materialize: LOCAL index ranges, REDUCE axis ranges
//!    - Substitute: All other ranges (inlined)
//!
//! 3. **Transformation**: Apply partial or full buffer removal
//!    - Partial: BUFFERIZE only materialized ranges, inline rest
//!    - Full: Remove BUFFERIZE entirely, inline all ranges
//!
//! # Cost Heuristics
//!
//! | Heuristic | Threshold | Action |
//! |-----------|-----------|--------|
//! | `accessed_buffers > 3` | Keep buffer | Complex multi-input |
//! | `out_in_ratio < 10` | Keep buffer | Efficient memory usage |
//! | `buffer_in_reduce` | Partial contiguous | Buffer reused in reduce |

use std::collections::HashSet;
use std::rc::Rc;

use morok_ir::{AddrSpace, Op, UOp, UOpKey};

/// Configuration for partial contiguous optimization.
#[derive(Debug, Clone, Copy)]
pub struct PcontigConfig {
    /// Enable partial contiguous optimization.
    ///
    /// - 0 = disabled
    /// - 1 = basic buffer removal
    /// - 2 = enabled (default)
    /// - 3+ = aggressive (force even with >3 buffers)
    pub level: u8,

    /// Maximum accessed buffers before keeping BUFFERIZE.
    ///
    /// Default: 3 (1-3 buffers = simple ops, 4+ = complex multi-input)
    pub max_buffers_threshold: usize,

    /// Maximum out_in_ratio before applying partial contiguous.
    ///
    /// Default: 10.0 (ratio < 10 = efficient buffer, >= 10 = wasteful)
    pub out_in_ratio_threshold: f64,
}

impl Default for PcontigConfig {
    fn default() -> Self {
        Self { level: 2, max_buffers_threshold: 3, out_in_ratio_threshold: 10.0 }
    }
}

/// Collect all accessed buffers in a computation tree.
///
/// Traverses the UOp tree and collects:
/// - BUFFER operations
/// - BUFFERIZE(GLOBAL) operations (stops traversal at boundaries)
/// - MSTACK, MSELECT operations
///
/// # Example
///
/// ```ignore
/// let buffers = collect_accessed_buffers(&computation);
/// if buffers.len() > 3 {
///     // Complex multi-input operation, keep buffer
/// }
/// ```
#[allow(clippy::mutable_key_type)]
pub fn collect_accessed_buffers(src: &Rc<UOp>) -> Vec<Rc<UOp>> {
    let mut buffers = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, buffers: &mut Vec<Rc<UOp>>, visited: &mut HashSet<UOpKey>) -> bool {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return true; // Already visited
        }

        match uop.op() {
            Op::Bufferize { opts, .. } if opts.addrspace == AddrSpace::Global => {
                buffers.push(Rc::clone(uop));
                return false; // Stop traversal - treat as atomic
            }
            Op::Buffer { .. } | Op::MStack { .. } | Op::MSelect { .. } => {
                buffers.push(Rc::clone(uop));
            }
            _ => {}
        }

        // Continue traversal
        for child in uop.op().sources() {
            visit(&child, buffers, visited);
        }

        true
    }

    visit(src, &mut buffers, &mut visited);

    // Deduplicate while preserving order
    let mut seen = HashSet::new();
    buffers.retain(|b| seen.insert(UOpKey(Rc::clone(b))));

    buffers
}

/// Collect all REDUCE operations in a computation tree.
///
/// # Example
///
/// ```ignore
/// let reduces = collect_reduces(&computation);
/// if !reduces.is_empty() {
///     // Check if buffer is accessed in reduce scope
/// }
/// ```
#[allow(clippy::mutable_key_type)]
pub fn collect_reduces(src: &Rc<UOp>) -> Vec<Rc<UOp>> {
    let mut reduces = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, reduces: &mut Vec<Rc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return;
        }

        if matches!(uop.op(), Op::Reduce { .. }) {
            reduces.push(Rc::clone(uop));
        }

        for child in uop.op().sources() {
            visit(&child, reduces, visited);
        }
    }

    visit(src, &mut reduces, &mut visited);
    reduces
}

/// Collect all INDEX operations in a computation tree.
///
/// # Example
///
/// ```ignore
/// let indexes = collect_indexes(&computation);
/// let local_indexes = collect_local_indexes(&indexes);
/// ```
#[allow(clippy::mutable_key_type)]
pub fn collect_indexes(src: &Rc<UOp>) -> Vec<Rc<UOp>> {
    let mut indexes = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, indexes: &mut Vec<Rc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return;
        }

        if matches!(uop.op(), Op::Index { .. }) {
            indexes.push(Rc::clone(uop));
        }

        for child in uop.op().sources() {
            visit(&child, indexes, visited);
        }
    }

    visit(src, &mut indexes, &mut visited);
    indexes
}

/// Calculate the size of a buffer in elements.
///
/// Returns `None` for symbolic shapes (can't compute product).
///
/// # Example
///
/// ```ignore
/// let size = calculate_buffer_size(&bufferize);
/// if let Some(n) = size {
///     println!("Buffer size: {} elements", n);
/// }
/// ```
pub fn calculate_buffer_size(buffer: &Rc<UOp>) -> Option<usize> {
    use morok_ir::ConstValue;

    match buffer.op() {
        Op::Bufferize { ranges, .. } => {
            // Product of all range sizes (element count)
            let mut product = 1usize;
            for range in ranges {
                match range.op() {
                    Op::Range { end, .. } => {
                        // Range size is the end value
                        match end.op() {
                            Op::Const(cv) => match cv.0 {
                                ConstValue::Int(n) if n > 0 => {
                                    product = product.checked_mul(n as usize)?;
                                }
                                _ => return None,
                            },
                            _ => return None, // Symbolic range
                        }
                    }
                    _ => return None,
                }
            }
            // Convert elements to bytes by multiplying by dtype size
            let element_size = buffer.dtype().bytes();
            Some(product.checked_mul(element_size)?)
        }
        Op::Buffer { size, .. } => {
            // Buffer has explicit size in bytes
            Some(*size)
        }
        Op::MStack { .. } | Op::MSelect { .. } => {
            // Estimate: assume size 1 (multi-buffer ops)
            Some(1)
        }
        _ => None,
    }
}

/// Calculate the output-to-input size ratio.
///
/// Returns `None` if any size is symbolic.
///
/// Formula: `(output_size + 1) / (sum(input_sizes) + 1)`
///
/// Interpretation:
/// - `< 10`: Output size comparable to inputs → efficient buffer
/// - `>= 10`: Output much larger than inputs → wasteful buffer
///
/// # Example
///
/// ```ignore
/// if let Some(ratio) = calculate_out_in_ratio(output_size, &input_buffers) {
///     if ratio >= 10.0 {
///         // Apply partial contiguous
///     }
/// }
/// ```
pub fn calculate_out_in_ratio(output_size: usize, input_buffers: &[Rc<UOp>]) -> Option<f64> {
    let mut input_sum = 0usize;

    for buf in input_buffers {
        match calculate_buffer_size(buf) {
            Some(size) => {
                input_sum = input_sum.checked_add(size)?;
            }
            None => return None, // Symbolic size
        }
    }

    // Add 1 to avoid division by zero
    let ratio = (output_size + 1) as f64 / (input_sum + 1) as f64;
    Some(ratio)
}

/// Check if any buffer is accessed within a reduce scope.
///
/// Creates a SINK of all reduce sources, then checks if the subgraph
/// contains any BUFFER or BUFFERIZE operations.
///
/// # Example
///
/// ```ignore
/// let reduces = collect_reduces(&computation);
/// if has_buffer_in_reduce(&reduces) {
///     // Buffer is reused in reduction, worth keeping
/// }
/// ```
#[allow(clippy::mutable_key_type)]
pub fn has_buffer_in_reduce(reduces: &[Rc<UOp>]) -> bool {
    if reduces.is_empty() {
        return false;
    }

    // Collect reduce sources
    let reduce_sources: Vec<Rc<UOp>> = reduces
        .iter()
        .filter_map(|r| if let Op::Reduce { src, .. } = r.op() { Some(Rc::clone(src)) } else { None })
        .collect();

    if reduce_sources.is_empty() {
        return false;
    }

    // Create SINK of all reduce sources
    let sink = UOp::sink(reduce_sources);

    // Traverse sink's subgraph looking for buffers
    let mut visited = HashSet::new();
    let mut found_buffer = false;

    fn visit(uop: &Rc<UOp>, found: &mut bool, visited: &mut HashSet<UOpKey>) -> bool {
        if *found {
            return false; // Early termination
        }

        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return true;
        }

        match uop.op() {
            Op::Buffer { .. } | Op::Bufferize { .. } => {
                *found = true;
                return false; // Early termination
            }
            _ => {}
        }

        for child in uop.op().sources() {
            if !visit(&child, found, visited) {
                return false; // Propagate early termination
            }
        }

        true
    }

    visit(&sink, &mut found_buffer, &mut visited);
    found_buffer
}

/// Filter indexes to only those accessing LOCAL bufferize operations.
///
/// # Example
///
/// ```ignore
/// let indexes = collect_indexes(&computation);
/// let local_indexes = collect_local_indexes(&indexes);
/// let exclude_ranges = extract_exclude_ranges(&local_indexes);
/// ```
pub fn collect_local_indexes(indexes: &[Rc<UOp>]) -> Vec<Rc<UOp>> {
    indexes
        .iter()
        .filter(|idx| {
            matches!(idx.op(), Op::Index { buffer, .. }
                if matches!(buffer.op(), Op::Bufferize { opts, .. }
                    if opts.addrspace == AddrSpace::Local))
        })
        .map(Rc::clone)
        .collect()
}

/// Extract ranges that should be excluded from substitution (kept as bufferize).
///
/// These are ranges from LOCAL INDEX operations - they need materialization
/// for efficient memory access patterns.
///
/// # Example
///
/// ```ignore
/// let local_indexes = collect_local_indexes(&indexes);
/// let exclude_ranges = extract_exclude_ranges(&local_indexes);
/// // exclude_ranges contains UOpKeys for ranges that must be materialized
/// ```
#[allow(clippy::mutable_key_type)]
pub fn extract_exclude_ranges(local_indexes: &[Rc<UOp>]) -> HashSet<UOpKey> {
    let mut exclude = HashSet::new();

    for idx in local_indexes {
        if let Op::Index { indices, .. } = idx.op() {
            // Add all index ranges to exclude set
            for range in indices {
                // Use in_scope_ranges to get all ranges this index depends on
                for r in range.in_scope_ranges() {
                    exclude.insert(r.clone());
                }
            }
        }
    }

    exclude
}

/// Partition ranges into "materialize" (keep buffered) vs "substitute" (inline).
///
/// Materialized ranges:
/// - Ranges in exclude_ranges (LOCAL indexes)
/// - Ranges with AxisType::REDUCE
///
/// Substituted ranges:
/// - All other ranges (will be inlined)
///
/// # Example
///
/// ```ignore
/// let (materialize, substitute) = partition_ranges(
///     &buf_ranges,
///     &idx_ranges,
///     &exclude_ranges,
/// );
/// // materialize: ranges to keep in BUFFERIZE
/// // substitute: ranges to inline
/// ```
#[allow(clippy::type_complexity, clippy::mutable_key_type)]
pub fn partition_ranges(
    buf_ranges: &[Rc<UOp>],
    idx_ranges: &[Rc<UOp>],
    exclude_ranges: &HashSet<UOpKey>,
) -> (Vec<(Rc<UOp>, Rc<UOp>)>, Vec<(Rc<UOp>, Rc<UOp>)>) {
    use morok_ir::AxisType;

    let mut materialize = Vec::new();
    let mut substitute = Vec::new();

    for (buf_rng, idx_rng) in buf_ranges.iter().zip(idx_ranges.iter()) {
        // Skip CONST ranges - they don't need substitution
        if matches!(buf_rng.op(), Op::Const(_)) {
            continue;
        }

        let buf_key = UOpKey(Rc::clone(buf_rng));

        // Check if this range should be materialized
        let should_materialize =
            // In exclude_ranges (LOCAL index)
            exclude_ranges.contains(&buf_key) ||
            // Has REDUCE axis in idx_range
            idx_rng.in_scope_ranges().iter().any(|r| {
                if let Op::Range { axis_type, .. } = r.0.op() {
                    matches!(axis_type, AxisType::Reduce)
                } else {
                    false
                }
            });

        let pair = (Rc::clone(buf_rng), Rc::clone(idx_rng));

        if should_materialize {
            materialize.push(pair);
        } else {
            substitute.push(pair);
        }
    }

    (materialize, substitute)
}

/// Apply the partial contiguous transformation.
///
/// 1. Substitute inlined ranges into src
/// 2. Create BUFFERIZE with only materialized ranges
/// 3. Create INDEX with materialized ranges
///
/// # Example
///
/// ```ignore
/// if let Some(result) = apply_partial_contiguous(&src, materialize, substitute) {
///     // Result is INDEX(BUFFERIZE(substituted_src, mat_ranges), mat_indices)
/// }
/// ```
#[allow(clippy::mutable_key_type)]
pub fn apply_partial_contiguous(
    src: &Rc<UOp>,
    materialize: Vec<(Rc<UOp>, Rc<UOp>)>,
    substitute: Vec<(Rc<UOp>, Rc<UOp>)>,
) -> Option<Rc<UOp>> {
    use std::collections::HashMap;

    // Must have something to substitute
    if substitute.is_empty() {
        return None;
    }

    // Build substitution map (UOpKey -> Rc<UOp>)
    let subs_map: HashMap<UOpKey, Rc<UOp>> = substitute.into_iter().map(|(k, v)| (UOpKey(k), v)).collect();

    // Substitute inlined ranges
    let substituted = src.substitute(&subs_map);

    // If no ranges to materialize, return substituted directly
    if materialize.is_empty() {
        return Some(substituted);
    }

    // Extract materialized ranges
    let (mat_buf_rngs, mat_idx_rngs): (Vec<_>, Vec<_>) = materialize.into_iter().unzip();

    // Create BUFFERIZE with materialized ranges
    use morok_ir::BufferizeOpts;
    let opts = BufferizeOpts::local();

    let bufferized = UOp::bufferize(substituted, mat_buf_rngs, opts);

    // Create INDEX with materialized ranges
    UOp::index(bufferized, mat_idx_rngs).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::{AxisType, BufferizeOpts, ConstValue};
    use smallvec::SmallVec;

    /// Helper: Create a BUFFER operation for testing.
    fn create_test_buffer(size: usize, dtype: DType, id: usize) -> Rc<UOp> {
        let unique = UOp::unique(Some(id));
        let device = UOp::device(morok_device::DeviceSpec::Cpu);
        UOp::new(Op::Buffer { unique, device, size }, dtype)
    }

    #[test]
    fn test_pcontig_config_default() {
        let config = PcontigConfig::default();
        assert_eq!(config.level, 2);
        assert_eq!(config.max_buffers_threshold, 3);
        assert_eq!(config.out_in_ratio_threshold, 10.0);
    }

    #[test]
    fn test_collect_accessed_buffers_empty() {
        // Constant has no buffers
        let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let buffers = collect_accessed_buffers(&const_val);
        assert_eq!(buffers.len(), 0);
    }

    #[test]
    fn test_collect_accessed_buffers_single_buffer() {
        // Single BUFFER operation
        let buffer = create_test_buffer(100, DType::Float32, 0);
        let buffers = collect_accessed_buffers(&buffer);
        assert_eq!(buffers.len(), 1);
        assert!(Rc::ptr_eq(&buffers[0], &buffer));
    }

    #[test]
    fn test_collect_accessed_buffers_multiple() {
        // Two buffers in ADD operation
        let buf1 = create_test_buffer(100, DType::Float32, 0);
        let buf2 = create_test_buffer(100, DType::Float32, 1);
        let add = buf1.try_add_op(&buf2).unwrap();

        let buffers = collect_accessed_buffers(&add);
        assert_eq!(buffers.len(), 2);
    }

    #[test]
    fn test_collect_accessed_buffers_stops_at_global_bufferize() {
        // Create inner buffer
        let inner_buf = create_test_buffer(100, DType::Float32, 0);

        // Wrap in GLOBAL bufferize
        let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
        let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
        let bufferize = UOp::bufferize(inner_buf, vec![range], opts);

        // Should only see the bufferize, not the inner buffer
        let buffers = collect_accessed_buffers(&bufferize);
        assert_eq!(buffers.len(), 1);
        assert!(Rc::ptr_eq(&buffers[0], &bufferize));
    }

    #[test]
    fn test_collect_accessed_buffers_deduplication() {
        // Use same buffer twice (b + b)
        let buf = create_test_buffer(100, DType::Float32, 0);
        let add = buf.try_add_op(&buf).unwrap();

        let buffers = collect_accessed_buffers(&add);
        assert_eq!(buffers.len(), 1); // Deduplicated
    }

    #[test]
    fn test_collect_reduces_empty() {
        // Constant has no reduces
        let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let reduces = collect_reduces(&const_val);
        assert_eq!(reduces.len(), 0);
    }

    #[test]
    fn test_collect_reduces_single() {
        // Create simple reduce
        let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Reduce);

        use morok_ir::ReduceOp;
        let reduce = UOp::reduce(const_val, SmallVec::from_iter([range]), ReduceOp::Add);

        let reduces = collect_reduces(&reduce);
        assert_eq!(reduces.len(), 1);
        assert!(Rc::ptr_eq(&reduces[0], &reduce));
    }

    #[test]
    fn test_collect_indexes_empty() {
        // Constant has no indexes
        let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let indexes = collect_indexes(&const_val);
        assert_eq!(indexes.len(), 0);
    }

    #[test]
    fn test_collect_indexes_single() {
        // Create INDEX operation
        let buffer = create_test_buffer(100, DType::Float32, 0);
        let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);

        let index = UOp::index(buffer, vec![range]).unwrap();

        let indexes = collect_indexes(&index);
        assert_eq!(indexes.len(), 1);
        assert!(Rc::ptr_eq(&indexes[0], &index));
    }

    #[test]
    fn test_calculate_buffer_size_concrete_ranges() {
        // BUFFERIZE with concrete ranges [10, 20, 30] → 10*20*30=6000 elements → 6000*4=24000 bytes (Float32)
        let range1 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
        let range2 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(20)), 1, AxisType::Loop);
        let range3 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(30)), 2, AxisType::Loop);

        let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
        let bufferize = UOp::bufferize(compute, vec![range1, range2, range3], opts);

        let size = calculate_buffer_size(&bufferize);
        assert_eq!(size, Some(24000)); // 6000 elements * 4 bytes = 24000 bytes
    }

    #[test]
    fn test_calculate_buffer_size_buffer_op() {
        // BUFFER with explicit size
        let buffer = create_test_buffer(12345, DType::Float32, 0);

        let size = calculate_buffer_size(&buffer);
        assert_eq!(size, Some(12345));
    }

    #[test]
    fn test_calculate_buffer_size_symbolic() {
        // Symbolic range → None
        let batch_size = UOp::define_var("batch".to_string(), 1, 128);
        let range = UOp::range_axis(batch_size, 0, AxisType::Loop);

        let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
        let bufferize = UOp::bufferize(compute, vec![range], opts);

        let size = calculate_buffer_size(&bufferize);
        assert_eq!(size, None);
    }

    #[test]
    #[ignore] // TODO: Implement mstack test when UOp::mstack constructor is available
    fn test_calculate_buffer_size_mstack() {
        // MSTACK should return 1
        let _buf1 = create_test_buffer(100, DType::Float32, 0);
        let _buf2 = create_test_buffer(100, DType::Float32, 1);
        // let mstack = UOp::mstack(vec![buf1, buf2]);

        // let size = calculate_buffer_size(&mstack);
        // assert_eq!(size, Some(1));
    }

    #[test]
    fn test_calculate_out_in_ratio_efficient() {
        // output=100, inputs=[90, 5, 5] → ratio = 101/101 = 1.0 < 10
        let output_size = 100;
        let input_buffers = vec![
            create_test_buffer(90, DType::Float32, 0),
            create_test_buffer(5, DType::Float32, 1),
            create_test_buffer(5, DType::Float32, 2),
        ];

        let ratio = calculate_out_in_ratio(output_size, &input_buffers).unwrap();
        assert!(ratio < 10.0);
        assert!((ratio - 1.0).abs() < 0.01); // Should be ~1.0
    }

    #[test]
    fn test_calculate_out_in_ratio_wasteful() {
        // output=10000, inputs=[10, 10, 10] → ratio = 10001/31 ≈ 322 > 10
        let output_size = 10000;
        let input_buffers = vec![
            create_test_buffer(10, DType::Float32, 0),
            create_test_buffer(10, DType::Float32, 1),
            create_test_buffer(10, DType::Float32, 2),
        ];

        let ratio = calculate_out_in_ratio(output_size, &input_buffers).unwrap();
        assert!(ratio > 10.0);
        assert!(ratio > 300.0); // Should be ~322
    }

    #[test]
    fn test_calculate_out_in_ratio_with_symbolic_bufferize() {
        // Bufferize with symbolic range → None
        let output_size = 100;

        let batch_size = UOp::define_var("batch".to_string(), 1, 128);
        let range = UOp::range_axis(batch_size, 0, AxisType::Loop);
        let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
        let symbolic_bufferize = UOp::bufferize(compute, vec![range], opts);

        let input_buffers = vec![symbolic_bufferize];

        let ratio = calculate_out_in_ratio(output_size, &input_buffers);
        assert_eq!(ratio, None);
    }

    #[test]
    fn test_has_buffer_in_reduce_positive() {
        // REDUCE(LOAD(BUFFER)) → true
        use morok_ir::ReduceOp;

        let buffer = create_test_buffer(100, DType::Float32, 0);
        let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Reduce);

        let reduce = UOp::reduce(buffer, SmallVec::from_iter([range]), ReduceOp::Add);

        let reduces = vec![reduce];
        assert!(has_buffer_in_reduce(&reduces));
    }

    #[test]
    fn test_has_buffer_in_reduce_negative() {
        // REDUCE(CONST) → false
        use morok_ir::ReduceOp;

        let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Reduce);

        let reduce = UOp::reduce(const_val, SmallVec::from_iter([range]), ReduceOp::Add);

        let reduces = vec![reduce];
        assert!(!has_buffer_in_reduce(&reduces));
    }

    #[test]
    fn test_has_buffer_in_reduce_empty() {
        // Empty reduces → false
        let reduces: Vec<Rc<UOp>> = vec![];
        assert!(!has_buffer_in_reduce(&reduces));
    }

    #[test]
    fn test_has_buffer_in_reduce_nested_bufferize() {
        // REDUCE(BUFFERIZE(...)) → true
        use morok_ir::ReduceOp;

        let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let range1 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
        let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
        let bufferize = UOp::bufferize(compute, vec![range1.clone()], opts);

        let range2 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 1, AxisType::Reduce);
        let reduce = UOp::reduce(bufferize, SmallVec::from_iter([range2]), ReduceOp::Add);

        let reduces = vec![reduce];
        assert!(has_buffer_in_reduce(&reduces));
    }
}
