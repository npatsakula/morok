//! Cost-based buffer optimization with partial contiguous support.
//!
//! - Cost analysis: buffer count, output/input ratio, reduce usage
//! - Range partitioning: materialize vs substitute
//! - Partial contiguous: selectively materialize dimensions

use std::collections::HashSet;
use std::rc::Rc;

use morok_ir::{AddrSpace, Op, UOp, UOpKey};

/// Configuration for partial contiguous optimization.
#[derive(Debug, Clone, Copy)]
pub struct PcontigConfig {
    /// 0=disabled, 1=basic, 2=enabled (default), 3+=aggressive
    pub level: u8,
    /// Max buffers before keeping BUFFERIZE (default: 3)
    pub max_buffers_threshold: usize,
    /// Max output/input ratio for partial contiguous (default: 10.0)
    pub out_in_ratio_threshold: f64,
}

impl Default for PcontigConfig {
    fn default() -> Self {
        Self { level: 2, max_buffers_threshold: 3, out_in_ratio_threshold: 10.0 }
    }
}

/// Collect BUFFER, BUFFERIZE(GLOBAL), MSTACK, and MSELECT operations in a tree.
/// Stops traversal at GLOBAL bufferize boundaries.
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

/// Calculate buffer size in bytes. Returns `None` for symbolic shapes.
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

/// Calculate output/input size ratio. Returns `None` for symbolic sizes.
/// Ratio < 10 suggests efficient buffer, >= 10 suggests wasteful.
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

/// Extract ranges that must be materialized (from LOCAL INDEX operations).
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

/// Partition ranges into materialize (LOCAL/REDUCE) vs substitute (inline).
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

/// Apply partial contiguous: substitute inlined ranges, bufferize the rest.
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
