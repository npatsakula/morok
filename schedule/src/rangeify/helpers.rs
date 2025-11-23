//! Helper functions for rangeify pattern matching and transformations.

use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, Op, SInt, UOp};
use std::rc::Rc;

/// Check if a constant value is the identity element for a given binary operation.
///
/// Identity elements:
/// - Add/Sub: 0
/// - Mul/Div: 1
/// - And: all bits set
/// - Or/Xor: 0
pub fn is_identity_value(value: &ConstValue, op: &BinaryOp, is_right: bool) -> bool {
    match (op, value) {
        // Addition: x + 0 = x, 0 + x = x
        (BinaryOp::Add, ConstValue::Int(0)) => true,
        (BinaryOp::Add, ConstValue::Float(f)) if *f == 0.0 => true,

        // Subtraction: x - 0 = x (only right identity)
        (BinaryOp::Sub, ConstValue::Int(0)) if is_right => true,
        (BinaryOp::Sub, ConstValue::Float(f)) if is_right && *f == 0.0 => true,

        // Multiplication: x * 1 = x, 1 * x = x
        (BinaryOp::Mul, ConstValue::Int(1)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 1.0 => true,

        // Division: x / 1 = x (only right identity)
        (BinaryOp::Idiv, ConstValue::Int(1)) if is_right => true,
        (BinaryOp::Fdiv, ConstValue::Float(f)) if is_right && *f == 1.0 => true,

        // Bitwise Or: x | 0 = x, 0 | x = x
        (BinaryOp::Or, ConstValue::Int(0)) => true,

        // Bitwise Xor: x ^ 0 = x, 0 ^ x = x
        (BinaryOp::Xor, ConstValue::Int(0)) => true,

        // Bitwise And: x & all_ones = x, all_ones & x = x
        (BinaryOp::And, ConstValue::Int(-1)) => true, // -1 has all bits set

        _ => false,
    }
}

/// Check if a constant value is the zero element for a given binary operation.
///
/// Zero elements (annihilators):
/// - Mul: 0 (x * 0 = 0)
/// - And: 0 (x & 0 = 0)
/// - Or: all bits set (x | -1 = -1)
pub fn is_zero_value(value: &ConstValue, op: &BinaryOp) -> bool {
    match (op, value) {
        // Multiplication: x * 0 = 0
        (BinaryOp::Mul, ConstValue::Int(0)) => true,
        (BinaryOp::Mul, ConstValue::Float(f)) if *f == 0.0 => true,

        // Bitwise And: x & 0 = 0
        (BinaryOp::And, ConstValue::Int(0)) => true,

        _ => false,
    }
}

/// Extract the constant value from a UOp if it's a CONST operation.
pub fn get_const_value(uop: &Rc<UOp>) -> Option<ConstValue> {
    match uop.op() {
        Op::Const(cv) => Some(cv.0),
        _ => None,
    }
}

/// Check if a UOp is a constant with a specific value.
pub fn is_const(uop: &Rc<UOp>, value: &ConstValue) -> bool {
    get_const_value(uop).as_ref() == Some(value)
}

/// Check if a UOp represents a zero-size tensor.
///
/// A tensor has zero size if any dimension in its shape is 0.
pub fn is_zero_size(uop: &Rc<UOp>) -> bool {
    uop.shape().ok().flatten().map(|shape| shape.iter().any(|dim| matches!(dim, SInt::Const(0)))).unwrap_or(false)
}

/// Check if a dtype is void (used for side-effecting operations).
pub fn is_void(dtype: &DType) -> bool {
    *dtype == DType::Void
}

/// Get the binary operation from a UOp if it's a BINARY operation.
pub fn get_binary_op(uop: &Rc<UOp>) -> Option<BinaryOp> {
    match uop.op() {
        Op::Binary(op, _, _) => Some(*op),
        _ => None,
    }
}

/// Apply a movement operation to transform input ranges.
///
/// This is the core function for converting movement ops to index transformations.
/// Each movement operation defines how output ranges map to input ranges.
///
/// # Arguments
///
/// * `op` - The movement operation to apply
/// * `in_shape` - The input shape (before movement)
/// * `rngs` - The output ranges (indices into the result)
///
/// # Returns
///
/// The transformed input ranges (indices into the source)
///
/// # Movement Op Transformations
///
/// - **SHRINK**: Add offset to each range: `rng[i] + begin[i]`
/// - **PERMUTE**: Reorder ranges: `rngs[inv_perm[i]]`
/// - **FLIP**: Reverse indices: `(shape[i]-1) - rng[i]` if flipped
/// - **EXPAND**: Replace with 0 for broadcast dims
/// - **PAD**: Add valid checks and subtract padding
/// - **RESHAPE**: Complex multi-dim index arithmetic (flatten → unflatten)
pub fn apply_movement_op(op: &Op, in_shape: &[morok_ir::SInt], rngs: &[Rc<UOp>]) -> Vec<Rc<UOp>> {
    use morok_ir::SInt;

    match op {
        // SHRINK: ranges[i] = rng[i] + begin[i]
        Op::Shrink { begins, .. } => {
            let begin_vals = extract_shape_values(begins);
            rngs.iter()
                .zip(begin_vals.iter())
                .map(|(rng, &begin)| {
                    if begin == 0 {
                        Rc::clone(rng)
                    } else {
                        let begin_uop = UOp::const_(DType::Index, ConstValue::Int(begin as i64));
                        rng.try_add_op(&begin_uop).unwrap()
                    }
                })
                .collect()
        }

        // PERMUTE: ranges = [rngs[inv_perm[i]] for i in 0..len]
        Op::Permute { axes, .. } => {
            let inv_perm = argsort(axes);
            inv_perm.iter().map(|&i| Rc::clone(&rngs[i])).collect()
        }

        // FLIP: ranges[i] = (shape[i]-1) - rng[i] if flip[i] else rng[i]
        Op::Flip { axes: flips, .. } => rngs
            .iter()
            .zip(in_shape.iter())
            .zip(flips.iter())
            .map(|((rng, shape), &flip)| {
                if !flip {
                    Rc::clone(rng)
                } else {
                    let shape_minus_1 = match shape {
                        SInt::Const(n) => UOp::const_(DType::Index, ConstValue::Int(*n as i64 - 1)),
                        SInt::Symbolic(uop) => {
                            let one = UOp::const_(DType::Index, ConstValue::Int(1));
                            uop.try_sub_op(&one).unwrap()
                        }
                    };
                    shape_minus_1.try_sub_op(rng).unwrap()
                }
            })
            .collect(),

        // EXPAND: ranges[i] = 0 if expanding else rng[i]
        Op::Expand { new_shape, .. } => {
            let new_shape_vals = extract_shape_from_uop(new_shape);
            rngs.iter()
                .zip(in_shape.iter())
                .zip(new_shape_vals.iter())
                .map(|((rng, in_sh), out_sh)| {
                    let expanding = match (in_sh, out_sh) {
                        (SInt::Const(1), SInt::Const(n)) if *n > 1 => true,
                        (SInt::Const(1), SInt::Symbolic(_)) => true,
                        _ => false,
                    };
                    if expanding { UOp::const_(DType::Index, ConstValue::Int(0)) } else { Rc::clone(rng) }
                })
                .collect()
        }

        // PAD: ranges[i] = valid_check.where(rng[i] - pad_begin, INVALID)
        Op::Pad { begin_pads, end_pads, .. } => {
            let begin_vals = extract_shape_values(begin_pads);
            let end_vals = extract_shape_values(end_pads);
            rngs.iter()
                .zip(in_shape.iter())
                .zip(begin_vals.iter().zip(end_vals.iter()))
                .map(|((rng, shape), (&begin, &end))| {
                    if begin == 0 && end == 0 {
                        return Rc::clone(rng);
                    }
                    let begin_uop = UOp::const_(DType::Index, ConstValue::Int(begin as i64));
                    let shape_plus_begin = match shape {
                        SInt::Const(n) => UOp::const_(DType::Index, ConstValue::Int(*n as i64 + begin as i64)),
                        SInt::Symbolic(uop) => uop.try_add_op(&begin_uop).unwrap(),
                    };
                    // rng >= begin  (use !(rng < begin) implemented as (rng < begin) XOR true)
                    let too_low = rng.try_cmplt(&begin_uop).unwrap();
                    let true_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
                    let valid_low = too_low.try_xor_op(&true_val).unwrap();
                    // rng < shape + begin
                    let valid_high = rng.try_cmplt(&shape_plus_begin).unwrap();
                    // valid = valid_low & valid_high
                    let valid = valid_low.try_and_op(&valid_high).unwrap();
                    // Subtract padding: rng - begin
                    let adjusted_rng = rng.try_sub_op(&begin_uop).unwrap();
                    // Use invalid marker for out-of-bounds regions (will be masked by valid check)
                    UOp::where_op(valid, adjusted_rng, UOp::invalid_marker()).unwrap()
                })
                .collect()
        }

        // RESHAPE: Complex multi-dimensional index arithmetic
        Op::Reshape { new_shape, .. } => {
            let new_shape_vals = extract_shape_from_uop(new_shape);
            // Step 1: Flatten output indices
            let mut acc = UOp::const_(DType::Index, ConstValue::Int(1));
            let mut axes_in = Vec::new();
            for (shape_dim, rng) in new_shape_vals.iter().zip(rngs.iter()).rev() {
                let weighted = acc.try_mul_op(rng).unwrap();
                axes_in.push(weighted);
                acc = match shape_dim {
                    SInt::Const(n) => {
                        let n_uop = UOp::const_(DType::Index, ConstValue::Int(*n as i64));
                        acc.try_mul_op(&n_uop).unwrap()
                    }
                    SInt::Symbolic(uop) => acc.try_mul_op(uop).unwrap(),
                };
            }
            let combined_axes = axes_in
                .into_iter()
                .reduce(|a, b| a.try_add_op(&b).unwrap())
                .unwrap_or_else(|| UOp::const_(DType::Index, ConstValue::Int(0)));
            // Step 2: Unflatten into input shape dimensions
            let mut axes_out = Vec::new();
            let mut combined = combined_axes;
            for shape_dim in in_shape.iter().rev() {
                let shape_uop = match shape_dim {
                    SInt::Const(n) => UOp::const_(DType::Index, ConstValue::Int(*n as i64)),
                    SInt::Symbolic(uop) => Rc::clone(uop),
                };
                let mod_result = combined.try_mod_op(&shape_uop).unwrap();
                axes_out.push(mod_result);
                combined = combined.try_idiv_op(&shape_uop).unwrap();
            }
            axes_out.reverse();
            axes_out
        }

        _ => panic!("apply_movement_op called with non-movement op: {:?}", op),
    }
}

/// Extract shape values from a UOp (for SHRINK begins/ends, PAD pads).
fn extract_shape_values(uop: &Rc<UOp>) -> Vec<usize> {
    match uop.op() {
        Op::Vectorize { elements } => elements
            .iter()
            .map(|elem| match elem.op() {
                Op::Const(cv) => match cv.0 {
                    ConstValue::Int(n) => n as usize,
                    _ => panic!("Expected int constant in vectorize"),
                },
                _ => panic!("Expected constant element in vectorize"),
            })
            .collect(),
        Op::Const(cv) => {
            // Single constant value - treat as 1-element vector
            match cv.0 {
                ConstValue::Int(n) => vec![n as usize],
                _ => panic!("Expected int constant"),
            }
        }
        _ => panic!("Expected vectorize or constant for shape values, got {:?}", uop.op()),
    }
}

/// Extract shape from a UOp (for RESHAPE new_shape, EXPAND new_shape).
fn extract_shape_from_uop(uop: &Rc<UOp>) -> Vec<morok_ir::SInt> {
    use morok_ir::SInt;
    match uop.op() {
        Op::Vectorize { elements } => elements
            .iter()
            .map(|elem| match elem.op() {
                Op::Const(cv) => match cv.0 {
                    ConstValue::Int(n) => SInt::Const(n as usize),
                    _ => SInt::Symbolic(Rc::clone(elem)),
                },
                _ => SInt::Symbolic(Rc::clone(elem)),
            })
            .collect(),
        Op::Const(cv) => {
            // Single constant - treat as 1-element shape
            match cv.0 {
                ConstValue::Int(n) => vec![SInt::Const(n as usize)],
                _ => panic!("Expected int constant for shape"),
            }
        }
        _ => panic!("Expected vectorize or constant for shape, got {:?}", uop.op()),
    }
}

/// Compute inverse permutation (argsort).
fn argsort(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Check if two range lists are structurally equal.
///
/// Two ranges are considered equal if they are pointer-equal (same UOp).
/// This is used to detect noop BUFFERIZE operations where INDEX uses
/// the same ranges.
pub fn ranges_equal(ranges1: &[Rc<UOp>], ranges2: &[Rc<UOp>]) -> bool {
    ranges1.len() == ranges2.len() && ranges1.iter().zip(ranges2).all(|(r1, r2)| Rc::ptr_eq(r1, r2))
}

/// Check if an operation should always run (never removed by buffer optimization).
///
/// These operations have important side effects or semantic requirements:
/// - CONTIGUOUS: Forces contiguous memory layout
/// - COPY: Moves data between devices
/// - ASSIGN: Writes to a specific location
/// - NOOP: Placeholder that must be preserved
pub fn is_always_run_op(op: &Op) -> bool {
    matches!(op, Op::Contiguous { .. } | Op::Copy { .. } | Op::Assign { .. } | Op::Noop)
}

/// Count the number of buffer-like operations accessed in a UOp tree.
///
/// Buffer-like operations include:
/// - BUFFER, BUFFER_VIEW: Actual buffers
/// - MSTACK, MSELECT: Multi-buffer operations
/// - DEFINE_GLOBAL, DEFINE_LOCAL: Global/local memory
///
/// This is used in cost-based buffer removal to decide if buffering
/// reduces the number of memory accesses.
#[allow(clippy::mutable_key_type)]
pub fn count_buffer_accesses(uop: &Rc<UOp>) -> usize {
    use morok_ir::UOpKey;
    use std::collections::HashSet;

    let mut visited = HashSet::new();
    let mut count = 0;

    fn visit(uop: &Rc<UOp>, visited: &mut HashSet<UOpKey>, count: &mut usize) {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return; // Already visited
        }

        // Check if this is a buffer-like operation
        if matches!(
            uop.op(),
            Op::Buffer { .. }
                | Op::BufferView { .. }
                | Op::MStack { .. }
                | Op::MSelect { .. }
                | Op::DefineGlobal(_)
                | Op::DefineLocal(_)
        ) {
            *count += 1;
        }

        // Recursively visit children
        for child in uop.op().sources() {
            visit(&child, visited, count);
        }
    }

    visit(uop, &mut visited, &mut count);
    count
}

/// Check if a range represents a dead axis (size 1).
///
/// A range is dead if it iterates 0 or 1 times (size ≤ 1).
///
/// Uses vmax analysis to detect both constant and symbolic dead ranges:
/// - Constant: RANGE(1) has vmax = 0
/// - Symbolic: Variable bounded to [1,1] has vmax = 0
/// - Arithmetic: Expression that simplifies to 1 has vmax = 0
///
/// For RANGE(end), the vmax is end - 1 (maximum loop variable value).
/// If vmax ≤ 0, the range iterates at most once (dead axis).
///
/// Dead axes can be removed from BUFFERIZE operations to simplify indexing.
pub fn is_dead_axis(range: &Rc<UOp>) -> bool {
    if !matches!(range.op(), Op::Range { .. }) {
        return false;
    }

    // Use vmax analysis to detect dead ranges
    // A range is dead if vmax ≤ 0 (iterates 0 or 1 time)
    // For RANGE(end), vmax = end - 1, so vmax ≤ 0 means end ≤ 1
    match range.vmax() {
        ConstValue::Int(v) => *v <= 0,
        ConstValue::UInt(v) => *v == 0, // UInt can't be negative
        _ => false,                     // Symbolic or non-numeric vmax - can't prove dead
    }
}

/// Check if an operation is cheap to inline (low recomputation cost).
///
/// Cheap operations include:
/// - Nullary ops: CONST, DEFINE_VAR, DEVICE, NOOP
/// - Unary ops: Simple transformations
/// - Binary ops: Arithmetic, comparisons
/// - Type conversions: CAST, BITCAST
/// - Indexing: GEP (vector element access)
///
/// Expensive operations (worth buffering):
/// - Memory ops: LOAD, STORE
/// - Reductions: REDUCE, REDUCE_AXIS
/// - Special ops: WMMA, CONTRACT
/// - Control flow: IF, RANGE (already buffered)
pub fn is_cheap_to_inline(op: &Op) -> bool {
    matches!(
        op,
        // Nullary - always cheap
        Op::Const(_)
            | Op::Unique(_)
            | Op::Device(_)
            | Op::Noop
            | Op::DefineVar { .. }
            | Op::DefineReg { .. }
            | Op::VConst { .. }
            // Simple operations - cheap to recompute
            | Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::Cast { .. }
            | Op::BitCast { .. }
            // Vector operations - cheap
            | Op::Gep { .. }
            | Op::Vectorize { .. }
            // Index/pointer operations - cheap
            | Op::PointerIndex { .. }
    )
}

/// Count how many times a UOp is used in a computation tree.
///
/// This helps determine if buffering is beneficial:
/// - Used once: buffering may not help (unless compute is expensive)
/// - Used multiple times: buffering avoids recomputation
///
/// Note: This is a simplified version that doesn't track parent relationships.
/// A more sophisticated implementation would build a use-def graph.
#[allow(clippy::mutable_key_type)]
pub fn count_uses(target: &Rc<UOp>, root: &Rc<UOp>) -> usize {
    use morok_ir::UOpKey;
    use std::collections::HashSet;

    let mut visited = HashSet::new();
    let mut count = 0;
    let target_key = UOpKey(Rc::clone(target));

    fn visit(uop: &Rc<UOp>, target_key: &UOpKey, visited: &mut HashSet<UOpKey>, count: &mut usize) {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return; // Already visited
        }

        // Check if this UOp uses the target
        for child in uop.op().sources() {
            let child_key = UOpKey(Rc::clone(&child));
            if child_key == *target_key {
                *count += 1;
            }
            visit(&child, target_key, visited, count);
        }
    }

    visit(root, &target_key, &mut visited, &mut count);
    count
}

/// Check if a UOp has no RANGE dependencies.
///
/// This is critical for reduce_collapse: after symbolic simplification,
/// we verify that all ranges have been eliminated before accepting the result.
///
/// # Algorithm
///
/// Uses `in_scope_ranges()` to get all ranges this UOp depends on, then
/// checks if any of them are actual RANGE operations.
///
/// # Example
///
/// ```ignore
/// // Has range dependency
/// let range = UOp::range_axis(end, 0, AxisType::Loop);
/// let sum = range.try_add_op(&const_5).unwrap();
/// assert!(!no_range(&sum)); // false - depends on range
///
/// // No range dependency
/// let const_val = UOp::const_(DType::Int32, ConstValue::Int(42));
/// assert!(no_range(&const_val)); // true - no ranges
/// ```
///
/// Based on Tinygrad's `no_range()` (tinygrad/codegen/simplify.py:66)
pub fn no_range(uop: &Rc<UOp>) -> bool {
    #[allow(clippy::mutable_key_type)]
    let in_scope_ranges = uop.in_scope_ranges();

    // Check if any of the in-scope ranges are actual RANGE operations
    !in_scope_ranges.iter().any(|key| matches!(key.0.op(), Op::Range { .. }))
}

/// Extract the size of a RANGE operation as an i64 constant.
///
/// This is used for computing closed-form sums when we know the range size.
/// Only works for constant-sized ranges; symbolic ranges return None.
///
/// # Arguments
///
/// * `range` - A RANGE UOp
///
/// # Returns
///
/// * `Some(size)` - The constant size if available
/// * `None` - If not a RANGE or if size is symbolic
///
/// # Example
///
/// ```ignore
/// let range = UOp::range_axis(
///     UOp::const_(DType::Index, ConstValue::Int(10)),
///     0,
///     AxisType::Loop
/// );
/// assert_eq!(range_size_as_i64(&range), Some(10));
/// ```
///
/// Used by reduce_collapse pattern matchers for bound computation.
pub fn range_size_as_i64(range: &Rc<UOp>) -> Option<i64> {
    if let Op::Range { end, .. } = range.op() {
        get_const_value(end).and_then(|cv| match cv {
            ConstValue::Int(n) => Some(n),
            ConstValue::UInt(n) => Some(n as i64),
            _ => None,
        })
    } else {
        None
    }
}

/// Check if all ranges in a list are structurally equal (ignoring validity masks).
///
/// Two ranges are considered equal if their index expressions are identical.
/// Validity masks (from padding/WHERE operations) are ignored.
///
/// This is used for range merging to determine if multiple consumers can share
/// the same indexing pattern.
///
/// # Arguments
///
/// * `ranges` - List of range UOps to compare
///
/// # Returns
///
/// * `true` - All ranges have identical index expressions
/// * `false` - At least one range differs
///
/// # Examples
///
/// ```ignore
/// // Identical ranges (same pointer)
/// let r1 = UOp::range_axis(end.clone(), 0, AxisType::Loop);
/// let r2 = r1.clone();
/// assert!(all_ranges_same(&[r1, r2]));
///
/// // Identical ranges (different pointers, same structure)
/// let end = UOp::const_(DType::Index, ConstValue::Int(10));
/// let r1 = UOp::range_axis(end.clone(), 0, AxisType::Loop);
/// let r2 = UOp::range_axis(end.clone(), 0, AxisType::Loop);
/// assert!(all_ranges_same(&[r1, r2])); // Structurally equal
///
/// // Different ranges
/// let r1 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Loop);
/// let r2 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(20)), 0, AxisType::Loop);
/// assert!(!all_ranges_same(&[r1, r2]));
/// ```
///
/// Used by range merging logic in indexing.rs.
pub fn all_ranges_same(ranges: &[Rc<UOp>]) -> bool {
    if ranges.is_empty() {
        return true;
    }

    // Extract index expressions (strip validity masks)
    let first_idx = ranges[0].get_idx();

    ranges.iter().skip(1).all(|r| {
        let idx = r.get_idx();
        // Check pointer equality first (fast path)
        Rc::ptr_eq(&first_idx, &idx) || uop_equal(&first_idx, &idx)
    })
}

/// Deep structural equality check for UOps.
///
/// Compares two UOps for structural equality, checking:
/// - Operation type and fields (via PartialEq on Op)
/// - Data type
/// - All sources recursively
///
/// This is used by range comparison to detect when two ranges have
/// the same structure even if they're different heap objects.
///
/// # Arguments
///
/// * `a`, `b` - UOps to compare
///
/// # Returns
///
/// * `true` - UOps are structurally identical
/// * `false` - UOps differ in structure
///
/// # Examples
///
/// ```ignore
/// // Same pointer - always equal
/// let x = UOp::const_(DType::Int32, ConstValue::Int(5));
/// assert!(uop_equal(&x, &x));
///
/// // Different pointers, same structure
/// let x1 = UOp::const_(DType::Int32, ConstValue::Int(5));
/// let x2 = UOp::const_(DType::Int32, ConstValue::Int(5));
/// assert!(uop_equal(&x1, &x2)); // hash-consing may make these the same
///
/// // Different values
/// let x1 = UOp::const_(DType::Int32, ConstValue::Int(5));
/// let x2 = UOp::const_(DType::Int32, ConstValue::Int(10));
/// assert!(!uop_equal(&x1, &x2));
/// ```
///
/// Note: This is a structural comparison, not pointer comparison.
/// Hash-consing may cause structurally equal UOps to have the same pointer.
pub fn uop_equal(a: &Rc<UOp>, b: &Rc<UOp>) -> bool {
    // Fast path: pointer equality
    if Rc::ptr_eq(a, b) {
        return true;
    }

    // Check operation type (discriminant)
    if std::mem::discriminant(a.op()) != std::mem::discriminant(b.op()) {
        return false;
    }

    // Check dtype
    if a.dtype() != b.dtype() {
        return false;
    }

    // Special case: Const operations need value comparison
    if let (Op::Const(cv_a), Op::Const(cv_b)) = (a.op(), b.op()) {
        return cv_a.0 == cv_b.0;
    }

    // For Range operations, we need to compare all fields including axis_id
    // which is not included in sources(). Use simplified equality:
    // just compare sources since that's what matters for range merging.
    // Different axis_ids with same end value are still compatible ranges.

    // Check sources recursively
    let a_srcs = a.op().sources();
    let b_srcs = b.op().sources();

    if a_srcs.len() != b_srcs.len() {
        return false;
    }

    a_srcs.iter().zip(b_srcs.iter()).all(|(sa, sb)| uop_equal(sa, sb))
}
