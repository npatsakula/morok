//! Helper functions for rangeify pattern matching and transformations.

use morok_dtype::DType;
use morok_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp};
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
    use morok_ir::SInt;
    uop.shape().map(|shape| shape.iter().any(|dim| matches!(dim, SInt::Const(0)))).unwrap_or(false)
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
                        UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(rng), begin_uop), DType::Index)
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
                            UOp::new(Op::Binary(BinaryOp::Sub, Rc::clone(uop), one), DType::Index)
                        }
                    };
                    UOp::new(Op::Binary(BinaryOp::Sub, shape_minus_1, Rc::clone(rng)), DType::Index)
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
                        SInt::Symbolic(uop) => {
                            UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(uop), begin_uop.clone()), DType::Index)
                        }
                    };
                    // rng >= begin  (use !(rng < begin) implemented as (rng < begin) XOR true)
                    let too_low = UOp::new(Op::Binary(BinaryOp::Lt, Rc::clone(rng), begin_uop.clone()), DType::Bool);
                    let true_val = UOp::const_(DType::Bool, ConstValue::Bool(true));
                    let valid_low = UOp::new(Op::Binary(BinaryOp::Xor, too_low, true_val), DType::Bool);
                    // rng < shape + begin
                    let valid_high = UOp::new(Op::Binary(BinaryOp::Lt, Rc::clone(rng), shape_plus_begin), DType::Bool);
                    // valid = valid_low & valid_high
                    let valid = UOp::new(Op::Binary(BinaryOp::And, valid_low, valid_high), DType::Bool);
                    // Subtract padding: rng - begin
                    let adjusted_rng = UOp::new(Op::Binary(BinaryOp::Sub, Rc::clone(rng), begin_uop), DType::Index);
                    // Use 0 as invalid value (will be masked by valid check later)
                    // TODO: Proper invalid handling - need to check Tinygrad's approach
                    let invalid_val = UOp::const_(DType::Index, ConstValue::Int(0));
                    UOp::new(Op::Ternary(TernaryOp::Where, valid, adjusted_rng, invalid_val), DType::Index)
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
                let weighted = UOp::new(Op::Binary(BinaryOp::Mul, acc.clone(), Rc::clone(rng)), DType::Index);
                axes_in.push(weighted);
                acc = match shape_dim {
                    SInt::Const(n) => UOp::new(
                        Op::Binary(BinaryOp::Mul, acc, UOp::const_(DType::Index, ConstValue::Int(*n as i64))),
                        DType::Index,
                    ),
                    SInt::Symbolic(uop) => UOp::new(Op::Binary(BinaryOp::Mul, acc, Rc::clone(uop)), DType::Index),
                };
            }
            let combined_axes = axes_in
                .into_iter()
                .reduce(|a, b| UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Index))
                .unwrap_or_else(|| UOp::const_(DType::Index, ConstValue::Int(0)));
            // Step 2: Unflatten into input shape dimensions
            let mut axes_out = Vec::new();
            let mut combined = combined_axes;
            for shape_dim in in_shape.iter().rev() {
                let shape_uop = match shape_dim {
                    SInt::Const(n) => UOp::const_(DType::Index, ConstValue::Int(*n as i64)),
                    SInt::Symbolic(uop) => Rc::clone(uop),
                };
                let mod_result = UOp::new(Op::Binary(BinaryOp::Mod, combined.clone(), shape_uop.clone()), DType::Index);
                axes_out.push(mod_result);
                combined = UOp::new(Op::Binary(BinaryOp::Idiv, combined, shape_uop), DType::Index);
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
/// A range is dead if:
/// - It's a RANGE(1) operation (loop from 0 to 1)
/// - The end value is a constant 1
///
/// Dead axes can be removed from BUFFERIZE operations to simplify indexing.
pub fn is_dead_axis(range: &Rc<UOp>) -> bool {
    // TODO: Enhance to detect provably-dead symbolic ranges
    // Currently only handles constant size-1 ranges
    if let Op::Range { end, .. } = range.op()
        && let Some(ConstValue::Int(1) | ConstValue::UInt(1)) = get_const_value(end)
    {
        return true;
    }
    false
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
