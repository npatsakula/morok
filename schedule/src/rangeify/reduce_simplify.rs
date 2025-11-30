//! Reduction simplification: reduce_unparented and reduce_collapse.

use std::collections::HashSet;
use std::rc::Rc;

use morok_ir::{Op, ReduceOp, UOp, UOpKey};
use smallvec::SmallVec;

/// Remove ranges from REDUCE that don't appear in source. ADD→mul by size, MUL→pow.
#[allow(clippy::mutable_key_type)]
pub fn reduce_unparented(reduce: &Rc<UOp>) -> Option<Rc<UOp>> {
    let Op::Reduce { src, ranges, reduce_op } = reduce.op() else {
        return None;
    };

    // Only handle ADD, MUL, MAX, MIN
    if !matches!(reduce_op, ReduceOp::Add | ReduceOp::Mul | ReduceOp::Max | ReduceOp::Min) {
        return None;
    }

    // All reduce ranges must be RANGE operations
    debug_assert!(
        ranges.iter().all(|r| matches!(r.op(), Op::Range { .. })),
        "reduce_unparented: Some reduce srcs aren't ranges"
    );

    // Get ranges that src depends on (in-scope ranges)
    let src_ranges = src.in_scope_ranges();

    // Partition ranges into parented (used by src) vs unparented (unused)
    let (parented, unparented) = partition_ranges(ranges, src_ranges);

    // No unparented ranges - nothing to optimize
    if unparented.is_empty() {
        return None;
    }

    // Build result starting with parented ranges only
    let mut result = if !parented.is_empty() || reduce.dtype() != src.dtype() {
        // Create REDUCE with only parented ranges
        UOp::reduce(Rc::clone(src), parented, *reduce_op)
    } else {
        // No parented ranges and same dtype - just return source
        Rc::clone(src)
    };

    // Apply transformations for each unparented range
    match reduce_op {
        ReduceOp::Add => {
            // sum(x, r) = x * r.size
            for range in &unparented {
                let size = get_range_size(range)?;
                let size_casted = cast_to_dtype(&size, &result.dtype())?;
                result = result.try_mul(&size_casted).ok()?;
            }
        }
        ReduceOp::Mul => {
            // prod(x, r) = x^r.size
            for range in &unparented {
                let size = get_range_size(range)?;
                let size_casted = cast_to_dtype(&size, &result.dtype())?;
                result = result.try_pow(&size_casted).ok()?;
            }
        }
        ReduceOp::Max | ReduceOp::Min => {
            // max/min(x, r) = x (constant over range)
            // Result already set correctly - ranges removed
        }
    }

    Some(result)
}

/// Partition ranges into parented (in src_ranges) and unparented.
#[allow(clippy::mutable_key_type)]
fn partition_ranges(
    ranges: &SmallVec<[Rc<UOp>; 4]>,
    src_ranges: &HashSet<UOpKey>,
) -> (SmallVec<[Rc<UOp>; 4]>, Vec<Rc<UOp>>) {
    let mut parented = SmallVec::new();
    let mut unparented = Vec::new();

    for range in ranges {
        let key = UOpKey(Rc::clone(range));
        if src_ranges.contains(&key) {
            parented.push(Rc::clone(range));
        } else {
            unparented.push(Rc::clone(range));
        }
    }

    (parented, unparented)
}

/// Extract size (end value) from RANGE.
fn get_range_size(range: &Rc<UOp>) -> Option<Rc<UOp>> {
    if let Op::Range { end, .. } = range.op() { Some(Rc::clone(end)) } else { None }
}

/// Lift range-independent computations outside REDUCE via symbolic simplification.
#[allow(clippy::mutable_key_type)]
pub fn reduce_collapse(reduce: &Rc<UOp>) -> Option<Rc<UOp>> {
    use std::collections::{HashMap, HashSet};

    // Only handle REDUCE operations
    let Op::Reduce { src, ranges, .. } = reduce.op() else {
        return None;
    };

    // Need at least one range to collapse
    if ranges.is_empty() {
        return None;
    }

    // Step 1: Create substitution map: RANGE → DEFINE_VAR
    // For each range, create a symbolic variable with bounds [0, size-1]
    let mut substitute_map: HashMap<UOpKey, Rc<UOp>> = HashMap::new();

    for (i, range) in ranges.iter().enumerate() {
        // Extract range size as constant i64
        let size_i64 = super::helpers::range_size_as_i64(range)?;

        // Can only collapse constant-sized ranges (symbolic ranges need runtime evaluation)
        if size_i64 <= 0 {
            return None;
        }

        // Create symbolic variable: Variable(f"idx{i}", 0, size-1)
        let var_name = format!("ridx{}", i); // "ridx" = reduction index
        let define_var = UOp::define_var(var_name, 0, size_i64 - 1);

        substitute_map.insert(UOpKey(Rc::clone(range)), define_var);
    }

    // Step 2: Apply substitution to src
    let substituted = src.substitute(&substitute_map);

    // Step 3: Apply symbolic simplification
    // This is where the magic happens - symbolic patterns eliminate range dependencies
    let matcher = crate::symbolic::symbolic_simple();
    let simplified = crate::rewrite::graph_rewrite(&matcher, substituted, &mut ());

    // Step 4: Verify range dependencies eliminated
    // Check if simplified expression still depends on any of the DEFINE_VARs we created
    // If it does, the optimization didn't eliminate the dependency - return None
    let vars_in_simplified: HashSet<UOpKey> =
        simplified.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::DefineVar { .. })).map(UOpKey).collect();

    // Check if any of our substituted vars remain in the simplified expression
    let has_var_dependency = substitute_map.values().any(|var| vars_in_simplified.contains(&UOpKey(Rc::clone(var))));

    if has_var_dependency {
        // Symbolic simplification didn't eliminate the range dependency
        return None;
    }

    // Also verify no RANGE operations remain (sanity check)
    if !super::helpers::no_range(&simplified) {
        return None;
    }

    // Step 5: Substitute back: DEFINE_VAR → RANGE
    // Create reverse mapping to restore original ranges (if any DEFINE_VARs remain)
    // Since we verified no var dependencies exist, this should be a no-op
    let reverse_map: HashMap<UOpKey, Rc<UOp>> =
        substitute_map.into_iter().map(|(range_key, var)| (UOpKey(var), range_key.0)).collect();

    let result = simplified.substitute(&reverse_map);

    Some(result)
}

/// Cast value to dtype, with broadcasting for vector types.
fn cast_to_dtype(value: &Rc<UOp>, target_dtype: &morok_dtype::DType) -> Option<Rc<UOp>> {
    use morok_dtype::DType;

    // Get the scalar type to cast to
    let scalar_type = match target_dtype {
        DType::Scalar(s) => DType::Scalar(*s),
        DType::Vector { scalar, .. } => DType::Scalar(*scalar),
        _ => return None, // Can't cast to pointer or image types
    };

    // Cast to scalar type
    let casted = UOp::cast(Rc::clone(value), scalar_type);

    // Broadcast if target is a vector (create vector with repeated elements)
    if target_dtype.is_vector() {
        let count = target_dtype.count();
        let elements: SmallVec<[Rc<UOp>; 4]> = (0..count).map(|_| casted.clone()).collect();
        Some(UOp::vectorize(elements))
    } else {
        Some(casted)
    }
}
