//! Test utilities for scheduler/optimizer tests.
//!
//! This module provides helper functions to create common test patterns
//! (reduces, matmul, etc.) and assertion utilities for validating scheduler state.

use morok_ir::{AxisId, AxisType, Op, ReduceOp, UOp};
use std::sync::Arc;

use crate::optimizer::Scheduler;

/// Creates a simple reduction pattern for testing.
///
/// Generates:
/// ```text
/// SINK[
///   REDUCE(op=reduce_op)[
///     CONST(value=1.0),
///     RANGE(size, axis_id=0, type=Reduce)
///   ]
/// ]
/// ```
///
/// # Arguments
/// * `size` - Size of the dimension to reduce
/// * `reduce_op` - Type of reduction (Add, Max, etc.)
///
/// # Returns
/// A UOp representing the reduction sink
pub fn create_simple_reduce(size: i64, reduce_op: ReduceOp) -> Arc<UOp> {
    use smallvec::smallvec;

    let size_uop = UOp::native_const(size as i32 as i64);
    // Use Reduce axis type for reduction dimensions
    let range = UOp::range_axis(size_uop, AxisId::Renumbered(0), AxisType::Reduce);
    let const_val = UOp::native_const(1.0f32);
    let reduce = const_val.reduce(smallvec![range], reduce_op);
    UOp::sink(vec![reduce])
}

/// Creates a reduction pattern with global axes for testing LOCAL operations.
///
/// Generates a pattern matching Tinygrad's `test_local_and_grouped_reduce`:
/// - Multiple Global axes (output dimensions)
/// - One Reduce axis (reduction dimension)
///
/// Pattern structure:
/// ```text
/// SINK[
///   REDUCE(op=reduce_op)[
///     CONST(value=1.0),
///     RANGE(reduce_size, axis_id=last, type=Reduce)
///   ]
/// ]
/// with Global axes for each output dimension
/// ```
///
/// # Arguments
/// * `global_sizes` - Sizes of Global axes (output shape)
/// * `reduce_size` - Size of Reduce axis
/// * `reduce_op` - Type of reduction (Add, Max, etc.)
///
/// # Returns
/// A UOp representing the reduction with global axes
pub fn create_reduce_with_globals(global_sizes: &[i64], reduce_size: i64, reduce_op: ReduceOp) -> Arc<UOp> {
    use smallvec::smallvec;

    // Create Global axes
    let mut all_axes = Vec::new();
    for (i, &size) in global_sizes.iter().enumerate() {
        let size_uop = UOp::native_const(size as i32 as i64);
        let axis = UOp::range_axis(size_uop, AxisId::Renumbered(i), AxisType::Global);
        all_axes.push(axis);
    }

    // Create Reduce axis
    let reduce_size_uop = UOp::native_const(reduce_size as i32 as i64);
    let reduce_axis = UOp::range_axis(reduce_size_uop, AxisId::Renumbered(global_sizes.len()), AxisType::Reduce);

    // Create reduction
    let const_val = UOp::native_const(1.0f32);
    let reduce = const_val.reduce(smallvec![reduce_axis], reduce_op);

    // Add all axes to sink
    all_axes.insert(0, reduce);
    UOp::sink(all_axes)
}

/// Creates a matmul pattern for testing.
///
/// Generates matrix multiplication C = A @ B where:
/// - A has shape (M, K)
/// - B has shape (K, N)
/// - C has shape (M, N)
///
/// Pattern:
/// ```text
/// SINK[
///   REDUCE(op=Add)[  // Reduce over K
///     ADD(
///       RANGE(m, axis_id=0, type=Global),
///       RANGE(k, axis_id=2, type=Global)
///     ),
///     RANGE(k, axis_id=2, type=Global)  // K dimension
///   ]
/// ]
/// with RANGE(n, axis_id=1, type=Global)  // N dimension
/// ```
///
/// # Arguments
/// * `m` - M dimension (rows of A, rows of C)
/// * `n` - N dimension (cols of B, cols of C)
/// * `k` - K dimension (cols of A, rows of B, reduction axis)
///
/// # Returns
/// A UOp representing the matmul sink
pub fn create_matmul_pattern(m: i64, n: i64, k: i64) -> Arc<UOp> {
    use smallvec::smallvec;

    // Create ranges for M, N, K dimensions
    let m_uop = UOp::native_const(m as i32 as i64);
    let n_uop = UOp::native_const(n as i32 as i64);
    let k_uop = UOp::native_const(k as i32 as i64);

    let m_range = UOp::range_axis(m_uop, AxisId::Renumbered(0), AxisType::Global);
    let n_range = UOp::range_axis(n_uop, AxisId::Renumbered(1), AxisType::Global);
    let k_range = UOp::range_axis(k_uop, AxisId::Renumbered(2), AxisType::Global);

    // Create a simple computation that uses all ranges
    // (simplified for testing - structure matters more than exact computation)
    let add_expr = m_range.try_add(&k_range).expect("ADD should succeed with same dtype");

    // Create reduction over K
    let reduce = add_expr.reduce(smallvec![k_range], ReduceOp::Add);

    // Create sink with all ranges
    UOp::sink(vec![reduce, m_range, n_range])
}

/// Creates a double reduction pattern (reduce two axes).
///
/// Generates:
/// ```text
/// SINK[
///   REDUCE(op=reduce_op)[
///     CONST(value=1.0),
///     RANGE(size1, axis_id=0, type=Reduce),
///     RANGE(size2, axis_id=1, type=Reduce)
///   ]
/// ]
/// ```
///
/// # Arguments
/// * `size1` - Size of first dimension to reduce
/// * `size2` - Size of second dimension to reduce
/// * `reduce_op` - Type of reduction (Add, Max, etc.)
///
/// # Returns
/// A UOp representing the double reduction sink
pub fn create_double_reduce(size1: i64, size2: i64, reduce_op: ReduceOp) -> Arc<UOp> {
    use smallvec::smallvec;

    let size1_uop = UOp::native_const(size1 as i32 as i64);
    let size2_uop = UOp::native_const(size2 as i32 as i64);

    // Reduction axes should be marked as Reduce from the start
    let range1 = UOp::range_axis(size1_uop, AxisId::Renumbered(0), AxisType::Reduce);
    let range2 = UOp::range_axis(size2_uop, AxisId::Renumbered(1), AxisType::Reduce);

    let const_val = UOp::native_const(1.0f32);
    let reduce = const_val.reduce(smallvec![range1, range2], reduce_op);

    UOp::sink(vec![reduce])
}

/// Creates a double reduction pattern with global axes.
///
/// Matches Tinygrad's structure for test_double_reduce:
/// - Tensor shape: (g1, r1, g2, r2) e.g., (8, 128, 8, 128)
/// - Reduction over axes (1, 3) -> Result shape: (8, 8)
///
/// Generates:
/// ```text
/// SINK[
///   REDUCE(op=reduce_op)[
///     CONST(value=1.0),
///     RANGE(reduce_size1, type=Reduce),
///     RANGE(reduce_size2, type=Reduce)
///   ],
///   RANGE(global_size1, type=Global),
///   RANGE(global_size2, type=Global)
/// ]
/// ```
///
/// # Arguments
/// * `global_sizes` - Sizes of Global axes (output dimensions) e.g., [8, 8]
/// * `reduce_sizes` - Sizes of Reduce axes e.g., [128, 128]
/// * `reduce_op` - Type of reduction (Add, Max, etc.)
///
/// # Returns
/// A UOp representing the double reduction with globals
pub fn create_double_reduce_with_globals(global_sizes: &[i64], reduce_sizes: &[i64], reduce_op: ReduceOp) -> Arc<UOp> {
    use smallvec::smallvec;

    assert_eq!(global_sizes.len(), 2, "Expected 2 global dimensions for double reduce");
    assert_eq!(reduce_sizes.len(), 2, "Expected 2 reduce dimensions for double reduce");

    // Create Global axes (output dimensions)
    let mut all_axes = Vec::new();
    let mut axis_id = 0;

    for &size in global_sizes {
        let size_uop = UOp::native_const(size as i32 as i64);
        let axis = UOp::range_axis(size_uop, AxisId::Renumbered(axis_id), AxisType::Global);
        all_axes.push(axis);
        axis_id += 1;
    }

    // Create Reduce axes
    let mut reduce_axes = smallvec![];
    for &size in reduce_sizes {
        let size_uop = UOp::native_const(size as i32 as i64);
        let axis = UOp::range_axis(size_uop, AxisId::Renumbered(axis_id), AxisType::Reduce);
        reduce_axes.push(axis);
        axis_id += 1;
    }

    // Create reduction
    let const_val = UOp::native_const(1.0f32);
    let reduce = const_val.reduce(reduce_axes, reduce_op);

    // Build sink with reduce first, then global axes
    all_axes.insert(0, reduce);
    UOp::sink(all_axes)
}

/// Creates a multi-dimensional elementwise pattern.
///
/// Generates:
/// ```text
/// SINK[
///   CONST(value=1.0),
///   RANGE(sizes[0], axis_id=0, type=Global),
///   RANGE(sizes[1], axis_id=1, type=Global),
///   ...
/// ]
/// ```
///
/// # Arguments
/// * `sizes` - Sizes of each dimension
///
/// # Returns
/// A UOp representing the elementwise sink
pub fn create_elementwise_pattern(sizes: &[i64]) -> Arc<UOp> {
    let const_val = UOp::native_const(1.0f32);

    let mut ops = vec![const_val];

    for (axis_id, &size) in sizes.iter().enumerate() {
        let size_uop = UOp::native_const(size as i32 as i64);
        let range = UOp::range_axis(size_uop, AxisId::Renumbered(axis_id), AxisType::Global);
        ops.push(range);
    }

    UOp::sink(ops)
}

/// Asserts that the scheduler has the expected axis types in order.
///
/// # Arguments
/// * `scheduler` - The scheduler to check
/// * `expected` - Expected axis types in priority order
///
/// # Panics
/// If the axis types don't match expectations
pub fn assert_axes_equal(scheduler: &Scheduler, expected: &[AxisType]) {
    // Extract actual axis types from ranges
    let actual: Vec<AxisType> = scheduler
        .rngs()
        .iter()
        .map(|r| {
            if let Op::Range { axis_type, .. } = r.op() {
                *axis_type
            } else {
                panic!("Expected Range operation");
            }
        })
        .collect();

    assert_eq!(actual.len(), expected.len(), "Expected {} axes, got {}: {:?}", expected.len(), actual.len(), actual);

    for (i, (actual_type, expected_type)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_type, expected_type,
            "Axis {} type mismatch: expected {:?}, got {:?}",
            i, expected_type, actual_type
        );
    }
}

/// Asserts that the scheduler has the expected shape.
///
/// # Arguments
/// * `scheduler` - The scheduler to check
/// * `expected` - Expected dimension sizes (-1 for symbolic/unknown)
///
/// # Panics
/// If the shape doesn't match expectations
pub fn assert_shape_equal(scheduler: &Scheduler, expected: &[i64]) {
    let actual = scheduler.full_shape();
    assert_eq!(
        actual.len(),
        expected.len(),
        "Expected {} dimensions, got {}: {:?}",
        expected.len(),
        actual.len(),
        actual
    );

    for (i, (&actual_size, &expected_size)) in actual.iter().zip(expected.iter()).enumerate() {
        // Allow -1 to match any size (symbolic)
        if expected_size != -1 {
            assert_eq!(
                actual_size, expected_size,
                "Dimension {} size mismatch: expected {}, got {}",
                i, expected_size, actual_size
            );
        }
    }
}

/// Asserts that the scheduler has the expected number of axes of given types.
///
/// # Arguments
/// * `scheduler` - The scheduler to check
/// * `axis_type` - The axis type to count
/// * `expected_count` - Expected number of axes of this type
///
/// # Panics
/// If the count doesn't match expectations
pub fn assert_axis_count(scheduler: &Scheduler, axis_type: AxisType, expected_count: usize) {
    let actual_count = scheduler.axes_of(&[axis_type]).len();
    assert_eq!(actual_count, expected_count, "Expected {} {:?} axes, got {}", expected_count, axis_type, actual_count);
}

/// Asserts that applying an optimization succeeds.
///
/// # Arguments
/// * `scheduler` - The scheduler to modify
/// * `opt` - The optimization to apply
///
/// # Panics
/// If the optimization fails
#[allow(dead_code)]
pub fn assert_opt_succeeds<'a>(scheduler: &'a mut Scheduler, opt: &crate::optimizer::Opt) -> &'a mut Scheduler {
    crate::optimizer::apply_opt(scheduler, opt, true)
        .unwrap_or_else(|e| panic!("Expected optimization {:?} to succeed, but got error: {:?}", opt, e));
    scheduler
}

/// Asserts that applying an optimization fails.
///
/// # Arguments
/// * `scheduler` - The scheduler to modify
/// * `opt` - The optimization to apply
///
/// # Panics
/// If the optimization succeeds (when it should fail)
#[allow(dead_code)]
pub fn assert_opt_fails(scheduler: &mut Scheduler, opt: &crate::optimizer::Opt) {
    let result = crate::optimizer::apply_opt(scheduler, opt, true);
    assert!(result.is_err(), "Expected optimization {:?} to fail, but it succeeded", opt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Opt, Renderer};

    #[test]
    fn test_create_simple_reduce() {
        let reduce = create_simple_reduce(32, ReduceOp::Add);
        assert!(matches!(reduce.op(), Op::Sink { .. }));
    }

    #[test]
    fn test_create_matmul_pattern() {
        let matmul = create_matmul_pattern(16, 16, 16);
        assert!(matches!(matmul.op(), Op::Sink { .. }));
    }

    #[test]
    fn test_create_double_reduce() {
        let reduce = create_double_reduce(8, 8, ReduceOp::Add);
        assert!(matches!(reduce.op(), Op::Sink { .. }));
    }

    #[test]
    fn test_create_elementwise_pattern() {
        let elem = create_elementwise_pattern(&[10, 20, 30]);
        assert!(matches!(elem.op(), Op::Sink { .. }));
    }

    #[test]
    fn test_assert_axes_equal() {
        let reduce = create_simple_reduce(16, ReduceOp::Add);
        let renderer = Renderer::cpu();
        let scheduler = Scheduler::new(reduce, renderer);

        // Should have one Reduce axis initially (reduction patterns start with Reduce type)
        assert_axes_equal(&scheduler, &[AxisType::Reduce]);
    }

    #[test]
    fn test_assert_shape_equal() {
        let reduce = create_simple_reduce(16, ReduceOp::Add);
        let renderer = Renderer::cpu();
        let scheduler = Scheduler::new(reduce, renderer);

        // Should have shape [16]
        assert_shape_equal(&scheduler, &[16]);
    }

    #[test]
    fn test_assert_axis_count() {
        // Use elementwise pattern to test UPCAST on Global axes
        let elem = create_elementwise_pattern(&[16]);
        let renderer = Renderer::cpu();
        let mut scheduler = Scheduler::new(elem, renderer);

        // Elementwise patterns have Global axes
        assert_axis_count(&scheduler, AxisType::Global, 1);
        assert_axis_count(&scheduler, AxisType::Upcast, 0);

        // After upcast, Global axis splits into (Global, Upcast)
        crate::optimizer::apply_opt(&mut scheduler, &Opt::upcast(0, 4), true).unwrap();
        assert_axis_count(&scheduler, AxisType::Global, 1);
        assert_axis_count(&scheduler, AxisType::Upcast, 1);
    }
}
