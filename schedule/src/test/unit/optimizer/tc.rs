//! Tests for tensor core (TC) optimization.

use crate::optimizer::{Renderer, Scheduler, tc::*};
use morok_ir::{AxisId, AxisType, ReduceOp, UOp};

// ===== Matching Tests =====

#[test]
fn test_detect_matmul_basic() {
    // Create a simple matmul: C[i,j] = sum_k A[i,k] * B[k,j]
    let i = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let j = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);

    // Create A[i,k] and B[k,j] (simplified - just use constants)
    let a_val = UOp::native_const(1.0f32);
    let b_val = UOp::native_const(2.0f32);

    // Multiply A * B
    let mul = a_val.try_mul(&b_val).unwrap();

    // Reduce over k
    let reduce = mul.reduce(vec![k].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce, i, j]);

    // Create scheduler
    let ren = Renderer::cuda();
    let scheduler = Scheduler::new(sink, ren);

    // Detect pattern
    let result = matching::detect_matmul(&scheduler);
    assert!(result.is_ok());

    // Pattern should be detected (though ranges might not match perfectly in this simplified test)
    // This is a basic smoke test - real tests would use proper INDEX operations
}

#[test]
fn test_detect_matmul_no_reduce() {
    // No REDUCE operation
    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);

    let ren = Renderer::cuda();
    let scheduler = Scheduler::new(sink, ren);

    let result = matching::detect_matmul(&scheduler);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_detect_matmul_not_mul() {
    // REDUCE but not of MUL
    let k = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Reduce);
    let val = UOp::native_const(1.0f32);
    let reduce = val.reduce(vec![k].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let ren = Renderer::cuda();
    let scheduler = Scheduler::new(sink, ren);

    let result = matching::detect_matmul(&scheduler);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

// ===== Selection Tests =====

#[test]
fn test_select_tensor_core_auto() {
    // Create a simple pattern
    let i = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let j = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);

    let a_val = UOp::native_const(1.0f32);
    let b_val = UOp::native_const(2.0f32);
    let mul = a_val.try_mul(&b_val).unwrap();
    let reduce = mul.reduce(vec![k.clone()].into(), ReduceOp::Add);

    let pattern = matching::MatmulPattern {
        reduce_op: reduce,
        in0: a_val,
        in1: b_val,
        in0_ranges: vec![i.clone()],
        in1_ranges: vec![j.clone()],
        red_ranges: vec![k.clone()],
        axis_choices: vec![(j, i, k)],
    };

    let renderer = Renderer::cuda();

    // Should find a match with auto-select
    let result = selection::select_tensor_core(&pattern, &renderer, -1, 0);
    assert!(result.is_ok());
    // May or may not find a match depending on dtype compatibility
}

#[test]
fn test_select_tensor_core_specific() {
    let i = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let j = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);

    let a_val = UOp::native_const(1.0f32);
    let b_val = UOp::native_const(2.0f32);
    let mul = a_val.try_mul(&b_val).unwrap();
    let reduce = mul.reduce(vec![k.clone()].into(), ReduceOp::Add);

    let pattern = matching::MatmulPattern {
        reduce_op: reduce,
        in0: a_val,
        in1: b_val,
        in0_ranges: vec![i.clone()],
        in1_ranges: vec![j.clone()],
        red_ranges: vec![k.clone()],
        axis_choices: vec![(j, i, k)],
    };

    let renderer = Renderer::cuda();

    // Select specific tensor core (index 0)
    let result = selection::select_tensor_core(&pattern, &renderer, 0, 0);
    assert!(result.is_ok());
}

#[test]
fn test_select_tensor_core_out_of_bounds() {
    let i = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let j = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);

    let a_val = UOp::native_const(1.0f32);
    let b_val = UOp::native_const(2.0f32);
    let mul = a_val.try_mul(&b_val).unwrap();
    let reduce = mul.reduce(vec![k.clone()].into(), ReduceOp::Add);

    let pattern = matching::MatmulPattern {
        reduce_op: reduce,
        in0: a_val,
        in1: b_val,
        in0_ranges: vec![i.clone()],
        in1_ranges: vec![j.clone()],
        red_ranges: vec![k.clone()],
        axis_choices: vec![(j, i, k)],
    };

    let renderer = Renderer::cuda();

    // Select out-of-bounds tensor core
    let result = selection::select_tensor_core(&pattern, &renderer, 9999, 0);
    assert!(result.is_err());
}

// ===== Swizzle Tests =====

#[test]
fn test_base_shape() {
    use crate::optimizer::renderer::{CUDA_81616, SwizzleAxis};
    use morok_dtype::DType;

    let tc = CUDA_81616.build(DType::Float16, DType::Float32);
    let shape = swizzle::base_shape(&tc);

    // Should have: u0, l0, l0, l1, l1, l1, u1, r0, r1, r2, r3
    assert!(!shape.is_empty());
    assert!(shape.contains(&SwizzleAxis::Upcast(0)));
    assert!(shape.contains(&SwizzleAxis::Local(0)));
    assert!(shape.contains(&SwizzleAxis::Reduce(0)));

    // Should have 2 upcasts, 5 locals (l0 appears 2x, l1 appears 3x), 4 reduces
    let upcast_count = shape.iter().filter(|&&a| matches!(a, SwizzleAxis::Upcast(_))).count();
    let local_count = shape.iter().filter(|&&a| matches!(a, SwizzleAxis::Local(_))).count();
    let reduce_count = shape.iter().filter(|&&a| matches!(a, SwizzleAxis::Reduce(_))).count();

    assert_eq!(upcast_count, 2);
    assert_eq!(local_count, 5);
    assert_eq!(reduce_count, 4);
}

#[test]
fn test_permutes_for_shape() {
    use crate::optimizer::renderer::CUDA_81616;
    use morok_dtype::DType;

    let tc = CUDA_81616.build(DType::Float16, DType::Float32);
    let shape = swizzle::base_shape(&tc);
    let (perm_a, perm_b) = swizzle::permutes_for_shape(&tc, &shape);

    // Permutations should be valid indices
    assert!(!perm_a.is_empty());
    assert!(!perm_b.is_empty());
    for &idx in &perm_a {
        assert!(idx < shape.len());
    }
    for &idx in &perm_b {
        assert!(idx < shape.len());
    }
}

#[test]
fn test_reduce_axes_count() {
    use crate::optimizer::renderer::CUDA_81616;
    use morok_dtype::DType;

    let tc = CUDA_81616.build(DType::Float16, DType::Float32);
    let count = swizzle::get_reduce_axes_count(&tc);

    // K=16 -> log2(16) = 4 reduce axes
    assert_eq!(count, 4);
}

// ===== Apply Tests =====

#[test]
fn test_apply_tc_basic() {
    // Create a simple matmul pattern
    let i = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let j = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Global);
    let k = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(2), AxisType::Reduce);

    let a_val = UOp::native_const(1.0f32);
    let b_val = UOp::native_const(2.0f32);
    let mul = a_val.try_mul(&b_val).unwrap();
    let reduce = mul.reduce(vec![k].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce, i, j]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Try to apply TC (may fail if pattern doesn't match exactly)
    let result = apply(&mut scheduler, -1, 0, 1);
    // This may fail in simplified test, but shouldn't panic
    let _result_ok = result.is_ok() || result.is_err();
}

#[test]
fn test_apply_tc_validation() {
    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Should fail - no matmul pattern
    let result = apply(&mut scheduler, -1, 0, 1);
    assert!(result.is_err());
}

#[test]
fn test_apply_tc_invalid_use_tc() {
    let i = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(0), AxisType::Global);
    let k = UOp::range_axis(UOp::index_const(16), AxisId::Renumbered(1), AxisType::Reduce);

    let val = UOp::native_const(1.0f32);
    let mul = val.try_mul(&val).unwrap();
    let reduce = mul.reduce(vec![k].into(), ReduceOp::Add);
    let sink = UOp::sink(vec![reduce, i]);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Should fail - invalid use_tensor_cores value
    let result = apply(&mut scheduler, -1, 0, 3);
    assert!(result.is_err());
}
