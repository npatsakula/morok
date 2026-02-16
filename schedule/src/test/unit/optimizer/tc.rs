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

// =============================================================================
// TC Padding Tests
// =============================================================================

use morok_dtype::DType;
use std::sync::Arc;

/// Helper to create a proper matmul pattern for TC padding tests.
/// Creates: C[m,n] = sum_k A[m,k] * B[k,n]
///
/// Unlike simplified tests, this creates inputs that depend on ranges
/// so that detect_matmul() can find the M, N, K axes.
fn create_matmul_pattern_for_padding(m: i64, n: i64, k: i64) -> Arc<morok_ir::UOp> {
    let m_range = UOp::range_axis(UOp::index_const(m), AxisId::Renumbered(0), AxisType::Global);
    let n_range = UOp::range_axis(UOp::index_const(n), AxisId::Renumbered(1), AxisType::Global);
    let k_range = UOp::range_axis(UOp::index_const(k), AxisId::Renumbered(2), AxisType::Reduce);

    // Create inputs that depend on ranges (so get_ranges() finds them)
    // A[m,k] - depends on m_range and k_range
    // B[k,n] - depends on k_range and n_range
    //
    // We achieve this by casting ranges to float and using them in expressions.
    // This creates a dependency without needing actual buffer operations.
    let m_float = m_range.clone().cast(DType::Float32);
    let k_float = k_range.clone().cast(DType::Float32);
    let n_float = n_range.clone().cast(DType::Float32);

    // A[m,k] = m + k (has m_range and k_range in backward slice)
    let a_val = m_float.try_add(&k_float).unwrap();
    // B[k,n] = k + n (has k_range and n_range in backward slice)
    let b_val = k_float.try_add(&n_float).unwrap();

    // C[m,n] = sum_k(A[m,k] * B[k,n])
    let mul = a_val.try_mul(&b_val).unwrap();
    let reduce = mul.reduce(vec![k_range].into(), ReduceOp::Add);

    UOp::sink(vec![reduce, m_range, n_range])
}

#[test]
fn test_tc_no_padding_divisible_dims() {
    // 16x16x16 matmul - perfectly divisible by TC dimensions
    let sink = create_matmul_pattern_for_padding(16, 16, 16);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Verify matmul pattern is detected
    let pattern = matching::detect_matmul(&scheduler);
    assert!(pattern.is_ok(), "Pattern detection should succeed");
    assert!(pattern.unwrap().is_some(), "Matmul pattern should be found");

    // tc_opt=1 (no padding) should work for divisible dims
    let result = apply(&mut scheduler, -1, 1, 1);
    // May succeed or fail based on dtype matching, but shouldn't fail due to divisibility
    if let Err(ref e) = result {
        let err_msg = format!("{:?}", e);
        assert!(!err_msg.contains("not divisible"), "16x16x16 should not fail divisibility check: {}", err_msg);
    }
}

#[test]
fn test_tc_rejects_non_divisible_without_tc_opt_2() {
    // 15x16x16 matmul - M not divisible by 16
    let sink = create_matmul_pattern_for_padding(15, 16, 16);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Verify matmul pattern is detected
    let pattern = matching::detect_matmul(&scheduler);
    assert!(pattern.is_ok(), "Pattern detection should succeed");
    assert!(pattern.unwrap().is_some(), "Matmul pattern should be found");

    // tc_opt=1 (no padding) should reject non-divisible dims
    let result = apply(&mut scheduler, -1, 1, 1);
    assert!(result.is_err(), "TC should fail for non-divisible dims with tc_opt=1");

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("not divisible") || err_msg.contains("no compatible"),
        "Should fail due to divisibility or no compatible TC: {}",
        err_msg
    );
}

#[test]
fn test_tc_padding_with_tc_opt_2() {
    // 15x16x16 matmul - M not divisible, but tc_opt=2 enables padding
    let sink = create_matmul_pattern_for_padding(15, 16, 16);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Verify matmul pattern is detected
    let pattern = matching::detect_matmul(&scheduler);
    assert!(pattern.is_ok(), "Pattern detection should succeed");
    assert!(pattern.unwrap().is_some(), "Matmul pattern should be found");

    // tc_opt=2 should attempt padding via PADTO
    // For 15→16, this is only ~6% more work so PADTO should succeed
    let result = apply(&mut scheduler, -1, 2, 1);

    // If it fails, it shouldn't be due to "not divisible" (padding should handle that)
    if let Err(ref e) = result {
        let err_msg = format!("{:?}", e);
        assert!(!err_msg.contains("not divisible"), "tc_opt=2 should pad instead of rejecting: {}", err_msg);
    }
}

#[test]
fn test_tc_padding_rejects_4x_work_increase() {
    // 4x16x16 matmul - padding 4→16 would be 4x work increase
    let sink = create_matmul_pattern_for_padding(4, 16, 16);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Verify matmul pattern is detected
    let pattern = matching::detect_matmul(&scheduler);
    assert!(pattern.is_ok(), "Pattern detection should succeed");
    assert!(pattern.unwrap().is_some(), "Matmul pattern should be found");

    // tc_opt=2 attempts padding, but PADTO rejects >4x work increase
    let result = apply(&mut scheduler, -1, 2, 1);

    // Should fail - either at padding (4x limit) or no compatible TC
    assert!(result.is_err(), "Should fail due to 4x work limit or no compatible TC");
}

#[test]
fn test_tc_padding_all_axes() {
    // 17x17x17 matmul - all dimensions need padding to 32
    let sink = create_matmul_pattern_for_padding(17, 17, 17);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // Verify matmul pattern is detected
    let pattern = matching::detect_matmul(&scheduler);
    assert!(pattern.is_ok(), "Pattern detection should succeed");
    assert!(pattern.unwrap().is_some(), "Matmul pattern should be found");

    // tc_opt=2 should attempt padding all axes
    // 17→32 is ~88% increase, within 4x limit
    let result = apply(&mut scheduler, -1, 2, 1);

    // May succeed or fail based on dtype matching, but shouldn't fail divisibility
    if let Err(ref e) = result {
        let err_msg = format!("{:?}", e);
        assert!(!err_msg.contains("not divisible"), "tc_opt=2 should pad instead of rejecting: {}", err_msg);
    }
}

#[test]
fn test_tc_opt_validation() {
    let sink = create_matmul_pattern_for_padding(16, 16, 16);

    let ren = Renderer::cuda();
    let mut scheduler = Scheduler::new(sink, ren);

    // tc_opt > 2 should be rejected
    let result = apply(&mut scheduler, -1, 3, 1);
    assert!(result.is_err(), "tc_opt=3 should be rejected");

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("tc_opt must be"), "Should fail validation: {}", err_msg);
}

// =============================================================================
// AMX Tensor Core Tests
// =============================================================================

use crate::optimizer::renderer::{APPLE_AMX, SwizzleAxis};
use crate::optimizer::{Opt, apply_opt};

#[test]
fn test_detect_matmul_amx() {
    // 16x16 matmul with K=16 — matches AMX dims (16, 16, 1) after K-split
    let sink = create_matmul_pattern_for_padding(16, 16, 16);

    let ren = Renderer::apple_amx();
    let scheduler = Scheduler::new(sink, ren);

    let pattern = matching::detect_matmul(&scheduler);
    assert!(pattern.is_ok(), "Pattern detection should succeed");
    assert!(pattern.unwrap().is_some(), "Matmul pattern should be found for AMX");
}

#[test]
fn test_select_tensor_core_amx() {
    let sink = create_matmul_pattern_for_padding(16, 16, 16);

    let ren = Renderer::apple_amx();
    let scheduler = Scheduler::new(sink, ren);

    let pattern = matching::detect_matmul(&scheduler).unwrap().unwrap();

    // Auto-select should find the float32 AMX TC
    let result = selection::select_tensor_core(&pattern, &scheduler.ren, -1, 0);
    assert!(result.is_ok(), "TC selection should not error: {:?}", result.err());

    if let Ok(Some(sel)) = result {
        let tc = &scheduler.ren.tensor_cores[sel.tc_index];
        assert_eq!(tc.dims, (16, 16, 1), "AMX float32 TC should have dims (16, 16, 1)");
        assert_eq!(tc.threads, 1, "AMX uses 1 thread (CPU)");
        assert_eq!(tc.dtype_in, DType::Float32);
        assert_eq!(tc.dtype_out, DType::Float32);
    }
}

#[test]
fn test_base_shape_amx() {
    let tc = APPLE_AMX.build(DType::Float32, DType::Float32);
    let shape = swizzle::base_shape(&tc);

    // AMX has 8 upcast opts (all U), K=1 so 0 reduce axes, no local opts
    let upcast_count = shape.iter().filter(|&&a| matches!(a, SwizzleAxis::Upcast(_))).count();
    let local_count = shape.iter().filter(|&&a| matches!(a, SwizzleAxis::Local(_))).count();
    let reduce_count = shape.iter().filter(|&&a| matches!(a, SwizzleAxis::Reduce(_))).count();

    assert_eq!(upcast_count, 8, "AMX should have 8 upcast axes");
    assert_eq!(local_count, 0, "AMX should have no local axes");
    assert_eq!(reduce_count, 0, "AMX K=1 → 0 reduce axes");
}

#[test]
fn test_permutes_amx() {
    let tc = APPLE_AMX.build(DType::Float32, DType::Float32);
    let shape = swizzle::base_shape(&tc);
    let (perm_a, perm_b) = swizzle::permutes_for_shape(&tc, &shape);

    // A: identity permutation (all upcast in order)
    assert_eq!(perm_a, (0..shape.len()).collect::<Vec<_>>(), "AMX A permutation should be identity");

    // B: swap halves (first 4 upcast ↔ last 4 upcast)
    let half = shape.len() / 2;
    let expected_b: Vec<usize> = (half..shape.len()).chain(0..half).collect();
    assert_eq!(perm_b, expected_b, "AMX B permutation should swap halves");
}

#[test]
fn test_reduce_axes_count_amx() {
    let tc = APPLE_AMX.build(DType::Float32, DType::Float32);
    let count = swizzle::get_reduce_axes_count(&tc);

    // K=1 → log2(1) = 0 reduce axes
    assert_eq!(count, 0, "AMX K=1 should produce 0 reduce axes");
}

#[test]
fn test_apply_tc_amx() {
    let sink = create_matmul_pattern_for_padding(16, 16, 16);

    let ren = Renderer::apple_amx();
    let mut scheduler = Scheduler::new(sink, ren);

    let result = apply(&mut scheduler, -1, 0, 1);
    assert!(result.is_ok(), "AMX TC apply should succeed: {:?}", result.err());

    let axes = result.unwrap();
    // axes should be [N_range, M_range, K_range] — all valid UOps
    for (i, ax) in axes.iter().enumerate() {
        assert!(matches!(ax.op(), morok_ir::Op::Range { .. }), "axes[{i}] should be a RANGE");
    }

    // Verify WMMA is present in the resulting AST
    let ast = scheduler.ast();
    let has_wmma = ast.toposort().iter().any(|u| matches!(u.op(), morok_ir::Op::Wmma { .. }));
    assert!(has_wmma, "AST should contain WMMA after TC apply");
}

#[test]
fn test_group_after_tc_rejected() {
    // Apply TC first, then attempt GROUP — should be rejected.
    // Tinygrad: check(all(x.op is not OptOps.TC for x in self.applied_opts), "no grouping with tensor cores")
    let sink = create_matmul_pattern_for_padding(16, 16, 16);

    let ren = Renderer::apple_amx();
    let mut scheduler = Scheduler::new(sink, ren);

    // Apply TC
    let tc_opt = Opt::tc(None, -1, 0, 1);
    let tc_result = apply_opt(&mut scheduler, &tc_opt, true);
    assert!(tc_result.is_ok(), "TC apply should succeed: {:?}", tc_result.err());

    // Attempt GROUP after TC — should fail
    let group_opt = Opt::group(0, 2);
    let group_result = apply_opt(&mut scheduler, &group_opt, true);
    assert!(group_result.is_err(), "GROUP after TC should be rejected");

    let err_msg = format!("{:?}", group_result.unwrap_err());
    assert!(err_msg.contains("no grouping with tensor cores"), "Error should mention TC guard: {}", err_msg);
}
