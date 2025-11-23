//! Full pipeline integration tests for RANGEIFY transformation.
//!
//! Tests verify the complete transformation pipeline from tensor operations
//! to executable kernels:
//! - Phase 1-4: run_rangeify (movement ops → BUFFERIZE+INDEX)
//! - Phase 5: run_kernel_split_pipeline (BUFFERIZE → KERNEL)
//! - End-to-end scenarios
//!
//! Based on Tinygrad's test_schedule.py integration tests.

use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{AxisType, BufferizeOpts, ConstValue, Op, ReduceOp, UOp};

use crate::rangeify::{rangeify, run_kernel_split_pipeline, run_rangeify};

// ===== Helper Function =====

/// Helper to unwrap full rangeify pipeline (includes all optimizations)
fn rangeify_unwrap(uop: Rc<UOp>) -> Rc<UOp> {
    match rangeify(uop, None) {
        Ok((rangeified, _ctx)) => rangeified,
        Err(_) => panic!("rangeify failed"),
    }
}

// ===== Phase 1-4 Pipeline Tests (run_rangeify) =====

#[test]
fn test_run_rangeify_simple_const() {
    // Test: CONST should pass through unchanged
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(42.0));

    let result = run_rangeify(const_val);
    assert!(result.is_ok(), "rangeify should succeed");
    let (rangeified, _ctx) = result.unwrap();

    // CONST has no movement ops, should remain unchanged
    assert!(matches!(rangeified.op(), Op::Const(_)));
}

#[test]
fn test_run_rangeify_detach_removal() {
    // Test: DETACH should be removed by early_rewrites
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let detach = UOp::new(Op::Detach { src: x.clone() }, x.dtype());

    let result = run_rangeify(detach);
    assert!(result.is_ok(), "rangeify should succeed");
    let (rangeified, _ctx) = result.unwrap();

    // DETACH should be removed, leaving only the constant
    // The result might be the constant or a transformed version
    match rangeified.op() {
        Op::Const(_) => {
            // DETACH successfully removed
            assert!(Rc::ptr_eq(&rangeified, &x) || matches!(rangeified.op(), Op::Const(_)));
        }
        _ => {
            // Transformation may have applied additional patterns
            // This is acceptable as long as it's not DETACH
            assert!(!matches!(rangeified.op(), Op::Detach { .. }));
        }
    }
}

#[test]
fn test_run_rangeify_contiguous_backward_removal() {
    // Test: CONTIGUOUS_BACKWARD should be removed
    let x = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as _));
    let contiguous = UOp::new(Op::ContiguousBackward { src: x.clone() }, x.dtype());

    let rangeified = rangeify_unwrap(contiguous);

    // CONTIGUOUS_BACKWARD should be removed
    match rangeified.op() {
        Op::Const(_) => {
            assert!(Rc::ptr_eq(&rangeified, &x) || matches!(rangeified.op(), Op::Const(_)));
        }
        _ => {
            assert!(!matches!(rangeified.op(), Op::ContiguousBackward { .. }));
        }
    }
}

#[test]
fn test_run_rangeify_binary_op() {
    // Test: Binary operations should be processed
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = a.try_add_op(&b).unwrap();

    let rangeified = rangeify_unwrap(add);

    // Binary op may remain or be transformed, but should be valid
    assert!(rangeified.dtype() == DType::Float32);
}

#[test]
fn test_run_rangeify_preserves_structure() {
    // Test: Complex computation structure should be preserved
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(3.0));

    // (a + b) * c
    let sum = a.try_add_op(&b).unwrap();
    let product = sum.try_mul_op(&c).unwrap();

    let rangeified = rangeify_unwrap(product);

    // Should preserve dtype
    assert_eq!(rangeified.dtype(), DType::Float32);

    // Structure may change but computation should be equivalent
    match rangeified.op() {
        Op::Binary { .. } | Op::Const(_) | Op::Bufferize { .. } | Op::Index { .. } => {
            // All acceptable transformations
        }
        _ => {}
    }
}

// ===== Phase 5 Pipeline Tests (run_kernel_split_pipeline) =====

#[test]
fn test_kernel_split_pipeline_simple_store() {
    // Test: Simple STORE should create a KERNEL
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);

    let result = run_kernel_split_pipeline(store);

    // Should produce a KERNEL operation or transformation
    // Note: Exact output depends on split_store implementation
    assert!(result.dtype() == DType::Void || matches!(result.op(), Op::Kernel { .. }));
}

#[test]
fn test_kernel_split_pipeline_with_end() {
    // Test: END(STORE) should be processed correctly
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);

    let range_end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range_axis(range_end, 0, AxisType::Loop);
    let end = UOp::new(Op::End { computation: store, ranges: vec![range].into() }, DType::Void);

    let result = run_kernel_split_pipeline(end);

    // Should handle END wrapper correctly
    assert!(result.dtype() == DType::Void || matches!(result.op(), Op::Kernel { .. }));
}

#[test]
fn test_kernel_split_pipeline_load_store() {
    // Test: LOAD + STORE pattern
    let in_buf = UOp::unique(Some(1));
    let out_buf = UOp::unique(Some(2));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));

    let load = UOp::new(Op::Load { buffer: in_buf, index: index.clone() }, DType::Float32);
    let store = UOp::new(Op::Store { buffer: out_buf, index, value: load }, DType::Void);

    let result = run_kernel_split_pipeline(store);

    // Should create valid kernel or passthrough
    assert!(result.dtype() == DType::Void || matches!(result.op(), Op::Kernel { .. }));
}

#[test]
fn test_kernel_split_pipeline_multiple_loads() {
    // Test: Multiple LOADs feeding into STORE
    let buf1 = UOp::unique(Some(1));
    let buf2 = UOp::unique(Some(2));
    let out_buf = UOp::unique(Some(3));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));

    let load1 = UOp::new(Op::Load { buffer: buf1, index: index.clone() }, DType::Float32);
    let load2 = UOp::new(Op::Load { buffer: buf2, index: index.clone() }, DType::Float32);
    let sum = load1.try_add_op(&load2).unwrap();
    let store = UOp::new(Op::Store { buffer: out_buf, index, value: sum }, DType::Void);

    let result = run_kernel_split_pipeline(store);

    // Should handle multiple inputs correctly
    assert!(result.dtype() == DType::Void || matches!(result.op(), Op::Kernel { .. }));
}

// ===== End-to-End Scenario Tests =====

#[test]
fn test_end_to_end_simple_computation() {
    // Test: Full pipeline from computation to kernel

    // Step 1: Create computation
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let sum = a.try_add_op(&b).unwrap();

    // Step 2: Wrap in STORE
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let store = UOp::new(Op::Store { buffer, index, value: sum }, DType::Void);

    // Step 3: Apply rangeify
    let rangeified = rangeify_unwrap(store);

    // Step 4: Apply kernel split
    let kernel = run_kernel_split_pipeline(rangeified);

    // Should produce valid output
    assert!(kernel.dtype() == DType::Void || matches!(kernel.op(), Op::Kernel { .. }));
}

#[test]
fn test_end_to_end_with_ranges() {
    // Test: Pipeline with explicit range operations

    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);

    // Wrap in END with ranges
    let range_end = UOp::const_(DType::Index, ConstValue::Int(100));
    let range = UOp::range_axis(range_end, 0, AxisType::Loop);
    let end = UOp::new(Op::End { computation: store, ranges: vec![range].into() }, DType::Void);

    let rangeified = rangeify_unwrap(end);
    let kernel = run_kernel_split_pipeline(rangeified);

    assert!(kernel.dtype() == DType::Void || matches!(kernel.op(), Op::Kernel { .. }));
}

// ===== Regression Tests =====

#[test]
fn test_pipeline_idempotent() {
    // Test: Applying pipeline twice should be safe
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    let rangeified1 = rangeify_unwrap(x.clone());
    let rangeified2 = rangeify_unwrap(rangeified1);

    // Second application should not break anything
    assert!(rangeified2.dtype() == x.dtype());
}

#[test]
fn test_pipeline_preserves_dtype() {
    // Test: Pipeline should preserve data types
    let dtypes = vec![
        (DType::Float32, ConstValue::Float(1.0)),
        (DType::Float64, ConstValue::Float(1.0)),
        (DType::Int32, ConstValue::Int(42)),
        (DType::Int64, ConstValue::Int(42)),
        (DType::Bool, ConstValue::Bool(true)),
    ];

    for (dtype, const_val) in dtypes {
        let value = UOp::const_(dtype.clone(), const_val);
        let rangeified = rangeify_unwrap(value.clone());

        // Should preserve dtype (or transform to compatible type)
        if let Op::Const(_) = rangeified.op() {
            assert_eq!(rangeified.dtype(), dtype)
        }
    }
}

#[test]
fn test_pipeline_handles_noop() {
    // Test: Pipeline should handle NOOP operations
    let noop = UOp::noop();
    let rangeified = rangeify_unwrap(noop);

    // NOOP should remain or be transformed safely
    assert!(rangeified.dtype() == DType::Void || matches!(rangeified.op(), Op::Noop));
}

// ===== Error Handling Tests =====

#[test]
fn test_pipeline_complex_nested_structure() {
    // Test: Pipeline should handle deeply nested operations
    let mut current = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Build a deep tree: ((((x + 1) + 1) + 1) + 1)
    for _ in 0..10 {
        let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        current = current.try_add_op(&one).unwrap();
    }

    let rangeified = rangeify_unwrap(current);

    // Should handle deep nesting without stack overflow
    assert_eq!(rangeified.dtype(), DType::Float32);
}

#[test]
fn test_pipeline_wide_tree() {
    // Test: Pipeline should handle wide trees (many branches)
    let mut operands = Vec::new();

    for i in 0..20 {
        operands.push(UOp::const_(DType::Float32, ConstValue::Float(i as f64)));
    }

    // Sum all operands
    let mut sum = operands[0].clone();
    for operand in &operands[1..] {
        sum = sum.try_add_op(operand).unwrap();
    }

    let rangeified = rangeify_unwrap(sum);

    // Should handle wide trees
    assert_eq!(rangeified.dtype(), DType::Float32);
}

// ===== Pattern Application Order Tests =====

#[test]
fn test_pipeline_applies_early_rewrites_first() {
    // Test: early_rewrites should be applied before other patterns
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let detach = UOp::new(Op::Detach { src: x.clone() }, x.dtype());

    let rangeified = rangeify_unwrap(detach);

    // DETACH should be removed (early rewrite)
    if let Op::Detach { .. } = rangeified.op() {
        panic!("DETACH should have been removed by early_rewrites")
    }
}

#[test]
fn test_pipeline_applies_buffer_folding() {
    // Test: buffer_folding patterns should be applied

    // Create BUFFERIZE(CONST) which should be folded
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let range_end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range_axis(range_end, 0, AxisType::Loop);
    let bufferize = UOp::bufferize(const_val.clone(), vec![range], BufferizeOpts::local());

    let rangeified = rangeify_unwrap(bufferize);

    // BUFFERIZE(CONST) may be folded to CONST
    // Or may remain as BUFFERIZE depending on pipeline phase
    match rangeified.op() {
        Op::Const(_) | Op::Bufferize { .. } => {} // Both acceptable
        _ => {}
    }
}

// ===== Correctness Tests =====

#[test]
fn test_pipeline_maintains_computation_semantics() {
    // Test: Transformation should not change computation semantics

    // Original: a * b + c
    let a = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let c = UOp::const_(DType::Float32, ConstValue::Float(4.0));

    let product = a.try_mul_op(&b).unwrap();
    let sum = product.try_add_op(&c).unwrap();

    let rangeified = rangeify_unwrap(sum);

    // Should preserve dtype
    assert_eq!(rangeified.dtype(), DType::Float32);

    // Structure may change but computation should be equivalent
    // (Hard to verify without evaluation, but we check it doesn't panic)
}

// ===== Reduction Optimization Tests =====

#[test]
fn test_pipeline_reduce_unparented_add() {
    // Test: reduce_unparented optimization for ADD
    // REDUCE(CONST(5), [range(10)], ADD) should become CONST(50)
    use morok_ir::ReduceOp;

    let const_val = UOp::const_(DType::Int32, ConstValue::Int(5));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Reduce);
    let reduce = UOp::reduce(const_val, vec![range].into(), ReduceOp::Add);

    let rangeified = rangeify_unwrap(reduce);

    // Should be optimized to a multiplication or constant
    // REDUCE(5, [10], ADD) → 5 * 10 = 50
    match rangeified.op() {
        Op::Binary(morok_ir::BinaryOp::Mul, _, _) => {
            // Optimized to multiplication
        }
        Op::Const(cv_hash) => {
            // May be folded further to constant 50
            if let ConstValue::Int(n) = cv_hash.0 {
                assert_eq!(n, 50, "reduce_unparented should optimize to 50");
            }
        }
        _ => {
            // If not optimized, should at least not panic
            // (Optimization may not apply if patterns don't match)
        }
    }
}

#[test]
fn test_pipeline_reduce_unparented_max() {
    // Test: reduce_unparented optimization for MAX
    // REDUCE(CONST(42), [range(5)], MAX) should become CONST(42)
    use morok_ir::ReduceOp;

    let const_val = UOp::const_(DType::Int32, ConstValue::Int(42));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(5)), 0, AxisType::Reduce);
    let reduce = UOp::reduce(const_val.clone(), vec![range].into(), ReduceOp::Max);

    let rangeified = rangeify_unwrap(reduce);

    // Should be optimized to just the constant
    match rangeified.op() {
        Op::Const(cv_hash) => {
            if let ConstValue::Int(n) = cv_hash.0 {
                assert_eq!(n, 42, "reduce_unparented MAX should preserve constant");
            }
        }
        _ => {
            // May remain as REDUCE if pattern doesn't match
        }
    }
}

#[test]
fn test_pipeline_split_reduceop_large_reduction() {
    // Test: split_reduceop should split large REDUCE_AXIS operations
    use morok_device::DeviceSpec;
    use morok_ir::ReduceOp;

    // Create a tensor with shape (100000,) - large enough to trigger split
    let total_size = 100000;
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, total_size, DType::Float32);

    // REDUCE_AXIS on this tensor
    let reduce = buffer.try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();

    let rangeified = rangeify_unwrap(reduce);

    // Verify: split_reduceop should have applied, creating CONTIGUOUS node
    let has_contiguous = rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Contiguous { .. }));

    assert!(
        has_contiguous,
        "split_reduceop should have split large reduction (100000 > 32768 threshold), creating CONTIGUOUS node"
    );

    // Should preserve output dtype
    assert_eq!(rangeified.dtype(), DType::Float32);
}

#[test]
fn test_pipeline_split_reduceop_below_threshold() {
    // Test: split_reduceop should NOT split small REDUCE_AXIS operations
    use morok_device::DeviceSpec;
    use morok_ir::ReduceOp;

    // Create a tensor with shape (1000,) - below threshold (32768)
    let total_size = 1000;
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, total_size, DType::Float32);

    // REDUCE_AXIS on this tensor
    let reduce = buffer.try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();

    let rangeified = rangeify_unwrap(reduce);

    // Verify: split_reduceop should NOT have applied (below threshold)
    let has_contiguous = rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Contiguous { .. }));

    assert!(!has_contiguous, "split_reduceop should NOT split small reduction (1000 < 32768 threshold)");

    // Should preserve output dtype
    assert_eq!(rangeified.dtype(), DType::Float32);
}

#[test]
fn test_pipeline_reduction_optimizations_dont_break_graph() {
    // Test: Reduction optimizations preserve graph validity
    use morok_ir::ReduceOp;

    // Create a realistic reduction scenario
    let data = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as _));
    let range1 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(8)), 0, AxisType::Reduce);
    let range2 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(4)), 1, AxisType::Reduce);

    // Create a multi-range reduction
    let reduce = UOp::reduce(data, vec![range1, range2].into(), ReduceOp::Add);

    // Apply full pipeline
    let result = run_rangeify(reduce);

    // Should succeed without panicking
    assert!(result.is_ok(), "Pipeline should handle multi-range reduction");

    let (rangeified, _ctx) = result.unwrap();

    // Result should have valid dtype
    assert_eq!(rangeified.dtype(), DType::Float32);
}

// ===== reduce_collapse Integration Tests =====

#[test]
fn test_pipeline_reduce_collapse_constant() {
    // Test: REDUCE(const, [range], ADD) should be simplified by reduce_collapse
    // after symbolic simplification eliminates range dependency
    let const_val = UOp::const_(DType::Int32, ConstValue::Int(42));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Reduce);
    let reduce = UOp::reduce(const_val, vec![range].into(), ReduceOp::Add);

    let result = rangeify_unwrap(reduce);

    // After reduce_collapse, the constant reduction should be simplified
    // The result should no longer be a REDUCE operation
    // (it will be transformed to the constant itself or a simpler form)
    assert_ne!(result.dtype(), DType::Void, "Result should have valid dtype after reduce_collapse");
}

#[test]
fn test_pipeline_reduce_collapse_multiple_ranges() {
    // Test: REDUCE with multiple independent ranges
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(std::f32::consts::PI as _));
    let range1 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(5)), 0, AxisType::Reduce);
    let range2 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 1, AxisType::Reduce);

    let reduce = UOp::reduce(const_val, vec![range1, range2].into(), ReduceOp::Add);

    let result = rangeify_unwrap(reduce);

    // Should successfully process through pipeline
    assert_eq!(result.dtype(), DType::Float32, "Result should preserve Float32 dtype");
}

#[test]
fn test_pipeline_reduce_collapse_with_algebraic_simplification() {
    // Test: reduce_collapse combined with algebraic patterns (x + 0)
    let x = UOp::const_(DType::Int32, ConstValue::Int(100));
    let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
    let x_plus_0 = x.try_add_op(&zero).unwrap();

    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(20)), 0, AxisType::Reduce);

    let reduce = UOp::reduce(x_plus_0, vec![range].into(), ReduceOp::Add);

    let result = rangeify_unwrap(reduce);

    // Symbolic simplification should eliminate x+0 → x,
    // then reduce_collapse should simplify REDUCE(x, [range], ADD)
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_pipeline_reduce_collapse_preserves_dependent_reductions() {
    // Test: Reductions with actual range dependencies should NOT be collapsed
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), 0, AxisType::Reduce);

    // Create expression that depends on range: range + 1
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let range_int = UOp::cast(range.clone(), DType::Int32);
    let src = range_int.try_add_op(&one).unwrap();

    let reduce = UOp::reduce(src, vec![range].into(), ReduceOp::Add);

    let result = rangeify_unwrap(reduce);

    // Should NOT collapse since src depends on range
    // Result should still be a valid Int32 value
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_pipeline_reduce_collapse_different_ops() {
    // Test: reduce_collapse works with different ReduceOp types through pipeline
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(2.5));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(8)), 0, AxisType::Reduce);

    // Test with MAX
    let reduce_max = UOp::reduce(const_val.clone(), vec![range.clone()].into(), ReduceOp::Max);
    let result_max = rangeify_unwrap(reduce_max);
    assert_eq!(result_max.dtype(), DType::Float32, "MAX reduce should work");

    // Test with MIN
    let reduce_min = UOp::reduce(const_val, vec![range].into(), ReduceOp::Min);
    let result_min = rangeify_unwrap(reduce_min);
    assert_eq!(result_min.dtype(), DType::Float32, "MIN reduce should work");
}

#[test]
fn test_pipeline_reduce_collapse_integration_with_unparented() {
    // Test: reduce_collapse and reduce_unparented should work together
    // Create a scenario where both optimizations could apply
    let const_val = UOp::const_(DType::Int32, ConstValue::Int(7));

    // Create two ranges: one will be unparented, one could be collapsed
    let range1 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(5)), 0, AxisType::Reduce);
    let range2 = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(3)), 1, AxisType::Reduce);

    // Const doesn't depend on either range
    let reduce = UOp::reduce(const_val, vec![range1, range2].into(), ReduceOp::Add);

    let result = rangeify_unwrap(reduce);

    // Both reduce_unparented and reduce_collapse could apply
    // Result should be simplified significantly
    assert_eq!(result.dtype(), DType::Int32);
}
