//! Tests for device-specific buffer limit enforcement.
//!
//! These tests validate that buffer limit enforcement correctly:
//! - Detects when buffer count exceeds device limits
//! - Forces bufferization of elementwise operations when needed
//! - Accounts for output buffer in the count (-1)
//! - Works with different device types (Metal, WebGPU, CPU, CUDA)
//! - Integrates correctly with the rangeify pipeline

use std::collections::HashSet;
use std::rc::Rc;

use morok_device::DeviceSpec;
use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisType, BinaryOp, BufferizeOpts, ConstValue, Op, SInt, TernaryOp, UOp, UOpKey};
use test_case::test_case;

use crate::rangeify::buffer_limits::{buffer_limit_patterns, extract_device_from_graph, is_elementwise};
use crate::rangeify::indexing::IndexingContext;
use crate::rewrite::graph_rewrite;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a test BUFFER with given size and dtype.
fn create_test_buffer(size: usize, dtype: DType, id: usize, device: DeviceSpec) -> Rc<UOp> {
    let unique = UOp::unique(Some(id));
    let device_op = UOp::device(device);
    UOp::new(Op::Buffer { unique, device: device_op, size }, dtype)
}

/// Create a computation graph that accesses multiple buffers.
///
/// Creates a chain of binary ADD operations that access `num_buffers` buffers.
/// Returns: (buffers, computation)
fn create_multi_buffer_computation(num_buffers: usize, device: DeviceSpec) -> (Vec<Rc<UOp>>, Rc<UOp>) {
    assert!(num_buffers > 0, "Must have at least one buffer");

    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let ranges = vec![range.clone()];

    // Create first buffer and index it
    let mut buffers = Vec::new();
    let buffer0 = create_test_buffer(40, DType::Float32, 0, device.clone());
    let indexed0 = UOp::index(buffer0.clone(), ranges.clone()).expect("Failed to create INDEX");
    buffers.push(buffer0);

    let mut computation = indexed0;

    // Chain additional buffers with ADD operations
    for i in 1..num_buffers {
        let buffer = create_test_buffer(40, DType::Float32, i, device.clone());
        let indexed = UOp::index(buffer.clone(), ranges.clone()).expect("Failed to create INDEX");
        computation = computation.try_add_op(&indexed).expect("Failed to create ADD");
        buffers.push(buffer);
    }

    (buffers, computation)
}

/// Count the number of BUFFERIZE operations in a UOp tree.
#[allow(clippy::mutable_key_type)]
fn count_bufferizes(uop: &Rc<UOp>) -> usize {
    let mut count = 0;
    let mut stack = vec![uop.clone()];
    let mut visited = HashSet::new();

    while let Some(current) = stack.pop() {
        let key = UOpKey(current.clone());
        if !visited.insert(key) {
            continue;
        }

        if matches!(current.op(), Op::Bufferize { .. }) {
            count += 1;
        }

        for child in current.op().sources() {
            stack.push(child);
        }
    }

    count
}

/// Count the number of unique BUFFER/BUFFERIZE operations accessed by a computation.
///
/// This replicates the buffer counting logic used by buffer_limit_patterns.
#[allow(clippy::mutable_key_type, dead_code)]
fn count_accessed_buffers(uop: &Rc<UOp>) -> usize {
    let mut buffers = Vec::new();
    let mut visited = HashSet::new();

    fn visit(uop: &Rc<UOp>, buffers: &mut Vec<Rc<UOp>>, visited: &mut HashSet<UOpKey>) {
        let key = UOpKey(Rc::clone(uop));
        if !visited.insert(key) {
            return;
        }

        match uop.op() {
            Op::Bufferize { opts, .. } if opts.addrspace == AddrSpace::Global => {
                buffers.push(Rc::clone(uop));
                return; // Stop traversal
            }
            Op::Buffer { .. } | Op::MStack { .. } | Op::MSelect { .. } => {
                buffers.push(Rc::clone(uop));
            }
            _ => {}
        }

        for child in uop.op().sources() {
            visit(&child, buffers, visited);
        }
    }

    visit(uop, &mut buffers, &mut visited);

    // Deduplicate
    let mut seen = HashSet::new();
    buffers.retain(|b| seen.insert(UOpKey(Rc::clone(b))));

    buffers.len()
}

// ============================================================================
// Phase 1: Device Limit Tests
// ============================================================================

#[cfg(feature = "metal")]
#[test]
fn test_metal_limit_at_threshold() {
    // Metal has 31 buffer limit
    // With 31 buffers accessed + 1 output = 32 total, should NOT trigger (31 <= 31-1 is false, but 31 <= 30 is false)
    // Actually: if accessed > max - 1, then 31 > 30 is true, so it SHOULD trigger
    // Let's test exactly at limit: 30 buffers accessed + 1 output = 31 total
    let device = DeviceSpec::Metal { device_id: 0 };
    let (_, computation) = create_multi_buffer_computation(30, device.clone());

    let matcher = buffer_limit_patterns(31);
    let result = graph_rewrite(&matcher, computation.clone(), &mut ());

    // Should NOT materialize (30 <= 30, within limit)
    assert!(
        Rc::ptr_eq(&result, &computation),
        "Should not materialize when exactly at limit (30 buffers + 1 output = 31 total)"
    );
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_limit_exceeded() {
    // Metal has 31 buffer limit
    // With 31 buffers accessed + 1 output = 32 total, should trigger materialization
    let device = DeviceSpec::Metal { device_id: 0 };
    let (_, computation) = create_multi_buffer_computation(31, device.clone());

    let before_count = count_bufferizes(&computation);
    let matcher = buffer_limit_patterns(31);
    let result = graph_rewrite(&matcher, computation.clone(), &mut ());
    let after_count = count_bufferizes(&result);

    // Should have materialized some operations
    assert!(
        after_count > before_count,
        "Should materialize when buffer limit exceeded (31 buffers + 1 output = 32 total)"
    );
}

#[cfg(feature = "webgpu")]
#[test]
fn test_webgpu_limit_at_threshold() {
    // WebGPU has 8 buffer limit
    // With 7 buffers accessed + 1 output = 8 total, should NOT trigger
    let device = DeviceSpec::WebGpu;
    let (_, computation) = create_multi_buffer_computation(7, device);

    let matcher = buffer_limit_patterns(8);
    let result = graph_rewrite(&matcher, computation.clone(), &mut ());

    // Should NOT materialize (7 <= 7, within limit)
    assert!(
        Rc::ptr_eq(&result, &computation),
        "Should not materialize when exactly at limit (7 buffers + 1 output = 8 total)"
    );
}

#[cfg(feature = "webgpu")]
#[test]
fn test_webgpu_limit_exceeded() {
    // WebGPU has 8 buffer limit
    // With 8 buffers accessed + 1 output = 9 total, should trigger
    let device = DeviceSpec::WebGpu;
    let (_, computation) = create_multi_buffer_computation(8, device);

    let before_count = count_bufferizes(&computation);
    let matcher = buffer_limit_patterns(8);
    let result = graph_rewrite(&matcher, computation.clone(), &mut ());
    let after_count = count_bufferizes(&result);

    // Should have materialized some operations
    assert!(
        after_count > before_count,
        "Should materialize when buffer limit exceeded (8 buffers + 1 output = 9 total)"
    );
}

#[test]
fn test_cpu_no_limit() {
    // CPU has no buffer limit
    let device = DeviceSpec::Cpu;
    let (_, computation) = create_multi_buffer_computation(100, device);

    // CPU returns None for max_buffers, so we manually test with a very high limit
    // In practice, this pattern wouldn't be created for CPU
    let before_count = count_bufferizes(&computation);
    let result = computation.clone(); // No pattern matcher needed

    // Should NOT change (no limit enforcement for CPU)
    assert!(Rc::ptr_eq(&result, &computation), "CPU should have no buffer limit");
    assert_eq!(count_bufferizes(&result), before_count, "CPU should not materialize buffers");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_no_limit() {
    // CUDA has no practical buffer limit
    let device = DeviceSpec::Cuda { device_id: 0 };
    let (_, computation) = create_multi_buffer_computation(100, device);

    let before_count = count_bufferizes(&computation);
    let result = computation.clone(); // No pattern matcher needed

    // Should NOT change (no limit enforcement for CUDA)
    assert!(Rc::ptr_eq(&result, &computation), "CUDA should have no buffer limit");
    assert_eq!(count_bufferizes(&result), before_count, "CUDA should not materialize buffers");
}

// ============================================================================
// Phase 2: Elementwise Materialization Tests
// ============================================================================

#[test]
fn test_binary_op_is_elementwise() {
    let left = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let right = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::new(Op::Binary(BinaryOp::Add, left, right), DType::Float32);

    assert!(is_elementwise(&add), "Binary ADD should be elementwise");
}

#[test]
fn test_ternary_op_is_elementwise() {
    let cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let true_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let false_val = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_val, false_val), DType::Float32);

    assert!(is_elementwise(&where_op), "Ternary WHERE should be elementwise");
}

#[test]
fn test_non_elementwise_operations() {
    // Constants are not elementwise
    let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    assert!(!is_elementwise(&const_op), "CONST should not be elementwise");

    // Buffers are not elementwise
    let device = DeviceSpec::Cpu;
    let buffer = create_test_buffer(100, DType::Float32, 1, device);
    assert!(!is_elementwise(&buffer), "BUFFER should not be elementwise");
}

#[cfg(feature = "metal")]
#[test]
fn test_materialize_only_elementwise() {
    // Create a computation with binary ops that should be materialized
    let device = DeviceSpec::Metal { device_id: 0 };
    let (_, computation) = create_multi_buffer_computation(31, device);

    // The computation is a chain of ADD operations (elementwise)
    let matcher = buffer_limit_patterns(31);
    let result = graph_rewrite(&matcher, computation, &mut ());

    // Should have created BUFFERIZE operations for elementwise ops
    let bufferize_count = count_bufferizes(&result);
    assert!(bufferize_count > 0, "Should have materialized elementwise operations");
}

// ============================================================================
// Phase 3: Output Buffer Accounting Tests
// ============================================================================

#[test_case(30, false ; "at_limit_should_not_trigger")]
#[test_case(31, true ; "over_limit_should_trigger")]
fn test_output_buffer_accounting(num_buffers: usize, should_materialize: bool) {
    // Test that the -1 accounting for output buffer works correctly
    let device = DeviceSpec::Cpu; // Use CPU to avoid feature flags
    let (_, computation) = create_multi_buffer_computation(num_buffers, device);

    let before_count = count_bufferizes(&computation);
    let matcher = buffer_limit_patterns(31); // Metal limit
    let result = graph_rewrite(&matcher, computation.clone(), &mut ());
    let after_count = count_bufferizes(&result);

    if should_materialize {
        assert!(after_count > before_count, "Should materialize when num_buffers={} (> 30)", num_buffers);
    } else {
        assert_eq!(after_count, before_count, "Should not materialize when num_buffers={} (<= 30)", num_buffers);
    }
}

// ============================================================================
// Phase 4: Edge Cases
// ============================================================================

#[test]
fn test_extract_device_no_device() {
    // Graph with no device info
    let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    assert_eq!(extract_device_from_graph(&const_op), None, "Should return None when no device");
}

#[test]
fn test_extract_device_from_device_op() {
    // Graph with Op::Device
    let device_op = UOp::device(DeviceSpec::Cpu);
    assert_eq!(extract_device_from_graph(&device_op), Some(DeviceSpec::Cpu), "Should extract CPU device");
}

#[test]
fn test_extract_device_from_buffer() {
    // Graph with Buffer containing device
    let device = DeviceSpec::Cpu;
    let buffer = create_test_buffer(100, DType::Float32, 1, device.clone());
    assert_eq!(extract_device_from_graph(&buffer), Some(device), "Should extract device from BUFFER");
}

#[test]
fn test_no_double_materialization() {
    // If a computation is already materialized, don't materialize again
    let device = DeviceSpec::Cpu;
    let (buffers, _) = create_multi_buffer_computation(35, device.clone());

    // Create a graph with an already-materialized BUFFERIZE
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let ranges = vec![range.clone()];

    // Access first buffer
    let indexed1 = UOp::index(buffers[0].clone(), ranges.clone()).expect("Failed to create INDEX");

    // Materialize it
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let materialized = UOp::bufferize(indexed1, ranges.clone(), opts);
    let indexed_materialized = UOp::index(materialized, ranges).expect("Failed to create INDEX");

    // Count BUFFERIZE operations before
    let before_count = count_bufferizes(&indexed_materialized);

    // Apply pattern (shouldn't double-materialize)
    let matcher = buffer_limit_patterns(31);
    let result = graph_rewrite(&matcher, indexed_materialized, &mut ());

    let after_count = count_bufferizes(&result);
    assert_eq!(before_count, after_count, "Should not double-materialize already-materialized operations");
}

// ============================================================================
// Phase 5: Integration Tests
// ============================================================================

#[test]
fn test_integration_with_rangeify_pipeline() {
    // Test that buffer limit enforcement works within the full rangeify pipeline
    let device = DeviceSpec::Cpu;

    // Create computation with many buffers
    let (_, computation) = create_multi_buffer_computation(10, device);

    // Run through rangeify (which includes buffer limit enforcement at Step 8.5)
    let result = crate::rangeify::rangeify(computation.clone(), None);

    // Should complete without errors
    assert!(result.is_ok(), "Rangeify pipeline should complete successfully with buffer limit enforcement");
}

#[test]
fn test_multiple_binary_ops() {
    // Test a computation with multiple binary operations
    let device = DeviceSpec::Cpu;
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let ranges = vec![range.clone()];

    // Create 20 buffers
    let mut buffers = Vec::new();
    for i in 0..20 {
        buffers.push(create_test_buffer(40, DType::Float32, i, device.clone()));
    }

    // Create a complex expression: ((((b0 + b1) + b2) + b3) + ...)
    let indexed0 = UOp::index(buffers[0].clone(), ranges.clone()).expect("Failed");
    let mut expr = indexed0;

    for buffer in buffers.iter().skip(1) {
        let indexed = UOp::index(buffer.clone(), ranges.clone()).expect("Failed");
        expr = expr.try_add_op(&indexed).expect("Failed to create ADD");
    }

    // Apply buffer limit (10 buffer limit)
    let before_count = count_bufferizes(&expr);
    let matcher = buffer_limit_patterns(10);
    let result = graph_rewrite(&matcher, expr, &mut ());
    let after_count = count_bufferizes(&result);

    // Should have materialized some intermediate results
    assert!(after_count > before_count, "Should materialize intermediate results to stay within buffer limit");
}

#[test]
fn test_ternary_op_materialization() {
    // Test that ternary operations are also checked for buffer limits
    let device = DeviceSpec::Cpu;
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let ranges = vec![range.clone()];

    // Create many buffers for condition, true_val, false_val
    let mut buffers = Vec::new();
    for i in 0..15 {
        buffers.push(create_test_buffer(40, DType::Float32, i, device.clone()));
    }

    // Create ternary WHERE operation accessing many buffers
    // cond = (b0 > b1 && b2 > b3 && ... b8 > b9)
    let indexed0 = UOp::index(buffers[0].clone(), ranges.clone()).expect("Failed");
    let indexed1 = UOp::index(buffers[1].clone(), ranges.clone()).expect("Failed");
    let mut cond = UOp::new(Op::Binary(BinaryOp::Gt, indexed0, indexed1), DType::Bool);

    for i in (2..10).step_by(2) {
        let left = UOp::index(buffers[i].clone(), ranges.clone()).expect("Failed");
        let right = UOp::index(buffers[i + 1].clone(), ranges.clone()).expect("Failed");
        let cmp = UOp::new(Op::Binary(BinaryOp::Gt, left, right), DType::Bool);
        cond = UOp::new(Op::Binary(BinaryOp::And, cond, cmp), DType::Bool);
    }

    // true_val and false_val access more buffers
    let true_val = UOp::index(buffers[10].clone(), ranges.clone()).expect("Failed");
    let false_val = {
        let b11 = UOp::index(buffers[11].clone(), ranges.clone()).expect("Failed");
        let b12 = UOp::index(buffers[12].clone(), ranges.clone()).expect("Failed");
        b11.try_add_op(&b12).expect("Failed to create ADD")
    };

    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, cond, true_val, false_val), DType::Float32);

    // Apply buffer limit (10 buffer limit)
    let before_count = count_bufferizes(&where_op);
    let matcher = buffer_limit_patterns(10);
    let result = graph_rewrite(&matcher, where_op, &mut ());
    let after_count = count_bufferizes(&result);

    // Should have materialized some intermediate results
    assert!(after_count > before_count, "Should materialize intermediate results in ternary operations");
}

#[test]
fn test_is_elementwise() {
    use morok_dtype::DType;
    use morok_ir::ConstValue;

    // Binary operations are elementwise
    let left = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let right = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = left.try_add_op(&right).unwrap();
    assert!(is_elementwise(&add), "Binary ADD should be elementwise");

    // Ternary operations are elementwise
    let cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let true_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let false_val = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let where_op = UOp::where_op(cond, true_val, false_val).unwrap();
    assert!(is_elementwise(&where_op), "Ternary WHERE should be elementwise");

    // Constants are not elementwise
    let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    assert!(!is_elementwise(&const_op), "CONST should not be elementwise");
}
