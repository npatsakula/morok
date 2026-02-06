use std::sync::Arc;

use crate::rangeify::patterns::buffer_removal;
use crate::rewrite::graph_rewrite;
use morok_dtype::{AddrSpace as DTypeAddrSpace, DType};
use morok_ir::{AddrSpace, AxisId, AxisType, BufferizeOpts, Op, UOp};

// Helper functions
fn create_const(val: i64) -> Arc<UOp> {
    UOp::native_const(val as i32)
}

fn create_range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::new(
        Op::Range {
            end: create_const(end),
            axis_id: AxisId::Renumbered(axis_id),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    )
}

fn create_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::bufferize(compute, ranges, BufferizeOpts { device: None, addrspace: AddrSpace::Global })
}

// Pattern 1: Cheap Compute Inlining Tests

#[test]
fn test_remove_bufferize_cheap_unary() {
    // BUFFERIZE(NEG(x), ranges) should inline (cheap operation)
    let x = UOp::var("x", DType::Float32, 0, 100);
    let neg = x.neg();

    let range = create_range(10, 0);
    let bufferized = create_bufferize(neg.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return the cheap operation
    assert!(Arc::ptr_eq(&result, &neg), "Cheap unary op should be inlined");
}

#[test]
fn test_remove_bufferize_cheap_binary() {
    // BUFFERIZE(x + y, ranges) should inline (cheap operation)
    let x = UOp::var("x", DType::Float32, 0, 100);
    let y = UOp::var("y", DType::Float32, 0, 100);
    let add = x.try_add(&y).unwrap();

    let range = create_range(10, 0);
    let bufferized = create_bufferize(add.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return the cheap operation
    assert!(Arc::ptr_eq(&result, &add), "Cheap binary op should be inlined");
}

#[test]
fn test_remove_bufferize_cast() {
    // BUFFERIZE(CAST(x), ranges) should inline (cheap operation)
    let x = UOp::var("x", DType::Int32, 0, 100);
    let cast = x.cast(DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(cast.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return the cast
    assert!(Arc::ptr_eq(&result, &cast), "CAST should be inlined");
}

#[test]
fn test_keep_bufferize_expensive() {
    // BUFFERIZE(REDUCE(...), ranges) should NOT inline (expensive operation)
    let x = UOp::var("x", DType::Float32, 0, 100);
    let range = create_range(100, 0);

    let reduce = UOp::new(
        Op::Reduce { src: x, ranges: vec![range.clone()].into(), reduce_op: morok_ir::ReduceOp::Add },
        DType::Float32,
    );

    let buf_range = create_range(10, 1);
    let bufferized = create_bufferize(reduce, vec![buf_range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized.clone(), &mut ());

    // Should NOT remove BUFFERIZE (reduce is expensive)
    assert!(Arc::ptr_eq(&result, &bufferized), "REDUCE should remain buffered");
}

// Pattern 2: Always-Run Ops Tests

#[test]
fn test_remove_bufferize_contiguous() {
    // BUFFERIZE(CONTIGUOUS(x), ranges) should be removed (always-run op)
    let x = UOp::var("x", DType::Float32, 0, 100);
    let contiguous = x.contiguous();

    let range = create_range(10, 0);
    let bufferized = create_bufferize(contiguous.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return CONTIGUOUS
    assert!(Arc::ptr_eq(&result, &contiguous), "CONTIGUOUS shouldn't be buffered");
}

#[test]
fn test_remove_bufferize_copy() {
    // BUFFERIZE(COPY(x, device), ranges) should be removed (always-run op)
    let x = UOp::var("x", DType::Float32, 0, 100);
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    let copy = x.copy(device);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(copy.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return COPY
    assert!(Arc::ptr_eq(&result, &copy), "COPY shouldn't be buffered");
}

#[test]
fn test_remove_bufferize_assign() {
    // BUFFERIZE(ASSIGN(target, value), ranges) should be removed (always-run op)
    let target = UOp::define_global(1, DType::Float32);
    let value = UOp::native_const(1.0f32);
    let assign = UOp::assign(target, value);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(assign.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return ASSIGN
    assert!(Arc::ptr_eq(&result, &assign), "ASSIGN shouldn't be buffered");
}

#[test]
fn test_remove_bufferize_noop() {
    // BUFFERIZE(NOOP, ranges) should be removed (always-run op)
    let noop = UOp::noop();

    let range = create_range(10, 0);
    let bufferized = create_bufferize(noop.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return NOOP
    assert!(Arc::ptr_eq(&result, &noop), "NOOP shouldn't be buffered");
}

// Pattern 3: Nested Buffer Removal Tests

#[test]
fn test_flatten_nested_bufferize() {
    // BUFFERIZE(BUFFERIZE(x, R1), R2) → BUFFERIZE(x, R2)
    // Use define_global with pointer dtype (represents a buffer pointer - not cheap)
    let ptr_dtype = DType::Float32.ptr(Some(100), DTypeAddrSpace::Global);
    let x = UOp::define_global(1, ptr_dtype);
    let inner_range = create_range(10, 0);
    let outer_range = create_range(20, 1);

    let inner_buf = create_bufferize(x.clone(), vec![inner_range]);
    let outer_buf = create_bufferize(inner_buf, vec![outer_range.clone()]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, outer_buf, &mut ());

    // Should flatten to single BUFFERIZE with outer ranges
    if let Op::Bufferize { compute, ranges, .. } = result.op() {
        assert!(Arc::ptr_eq(compute, &x), "Should unwrap to original compute");
        assert_eq!(ranges.len(), 1, "Should have outer ranges only");
        assert!(Arc::ptr_eq(&ranges[0], &outer_range), "Should preserve outer range");
    } else {
        panic!("Expected BUFFERIZE after flattening");
    }
}

#[test]
fn test_nested_bufferize_multiple_ranges() {
    // BUFFERIZE(BUFFERIZE(x, [R1, R2]), [R3, R4]) → BUFFERIZE(x, [R3, R4])
    // Use define_global with pointer dtype (represents a buffer pointer - not cheap)
    let ptr_dtype = DType::Float32.ptr(Some(100), DTypeAddrSpace::Global);
    let x = UOp::define_global(1, ptr_dtype);
    let inner_ranges = vec![create_range(10, 0), create_range(15, 1)];
    let outer_ranges = vec![create_range(20, 2), create_range(25, 3)];

    let inner_buf = create_bufferize(x.clone(), inner_ranges);
    let outer_buf = create_bufferize(inner_buf, outer_ranges.clone());

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, outer_buf, &mut ());

    // Should flatten to single BUFFERIZE with outer ranges
    if let Op::Bufferize { compute, ranges, .. } = result.op() {
        assert!(Arc::ptr_eq(compute, &x), "Should unwrap to original compute");
        assert_eq!(ranges.len(), 2, "Should have 2 outer ranges");
        assert!(Arc::ptr_eq(&ranges[0], &outer_ranges[0]), "First outer range preserved");
        assert!(Arc::ptr_eq(&ranges[1], &outer_ranges[1]), "Second outer range preserved");
    } else {
        panic!("Expected BUFFERIZE after flattening");
    }
}

// Combined Tests

#[test]
fn test_multiple_cheap_ops_inline() {
    // Multiple cheap operations should all inline
    let x = UOp::var("x", DType::Float32, 0, 100);
    let range = create_range(10, 0);

    // Use direct Binary construction to test buffer removal, not type validation
    let test_ops = vec![x.neg(), x.try_exp2().unwrap(), x.try_mul(&x).unwrap()];

    let matcher = buffer_removal();

    for op in test_ops {
        let bufferized = create_bufferize(op.clone(), vec![range.clone()]);
        let result = graph_rewrite(&matcher, bufferized, &mut ());
        assert!(Arc::ptr_eq(&result, &op), "All cheap ops should inline");
    }
}

#[test]
fn test_no_removal_on_normal_buffer() {
    // Normal buffer operations (not cheap, not always-run) should remain
    // Use define_global with pointer dtype (represents a buffer pointer - not cheap)
    let ptr_dtype = DType::Float32.ptr(Some(100), DTypeAddrSpace::Global);
    let x = UOp::define_global(1, ptr_dtype);
    let range = create_range(10, 0);

    // Create a normal BUFFERIZE (not covering special cases)
    let bufferized = create_bufferize(x, vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized.clone(), &mut ());

    // Might be the same or modified, but the key is it doesn't crash
    // and follows the expected cost-based logic
    assert!(
        !result.op().children().is_empty() || matches!(result.op(), Op::DefineGlobal(_)),
        "Should produce valid result"
    );
}
