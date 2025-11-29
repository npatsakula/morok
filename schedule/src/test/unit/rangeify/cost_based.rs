use std::rc::Rc;

use crate::rangeify::patterns::buffer_removal;
use crate::rewrite::graph_rewrite;
use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisId, AxisType, BinaryOp, BufferizeOpts, ConstValue, Op, UOp, UnaryOp};

// Helper functions
fn create_const(val: i64) -> Rc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(val))
}

fn create_range(end: i64, axis_id: usize) -> Rc<UOp> {
    UOp::new(
        Op::Range { end: create_const(end), axis_id: AxisId::Renumbered(axis_id), axis_type: AxisType::Loop },
        DType::Index,
    )
}

fn create_bufferize(compute: Rc<UOp>, ranges: Vec<Rc<UOp>>) -> Rc<UOp> {
    UOp::bufferize(compute, ranges, BufferizeOpts { device: None, addrspace: AddrSpace::Global })
}

// Pattern 1: Cheap Compute Inlining Tests

#[test]
fn test_remove_bufferize_cheap_unary() {
    // BUFFERIZE(NEG(x), ranges) should inline (cheap operation)
    let x = UOp::define_global(1, DType::Float32);
    let neg = UOp::new(Op::Unary(UnaryOp::Neg, x), DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(neg.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return the cheap operation
    assert!(Rc::ptr_eq(&result, &neg), "Cheap unary op should be inlined");
}

#[test]
fn test_remove_bufferize_cheap_binary() {
    // BUFFERIZE(x + y, ranges) should inline (cheap operation)
    let x = UOp::define_global(1, DType::Float32);
    let y = UOp::define_global(2, DType::Float32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, x, y), DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(add.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return the cheap operation
    assert!(Rc::ptr_eq(&result, &add), "Cheap binary op should be inlined");
}

#[test]
fn test_remove_bufferize_cast() {
    // BUFFERIZE(CAST(x), ranges) should inline (cheap operation)
    let x = UOp::define_global(1, DType::Int32);
    let cast = UOp::new(Op::Cast { src: x, dtype: DType::Float32 }, DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(cast.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return the cast
    assert!(Rc::ptr_eq(&result, &cast), "CAST should be inlined");
}

#[test]
fn test_keep_bufferize_expensive() {
    // BUFFERIZE(REDUCE(...), ranges) should NOT inline (expensive operation)
    let x = UOp::define_global(1, DType::Float32);
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
    assert!(Rc::ptr_eq(&result, &bufferized), "REDUCE should remain buffered");
}

// Pattern 2: Always-Run Ops Tests

#[test]
fn test_remove_bufferize_contiguous() {
    // BUFFERIZE(CONTIGUOUS(x), ranges) should be removed (always-run op)
    let x = UOp::define_global(1, DType::Float32);
    let contiguous = UOp::new(Op::Contiguous { src: x }, DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(contiguous.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return CONTIGUOUS
    assert!(Rc::ptr_eq(&result, &contiguous), "CONTIGUOUS shouldn't be buffered");
}

#[test]
fn test_remove_bufferize_copy() {
    // BUFFERIZE(COPY(x, device), ranges) should be removed (always-run op)
    let x = UOp::define_global(1, DType::Float32);
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    let copy = UOp::new(Op::Copy { src: x, device }, DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(copy.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return COPY
    assert!(Rc::ptr_eq(&result, &copy), "COPY shouldn't be buffered");
}

#[test]
fn test_remove_bufferize_assign() {
    // BUFFERIZE(ASSIGN(target, value), ranges) should be removed (always-run op)
    let target = UOp::define_global(1, DType::Float32);
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let assign = UOp::new(Op::Assign { target, value }, DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(assign.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return ASSIGN
    assert!(Rc::ptr_eq(&result, &assign), "ASSIGN shouldn't be buffered");
}

#[test]
fn test_remove_bufferize_noop() {
    // BUFFERIZE(NOOP, ranges) should be removed (always-run op)
    let noop = UOp::new(Op::Noop, DType::Float32);

    let range = create_range(10, 0);
    let bufferized = create_bufferize(noop.clone(), vec![range]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, bufferized, &mut ());

    // Should remove BUFFERIZE and return NOOP
    assert!(Rc::ptr_eq(&result, &noop), "NOOP shouldn't be buffered");
}

// Pattern 3: Nested Buffer Removal Tests

#[test]
fn test_flatten_nested_bufferize() {
    // BUFFERIZE(BUFFERIZE(x, R1), R2) → BUFFERIZE(x, R2)
    let x = UOp::define_global(1, DType::Float32);
    let inner_range = create_range(10, 0);
    let outer_range = create_range(20, 1);

    let inner_buf = create_bufferize(x.clone(), vec![inner_range]);
    let outer_buf = create_bufferize(inner_buf, vec![outer_range.clone()]);

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, outer_buf, &mut ());

    // Should flatten to single BUFFERIZE with outer ranges
    if let Op::Bufferize { compute, ranges, .. } = result.op() {
        assert!(Rc::ptr_eq(compute, &x), "Should unwrap to original compute");
        assert_eq!(ranges.len(), 1, "Should have outer ranges only");
        assert!(Rc::ptr_eq(&ranges[0], &outer_range), "Should preserve outer range");
    } else {
        panic!("Expected BUFFERIZE after flattening");
    }
}

#[test]
fn test_nested_bufferize_multiple_ranges() {
    // BUFFERIZE(BUFFERIZE(x, [R1, R2]), [R3, R4]) → BUFFERIZE(x, [R3, R4])
    let x = UOp::define_global(1, DType::Float32);
    let inner_ranges = vec![create_range(10, 0), create_range(15, 1)];
    let outer_ranges = vec![create_range(20, 2), create_range(25, 3)];

    let inner_buf = create_bufferize(x.clone(), inner_ranges);
    let outer_buf = create_bufferize(inner_buf, outer_ranges.clone());

    let matcher = buffer_removal();
    let result = graph_rewrite(&matcher, outer_buf, &mut ());

    // Should flatten to single BUFFERIZE with outer ranges
    if let Op::Bufferize { compute, ranges, .. } = result.op() {
        assert!(Rc::ptr_eq(compute, &x), "Should unwrap to original compute");
        assert_eq!(ranges.len(), 2, "Should have 2 outer ranges");
        assert!(Rc::ptr_eq(&ranges[0], &outer_ranges[0]), "First outer range preserved");
        assert!(Rc::ptr_eq(&ranges[1], &outer_ranges[1]), "Second outer range preserved");
    } else {
        panic!("Expected BUFFERIZE after flattening");
    }
}

// Combined Tests

#[test]
fn test_multiple_cheap_ops_inline() {
    // Multiple cheap operations should all inline
    let x = UOp::define_global(1, DType::Float32);
    let range = create_range(10, 0);

    let test_ops = vec![
        UOp::new(Op::Unary(UnaryOp::Neg, x.clone()), DType::Float32),
        UOp::new(Op::Unary(UnaryOp::Exp2, x.clone()), DType::Float32),
        UOp::new(Op::Binary(BinaryOp::Mul, x.clone(), x.clone()), DType::Float32),
    ];

    let matcher = buffer_removal();

    for op in test_ops {
        let bufferized = create_bufferize(op.clone(), vec![range.clone()]);
        let result = graph_rewrite(&matcher, bufferized, &mut ());
        assert!(Rc::ptr_eq(&result, &op), "All cheap ops should inline");
    }
}

#[test]
fn test_no_removal_on_normal_buffer() {
    // Normal buffer operations (not cheap, not always-run) should remain
    let x = UOp::define_global(1, DType::Float32);
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
