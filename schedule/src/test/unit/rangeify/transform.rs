use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::{AxisType, ConstValue, Op, UOp};

use crate::rangeify::{
    IndexingContext,
    transform::{should_remove_movement_op, transform_single_source, transform_sources_with_bufferize},
};

#[test]
fn test_transform_buffer_source() {
    // Create an actual BUFFER operation and a consumer with ranges
    let unique = UOp::unique(Some(0));
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    let buffer = UOp::new(
        Op::Buffer {
            unique,
            device,
            size: 40, // 10 floats * 4 bytes
        },
        DType::Float32,
    );

    let range = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(10)), axis_id: 0, axis_type: AxisType::Loop },
        DType::Index,
    );

    // Create consumer (e.g., a binary op using the buffer)
    let five = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let consumer = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, buffer.clone(), five.clone()), DType::Float32);

    // Setup context with ranges for consumer
    let mut ctx = IndexingContext::new();
    // Consumer has ranges assigned (input_ranges same as output_ranges)
    ctx.set_ranges(&consumer, vec![range.clone()], vec![range.clone()]);

    // Transform sources
    let new_sources = transform_sources_with_bufferize(&consumer, &ctx);

    assert!(new_sources.is_some());
    let new_sources = new_sources.unwrap();
    assert_eq!(new_sources.len(), 2);

    // First source (buffer) should be wrapped in INDEX
    assert!(matches!(new_sources[0].op(), Op::Index { .. }));

    // Second source (const) should be unchanged
    assert!(Rc::ptr_eq(&new_sources[1], &five));
}

#[test]
fn test_transform_realizable_source() {
    // Create a source that needs realization
    let x = UOp::define_global(1, DType::Float32);
    let consumer = UOp::new(Op::Unary(morok_ir::UnaryOp::Neg, x.clone()), DType::Float32);

    // Create ranges
    let range = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(5)), axis_id: 0, axis_type: AxisType::Loop },
        DType::Index,
    );

    // Setup context
    let mut ctx = IndexingContext::new();
    ctx.set_ranges(&x, vec![range.clone()], vec![range.clone()]);
    ctx.set_ranges(&consumer, vec![range.clone()], vec![range.clone()]);
    ctx.mark_realize(&x, vec![0]);

    // Transform
    let new_src = transform_single_source(&consumer, &x, std::slice::from_ref(&range), &ctx);

    // Should be INDEX(BUFFERIZE(x))
    if let Op::Index { buffer, .. } = new_src.op() {
        assert!(matches!(buffer.op(), Op::Bufferize { .. }));
    } else {
        panic!("Expected INDEX operation");
    }
}

#[test]
fn test_should_remove_movement_op() {
    let src = UOp::define_global(0, DType::Float32);
    let permute = UOp::new(Op::Permute { src: src.clone(), axes: vec![1, 0] }, DType::Float32);

    let mut ctx = IndexingContext::new();

    // Initially should not remove
    assert!(!should_remove_movement_op(&permute, &ctx));

    // After assigning ranges, should remove
    let range = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(5)), axis_id: 0, axis_type: AxisType::Loop },
        DType::Index,
    );
    ctx.set_ranges(&permute, vec![range.clone()], vec![range.clone()]);

    assert!(should_remove_movement_op(&permute, &ctx));
}

#[test]
fn test_no_transform_for_normal_source() {
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let y = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::new(Op::Binary(morok_ir::BinaryOp::Add, x.clone(), y.clone()), DType::Float32);

    let ctx = IndexingContext::new();

    // No ranges assigned, no transformation should happen
    let result = transform_sources_with_bufferize(&add, &ctx);
    assert!(result.is_none());
}

#[test]
fn test_rangeify_with_symbolic_simplification() {
    // This test verifies that symbolic simplification is integrated into rangeify.
    // We create a computation with a PERMUTE operation that will create index expressions,
    // and ensure the full pipeline (including symbolic simplification) runs successfully.

    // Create a simple PERMUTE operation: swap axes
    let src = UOp::define_global(0, DType::Float32);
    let permute = UOp::new(Op::Permute { src: src.clone(), axes: vec![1, 0] }, DType::Float32);

    // Run full rangeify pipeline (includes symbolic simplification in Step 8)
    let (result, _ctx) = crate::rangeify::rangeify(permute).unwrap();

    // Verify the pipeline completed successfully without panicking.
    // This is primarily a smoke test to ensure symbolic simplification
    // is integrated and doesn't break the rangeify pipeline.
    // We don't make strong assertions about the result structure since
    // rangeify behavior depends on the complexity of the input graph.

    // At minimum, verify we got a result back
    assert!(result.dtype() == DType::Float32);
}
