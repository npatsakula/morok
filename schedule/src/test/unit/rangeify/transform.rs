use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, Op, UOp};

use crate::rangeify::{
    IndexingContext,
    transform::{should_remove_movement_op, transform_single_source, transform_sources_with_bufferize},
};

#[test]
fn test_transform_buffer_source() {
    // Create two BUFFER operations with the same shape for a valid binary op
    let buffer1 = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 40, DType::Float32);
    let buffer2 = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 40, DType::Float32);

    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Loop);

    // Create consumer - adding two buffers of the same shape
    let consumer = buffer1.try_add(&buffer2).unwrap();

    // Setup context with ranges for consumer
    let mut ctx = IndexingContext::new();
    ctx.set_ranges(&consumer, vec![range.clone()], vec![range.clone()]);

    // Transform sources
    let new_sources = transform_sources_with_bufferize(&consumer, &ctx);

    assert!(new_sources.is_some());
    let new_sources = new_sources.unwrap();
    assert_eq!(new_sources.len(), 2);

    // Both buffer sources should be wrapped in INDEX
    assert!(matches!(new_sources[0].op(), Op::Index { .. }));
    assert!(matches!(new_sources[1].op(), Op::Index { .. }));
}

#[test]
fn test_transform_realizable_source() {
    // Create a source that needs realization
    let x = UOp::define_global(1, DType::Float32);
    let consumer = x.neg();

    // Create ranges
    let range = UOp::new(
        Op::Range { end: UOp::index_const(5), axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop },
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
        Op::Range { end: UOp::index_const(5), axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop },
        DType::Index,
    );
    ctx.set_ranges(&permute, vec![range.clone()], vec![range.clone()]);

    assert!(should_remove_movement_op(&permute, &ctx));
}

#[test]
fn test_no_transform_for_normal_source() {
    let x = UOp::native_const(1.0f32);
    let y = UOp::native_const(2.0f32);
    // Use direct Binary construction - this test checks transform behavior, not arithmetic
    let add = x.try_add(&y).unwrap();

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
    let (result, _ctx) = crate::rangeify::rangeify(permute, None).unwrap();

    // Verify the pipeline completed successfully without panicking.
    // This is primarily a smoke test to ensure symbolic simplification
    // is integrated and doesn't break the rangeify pipeline.
    // We don't make strong assertions about the result structure since
    // rangeify behavior depends on the complexity of the input graph.

    // At minimum, verify we got a result back
    assert!(result.dtype() == DType::Float32);
}
