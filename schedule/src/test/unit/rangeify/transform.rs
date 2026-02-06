use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, Op, UOp};

use crate::rangeify::{
    IndexingContext,
    transforms::{transform_single_source, transform_sources_with_bufferize},
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
    let new_sources = transform_sources_with_bufferize(&consumer, &mut ctx);

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
        Op::Range {
            end: UOp::index_const(5),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );

    // Setup context
    let mut ctx = IndexingContext::new();
    ctx.set_ranges(&x, vec![range.clone()], vec![range.clone()]);
    ctx.set_ranges(&consumer, vec![range.clone()], vec![range.clone()]);
    ctx.mark_realize(&x, vec![0]);

    // Transform
    let new_src = transform_single_source(&consumer, &x, std::slice::from_ref(&range), &mut ctx);

    // Should be INDEX(BUFFERIZE(x))
    if let Op::Index { buffer, .. } = new_src.op() {
        assert!(matches!(buffer.op(), Op::Bufferize { .. }));
    } else {
        panic!("Expected INDEX operation");
    }
}

#[test]
fn test_no_transform_for_normal_source() {
    let x = UOp::native_const(1.0f32);
    let y = UOp::native_const(2.0f32);
    // Use direct Binary construction - this test checks transform behavior, not arithmetic
    let add = x.try_add(&y).unwrap();

    let mut ctx = IndexingContext::new();

    // No ranges assigned, no transformation should happen
    let result = transform_sources_with_bufferize(&add, &mut ctx);
    assert!(result.is_none());
}

#[test]
fn test_transform_movement_chain_on_buffer() {
    // Test that RESHAPE(BUFFER) is transformed to INDEX(BUFFER, transformed_indices)
    // The RESHAPE is eliminated and indices are computed to achieve the same memory access pattern
    let buffer = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 12, DType::Float32);

    // RESHAPE(BUFFER) to 3x4 shape
    let reshape_shape = UOp::vectorize(vec![UOp::index_const(3), UOp::index_const(4)].into());
    let reshape = UOp::new(Op::Reshape { src: buffer.clone(), new_shape: reshape_shape }, DType::Float32);

    // Verify is_movement works
    assert!(reshape.op().is_movement(), "RESHAPE should be identified as movement op");

    // Create an ADD that uses the reshaped buffer
    // Create another RESHAPE(BUFFER) with same shape for valid binary op
    let buffer2 = UOp::new_buffer(morok_device::DeviceSpec::Cpu, 12, DType::Float32);
    let reshape_shape2 = UOp::vectorize(vec![UOp::index_const(3), UOp::index_const(4)].into());
    let reshape2 = UOp::new(Op::Reshape { src: buffer2.clone(), new_shape: reshape_shape2 }, DType::Float32);
    let add = reshape.try_add(&reshape2).unwrap();

    // Set up context with ranges for add (simulating what indexing.rs does)
    let range0 = UOp::range_axis(UOp::index_const(3), AxisId::Renumbered(0), AxisType::Loop);
    let range1 = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(1), AxisType::Loop);

    let mut ctx = IndexingContext::new();
    ctx.set_ranges(&add, vec![range0.clone(), range1.clone()], vec![range0.clone(), range1.clone()]);

    // Transform sources
    let new_sources = transform_sources_with_bufferize(&add, &mut ctx);

    // RESHAPE(BUFFER) should be transformed to INDEX(BUFFER, transformed_indices)
    // RESHAPE is eliminated - indices are transformed to achieve the same effect
    assert!(new_sources.is_some(), "Transform should happen for movement chain on buffer");
    let new_sources = new_sources.unwrap();
    assert_eq!(new_sources.len(), 2);

    // First source should be wrapped in INDEX
    assert!(
        matches!(new_sources[0].op(), Op::Index { .. }),
        "RESHAPE(BUFFER) should be transformed to INDEX, got: {:?}",
        new_sources[0].op()
    );

    // Verify the INDEX directly wraps the BUFFER (RESHAPE eliminated)
    if let Op::Index { buffer: idx_buffer, indices, .. } = new_sources[0].op() {
        assert!(
            matches!(idx_buffer.op(), Op::Buffer { .. }),
            "INDEX should wrap BUFFER directly (RESHAPE eliminated), got: {:?}",
            idx_buffer.op()
        );
        // Indices should be transformed (not just the original ranges)
        // For 3x4 reshape of a 12-element buffer: index = range0 * 4 + range1
        assert_eq!(indices.len(), 1, "Transformed indices should be flattened to 1D");
    }
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
