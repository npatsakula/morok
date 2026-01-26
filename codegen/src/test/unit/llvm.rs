//! LLVM renderer tests for loop and reduction codegen.

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, ConstValue, ReduceOp, UOp};
use smallvec::SmallVec;

use crate::llvm::text::render;

/// Test basic RANGE/END loop codegen.
///
/// Creates the equivalent of:
/// ```c
/// for (int i = 0; i < 10; i++) {
///     // empty body
/// }
/// ```
#[test]
fn test_range_end_basic() {
    // Create range: for i in 0..10
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);

    // Create a NOOP as the computation (empty loop body)
    let noop = UOp::noop();

    // End the loop - END wraps computation and references the range
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = UOp::end(noop, ranges);

    // Wrap in SINK
    let sink = UOp::sink(vec![end_op]);

    // Render to LLVM IR
    let result = render(&sink, Some("test_loop"));
    if let Err(ref e) = result {
        eprintln!("Codegen failed: {:?}", e);
    }
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());

    let kernel = result.unwrap();
    let ir = &kernel.code;

    // Verify loop structure in generated IR
    // Block names use uop_id which varies, so just check for the patterns
    assert!(ir.contains("loop_entry_"), "Missing entry block:\n{}", ir);
    assert!(ir.contains("loop_latch_"), "Missing latch block:\n{}", ir);
    assert!(ir.contains("loop_body_"), "Missing body block:\n{}", ir);
    assert!(ir.contains("loop_footer_"), "Missing footer block:\n{}", ir);
    assert!(ir.contains("loop_exit_"), "Missing exit block:\n{}", ir);
    assert!(ir.contains("phi i64"), "Missing PHI node:\n{}", ir);
}

/// Test basic REDUCE codegen with sum operation.
///
/// Creates the equivalent of:
/// ```c
/// float acc = 0.0;
/// for (int i = 0; i < 10; i++) {
///     acc += 5.0;  // constant value
/// }
/// return acc;  // should be 50.0
/// ```
#[test]
fn test_reduce_add_basic() {
    // Create reduction: sum of constant 5.0 over range 0..10
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let range =
        UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = UOp::reduce(const_val, smallvec::smallvec![range.clone()], ReduceOp::Add);

    // END op closes the loop - required for proper codegen
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = UOp::end(reduce, ranges);

    // Wrap in SINK
    let sink = UOp::sink(vec![end_op]);

    // Render to LLVM IR
    let result = render(&sink, Some("test_reduce_add"));
    if let Err(ref e) = result {
        eprintln!("Codegen failed: {:?}", e);
    }
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());

    let kernel = result.unwrap();
    let ir = &kernel.code;

    // Verify loop structure (uses standard loop blocks with alloca accumulator, like Tinygrad's DEFINE_REG)
    assert!(ir.contains("loop_latch_"), "Missing loop latch block:\n{}", ir);
    assert!(ir.contains("loop_body_"), "Missing loop body block:\n{}", ir);
    assert!(ir.contains("loop_exit_"), "Missing loop exit block:\n{}", ir);
    // Verify accumulator alloca
    assert!(ir.contains("alloca float"), "Missing reduce accumulator alloca:\n{}", ir);
    // Verify reduce add operation
    assert!(ir.contains("fadd"), "Missing fadd instruction:\n{}", ir);
}

/// Test REDUCE codegen with max operation.
#[test]
fn test_reduce_max() {
    // Create reduction: max of constant over range
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(5)), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = UOp::reduce(const_val, smallvec::smallvec![range.clone()], ReduceOp::Max);

    // END op closes the loop
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = UOp::end(reduce, ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render(&sink, Some("test_reduce_max"));
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());

    let kernel = result.unwrap();
    let ir = &kernel.code;

    // Verify max intrinsic is called
    assert!(ir.contains("llvm.maxnum.f") || ir.contains("maxnum"), "Missing maxnum intrinsic:\n{}", ir);
}

/// Test REDUCE codegen with empty ranges (no reduction).
#[test]
fn test_reduce_empty_ranges() {
    // Create reduction with empty ranges - should just return source
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let reduce = UOp::reduce(const_val, smallvec::smallvec![], ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let result = render(&sink, Some("test_reduce_empty"));
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());
}
