//! C renderer tests for code generation verification.

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, ConstValue, ReduceOp, UOp};
use smallvec::SmallVec;

use crate::c::render;

#[test]
fn test_range_end_basic() {
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);
    let noop = UOp::noop();
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = noop.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render(&sink, Some("test_loop")).expect("C codegen failed");

    assert!(result.code.contains("for"), "Missing for loop:\n{}", result.code);
    assert!(result.code.contains("ridx0"), "Missing loop variable:\n{}", result.code);
    assert!(result.code.contains("< 10"), "Missing loop bound:\n{}", result.code);
}

#[test]
fn test_reduce_add_basic() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let range =
        UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render(&sink, Some("test_reduce")).expect("C codegen failed");

    assert!(result.code.contains("acc"), "Missing accumulator:\n{}", result.code);
    assert!(result.code.contains("for"), "Missing for loop:\n{}", result.code);
    assert!(result.code.contains("+="), "Missing accumulation:\n{}", result.code);
    assert!(result.code.contains("0.0f"), "Missing identity value:\n{}", result.code);
}

#[test]
fn test_reduce_max() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(5)), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Max);
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render(&sink, Some("test_reduce_max")).expect("C codegen failed");

    assert!(result.code.contains("fmaxf"), "Missing fmaxf:\n{}", result.code);
}

#[test]
fn test_reduce_empty_ranges() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let reduce = const_val.reduce(smallvec::smallvec![], ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let result = render(&sink, Some("test_reduce_empty"));
    assert!(result.is_ok(), "C codegen failed: {:?}", result.err());
}
