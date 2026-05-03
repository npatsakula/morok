//! LLVM renderer tests for loop and reduction codegen.

use morok_dtype::{DType, DeviceSpec};
use morok_ir::{AxisId, AxisType, ConstValue, ReduceOp, UOp};
use smallvec::SmallVec;

use crate::llvm::text::render;

fn render_linearized(root: &std::sync::Arc<UOp>, name: Option<&str>) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(root.clone()).into());
    render(&linear, name)
}

#[test]
fn test_render_linear_input_succeeds() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());

    let rendered = render(&linear, Some("test_linear")).expect("LLVM codegen from LINEAR should succeed");
    assert!(rendered.code.contains("test_linear"));
}

#[test]
fn test_render_rejects_non_linear_inputs() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let program = UOp::program(sink.clone(), UOp::device(DeviceSpec::Cpu), None, None, None);

    let err = render(&program, Some("test_program_input")).expect_err("PROGRAM input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");

    let err = render(&sink, Some("test_sink_input")).expect_err("SINK input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");
}

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
    let end_op = noop.end(ranges);

    // Wrap in SINK
    let sink = UOp::sink(vec![end_op]);

    // Render to LLVM IR
    let result = render_linearized(&sink, Some("test_loop"));
    if let Err(ref e) = result {
        eprintln!("Codegen failed: {:?}", e);
    }
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());

    let kernel = result.unwrap();
    let ir = &kernel.code;

    // Verify loop structure in generated IR (Tinygrad-style: entry/latch/body/footer/exit)
    // Block names use axis_id which varies, so just check for the patterns
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

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);

    // END op closes the loop - required for proper codegen
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);

    // Wrap in SINK
    let sink = UOp::sink(vec![end_op]);

    // Render to LLVM IR
    let result = render_linearized(&sink, Some("test_reduce_add"));
    if let Err(ref e) = result {
        eprintln!("Codegen failed: {:?}", e);
    }
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());

    let kernel = result.unwrap();
    let ir = &kernel.code;

    // Verify loop structure (Tinygrad-style: entry/latch/body/footer/exit with alloca accumulator)
    assert!(ir.contains("loop_entry_"), "Missing loop entry block:\n{}", ir);
    assert!(ir.contains("loop_latch_"), "Missing loop latch block:\n{}", ir);
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

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Max);

    // END op closes the loop
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render_linearized(&sink, Some("test_reduce_max"));
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
    let reduce = const_val.reduce(smallvec::smallvec![], ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let result = render_linearized(&sink, Some("test_reduce_empty"));
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());
}

#[test]
fn test_multi_index_requires_linearization() {
    let ptr_dtype = DType::Float32.ptr(None, morok_dtype::AddrSpace::Global);
    let buffer = UOp::param(0, 1024, ptr_dtype, None);
    let i = UOp::const_(DType::Index, ConstValue::Int(1));
    let j = UOp::const_(DType::Index, ConstValue::Int(2));
    let index = UOp::index().buffer(buffer).indices(vec![i, j]).call().unwrap();
    let sink = UOp::sink(vec![index]);

    let linear = UOp::linear(sink.toposort().into());
    let err = render(&linear, Some("test_multi_index_requires_linearization"))
        .expect_err("multi-index INDEX must surface as InvalidGraph");
    assert!(
        matches!(&err, crate::Error::InvalidGraph { reason } if reason.contains("linearized INDEX")),
        "expected InvalidGraph(linearized INDEX), got {err:?}",
    );
}

#[test]
fn test_custom_is_explicitly_unsupported_in_llvm_backend() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let custom = UOp::custom(smallvec::smallvec![one], "add i32 {0}, 3".to_string(), DType::Int32);
    let sink = UOp::sink(vec![custom]);

    let err = render_linearized(&sink, Some("test_custom_unsupported")).expect_err("LLVM backend must reject CUSTOM");
    assert!(format!("{err}").contains("does not support CUSTOM/CUSTOMI"), "unexpected error: {err}");
}

#[test]
fn test_customi_is_explicitly_unsupported_in_llvm_backend() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c = UOp::const_(DType::Int32, ConstValue::Int(3));
    let inline = UOp::customi(smallvec::smallvec![a, b, c], "{2}".to_string(), DType::Int32);
    let sink = UOp::sink(vec![inline]);

    let err = render_linearized(&sink, Some("test_customi_unsupported")).expect_err("LLVM backend must reject CUSTOMI");
    assert!(format!("{err}").contains("does not support CUSTOM/CUSTOMI"), "unexpected error: {err}");
}
