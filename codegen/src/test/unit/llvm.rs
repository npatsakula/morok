//! LLVM renderer tests for loop and reduction codegen.

use smallvec::SmallVec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, Op, ParamArg, UOp};

use crate::llvm::common::lconst;
use crate::llvm::text::render;

fn render_linearized(root: &std::sync::Arc<UOp>, name: Option<&str>) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root.clone()).into());
    render(&linear, name)
}

fn volatile_param(slot: usize, size: usize) -> std::sync::Arc<UOp> {
    let mut arg = ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, None);
    arg.volatile = true;
    UOp::new(Op::Param { shape: UOp::native_const(size as i32), arg: arg.into() }, DType::Float32)
}

fn shrink4(src: std::sync::Arc<UOp>) -> std::sync::Arc<UOp> {
    UOp::new(Op::Shrink { src, offsets: UOp::native_const(0i32), sizes: UOp::native_const(4i32) }, DType::Float32)
}

#[test]
fn test_render_linear_input_succeeds() {
    let sink = UOp::sink(vec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());

    let rendered = render(&linear, Some("test_linear")).expect("LLVM codegen from LINEAR should succeed");
    assert!(rendered.code.contains("test_linear"));
}

#[test]
fn llvm_constants_use_committed_storage_bits() {
    let half = UOp::const_(DType::Float16, ConstValue::Float(1.0 / 123_008.0));
    let Op::Const(half) = half.op() else { unreachable!() };
    assert_eq!(lconst(&half.0, &DType::Float16), "0xH0088");

    let fp8 = UOp::const_(DType::FP8E4M3, ConstValue::Float(1.1875));
    let Op::Const(fp8) = fp8.op() else { unreachable!() };
    assert_eq!(lconst(&fp8.0, &DType::FP8E4M3), "58");
}

#[test]
fn grouped_shrink_renders_single_vector_load_and_store() {
    let output = shrink4(UOp::param(0, 8, DType::Float32, None));
    let input = shrink4(UOp::param(1, 8, DType::Float32, None));
    let sink = UOp::sink(vec![output.store(UOp::load().index(input).call())]);

    let rendered = render_linearized(&sink, Some("grouped_memory")).expect("render grouped LLVM memory");
    assert_eq!(rendered.code.matches("load <4 x float>").count(), 1, "{}", rendered.code);
    assert_eq!(rendered.code.matches("store <4 x float>").count(), 1, "{}", rendered.code);
}

#[test]
fn volatile_scalar_and_grouped_memory_accesses_render_explicitly() {
    let index = UOp::native_const(0i32);
    let scalar_input = UOp::index().buffer(volatile_param(1, 1)).indices(vec![index.clone()]).call().unwrap();
    let scalar_output = UOp::index().buffer(volatile_param(0, 1)).indices(vec![index]).call().unwrap();
    let scalar = UOp::sink(vec![scalar_output.store(UOp::load().index(scalar_input).call())]);
    let scalar = render_linearized(&scalar, Some("volatile_scalar")).expect("render volatile scalar LLVM");
    assert!(scalar.code.contains("load volatile float"), "{}", scalar.code);
    assert!(scalar.code.contains("store volatile float"), "{}", scalar.code);

    let grouped_input = shrink4(volatile_param(1, 8));
    let grouped_output = shrink4(volatile_param(0, 8));
    let grouped = UOp::sink(vec![grouped_output.store(UOp::load().index(grouped_input).call())]);
    let grouped = render_linearized(&grouped, Some("volatile_grouped")).expect("render volatile grouped LLVM");
    assert!(grouped.code.contains("load volatile <4 x float>"), "{}", grouped.code);
    assert!(grouped.code.contains("store volatile <4 x float>"), "{}", grouped.code);
}

#[test]
fn test_render_rejects_non_linear_inputs() {
    let sink = UOp::sink(vec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let program = UOp::program(sink.clone(), info, None, None, None);

    let err = render(&program, Some("test_program_input")).expect_err("PROGRAM input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");

    let err = render(&sink, Some("test_sink_input")).expect_err("SINK input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");
}

/// A `for i in 0..10` loop lowers to the Tinygrad block layout
/// (entry / latch / body / footer / exit) with the induction variable in a PHI.
/// Block names embed the axis id, so only the prefixes are pinned.
#[test]
fn test_range_end_basic() {
    let end = UOp::const_(DType::Int64, ConstValue::Int(10));
    let range = UOp::new(
        Op::Range { end, axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop, deps: SmallVec::new() },
        DType::Int64,
    );
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let sink = UOp::sink(vec![UOp::noop().end(ranges)]);

    let kernel = render_linearized(&sink, Some("test_loop")).expect("render loop");

    for needle in ["loop_entry_", "loop_latch_", "loop_body_", "loop_footer_", "loop_exit_", "phi i64"] {
        assert!(kernel.code.contains(needle), "missing {needle}:\n{}", kernel.code);
    }
}

#[test]
fn test_multi_index_requires_linearization() {
    let buffer = UOp::param(0, 1024, DType::Float32, None);
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

/// A REG buffer is storage, not a vector value: indexing it is a GEP + load,
/// including when the buffer arrives through an AFTER edge.
#[test_case::test_case(false; "direct")]
#[test_case::test_case(true; "through after")]
fn shaped_reg_index_renders_as_memory_load(through_after: bool) {
    let reg = UOp::buffer(0, 4, DType::Float32, AddrSpace::Reg, None);
    let buffer = if through_after {
        let after = reg.after(smallvec::smallvec![UOp::noop()]);
        assert_eq!(after.addrspace(), Some(AddrSpace::Reg));
        after
    } else {
        reg
    };
    let index =
        UOp::index().buffer(buffer).indices(vec![UOp::const_(DType::Int32, ConstValue::Int(2))]).call().unwrap();
    let sink = UOp::sink(vec![UOp::load().index(index).call()]);

    let result = render_linearized(&sink, Some("shaped_reg_load")).expect("render REG load");
    assert!(result.code.contains("getelementptr inbounds float, ptr %reg0, i32 2"), "{}", result.code);
    assert!(result.code.contains("load float, ptr"), "{}", result.code);
    assert!(!result.code.contains("extractelement <4 x float> %reg0"), "{}", result.code);
}

/// A typed CUSTOM renders its template as the RHS of an SSA assignment; a
/// CUSTOMI is inlined as an operand string into its consumer instead.
#[test]
fn custom_renders_a_typed_rhs_and_customi_inlines_into_its_consumer() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let custom = UOp::custom(smallvec::smallvec![one.clone()], "add i32 {0}, 3".to_string(), DType::Int32);
    let typed = render_linearized(&UOp::sink(vec![custom]), Some("test_custom")).expect("render CUSTOM");
    assert!(typed.code.contains("= add i32 1, 3"), "typed CUSTOM should render its RHS:\n{}", typed.code);

    // `{2}` selects the third dep (const 3).
    let two = UOp::const_(DType::Int32, ConstValue::Int(2));
    let three = UOp::const_(DType::Int32, ConstValue::Int(3));
    let inline = UOp::customi(smallvec::smallvec![one, two, three], "{2}".to_string(), DType::Int32);
    let consumer = UOp::custom(smallvec::smallvec![inline], "add i32 {0}, 10".to_string(), DType::Int32);
    let inlined = render_linearized(&UOp::sink(vec![consumer]), Some("test_customi")).expect("render CUSTOMI");
    assert!(inlined.code.contains("= add i32 3, 10"), "CUSTOMI should inline into the consumer:\n{}", inlined.code);
}

/// A gated LOAD renders as branch + phi, and the phi is typed by the load's
/// shape — a scalar-dtype shape-`[4]` load phis `<4 x float>`, not `float`.
///
/// The unpaired-alt/gate and non-bool-gate branches of the renderer's guard are
/// unreachable: `UOp::new` already asserts both (`ir/src/uop/hash_consing.rs`
/// `new_tagged`), so they are defense-in-depth, not testable states.
#[test_case::test_case(1, "= phi float"; "scalar gated load")]
#[test_case::test_case(4, "= phi <4 x float>"; "grouped gated load")]
fn llvm_gated_load_phis_the_shaped_load_type(lanes: usize, expected: &str) {
    let address = |slot| {
        if lanes == 1 {
            UOp::index()
                .buffer(UOp::param(slot, 8, DType::Float32, None))
                .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
                .call()
                .unwrap()
        } else {
            UOp::new(
                Op::Shrink {
                    src: UOp::param(slot, 8, DType::Float32, None),
                    offsets: UOp::native_const(0i32),
                    sizes: UOp::native_const(lanes as i32),
                },
                DType::Float32,
            )
        }
    };
    let alt = UOp::const_(DType::Float32, ConstValue::Float(7.0)).broadcast(lanes);
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let load = UOp::load().index(address(1)).alt(alt).gate(gate).call();

    let rendered =
        render_linearized(&UOp::sink(vec![address(0).store(load)]), Some("gated_load")).expect("render gated load");
    assert!(rendered.code.contains(expected), "{}", rendered.code);
}
