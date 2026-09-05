//! Metal Shading Language dialect tests for the C-family renderer.

use smallvec::SmallVec;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::{AxisId, AxisType, ConstValue, Op, ParamArg, ReduceOp, RendererDevice, UOp, WmmaMetadata};

use crate::c::types::c_const;
use crate::c::{CDialect, render, render_metal};
use svod_ir::ops;

fn render_linearized(root: &std::sync::Arc<UOp>, name: &str) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root.clone()).into());
    render_metal(&linear, Some(name))
}

fn render_ok(root: &std::sync::Arc<UOp>, name: &str) -> String {
    let rendered = render_linearized(root, name).expect("render MSL");
    assert_msl_compiles(&rendered.code);
    rendered.code
}

fn concrete_range(end: i64, axis_type: AxisType) -> std::sync::Arc<UOp> {
    let end = UOp::const_(DType::Int32, ConstValue::Int(end));
    UOp::new(
        Op::Range(ops::Range { end, axis_id: AxisId::Renumbered(0), axis_type, deps: SmallVec::new() }),
        DType::Int32,
    )
}

fn slotted_var(name: &str, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), 0, 16, DType::Int32);
    let Op::Param(ops::Param { shape, arg }) = var.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), DType::Int32)
}

fn volatile_param(slot: usize, size: usize) -> std::sync::Arc<UOp> {
    let mut arg = ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, None);
    arg.volatile = true;
    UOp::new(Op::Param(ops::Param { shape: UOp::index_const(size as i64), arg: arg.into() }), DType::Float32)
}

fn shrink(src: std::sync::Arc<UOp>, dtype: DType, lanes: i64) -> std::sync::Arc<UOp> {
    UOp::new(
        Op::Shrink(ops::Shrink {
            src,
            offsets: UOp::const_(DType::Int32, ConstValue::Int(0)),
            sizes: UOp::const_(DType::Int32, ConstValue::Int(lanes)),
        }),
        dtype,
    )
}

fn element(buffer: std::sync::Arc<UOp>, lane: i64) -> std::sync::Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![UOp::const_(DType::Index, ConstValue::Int(lane))]).call().unwrap()
}

fn load_scalar(slot: usize, dtype: DType) -> std::sync::Arc<UOp> {
    UOp::load().index(element(UOp::param(slot, 4, dtype, None), 0)).call()
}

/// `data0[0] = f(data1[0])` for a unary transcendental / cast on `dtype`.
fn unary_sink(dtype: DType, f: impl FnOnce(&std::sync::Arc<UOp>) -> std::sync::Arc<UOp>) -> std::sync::Arc<UOp> {
    let value = f(&load_scalar(1, dtype.clone()));
    let out_dtype = value.dtype();
    UOp::sink(vec![element(UOp::param(0, 4, out_dtype, None), 0).store(value)])
}

#[test]
fn metal_signature_uses_device_and_constant_qualifiers() {
    let sink = UOp::sink(vec![
        slotted_var("high", 3),
        UOp::param(2, 1, DType::Float32, None),
        slotted_var("low", 1),
        UOp::param(0, 1, DType::Float32, None),
    ]);
    let rendered = render_linearized(&sink, "mixed_abi").expect("render mixed ABI");
    assert_eq!(rendered.buffer_args.iter().map(|arg| arg.index).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(rendered.var_names, vec!["low", "high"]);
    assert_eq!(rendered.abi.len(), 4, "launch ids must not enter the ABI");
    assert!(
        rendered.code.contains(
            "kernel void mixed_abi(device float* data0, constant int& data1, device float* data2, constant int& data3, \
             uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])"
        ),
        "{}",
        rendered.code
    );
    assert_msl_compiles(&rendered.code);
}

#[test]
fn metal_volatile_buffer_keeps_the_qualifier() {
    let sink = UOp::sink(vec![volatile_param(0, 1), UOp::param(1, 1, DType::Float32, None)]);
    let code = render_ok(&sink, "volatile_params");
    assert!(
        code.contains("kernel void volatile_params(volatile device float* data0, device float* data1, uint3 gid"),
        "{code}"
    );
}

#[test]
fn metal_prelude_has_no_clang_artifacts() {
    let output = shrink(UOp::param(0, 8, DType::Float32, None), DType::Float32, 4);
    let input = shrink(UOp::param(1, 8, DType::Float32, None), DType::Float32, 4);
    let code = render_ok(&UOp::sink(vec![output.store(UOp::load().index(input).call())]), "grouped_memory");
    assert!(code.starts_with("#include <metal_stdlib>\nusing namespace metal;\n"), "{code}");
    for forbidden in ["typedef", "restrict", "ext_vector_type", "stdbool", "__builtin_"] {
        assert!(!code.contains(forbidden), "unexpected {forbidden}:\n{code}");
    }
    assert!(code.contains("float4 val0 = *((device float4*)(data1 + 0));"), "{code}");
    assert!(code.contains("*((device float4*)(data0 + 0)) = val0;"), "{code}");
}

#[test_case::test_case(DType::Bool, "bool")]
#[test_case::test_case(DType::Int8, "char")]
#[test_case::test_case(DType::UInt8, "uchar")]
#[test_case::test_case(DType::Int16, "short")]
#[test_case::test_case(DType::UInt16, "ushort")]
#[test_case::test_case(DType::Int32, "int")]
#[test_case::test_case(DType::UInt32, "uint")]
#[test_case::test_case(DType::Int64, "long")]
#[test_case::test_case(DType::UInt64, "ulong")]
#[test_case::test_case(DType::Float16, "half")]
#[test_case::test_case(DType::BFloat16, "bfloat")]
#[test_case::test_case(DType::Float32, "float")]
fn metal_scalar_type_names(dtype: DType, name: &str) {
    let sink = UOp::sink(vec![element(UOp::param(0, 4, dtype.clone(), None), 0).store(load_scalar(1, dtype))]);
    let code = render_ok(&sink, "copy");
    assert!(code.contains(&format!("device {name}* data0")), "{code}");
    assert!(code.contains(&format!("{name} val0 = *(data1 + 0L);")), "{code}");
}

#[test_case::test_case(DType::Float32, 4, "float4")]
#[test_case::test_case(DType::Float16, 4, "half4")]
#[test_case::test_case(DType::Int32, 2, "int2")]
#[test_case::test_case(DType::UInt8, 4, "uchar4")]
fn metal_vector_types_are_native(dtype: DType, lanes: i64, name: &str) {
    let output = shrink(UOp::param(0, 8, dtype.clone(), None), dtype.clone(), lanes);
    let input = shrink(UOp::param(1, 8, dtype.clone(), None), dtype, lanes);
    let code = render_ok(&UOp::sink(vec![output.store(UOp::load().index(input).call())]), "vectors");
    assert!(code.contains(&format!("{name} val0 = *((device {name}*)(data1 + 0));")), "{code}");
    assert!(code.contains(&format!("*((device {name}*)(data0 + 0)) = val0;")), "{code}");
    assert!(!code.contains("typedef"), "{code}");
}

#[test_case::test_case(ConstValue::Float(-3.2), DType::Float32 => "-3.2e0f")]
#[test_case::test_case(ConstValue::Float(1.0), DType::Float16 => "((half)1.0f)")]
#[test_case::test_case(ConstValue::Float(1.0), DType::BFloat16 => "((bfloat)1.0f)")]
#[test_case::test_case(ConstValue::Float(f64::INFINITY), DType::Float32 => "INFINITY")]
#[test_case::test_case(ConstValue::Float(f64::NEG_INFINITY), DType::Float32 => "-INFINITY")]
#[test_case::test_case(ConstValue::Float(f64::INFINITY), DType::Float16 => "((half)INFINITY)")]
#[test_case::test_case(ConstValue::Float(f64::NAN), DType::Float32 => "NAN")]
#[test_case::test_case(ConstValue::Float(f64::NAN), DType::BFloat16 => "((bfloat)NAN)")]
#[test_case::test_case(ConstValue::Int(5), DType::Int64 => "5L")]
#[test_case::test_case(ConstValue::Int(5), DType::Index => "5L")]
#[test_case::test_case(ConstValue::UInt(5), DType::UInt64 => "5UL")]
#[test_case::test_case(ConstValue::UInt(5), DType::UInt32 => "5u")]
#[test_case::test_case(ConstValue::Int(5), DType::Int32 => "5")]
fn metal_constants(value: ConstValue, dtype: DType) -> String {
    c_const(&value, &dtype, CDialect::Metal)
}

#[test_case::test_case("gidx0", "int gidx0 = gid.x;")]
#[test_case::test_case("gidx1", "int gidx1 = gid.y;")]
#[test_case::test_case("gidx2", "int gidx2 = gid.z;")]
#[test_case::test_case("lidx0", "int lidx0 = lid.x;")]
#[test_case::test_case("lidx1", "int lidx1 = lid.y;")]
#[test_case::test_case("lidx2", "int lidx2 = lid.z;")]
#[test_case::test_case("idx0", "int idx0 = gid.x;")]
#[test_case::test_case("idx1", "int idx1 = gid.y;")]
fn metal_special_axes(name: &str, expected: &str) {
    let x = UOp::param(1, 8, DType::Float32, None);
    let y = UOp::param(0, 8, DType::Float32, None);
    let axis = UOp::special_dtype(UOp::const_(DType::Int32, ConstValue::Int(8)), name.to_string(), DType::Int32);
    let load = UOp::load().index(UOp::index().buffer(x).indices(vec![axis.clone()]).call().unwrap()).call();
    let store = UOp::index().buffer(y).indices(vec![axis]).call().unwrap().store(load);
    let code = render_ok(&UOp::sink(vec![store]), "special");
    assert!(code.contains(expected), "{code}");
}

#[test]
fn metal_special_with_malformed_name_is_invalid_graph() {
    let axis = UOp::special_dtype(UOp::const_(DType::Int32, ConstValue::Int(8)), "zzz".to_string(), DType::Int32);
    let store = element(UOp::param(0, 8, DType::Int32, None), 0).store(axis);
    let err = render_linearized(&UOp::sink(vec![store]), "bad_special").expect_err("malformed SPECIAL must fail");
    assert!(
        matches!(&err, crate::Error::InvalidGraph { reason } if reason.contains("malformed SPECIAL")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn metal_barrier_emits_threadgroup_barrier() {
    let barrier = UOp::noop().barrier(SmallVec::new());
    let code = render_ok(&UOp::sink(vec![barrier]), "barrier");
    assert!(code.contains("  threadgroup_barrier(mem_flags::mem_threadgroup);"), "{code}");
}

#[test]
fn metal_local_buffer_is_threadgroup_memory() {
    let local = UOp::buffer(42, 16, DType::Float32, AddrSpace::Local, None);
    let store = element(local.clone(), 0).store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
    let code = render_ok(&UOp::sink(vec![store]), "local");
    let (head, body) = code.split_once("{\n").expect("kernel body");
    assert!(head.contains("kernel void local("), "{code}");
    assert!(body.starts_with("  threadgroup __attribute__((aligned(16))) float local42[16];\n"), "{code}");
    assert!(code.contains("*(local42 + 0L) = 1.0f;"), "{code}");
}

#[test]
fn metal_local_vector_access_carries_the_address_space() {
    let local = UOp::buffer(3, 16, DType::Float32, AddrSpace::Local, None);
    let output = shrink(UOp::param(0, 8, DType::Float32, None), DType::Float32, 4);
    let value = UOp::load().index(shrink(local, DType::Float32, 4)).call();
    let code = render_ok(&UOp::sink(vec![output.store(value)]), "local_vec");
    assert!(code.contains("*((threadgroup float4*)(local3 + 0))"), "{code}");
    assert!(code.contains("*((device float4*)(data0 + 0)) ="), "{code}");
}

#[test]
fn metal_reg_buffer_is_a_thread_array() {
    let reg = UOp::buffer(7, 4, DType::Float32, AddrSpace::Reg, None);
    let store = element(reg, 0).store(UOp::const_(DType::Float32, ConstValue::Float(2.0)));
    let code = render_ok(&UOp::sink(vec![store]), "reg");
    assert!(code.contains("  float reg0[4];"), "{code}");
    assert!(!code.contains("threadgroup float reg0"), "{code}");
}

#[test]
fn metal_stack_uses_vector_constructor() {
    let input = shrink(UOp::param(0, 8, DType::Float32, None), DType::Float32, 4);
    let lanes = (0..4).map(|lane| UOp::load().index(element(input.clone(), lane)).call()).collect();
    let output = shrink(UOp::param(1, 8, DType::Float32, None), DType::Float32, 4);
    let code = render_ok(&UOp::sink(vec![output.store(UOp::stack(lanes).detach())]), "stack");
    assert!(code.contains("float4(val0, val1, val2, val3)"), "{code}");
    assert!(!code.contains("(float4){"), "{code}");
}

#[test]
fn metal_vector_constant_uses_constructor_and_store_width_follows_value() {
    let index = element(UOp::param(0, 8, DType::Float32, None), 0);
    let value = UOp::vconst(vec![ConstValue::Float(1.0); 4], DType::Float32);
    let code = render_ok(&UOp::sink(vec![index.store(value)]), "vconst");
    assert!(code.contains("*((device float4*)(data0 + 0L)) = float4(1.0f, 1.0f, 1.0f, 1.0f);"), "{code}");
}

#[test]
fn metal_bitcast_uses_as_type() {
    let code = render_ok(&unary_sink(DType::Float32, |x| x.bitcast(DType::Int32)), "bitcast");
    assert!(code.contains("as_type<int>((float)(val0))"), "{code}");
    assert!(!code.contains("__builtin_bit_cast"), "{code}");
}

#[test]
fn metal_address_casts_carry_the_address_space() {
    let address = element(UOp::param(0, 1, DType::Float32, None), 0).cast(DType::Int32);
    let output = element(UOp::param(1, 1, DType::Int32, None), 0);
    let code = render_ok(&UOp::sink(vec![output.store(UOp::load().index(address).call())]), "address_cast");
    assert!(code.contains("((device int*)(data0 + 0L))"), "{code}");
}

#[test]
fn metal_hoists_addresses_with_their_address_space() {
    let param = UOp::param(0, 8, DType::Float32, None);
    let bound = UOp::const_(DType::Int32, ConstValue::Int(4));
    let range = UOp::range_axis_dtype(bound.clone(), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    let index = UOp::index().buffer(param.clone()).indices(vec![range.clone()]).call().unwrap();
    let load = UOp::load().index(index.clone()).call();
    let end = load.end(smallvec::smallvec![range.clone()]);
    let store = index.clone().store(load.clone());
    let linear = UOp::linear(smallvec::smallvec![bound, param, range, index, load, end, store]);

    let rendered = render_metal(&linear, Some("escaping_address")).expect("render escaping address");
    assert!(rendered.code.contains("  device float* bidx0;"), "{}", rendered.code);
    assert_msl_compiles(&rendered.code);
}

#[test_case::test_case(DType::Float32, |x: &std::sync::Arc<UOp>| x.try_sqrt().unwrap(), "= sqrt(val0);"; "sqrt f32")]
#[test_case::test_case(DType::Float32, |x: &std::sync::Arc<UOp>| x.try_sin().unwrap(), "= precise::sin(val0);"; "sin uses precise")]
#[test_case::test_case(DType::Float32, |x: &std::sync::Arc<UOp>| x.try_exp2().unwrap(), "= exp2(val0);"; "exp2 f32")]
#[test_case::test_case(DType::Float32, |x: &std::sync::Arc<UOp>| x.try_log2().unwrap(), "= log2(val0);"; "log2 f32")]
#[test_case::test_case(DType::Float16, |x: &std::sync::Arc<UOp>| x.try_sqrt().unwrap(), "= ((half)sqrt(val0));"; "half rounds back")]
#[test_case::test_case(DType::BFloat16, |x: &std::sync::Arc<UOp>| x.try_sqrt().unwrap(), "= ((bfloat)sqrt((float)val0));"; "bfloat promotes")]
#[test_case::test_case(DType::Float32, |x: &std::sync::Arc<UOp>| x.try_rsqrt().unwrap(), "= (1.0f / sqrt(val0));"; "rsqrt")]
#[test_case::test_case(DType::Float32, |x: &std::sync::Arc<UOp>| x.try_mul(x).unwrap().try_add(x).unwrap(), "= ((val0 * val0) + val0);"; "arithmetic")]
fn metal_math_functions(dtype: DType, f: fn(&std::sync::Arc<UOp>) -> std::sync::Arc<UOp>, expected: &str) {
    let code = render_ok(&unary_sink(dtype, f), "math");
    assert!(code.contains(expected), "{code}");
    assert!(!code.contains("__builtin_"), "{code}");
}

#[test]
fn metal_narrow_int_arithmetic_casts_back() {
    let a = load_scalar(1, DType::Int8);
    let sink = UOp::sink(vec![
        element(UOp::param(0, 4, DType::Int32, None), 0).store(a.try_add(&a).unwrap().cast(DType::Int32)),
    ]);
    let code = render_ok(&sink, "narrow");
    assert!(code.contains("((char)(val0 + val0))"), "{code}");
}

fn reduce_sink(value: f64, extent: i64, op: ReduceOp) -> std::sync::Arc<UOp> {
    let range = concrete_range(extent, AxisType::Reduce);
    let reduce = UOp::const_(DType::Float32, ConstValue::Float(value)).reduce(smallvec::smallvec![range.clone()], op);
    UOp::sink(vec![reduce.end(smallvec::smallvec![range])])
}

#[test_case::test_case(reduce_sink(5.0, 10, ReduceOp::Add), &["float acc", " = 0.0f;", "for (int ridx0 = 0; ridx0 < 10; ridx0++) {", " += 5.0f;"]; "sum")]
#[test_case::test_case(reduce_sink(3.0, 5, ReduceOp::Max), &[" = -INFINITY;", " = fmax(acc"]; "max")]
#[test_case::test_case(reduce_sink(3.0, 5, ReduceOp::Min), &[" = INFINITY;", " = fmin(acc"]; "min")]
fn metal_reduce_structure(sink: std::sync::Arc<UOp>, needles: &[&str]) {
    let code = render_ok(&sink, "reduce");
    for needle in needles {
        assert!(code.contains(needle), "missing {needle}:\n{code}");
    }
    assert!(!code.contains("__builtin_inf"), "{code}");
}

fn metal_wmma_meta(dims: (usize, usize, usize), dtype_in: DType, dtype_out: DType) -> WmmaMetadata {
    let name = |dtype: &DType| match dtype.base() {
        ScalarDType::Float32 => "float",
        ScalarDType::Float16 => "half",
        ScalarDType::BFloat16 => "bfloat",
        other => panic!("unexpected WMMA dtype {other:?}"),
    };
    WmmaMetadata {
        name: format!("WMMA_{}_{}_{}_{}_{}", dims.0, dims.1, dims.2, name(&dtype_in), name(&dtype_out)),
        dims,
        dtype_in,
        dtype_out,
        device: RendererDevice::Metal,
        threads: 32,
        upcast_axes: None,
        reduce_axes: vec![],
    }
}

/// `D = A*B + C` over 2-lane per-thread fragments loaded from three buffers.
fn wmma_sink(meta: WmmaMetadata, c_slot: usize) -> std::sync::Arc<UOp> {
    let a = load_scalar(0, meta.dtype_in.clone()).broadcast(2);
    let b = load_scalar(1, meta.dtype_in.clone()).broadcast(2);
    let c = load_scalar(c_slot, meta.dtype_out.clone()).broadcast(2);
    UOp::sink(vec![UOp::wmma(a, b, c, meta)])
}

#[test_case::test_case(DType::Float32, DType::Float32, "float", "float")]
#[test_case::test_case(DType::Float16, DType::Float32, "half", "float")]
#[test_case::test_case(DType::Float16, DType::Float16, "half", "half")]
#[test_case::test_case(DType::BFloat16, DType::Float32, "bfloat", "float")]
#[test_case::test_case(DType::BFloat16, DType::BFloat16, "bfloat", "bfloat")]
fn metal_wmma_lowers_to_simdgroup_matrix(dtype_in: DType, dtype_out: DType, inp: &str, out: &str) {
    let code = render_ok(&wmma_sink(metal_wmma_meta((8, 8, 8), dtype_in, dtype_out), 2), "wmma");
    let name = format!("WMMA_8_8_8_{inp}_{out}");
    for needle in [
        format!("{out}2 __{name}({inp}2 a, {inp}2 b, {out}2 c){{"),
        format!("  simdgroup_{inp}8x8 mat_a, mat_b; simdgroup_{out}8x8 mat_c;"),
        "  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);".to_string(),
        format!("  return {out}2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);"),
        format!("{out}2 wmma"),
        format!(" = __{name}("),
    ] {
        assert!(code.contains(&needle), "missing {needle}:\n{code}");
    }
    let (prefix, _) = code.split_once("kernel void").expect("kernel");
    assert!(prefix.contains(&format!("__{name}(")), "helper must precede the kernel:\n{code}");
}

#[test]
fn metal_wmma_helpers_are_deduped_per_shape() {
    let meta = metal_wmma_meta((8, 8, 8), DType::Float16, DType::Float32);
    let a = load_scalar(0, DType::Float16).broadcast(2);
    let b = load_scalar(1, DType::Float16).broadcast(2);
    let first = UOp::wmma(a.clone(), b.clone(), load_scalar(2, DType::Float32).broadcast(2), meta.clone());
    let second = UOp::wmma(a, b, load_scalar(3, DType::Float32).broadcast(2), meta);
    let code = render_ok(&UOp::sink(vec![first, second]), "wmma_dedup");
    assert_eq!(code.matches("float2 __WMMA_8_8_8_half_float(").count(), 1, "{code}");
    assert_eq!(code.matches("= __WMMA_8_8_8_half_float(").count(), 2, "{code}");
}

#[test]
fn metal_wmma_rejects_non_simdgroup_shapes() {
    let err = render_linearized(&wmma_sink(metal_wmma_meta((16, 16, 16), DType::Float16, DType::Float32), 2), "wmma16")
        .expect_err("16x16x16 has no Metal lowering");
    assert!(
        matches!(&err, crate::Error::InvalidGraph { reason } if reason.contains("8x8x8")),
        "unexpected error: {err:?}"
    );
}

#[test_case::test_case(UOp::const_(DType::Float64, ConstValue::Float(1.0)), "no double type"; "float64")]
#[test_case::test_case(UOp::const_(DType::FP8E4M3, ConstValue::Float(1.0)), "no fp8 type"; "fp8")]
#[test_case::test_case(UOp::const_(DType::FP8E5M2FNUZ, ConstValue::Float(1.0)), "Metal renderer does not support FP8E5M2FNUZ"; "fnuz")]
#[test_case::test_case(UOp::vconst(vec![ConstValue::Float(0.0); 8], DType::Float32), "only 2-, 3- and 4-component vectors"; "wide vector")]
fn metal_rejects_unsupported_dtypes(node: std::sync::Arc<UOp>, reason: &str) {
    let err = render_metal(&UOp::linear(vec![node].into()), Some("bad_dtype")).expect_err("must be rejected");
    assert!(
        matches!(&err, crate::Error::TypeError { reason: actual } if actual.contains(reason)),
        "unexpected error: {err:?}"
    );
}

/// `out[i] = a[i] + b[i]` over 1024 floats as the optimizer lays it out for
/// Metal: 256 threadgroups of 4 threads.
#[test]
fn metal_golden_elementwise_add() {
    let g = UOp::special_dtype(UOp::const_(DType::Int32, ConstValue::Int(256)), "gidx0".to_string(), DType::Int32);
    let l = UOp::special_dtype(UOp::const_(DType::Int32, ConstValue::Int(4)), "lidx0".to_string(), DType::Int32);
    let idx = g.try_mul(&UOp::const_(DType::Int32, ConstValue::Int(4))).unwrap().try_add(&l).unwrap();
    let at = |slot| {
        UOp::index().buffer(UOp::param(slot, 1024, DType::Float32, None)).indices(vec![idx.clone()]).call().unwrap()
    };
    let a = UOp::load().index(at(1)).call();
    let b = UOp::load().index(at(2)).call();
    let code = render_ok(&UOp::sink(vec![at(0).store(a.try_add(&b).unwrap())]), "add1024");

    let expected = [
        "kernel void add1024(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {",
        "  int gidx0 = gid.x;",
        "  int lidx0 = lid.x;",
        "  int alu0 = ((gidx0 * 4) + lidx0);",
        "  float val1 = *(data1 + alu0);",
        "  float val2 = *(data2 + alu0);",
        "  *(data0 + alu0) = (val1 + val2);",
        "}",
    ];
    let mut rest = code.as_str();
    for line in expected {
        let Some(pos) = rest.find(line) else { panic!("missing/out of order {line:?}:\n{code}") };
        rest = &rest[pos + line.len()..];
    }
}

/// `out[g] = sum(local[0..16])` with the threadgroup filled by the 16 threads
/// and a barrier before the reduction reads it.
#[test]
fn metal_golden_threadgroup_reduction() {
    let g = UOp::special_dtype(UOp::const_(DType::Int32, ConstValue::Int(64)), "gidx0".to_string(), DType::Int32);
    let l = UOp::special_dtype(UOp::const_(DType::Int32, ConstValue::Int(16)), "lidx0".to_string(), DType::Int32);
    let local = UOp::buffer(0, 16, DType::Float32, AddrSpace::Local, None);
    let input = UOp::param(1, 1024, DType::Float32, None);
    let output = UOp::param(0, 64, DType::Float32, None);

    let flat = g.try_mul(&UOp::const_(DType::Int32, ConstValue::Int(16))).unwrap().try_add(&l).unwrap();
    let load = UOp::load().index(UOp::index().buffer(input).indices(vec![flat]).call().unwrap()).call();
    let fill = UOp::index().buffer(local.clone()).indices(vec![l]).call().unwrap().store(load);
    let barrier = fill.barrier(SmallVec::new());
    let ready = local.after(smallvec::smallvec![barrier]);

    let range = concrete_range(16, AxisType::Reduce);
    let partial = UOp::load().index(UOp::index().buffer(ready).indices(vec![range.clone()]).call().unwrap()).call();
    let sum = partial.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);
    let store = UOp::index().buffer(output).indices(vec![g]).call().unwrap().store(sum.clone());
    let code = render_ok(&UOp::sink(vec![sum.end(smallvec::smallvec![range]), store]), "local_sum");

    let expected = [
        "  threadgroup __attribute__((aligned(16))) float local0[16];",
        "  float acc",
        " = 0.0f;",
        "  int gidx0 = gid.x;",
        "  int lidx0 = lid.x;",
        "  *(local0 + lidx0) = val",
        "  threadgroup_barrier(mem_flags::mem_threadgroup);",
        "  for (int ridx0 = 0; ridx0 < 16; ridx0++) {",
        " = *(local0 + ridx0);",
        " += val",
        "  }",
        "  *(data0 + gidx0) = acc",
    ];
    let mut rest = code.as_str();
    for line in expected {
        let Some(pos) = rest.find(line) else { panic!("missing/out of order {line:?}:\n{code}") };
        rest = &rest[pos + line.len()..];
    }
}

#[test]
fn clang_dialect_output_is_unchanged() {
    let output = shrink(UOp::param(0, 8, DType::Float32, None), DType::Float32, 4);
    let input = shrink(UOp::param(1, 8, DType::Float32, None), DType::Float32, 4);
    let sink = UOp::sink(vec![output.store(UOp::load().index(input).call())]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());
    let clang = render(&linear, Some("grouped")).expect("clang").code;
    let metal = render_metal(&linear, Some("grouped")).expect("metal").code;
    for needle in ["#include <stdbool.h>", "* restrict ", "typedef float float4", "void grouped("] {
        assert!(clang.contains(needle), "missing {needle}:\n{clang}");
    }
    for needle in ["kernel void", "device ", "metal_stdlib"] {
        assert!(!clang.contains(needle), "unexpected {needle}:\n{clang}");
        assert!(metal.contains(needle), "missing {needle}:\n{metal}");
    }
}

/// Pipe `src` through `xcrun metal -fsyntax-only` and assert it parses.
/// Returns without asserting when the Metal toolchain is unusable: `xcrun --find
/// metal` resolves even when the Metal Toolchain component is not installed
/// ("cannot execute tool 'metal' due to missing Metal Toolchain"), so presence
/// is not a usable signal — only a probe compile is. The probe also skips
/// hosts without `xcrun` and toolchains older than MSL 3.1 (needed for the
/// `bfloat` simdgroup helpers). Mirrors `assert_c_compiles` in `c.rs`.
fn assert_msl_compiles(src: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let run = |src: &str| -> Option<(bool, String)> {
        let mut child = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-x", "metal", "-std=metal3.1", "-fno-fast-math", "-fsyntax-only", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;
        child.stdin.take()?.write_all(src.as_bytes()).ok()?;
        let output = child.wait_with_output().ok()?;
        Some((output.status.success(), String::from_utf8_lossy(&output.stderr).into_owned()))
    };

    let probe = "#include <metal_stdlib>\nusing namespace metal;\n\
                 kernel void p(device float* a, uint3 gid [[threadgroup_position_in_grid]]) { a[gid.x] = 1.0f; }\n";
    let Some((true, _)) = run(probe) else { return };

    let (ok, stderr) = run(src).expect("metal probe already succeeded");
    assert!(ok, "metal rejected the emitted MSL:\n{src}\n--- metal stderr ---\n{stderr}");
}
