use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{BinaryOp, CallInfo, Error, Op, UOp};
use test_case::test_case;

use crate::rangeify::{rangeify, transforms::resolve_calls};
use svod_ir::ops;

/// Helper: peel a TUPLE wrapper. Per tinygrad, FUNCTION bodies are TUPLE-wrapped
/// value producers; the inlined result of `resolve_function` is the substituted TUPLE.
fn peel_tuple(uop: &Arc<UOp>) -> &Arc<UOp> {
    match uop.op() {
        Op::Tuple(ops::Tuple { src }) if src.len() == 1 => &src[0],
        _ => uop,
    }
}

#[test]
fn test_resolve_call_inlines_function() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a1 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let function = body.function(smallvec![a0.clone(), a1.clone()], CallInfo::default());

    let resolved = resolve_calls(function).expect("resolve_calls should succeed");
    // Inlined body is wrapped in a single-element TUPLE per tinygrad invariant.
    let inlined = peel_tuple(&resolved);
    match inlined.op() {
        Op::Binary(BinaryOp::Add, lhs, rhs) => {
            assert!(Arc::ptr_eq(lhs, &a0));
            assert!(Arc::ptr_eq(rhs, &a1));
        }
        op => panic!("expected inlined add body, got {op:?}"),
    }
    assert!(!resolved.toposort().iter().any(|u| matches!(u.op(), Op::Function(..))));
}

fn value_body() -> Arc<UOp> {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    p0.try_add(&UOp::param(1, 8, DType::Float32, None)).unwrap()
}

/// PROGRAM and SINK are in tinygrad's `_OPAQUE_CALL_BODIES` (`ops.py:933`); a
/// plain value body is opaque too once it is wrapped in CALL rather than
/// FUNCTION, and a nested CALL stays nested.
fn program_body() -> Arc<UOp> {
    let sink = UOp::sink(vec![]);
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    UOp::program(sink, info, None, None, None)
}

fn kernel_sink_body() -> Arc<UOp> {
    UOp::sink_with_info(vec![UOp::native_const(1.0f32)], svod_ir::KernelInfo::default())
}

fn sink_with_unrelated_metadata() -> Arc<UOp> {
    #[derive(Debug)]
    struct UnrelatedMarker;
    UOp::sink(vec![UOp::param(0, 8, DType::Float32, None)]).with_metadata(UnrelatedMarker)
}

fn nested_call_body() -> Arc<UOp> {
    UOp::native_const(1.0f32).call(smallvec![], CallInfo::default())
}

#[test_case(super::value_body ; "arithmetic body")]
#[test_case(super::program_body ; "program body")]
#[test_case(super::kernel_sink_body ; "kernel sink body")]
#[test_case(super::sink_with_unrelated_metadata ; "sink carrying unrelated metadata")]
#[test_case(super::nested_call_body ; "nested call body")]
fn a_call_body_is_never_inlined(build: fn() -> Arc<UOp>) {
    let body = build();
    let args: smallvec::SmallVec<[Arc<UOp>; 4]> =
        (0..2).map(|_| UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32)).collect();
    let call = body.clone().call(args.clone(), CallInfo::default());

    let resolved = resolve_calls(call).expect("resolve_calls should succeed");
    let Op::Call(ops::Call { body: resolved_body, args: resolved_args, .. }) = resolved.op() else {
        panic!("expected the CALL to survive, got {}", resolved.tree())
    };
    assert!(Arc::ptr_eq(resolved_body, &body), "the body must be untouched");
    assert!(resolved_args.iter().zip(&args).all(|(a, b)| Arc::ptr_eq(a, b)));
}

#[test]
fn test_resolve_call_preserves_precompile_function() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let body = p0.try_sqrt().unwrap();
    let arg = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let info = CallInfo { precompile: true, ..CallInfo::default() };
    let function = body.function(smallvec![arg.clone()], info);

    let resolved = resolve_calls(function).expect("resolve_calls should succeed");
    match resolved.op() {
        Op::Function(ops::Function { body, args, info }) => {
            // FUNCTION body is now TUPLE-wrapped per tinygrad invariant.
            let Op::Tuple(ops::Tuple { src }) = body.op() else {
                panic!("expected FUNCTION body to be TUPLE, got {:?}", body.op())
            };
            assert_eq!(src.len(), 1);
            assert!(matches!(src[0].op(), Op::Unary(_, _)));
            assert_eq!(args.len(), 1);
            assert!(Arc::ptr_eq(&args[0], &arg));
            assert!(info.precompile);
        }
        op => panic!("expected precompile FUNCTION to be preserved, got {op:?}"),
    }
}

#[test]
fn test_resolve_call_preserves_precompile_gettuple_and_actual() {
    let formal = UOp::param(0, 8, DType::Float32, None);
    let actual = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let info = CallInfo { precompile: true, ..CallInfo::default() };
    let function = formal.function(smallvec![actual.clone()], info);
    let gettuple = function.try_gettuple(0).unwrap();

    let resolved = resolve_calls(gettuple).expect("precompiled FUNCTION must remain opaque");
    let Op::GetTuple(ops::GetTuple { src, index: 0 }) = resolved.op() else { panic!("expected GETTUPLE root") };
    let Op::Function(ops::Function { args, info, .. }) = src.op() else { panic!("expected preserved FUNCTION source") };
    assert!(info.precompile);
    assert_eq!(args.len(), 1);
    assert!(Arc::ptr_eq(&args[0], &actual));
}

#[test]
fn test_resolve_call_keeps_nested_function_under_opaque_outer_function() {
    let nested_formal = UOp::param(0, 8, DType::Float32, None);
    let nested_actual = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let nested = nested_formal.function(smallvec![nested_actual], CallInfo::default());
    let outer_info = CallInfo { precompile: true, ..CallInfo::default() };
    let outer = nested.function(smallvec![], outer_info);

    let resolved = resolve_calls(outer).expect("opaque outer FUNCTION must preserve its body");
    let Op::Function(ops::Function { body, info, .. }) = resolved.op() else { panic!("expected outer FUNCTION") };
    assert!(info.precompile);
    assert!(
        body.toposort()
            .iter()
            .any(|node| matches!(node.op(), Op::Function(ops::Function { info, .. }) if !info.precompile))
    );
}

/// Tinygrad parity: BIND is value-producing (not in `_OPAQUE_CALL_BODIES`), so a
/// FUNCTION wrapping it gets inlined like any value body. With no PARAMs the
/// substitution is a no-op and the resolved value is the TUPLE-wrapped body.
#[test]
fn test_resolve_call_inlines_bind_body_function() {
    let var = UOp::define_var("N".to_string(), 0, 32);
    let bind = var.bind(UOp::index_const(8));
    let function = bind.function(smallvec![], CallInfo::default());

    let resolved = resolve_calls(function).expect("resolve_calls should succeed");
    let inlined = peel_tuple(&resolved);
    assert!(matches!(inlined.op(), Op::Bind(..)));
    assert!(!resolved.toposort().iter().any(|u| matches!(u.op(), Op::Function(..))));
}

/// SINK with non-kernel metadata still requires CALL (opaque body).
#[test]
fn test_resolve_call_allows_non_contiguous_param_slots_with_unused_args() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p2 = UOp::param(2, 8, DType::Float32, None);
    let body = p0.try_add(&p2).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a1 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a2 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let function = body.function(smallvec![a0.clone(), a1, a2.clone()], CallInfo::default());

    let resolved = resolve_calls(function).expect("unused argument slots should be allowed");
    assert!(resolved.toposort().iter().all(|u| !matches!(u.op(), Op::Param(..))));
    assert!(resolved.toposort().iter().any(|u| Arc::ptr_eq(u, &a0)));
    assert!(resolved.toposort().iter().any(|u| Arc::ptr_eq(u, &a2)));
}

/// Every actual argument must line up with its formal PARAM: a missing slot, a
/// different extent and a different dtype are all typed errors.
#[test]
fn a_mismatched_actual_argument_is_a_typed_error() {
    let two_params =
        value_body().function(smallvec![UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32)], CallInfo::default());
    let sqrt = |buffer| {
        UOp::param(0, 8, DType::Float32, None).try_sqrt().unwrap().function(smallvec![buffer], CallInfo::default())
    };
    let wrong_shape = sqrt(UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32));
    let wrong_dtype = sqrt(UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Int32));

    let err = |f| match resolve_calls(f) {
        Err(err) => err,
        Ok(resolved) => panic!("expected a typed error, got {}", resolved.tree()),
    };
    assert!(matches!(err(two_params), Error::CallFormalSlotMissing { slot: 1, arg_count: 1 }));
    assert!(matches!(err(wrong_shape), Error::CallArgShapeMismatch { arg_index: 0, .. }));
    assert!(matches!(err(wrong_dtype), Error::CallArgDTypeMismatch { arg_index: 0, .. }));
}

#[test]
fn test_rangeify_pipeline_runs_resolve_call() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a1 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let function = body.function(smallvec![a0, a1], CallInfo::default());

    let (out, _ctx) = rangeify(function).expect("rangeify should succeed");
    assert!(!out.toposort().iter().any(|u| matches!(u.op(), Op::Function(..))));
}

#[test]
fn test_rangeify_consumes_expression_valued_function_result_shape() {
    let p1 = UOp::scalar_param(1, Some("p1".into()), DType::WeakInt, 1, 8);
    let extent = p1.try_add(&p1).unwrap();
    let formal = UOp::param_with_shape(0, &smallvec![svod_ir::SInt::Symbolic(extent)], DType::Float32, None);
    let actual_dim = UOp::define_var("actual".into(), 1, 8);
    let actual_extent = actual_dim.try_add(&actual_dim).unwrap();
    let actual =
        UOp::param_with_shape(7, &smallvec![svod_ir::SInt::Symbolic(actual_extent.clone())], DType::Float32, None);
    let output = formal.function(smallvec![actual, actual_dim], CallInfo::default()).try_gettuple(0).unwrap();

    assert_eq!(output.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Symbolic(actual_extent)]);
    let (resolved, _) = rangeify(output).expect("rangeify should consume substituted call shape");
    assert!(resolved.toposort().iter().all(|node| !matches!(node.op(), Op::Function(..))));
}

#[test]
fn test_rangeify_preserves_kernel_call_body_boundaries() {
    let detached = UOp::native_const(1.0f32).detach();
    let body = UOp::sink_with_info(vec![detached], svod_ir::KernelInfo::default());
    let function = body.call(smallvec![], CallInfo::default());

    let (out, _ctx) = rangeify(function).expect("rangeify should succeed");
    let call_node =
        out.toposort().into_iter().find(|u| matches!(u.op(), Op::Call(..))).expect("kernel call should be preserved");

    let Op::Call(ops::Call { body, .. }) = call_node.op() else { unreachable!("filtered to call node") };
    assert!(
        body.toposort().iter().any(|u| matches!(u.op(), Op::Detach(..))),
        "call-preserving rangeify should not rewrite inside preserved kernel call bodies"
    );
}

#[test]
fn test_resolve_call_does_not_inline_function_inside_opaque_call_body() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let nested_body = p0.try_add(&p1).unwrap();
    let nested_function = nested_body.function(
        smallvec![
            UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32),
            UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32)
        ],
        CallInfo::default(),
    );
    let opaque_call = nested_function.call(smallvec![], CallInfo::default());

    let resolved = resolve_calls(opaque_call).expect("resolve_calls should preserve call body");
    let Op::Call(ops::Call { body, .. }) = resolved.op() else { panic!("expected CALL root") };
    assert!(
        body.toposort().iter().any(|u| matches!(u.op(), Op::Function(..))),
        "FUNCTION inside CALL body must remain unresolved"
    );
}
