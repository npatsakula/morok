//! Symbolic-shape support across the reductions.
//!
//! Every reduction must accept a tensor whose *non-reduced* dims are symbolic
//! (a JIT batch, say) and carry them into the output shape. Only the *reduced*
//! axis has to be concrete, and only for the ops that need its extent as a
//! number: the divisor of `mean`/`var`/`std`, the tie-break ramp of
//! `argmax`/`argmin`, and the pad/pool extents of `cumsum`/`cumprod`.

use svod_dtype::DType;
use svod_ir::SInt;
use test_case::test_case;

use crate::error::ErrorKind;
use crate::test::helpers::*;
use crate::{Result, Tensor, Variable};

/// `[b, 3]` with `b` a batch variable bound to 4 — nothing is realized.
fn symbolic() -> Tensor {
    Tensor::empty_dynamic(&[Variable::new("b", 1, 8).bind(4).unwrap().as_sint(), SInt::Const(3)], DType::Float32)
}

/// `[2, 3]`, the concrete counterpart of [`symbolic`].
fn concrete() -> Tensor {
    Tensor::from_slice([1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0]).try_reshape([2, 3]).unwrap()
}

/// The reductions under test, applied along one axis.
fn reduce(op: &str, t: &Tensor, axis: isize, keepdim: bool) -> Result<Tensor> {
    match op {
        "sum" => t.sum_with().axes(axis).keepdim(keepdim).call(),
        "mean" => t.mean_with().axes(axis).keepdim(keepdim).call(),
        "var" => t.var_with().axes(axis).keepdim(keepdim).call(),
        "std" => t.std_with().axes(axis).keepdim(keepdim).call(),
        "var_mean" => t.var_mean_with().axes(axis).keepdim(keepdim).call().map(|(var, _)| var),
        "std_mean" => t.std_mean_with().axes(axis).keepdim(keepdim).call().map(|(std, _)| std),
        "argmax" => t.argmax_with().axis(axis).keepdim(keepdim).call(),
        "argmin" => t.argmin_with().axis(axis).keepdim(keepdim).call(),
        "any" => t.any_with().axes(axis).keepdim(keepdim).call(),
        "all" => t.all_with().axes(axis).keepdim(keepdim).call(),
        _ => unreachable!("unknown reduction {op}"),
    }
}

/// The cumulative ops, which keep the input shape.
fn cumulative(op: &str, t: &Tensor, axis: isize) -> Result<Tensor> {
    match op {
        "cumsum" => t.cumsum(axis),
        "cumprod" => t.cumprod(axis),
        "cumsum_exclusive" => t.cumsum_with().axis(axis).exclusive(true).call(),
        "cumprod_exclusive" => t.cumprod_with().axis(axis).exclusive(true).call(),
        "cumsum_reverse" => t.cumsum_with().axis(axis).reverse(true).call(),
        "cumprod_reverse" => t.cumprod_with().axis(axis).reverse(true).call(),
        _ => unreachable!("unknown cumulative op {op}"),
    }
}

// =========================================================================
// Non-reduced symbolic dims survive
// =========================================================================

#[test_case("sum")]
#[test_case("mean")]
#[test_case("var")]
#[test_case("std")]
#[test_case("var_mean")]
#[test_case("std_mean")]
#[test_case("argmax")]
#[test_case("argmin")]
#[test_case("any")]
#[test_case("all")]
fn a_symbolic_batch_survives_a_reduction_over_a_concrete_axis(op: &str) {
    let shape = reduce(op, &symbolic(), -1, false).unwrap().shape().unwrap();
    assert_eq!(shape.len(), 1, "{op}: {shape:?}");
    assert!(shape[0].is_symbolic(), "{op}: the batch must stay symbolic, got {shape:?}");
}

#[test_case("sum")]
#[test_case("mean")]
#[test_case("var")]
#[test_case("std")]
#[test_case("var_mean")]
#[test_case("std_mean")]
#[test_case("argmax")]
#[test_case("argmin")]
#[test_case("any")]
#[test_case("all")]
fn a_symbolic_batch_survives_a_keepdim_reduction(op: &str) {
    let shape = reduce(op, &symbolic(), -1, true).unwrap().shape().unwrap();
    assert_eq!(shape.len(), 2, "{op}: {shape:?}");
    assert!(shape[0].is_symbolic(), "{op}: the batch must stay symbolic, got {shape:?}");
    assert_eq!(shape[1].as_const(), Some(1), "{op}: {shape:?}");
}

#[test_case("cumsum")]
#[test_case("cumprod")]
#[test_case("cumsum_exclusive")]
#[test_case("cumprod_exclusive")]
#[test_case("cumsum_reverse")]
#[test_case("cumprod_reverse")]
fn a_symbolic_batch_survives_a_cumulative_op_over_a_concrete_axis(op: &str) {
    let shape = cumulative(op, &symbolic(), -1).unwrap().shape().unwrap();
    assert_eq!(shape.len(), 2, "{op}: {shape:?}");
    assert!(shape[0].is_symbolic(), "{op}: the batch must stay symbolic, got {shape:?}");
    assert_eq!(shape[1].as_const(), Some(3), "{op}: {shape:?}");
}

// =========================================================================
// The same ops on the concrete counterpart
// =========================================================================

#[test_case("sum")]
#[test_case("mean")]
#[test_case("var")]
#[test_case("std")]
#[test_case("var_mean")]
#[test_case("std_mean")]
#[test_case("argmax")]
#[test_case("argmin")]
#[test_case("any")]
#[test_case("all")]
fn a_concrete_tensor_reduces_to_the_same_rank(op: &str) {
    assert_eq!(reduce(op, &concrete(), -1, false).unwrap().dims().unwrap(), vec![2], "{op}");
    assert_eq!(reduce(op, &concrete(), -1, true).unwrap().dims().unwrap(), vec![2, 1], "{op}");
}

#[test_case("cumsum")]
#[test_case("cumprod")]
#[test_case("cumsum_exclusive")]
#[test_case("cumprod_exclusive")]
#[test_case("cumsum_reverse")]
#[test_case("cumprod_reverse")]
fn a_concrete_cumulative_op_keeps_the_input_shape(op: &str) {
    assert_eq!(cumulative(op, &concrete(), -1).unwrap().dims().unwrap(), vec![2, 3], "{op}");
}

// =========================================================================
// A symbolic *reduced* axis: supported where the algorithm allows it
// =========================================================================

#[test_case("sum")]
#[test_case("any")]
#[test_case("all")]
fn a_plain_reduce_accepts_a_symbolic_reduced_axis(op: &str) {
    // No divisor, no index ramp, no pad extent — the reduce is shape-agnostic.
    assert_eq!(reduce(op, &symbolic(), 0, false).unwrap().dims().unwrap(), vec![3], "{op}");
}

#[test_case("mean"; "mean needs the element count as a divisor")]
#[test_case("var"; "var needs the element count as a divisor")]
#[test_case("std"; "std needs the element count as a divisor")]
#[test_case("var_mean"; "var_mean needs the element count as a divisor")]
#[test_case("std_mean"; "std_mean needs the element count as a divisor")]
#[test_case("argmax"; "argmax needs a tie-break ramp of the axis extent")]
#[test_case("argmin"; "argmin needs a tie-break ramp of the axis extent")]
fn a_symbolic_reduced_axis_is_rejected_with_a_clear_error(op: &str) {
    let err = reduce(op, &symbolic(), 0, false).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::SymbolicShapeUnsupported { .. }), "{op}: {err:?}");
}

#[test_case("cumsum")]
#[test_case("cumprod")]
#[test_case("cumsum_exclusive")]
#[test_case("cumprod_exclusive")]
#[test_case("cumsum_reverse")]
#[test_case("cumprod_reverse")]
fn a_cumulative_op_rejects_a_symbolic_axis(op: &str) {
    // pad → pool → reduce needs the axis extent as a kernel size.
    let err = cumulative(op, &symbolic(), 0).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::SymbolicShapeUnsupported { .. }), "{op}: {err:?}");
}

#[test_case("argmax")]
#[test_case("argmin")]
fn a_flattening_arg_reduce_rejects_any_symbolic_dim(op: &str) {
    // `axis = None` folds every dim into the reduced one.
    let t = symbolic();
    let err = if op == "argmax" { t.argmax(None) } else { t.argmin(None) }.unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::SymbolicShapeUnsupported { .. }), "{op}: {err:?}");
}

// =========================================================================
// Values on the concrete path
// =========================================================================

crate::codegen_tests! {
    fn concrete_values_match_the_reference(config) {
        test_setup();
        let t = concrete();
        assert_eq!(t.argmax(-1).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [1, 2]);
        assert_eq!(t.argmin(-1).unwrap().realize_with_and(&config).as_vec::<i32>().unwrap(), [0, 1]);
        assert_close_f32(&t.mean(-1isize).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(), &[3.0, 4.0], 1e-6);
        assert_close_f32(
            &t.cumsum(-1).unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(),
            &[1.0, 6.0, 9.0, 4.0, 6.0, 12.0],
            1e-6,
        );
        assert_close_f32(
            &t.cumsum_with().axis(-1).exclusive(true).call().unwrap().realize_with_and(&config).as_vec::<f32>().unwrap(),
            &[0.0, 1.0, 6.0, 0.0, 4.0, 6.0],
            1e-6,
        );
    }
}
