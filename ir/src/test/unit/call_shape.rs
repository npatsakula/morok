use std::sync::Arc;

use smallvec::{SmallVec, smallvec};

use crate::ops;
use crate::{CallInfo, DType, Error, Op, SInt, UOp};

fn symbolic_buffer(slot: usize, shape: SmallVec<[SInt; 4]>) -> Arc<UOp> {
    UOp::param_with_shape(slot, &shape, DType::Float32, None)
}

#[test]
fn function_result_shape_substitutes_direct_param() {
    let formal_dim = UOp::scalar_param(1, Some("p1".into()), DType::WeakInt, 1, 8);
    let formal = symbolic_buffer(0, smallvec![SInt::Symbolic(formal_dim)]);
    let n = UOp::define_var("n".into(), 1, 8);
    let bound = n.bind(UOp::index_const(5));
    let actual = symbolic_buffer(9, smallvec![SInt::Symbolic(bound.clone())]);

    let result =
        formal.try_function(smallvec![actual, bound.clone()], CallInfo::default()).unwrap().try_gettuple(0).unwrap();
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Symbolic(bound)]);
}

#[test]
fn function_result_shape_substitutes_nested_expression_and_repeated_formal() {
    let p2 = UOp::scalar_param(2, Some("p2".into()), DType::WeakInt, 1, 8);
    let extent = p2.try_add(&p2).unwrap().try_mul(&p2.const_like(2i32)).unwrap();
    let formal = symbolic_buffer(0, smallvec![SInt::Symbolic(extent), SInt::Const(4)]);

    let actual_dim = UOp::define_var("actual".into(), 1, 8);
    let actual_extent = actual_dim.try_add(&actual_dim).unwrap().try_mul(&actual_dim.const_like(2i32)).unwrap();
    let actual = symbolic_buffer(7, smallvec![SInt::Symbolic(actual_extent.clone()), SInt::Const(4)]);
    let unused = UOp::native_const(123i32);

    let result = formal
        .try_function(smallvec![actual, unused, actual_dim], CallInfo::default())
        .unwrap()
        .try_gettuple(0)
        .unwrap();
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[SInt::Symbolic(actual_extent), SInt::Const(4)]);
}

#[test]
fn function_tuple_outputs_substitute_each_selected_shape() {
    let p1 = UOp::scalar_param(1, Some("p1".into()), DType::WeakInt, 1, 8);
    let formal = symbolic_buffer(0, smallvec![SInt::Symbolic(p1), SInt::Const(4)]);
    let body = UOp::tuple(smallvec![formal.clone(), formal.try_permute(vec![1, 0]).unwrap()]);

    let actual_dim = UOp::define_var("actual".into(), 1, 8);
    let actual = symbolic_buffer(5, smallvec![SInt::Symbolic(actual_dim.clone()), SInt::Const(4)]);
    let function = body.try_function(smallvec![actual, actual_dim.clone()], CallInfo::default()).unwrap();

    assert_eq!(function.shape().unwrap(), None);
    assert_eq!(
        function.try_gettuple(0).unwrap().shape().unwrap().unwrap().as_slice(),
        &[SInt::Symbolic(actual_dim.clone()), SInt::Const(4)]
    );
    assert_eq!(
        function.try_gettuple(1).unwrap().shape().unwrap().unwrap().as_slice(),
        &[SInt::Const(4), SInt::Symbolic(actual_dim)]
    );
}

#[test]
fn typed_call_is_scalar_and_void_call_is_shapeless_without_entering_body() {
    let hidden = UOp::param(4, 32, DType::Float32, None);
    let body = UOp::sink(vec![hidden]);
    let arg = UOp::native_const(1i32);
    let void_call = body.call(smallvec![arg.clone()], CallInfo::default());
    let typed_call = body.call_typed(smallvec![arg], CallInfo::default(), DType::Int32);

    assert_eq!(void_call.shape().unwrap(), None);
    assert!(typed_call.shape().unwrap().unwrap().is_empty());
}

#[test]
fn function_shape_keeps_nested_opaque_call_body_untouched() {
    let hidden = UOp::param(3, 8, DType::Float32, None);
    let opaque_body = UOp::sink(vec![hidden]);
    let opaque = opaque_body.call(smallvec![], CallInfo::default());
    let formal = UOp::param(0, 8, DType::Float32, None);
    let output = formal.after(smallvec![opaque]);
    let actual = UOp::param(9, 8, DType::Float32, None);
    let function = output.try_function(smallvec![actual], CallInfo::default()).unwrap();

    assert_eq!(function.try_gettuple(0).unwrap().shape().unwrap().unwrap().as_slice(), &[SInt::Const(8)]);
    let Op::Function(ops::Function { body, .. }) = function.op() else { panic!("expected FUNCTION") };
    assert!(body.toposort().iter().any(|node| Arc::ptr_eq(node, &opaque_body)));
}

#[test]
fn function_shape_reports_missing_mismatch_and_unsupported_actuals() {
    let p2 = UOp::scalar_param(2, Some("p2".into()), DType::WeakInt, 1, 8);
    let formal = symbolic_buffer(0, smallvec![SInt::Symbolic(p2)]);
    let actual = symbolic_buffer(7, smallvec![SInt::Const(8)]);
    let missing = formal.function(smallvec![actual], CallInfo::default()).try_gettuple(0).unwrap();
    assert!(matches!(missing.shape(), Err(Error::CallFormalSlotMissing { slot: 2, arg_count: 1 })));

    let formal = UOp::param(0, 8, DType::Float32, None);
    let wrong_shape = UOp::param(3, 4, DType::Float32, None);
    assert!(matches!(
        formal.try_function(smallvec![wrong_shape], CallInfo::default()),
        Err(Error::CallArgShapeMismatch { arg_index: 0, .. })
    ));
    let wrong_dtype = UOp::param(3, 8, DType::Int32, None);
    assert!(matches!(
        formal.try_function(smallvec![wrong_dtype], CallInfo::default()),
        Err(Error::CallArgDTypeMismatch { arg_index: 0, .. })
    ));

    // A free variable in a formal shape reads slot -1, i.e. the last actual; a void actual
    // there has no shape to substitute.
    let free_shape = smallvec![SInt::Symbolic(UOp::variable("free".into(), 1, 8, DType::WeakInt))];
    let no_outputs = UOp::tuple(smallvec![]);
    assert!(matches!(
        crate::shape::substitute_selected_shape(&free_shape, &no_outputs, &[UOp::sink(vec![])]),
        Err(Error::CallShapeSubstitutionUnsupported { slot: -1, .. })
    ));

    let actual = UOp::define_var("actual".into(), 1, 8);
    let substituted =
        crate::shape::substitute_selected_shape(&free_shape, &no_outputs, &[UOp::native_const(0i32), actual.clone()])
            .unwrap();
    assert_eq!(substituted.as_slice(), &[SInt::Symbolic(actual)]);
}

#[test]
fn function_validation_rejects_different_symbolic_max_shapes() {
    let formal_dim = UOp::scalar_param(1, Some("formal".into()), DType::WeakInt, 1, 8);
    let formal = symbolic_buffer(0, smallvec![SInt::Symbolic(formal_dim)]);
    let actual_dim = UOp::define_var("actual".into(), 1, 16);
    let actual = symbolic_buffer(7, smallvec![SInt::Symbolic(actual_dim.clone())]);

    assert!(matches!(
        formal.try_function(smallvec![actual, actual_dim], CallInfo::default()),
        Err(Error::CallArgShapeMismatch { arg_index: 0, .. })
    ));
}

#[test]
fn selected_fixed_output_ignores_missing_formal_in_other_output() {
    let selected = UOp::param(0, 4, DType::Float32, None);
    let missing_dim = UOp::scalar_param(3, Some("missing".into()), DType::WeakInt, 1, 8);
    let unselected = symbolic_buffer(1, smallvec![SInt::Symbolic(missing_dim)]);
    let actual = UOp::param(9, 4, DType::Float32, None);
    let function = UOp::tuple(smallvec![selected, unselected]).function(smallvec![actual], CallInfo::default());

    assert_eq!(function.try_gettuple(0).unwrap().shape().unwrap().unwrap().as_slice(), &[SInt::Const(4)]);
    assert!(matches!(
        function.try_gettuple(1).unwrap().shape(),
        Err(Error::CallFormalSlotMissing { slot: 3, arg_count: 1 })
    ));
}

#[test]
fn substituted_shape_is_deterministic_and_actuals_participate_in_identity() {
    let p1 = UOp::scalar_param(1, Some("p1".into()), DType::WeakInt, 1, 8);
    let formal = symbolic_buffer(0, smallvec![SInt::Symbolic(p1)]);
    let actual = symbolic_buffer(5, smallvec![SInt::Symbolic(UOp::define_var("a".into(), 1, 8))]);
    let a = UOp::define_var("a".into(), 1, 8);
    let b = UOp::define_var("b".into(), 1, 8);

    let fa = formal.function(smallvec![actual.clone(), a], CallInfo::default());
    let fa_again = formal.function(smallvec![actual.clone(), UOp::define_var("a".into(), 1, 8)], CallInfo::default());
    let fb = formal.function(smallvec![actual, b], CallInfo::default());
    assert!(Arc::ptr_eq(&fa, &fa_again));
    assert!(!Arc::ptr_eq(&fa, &fb));
    assert_ne!(fa.try_gettuple(0).unwrap().content_hash, fb.try_gettuple(0).unwrap().content_hash);
}
