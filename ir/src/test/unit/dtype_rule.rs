use crate::ops;
use crate::{BinaryOp, ConstValue, ConstValueHash, DType, Op, TernaryOp, UOp, UnaryOp, dtype_from_op};
use svod_dtype::{AddrSpace, DeviceSpec};

/// Literals and loop indices stay weak until lowering; INVALID has exactly one produced
/// dtype whatever it is used as.
#[test]
fn produced_dtypes_of_leaves() {
    let end = UOp::index_const(8);
    for (label, op, expected) in [
        ("int literal", Op::Const(ConstValueHash(ConstValue::Int(1))), DType::WeakInt),
        ("float literal", Op::Const(ConstValueHash(ConstValue::Float(1.0))), DType::WeakFloat),
        ("bool literal", Op::Const(ConstValueHash(ConstValue::Bool(true))), DType::Bool),
        ("range", UOp::range_const(8, 0).op().clone(), DType::WeakInt),
        ("special", UOp::special(end, "gidx0".to_string()).op().clone(), DType::WeakInt),
        ("invalid", UOp::invalid_marker().op().clone(), DType::Bool),
    ] {
        assert_eq!(dtype_from_op(&op), Some(expected), "{label}");
    }
}

#[test]
fn alu_dtype_uses_current_promotion_rules() {
    let weak = UOp::new(Op::Const(ConstValueHash(ConstValue::Int(1))), DType::WeakInt);
    let strong = UOp::const_(DType::Int16, ConstValue::Int(2));

    assert_eq!(dtype_from_op(&Op::Binary(BinaryOp::Add, weak.clone(), strong.clone())), Some(DType::Int16));
    assert_eq!(dtype_from_op(&Op::Binary(BinaryOp::Lt, weak, strong)), Some(DType::Bool));
}

/// `with_sources` re-derives the parent dtype from the new sources rather than carrying the
/// old one over — except for INVALID, which is polymorphic and never retypes its parent.
#[test]
fn source_rewrite_rederives_parent_dtype() {
    let int = UOp::const_(DType::Int16, ConstValue::Int(1));
    let add = UOp::new(Op::Binary(BinaryOp::Add, int.clone(), int.clone()), DType::Int16);
    let float = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    assert_eq!(add.with_sources(vec![int.clone(), float]).dtype(), DType::Float32);
    assert_eq!(add.with_sources(vec![int, UOp::invalid_marker()]).dtype(), DType::Int16);

    let index = |dtype| {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, dtype);
        UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap()
    };
    let load = UOp::load().index(index(DType::Float32)).call();
    assert_eq!(load.with_sources(vec![index(DType::Float64)]).dtype(), DType::Float64);
}

/// A rebuilt ALU takes its dtype and its lane count from the new sources: the legacy vector
/// result dtype on the node being rebuilt is discarded, and the lanes become a shape.
#[test]
fn alu_reconstruction_does_not_preserve_legacy_vector_result_dtype() {
    let old_float = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let old_bool = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let floats = UOp::stack(vec![old_float.clone(), old_float.clone()].into());
    let bools = UOp::stack(vec![old_bool.clone(), old_bool.clone()].into());
    let vector_float = DType::Float32.vec(2).unwrap();

    let unary =
        UOp::new(Op::Unary(UnaryOp::Sqrt, old_float.clone()), vector_float.clone()).with_sources(vec![floats.clone()]);
    let binary = UOp::new(Op::Binary(BinaryOp::Add, old_float.clone(), old_float.clone()), vector_float.clone())
        .with_sources(vec![floats.clone(), floats.clone()]);
    let comparison =
        UOp::new(Op::Binary(BinaryOp::Lt, old_float.clone(), old_float.clone()), DType::Bool.vec(2).unwrap())
            .with_sources(vec![floats.clone(), floats.clone()]);
    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, old_bool, old_float.clone(), old_float), vector_float)
        .with_sources(vec![bools, floats.clone(), floats]);

    for result in [&unary, &binary, &where_op] {
        assert_eq!(result.dtype(), DType::Float32);
        assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
    }
    assert_eq!(comparison.dtype(), DType::Bool);
    assert_eq!(comparison.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
}

/// An explicit dtype request on INDEX is honoured only when it is the weak equivalent of
/// the buffer's dtype; a concrete request must match exactly, and LOAD takes no request.
#[test]
fn weak_equivalent_explicit_dtype_is_only_an_index_exception() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let offset = UOp::index_const(0);
    let weak_index =
        UOp::index().buffer(buffer.clone()).indices(vec![offset.clone()]).dtype(DType::WeakFloat).call().unwrap();
    assert_eq!(weak_index.dtype(), DType::WeakFloat);

    let narrow = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float16);
    assert!(UOp::index().buffer(narrow).indices(vec![offset.clone()]).dtype(DType::Float32).call().is_err());

    let strong_index = UOp::index().buffer(buffer).indices(vec![offset]).call().unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            UOp::load().index(strong_index).dtype(DType::WeakFloat).call()
        }))
        .is_err()
    );
}

/// INVALID is a single interned node whatever dtype it is requested at, and reaching an
/// operand position never retypes the operation.
#[test]
fn invalid_is_canonical_and_polymorphic() {
    let invalid = UOp::invalid_marker();
    assert_eq!(invalid.dtype(), DType::Bool);
    assert!(std::sync::Arc::ptr_eq(&invalid, &UOp::const_(DType::Float32, ConstValue::Invalid)));

    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let add = value.try_add(&invalid).unwrap();
    let Op::Binary(_, _, rhs) = add.op() else { panic!("expected binary operation") };
    assert!(std::sync::Arc::ptr_eq(rhs, &invalid));
    assert_eq!(add.dtype(), DType::Float32);
}

/// Indexing an image-shaped PARAM yields the four-channel load dtype; a BUFFER, or a PARAM
/// whose shape is not image-shaped, keeps the element dtype.
#[test]
fn index_dtype_matches_target_param_image_exception() {
    let shape = |dims: &[usize]| crate::shape::shape_to_uop(&dims.iter().map(|d| (*d).into()).collect());
    let param = |slot, dims: &[usize]| {
        let arg = crate::ParamArg::buffer(slot, DType::Float16, AddrSpace::Global, None);
        UOp::new(Op::Param(ops::Param { shape: shape(dims), arg: arg.into() }), DType::Float16)
    };
    let offset = UOp::index_const(0);
    let index_dtype = |buffer| UOp::index().buffer(buffer).indices(vec![offset.clone()]).call().unwrap().dtype();

    let image_arg = crate::ParamArg::buffer(0, DType::Float16, AddrSpace::Global, None);
    let buffer = UOp::new(Op::Buffer(ops::Buffer { shape: shape(&[2, 3, 4]), arg: image_arg.into() }), DType::Float16);

    assert_eq!(index_dtype(param(0, &[2, 3, 4])), DType::Float32, "image-shaped PARAM");
    assert_eq!(index_dtype(buffer), DType::Float16, "BUFFER is never image-shaped");
    assert_eq!(index_dtype(param(1, &[2, 3, 5])), DType::Float16, "wrong channel count");
    assert_eq!(index_dtype(param(2, &[3, 4])), DType::Float16, "wrong rank");
}
