//! Shape/dtype accessors ([`Tensor::dims`], [`Tensor::dim`], [`Tensor::dim_const`],
//! [`Tensor::dtype`]) and the metadata-only [`Debug`] rendering.

use svod_dtype::DType;
use svod_ir::SInt;
use test_case::test_case;

use crate::error::Error;
use crate::{Tensor, Variable};

fn concrete() -> Tensor {
    Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([2, 3]).unwrap()
}

/// Shape `[batch, 4]` with `batch` symbolic — nothing is realized.
fn symbolic() -> Tensor {
    Tensor::empty_dynamic(&[Variable::new("batch", 1, 8).as_sint(), SInt::Const(4)], DType::Float32)
}

// =========================================================================
// dims
// =========================================================================

#[test]
fn dims_returns_the_concrete_extents() {
    assert_eq!(concrete().dims().unwrap(), vec![2, 3]);
    assert_eq!(Tensor::from_slice([1.0f32, 2.0]).dims().unwrap(), vec![2]);
}

#[test]
fn dims_rejects_a_symbolic_shape() {
    let err = symbolic().dims().unwrap_err();
    assert!(matches!(err, Error::UOp { .. }), "{err:?}");
}

// =========================================================================
// dim / dim_const
// =========================================================================

#[test_case(0, 2; "first axis")]
#[test_case(1, 3; "second axis")]
#[test_case(-1, 3; "last axis, negative index")]
#[test_case(-2, 2; "second-to-last axis, negative index")]
fn dim_and_dim_const_agree_on_each_axis(axis: isize, expected: usize) {
    let t = concrete();
    assert_eq!(t.dim_const(axis).unwrap(), expected);
    assert_eq!(t.dim(axis).unwrap().as_const(), Some(expected));
}

#[test_case(2; "one past the last axis")]
#[test_case(9; "far past the last axis")]
#[test_case(-3; "one before the first axis")]
fn dim_const_rejects_an_out_of_range_axis(axis: isize) {
    let err = concrete().dim_const(axis).unwrap_err();
    assert!(matches!(err, Error::AxisOutOfRange { ndim: 2, .. }), "{err:?}");
}

#[test]
fn dim_const_rejects_a_symbolic_axis_that_dim_still_reports() {
    let t = symbolic();
    assert_eq!(t.dim_const(1).unwrap(), 4, "the concrete axis is still readable");

    let err = t.dim_const(0).unwrap_err();
    assert!(matches!(err, Error::NonConstDim { axis: 0, .. }), "{err:?}");
    assert!(err.to_string().contains("dimension 0 is symbolic"), "{err}");
    // `dim` keeps the symbolic extent rather than failing.
    assert!(t.dim(0).unwrap().as_const().is_none());
}

// =========================================================================
// dtype
// =========================================================================

#[test_case(DType::Float32; "f32")]
#[test_case(DType::Int32; "i32")]
#[test_case(DType::Int64; "i64")]
#[test_case(DType::Bool; "bool")]
fn dtype_reports_the_element_type(dtype: DType) {
    assert_eq!(Tensor::zeros(&[2, 2], dtype.clone()).unwrap().dtype(), dtype);
}

#[test]
fn dtype_follows_a_cast() {
    let t = Tensor::from_slice([1.0f32, 2.0]);
    assert_eq!(t.dtype(), DType::Float32);
    assert_eq!(t.cast(DType::Int32).unwrap().dtype(), DType::Int32);
}

// =========================================================================
// Debug
// =========================================================================

#[test]
fn debug_prints_metadata_and_never_the_data() {
    let s = format!("{:?}", concrete());
    assert!(s.contains("shape: [2, 3]"), "{s}");
    assert!(s.contains("Float32"), "{s}");
    assert!(s.contains("realized: true"), "{s}");
    // Element values must not leak — that would force a device read.
    assert!(!s.contains("6.0"), "{s}");
}

#[test]
fn debug_reports_a_symbolic_shape_and_an_unrealized_buffer() {
    let s = format!("{:?}", symbolic());
    assert!(s.contains("shape: symbolic"), "{s}");
    assert!(s.contains("realized: false"), "{s}");
}

#[test]
fn debug_reports_an_unrealized_lazy_graph() {
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let s = format!("{:?}", &a + &a);
    assert!(s.contains("shape: [2]"), "{s}");
    assert!(s.contains("realized: false"), "{s}");
}

// =========================================================================
// index_select — a symbolic index length is an error, not a panic
// =========================================================================

#[test]
fn index_select_rejects_a_symbolic_index_length() {
    let src = Tensor::zeros(&[4, 3], DType::Float32).unwrap();
    let index = Tensor::empty_dynamic(&[Variable::new("k", 1, 4).as_sint()], DType::Int32);

    let err = src.index_select(0, &index).unwrap_err();
    assert!(matches!(err, Error::NonConstDim { axis: 0, .. }), "{err:?}");
}

#[test]
fn index_select_rejects_a_rank_0_index() {
    let src = Tensor::zeros(&[4, 3], DType::Float32).unwrap();
    let index = Tensor::zeros(&[], DType::Int32).unwrap();
    // Previously an out-of-bounds slice index on the index tensor's shape.
    assert!(src.index_select(0, &index).is_err());
}
