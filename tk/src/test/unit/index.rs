//! Pure graph-shape tests for flat tile addressing.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, Op, UOp};

use crate::index::{Idx, flat_offset, flat_ptr, strides};

#[test]
fn test_strides_row_major() {
    assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
    assert_eq!(strides(&[16]), vec![1]);
    assert_eq!(strides(&[]), Vec::<i64>::new());
}

#[test]
fn test_flat_offset_all_const_folds() {
    // shape [2, 16], idx [1, 3] -> 1*16 + 3*1 = 19, folded to a single Const.
    let off = flat_offset(&[2, 16], &[Idx::Const(1), Idx::Const(3)]);
    match off.op() {
        Op::Const(c) => assert!(matches!(c.0, ConstValue::Int(19)), "expected Const(19), got {:?}", c.0),
        other => panic!("expected folded Const, got {other:?}"),
    }
}

#[test]
fn test_flat_offset_dynamic_index_survives() {
    // A dynamic (range) index must not fold away; the const part still folds.
    let r = UOp::range_const(16, 0);
    let off = flat_offset(&[2, 16], &[Idx::Const(1), Idx::from(&r)]);

    assert_eq!(off.dtype(), DType::WeakInt);
    assert!(!matches!(off.op(), Op::Const(_)), "offset with a dynamic index must not fold to a constant");
    assert!(off.toposort().iter().any(|u| Arc::ptr_eq(u, &r)), "the dynamic range must appear in the offset graph");
}

#[test]
fn test_flat_offset_unit_stride_skips_mul() {
    // Last axis has stride 1, so a lone dynamic index is returned as-is.
    let r = UOp::range_const(8, 0);
    let off = flat_offset(&[8], &[Idx::from(&r)]);
    assert!(Arc::ptr_eq(&off, &r), "unit-stride single dynamic index should pass through unchanged");
}

#[test]
fn test_flat_ptr_unwraps_reshape_to_param() {
    // A multi-dim placeholder is RESHAPE(PARAM); flat_ptr must return the flat
    // PARAM and its element dtype.
    let ph = UOp::placeholder_like(
        &UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 12, DType::Float32),
        0,
        svod_dtype::AddrSpace::Global,
    )
    .expect("placeholder");
    // 1-D buffer reshaped to (3, 4) for the test.
    let ph = ph
        .try_reshape(&svod_ir::shape::Shape::from_iter([svod_ir::SInt::Const(3), svod_ir::SInt::Const(4)]))
        .expect("reshape");
    assert!(matches!(ph.op(), Op::Reshape(..)), "precondition: placeholder view is a reshape");

    let (buf, elem) = flat_ptr(&ph);
    assert!(!matches!(buf.op(), Op::Reshape(..)), "flat_ptr must unwrap the reshape");
    assert_eq!(elem, DType::Float32, "element dtype is the pointer base");
}
