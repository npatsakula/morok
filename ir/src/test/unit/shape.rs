use smallvec::smallvec;

use morok_dtype::DType;

use crate::{ConstValue, SInt, UOp, shape::*};

#[test]
fn test_is_static() {
    let static_shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
    assert!(is_static(&static_shape));

    // Note: Const UOps are automatically simplified to SInt::Const
    // To get a truly symbolic dimension, we need a non-const UOp
    // For now, just test with concrete values
    let also_static = smallvec![SInt::from(3), SInt::from(10)];
    assert!(is_static(&also_static));
}

#[test]
fn test_to_static() {
    let shape = smallvec![SInt::from(3), SInt::from(4)];
    assert_eq!(to_static(&shape), Some(smallvec![3, 4]));

    // Note: Const UOps are automatically simplified, so we'd need
    // a truly symbolic UOp (like from an operation) to test dynamic shapes
    // For now, just verify static conversion works
    let shape2 = smallvec![SInt::from(5), SInt::from(6), SInt::from(7)];
    assert_eq!(to_static(&shape2), Some(smallvec![5, 6, 7]));
}

#[test]
fn test_ndim() {
    let shape: Shape = smallvec![SInt::from(3), SInt::from(4), SInt::from(5)];
    assert_eq!(shape.len(), 3);
}

#[test]
fn test_shape_product() {
    let shape: Shape = smallvec![SInt::from(2), SInt::from(3), SInt::from(4)];
    let product = crate::sint_prod(&shape);
    assert_eq!(product.as_const(), Some(24));
}

#[test]
fn test_validate_shape() {
    assert!(validate_shape(&[1, 2, 3]).is_ok());
    assert!(validate_shape(&[1, -2, 3]).is_err());
    assert!(validate_shape(&[1, 0, 3]).is_err());
}

#[test]
fn test_shapes_equal() {
    let shape1 = smallvec![SInt::from(3), SInt::from(4)];
    let shape2 = smallvec![SInt::from(3), SInt::from(4)];
    assert!(shapes_equal(&shape1, &shape2));

    let shape3 = smallvec![SInt::from(3), SInt::from(5)];
    assert!(!shapes_equal(&shape1, &shape3));
}

#[test]
fn test_all_shapes_equal() {
    let shape1 = smallvec![SInt::from(3), SInt::from(4)];
    let shape2 = smallvec![SInt::from(3), SInt::from(4)];
    let shape3 = smallvec![SInt::from(3), SInt::from(4)];
    assert!(all_shapes_equal(&[shape1, shape2, shape3]));
}

#[test]
fn test_align_shapes_left() {
    let shape1 = smallvec![SInt::from(5)];
    let shape2 = smallvec![SInt::from(3), SInt::from(5)];
    let aligned = align_shapes_left(&[shape1, shape2]);

    assert_eq!(aligned.len(), 2);
    assert_eq!(aligned[0].len(), 2);
    assert_eq!(aligned[0][0].as_const(), Some(1));
    assert_eq!(aligned[0][1].as_const(), Some(5));
}

#[test]
fn test_can_broadcast() {
    let shape1 = smallvec![SInt::from(1), SInt::from(5)];
    let shape2 = smallvec![SInt::from(3), SInt::from(5)];
    assert!(can_broadcast(&shape1, &shape2));

    let shape3 = smallvec![SInt::from(3), SInt::from(4)];
    assert!(!can_broadcast(&shape1, &shape3));
}

#[test]
fn test_broadcast_shape() {
    let shape1 = smallvec![SInt::from(1), SInt::from(5)];
    let shape2 = smallvec![SInt::from(3), SInt::from(5)];
    let result = broadcast_shape(&shape1, &shape2).unwrap();

    assert_eq!(result[0].as_const(), Some(3));
    assert_eq!(result[1].as_const(), Some(5));
}

#[test]
fn test_broadcast_shape_error() {
    let shape1 = smallvec![SInt::from(3), SInt::from(4)];
    let shape2 = smallvec![SInt::from(3), SInt::from(5)];
    assert!(broadcast_shape(&shape1, &shape2).is_err());
}

#[test]
fn test_broadcast_shapes_multiple() {
    let shape1 = smallvec![SInt::from(1), SInt::from(5)];
    let shape2 = smallvec![SInt::from(3), SInt::from(1)];
    let shape3 = smallvec![SInt::from(3), SInt::from(5)];

    let result = broadcast_shapes(&[shape1, shape2, shape3]).unwrap();
    assert_eq!(result[0].as_const(), Some(3));
    assert_eq!(result[1].as_const(), Some(5));
}

// =====================================================================
// Shape Inference Tests
// =====================================================================

#[test]
fn test_infer_const_shape() {
    let scalar = UOp::native_const(42.0f32);
    let shape = scalar.shape().unwrap().expect("Const should have shape");
    assert_eq!(shape.len(), 0); // Scalar has empty shape
}

#[test]
fn test_infer_vconst_shape() {
    let values = vec![ConstValue::Float(1.0), ConstValue::Float(2.0), ConstValue::Float(3.0), ConstValue::Float(4.0)];
    let vec = UOp::new(crate::Op::VConst { values: values.clone() }, DType::Float32.vec(4));
    // VConst is a kernel-level op and returns None (matches Tinygrad)
    assert!(vec.shape().unwrap().is_none(), "VConst should return None for shape");
}

#[test]
fn test_infer_unary_shape() {
    let val = UOp::native_const(5.0f32);
    let neg = val.neg();
    let shape = neg.shape().unwrap().expect("Unary should have shape");
    assert_eq!(shape.len(), 0); // Preserves scalar shape
}

#[test]
fn test_infer_binary_shape() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();
    let shape = add.shape().unwrap().expect("Binary should have shape");
    assert_eq!(shape.len(), 0); // Both scalars -> scalar result
}

#[test]
fn test_infer_cast_shape() {
    let val = UOp::native_const(1.5f32);
    let cast = UOp::cast(val, DType::Int32);
    let shape = cast.shape().unwrap().expect("Cast should preserve shape");
    assert_eq!(shape.len(), 0);
}

#[test]
fn test_shape_caching() {
    let val = UOp::native_const(1.0f32);
    // First access computes shape
    let shape1 = val.shape().unwrap().expect("Should have shape");
    // Second access uses cached value (same pointer)
    let shape2 = val.shape().unwrap().expect("Should have cached shape");
    assert!(std::ptr::eq(shape1, shape2), "Shape should be cached");
}

// =====================================================================
// shape_to_uop Tests
// =====================================================================

#[test]
fn test_shape_to_uop_empty() {
    use crate::op::Op;

    // Empty shape should create Vectorize with empty elements
    let empty_shape = smallvec![];
    let shape_uop = shape_to_uop(&empty_shape);

    // Should be Vectorize, not VConst
    if let Op::Vectorize { elements } = shape_uop.op() {
        assert_eq!(elements.len(), 0, "Empty shape should have empty Vectorize");
    } else {
        panic!("Expected Vectorize for empty shape, got {:?}", shape_uop.op());
    }
}

#[test]
fn test_shape_to_uop_non_empty() {
    use crate::op::Op;

    // Non-empty shape should create Vectorize with elements
    let shape = smallvec![SInt::from(3), SInt::from(4)];
    let shape_uop = shape_to_uop(&shape);

    // Should be Vectorize with correct number of elements
    if let Op::Vectorize { elements } = shape_uop.op() {
        assert_eq!(elements.len(), 2, "Shape [3, 4] should have 2 elements");
    } else {
        panic!("Expected Vectorize, got {:?}", shape_uop.op());
    }
}

#[test]
fn test_shape_to_uop_consistency() {
    use crate::op::Op;

    // Both empty and non-empty shapes should use Vectorize
    let empty = smallvec![];
    let non_empty = smallvec![SInt::from(5)];

    let empty_uop = shape_to_uop(&empty);
    let non_empty_uop = shape_to_uop(&non_empty);

    // Both should be Vectorize operations
    assert!(matches!(empty_uop.op(), Op::Vectorize { .. }), "Empty should be Vectorize");
    assert!(matches!(non_empty_uop.op(), Op::Vectorize { .. }), "Non-empty should be Vectorize");
}
