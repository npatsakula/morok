use crate::*;

#[test]
fn test_broadcast_same_shape() {
    // No broadcasting needed - shapes already match
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let result = a.try_add(&b).unwrap();
    assert_eq!(result.shape().unwrap().len(), 1);
    assert_eq!(result.shape().unwrap()[0].as_const().unwrap(), 3);
}

#[test]
fn test_broadcast_scalar_vector() {
    // Scalar + Vector: [1] + [3] -> [3]
    let scalar = Tensor::from_slice([5.0f32]);
    let vector = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let result = scalar.try_add(&vector).unwrap();
    assert_eq!(result.shape().unwrap().len(), 1);
    assert_eq!(result.shape().unwrap()[0].as_const().unwrap(), 3);
}

#[test]
fn test_broadcast_matrix_row() {
    // Matrix + Row: [2, 3] + [1, 3] -> [2, 3]
    let matrix = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    let row = Tensor::from_slice([1.0f32; 3]).try_reshape(&[1, 3]).unwrap();
    let result = matrix.try_add(&row).unwrap();
    let shape = result.shape().unwrap();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0].as_const().unwrap(), 2);
    assert_eq!(shape[1].as_const().unwrap(), 3);
}

#[test]
fn test_broadcast_matrix_column() {
    // Matrix + Column: [2, 3] + [2, 1] -> [2, 3]
    let matrix = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    let column = Tensor::from_slice([1.0f32; 2]).try_reshape(&[2, 1]).unwrap();
    let result = matrix.try_add(&column).unwrap();
    let shape = result.shape().unwrap();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0].as_const().unwrap(), 2);
    assert_eq!(shape[1].as_const().unwrap(), 3);
}

#[test]
fn test_broadcast_vector_to_matrix() {
    // Vector broadcast to matrix: [3] + [2, 3] -> [2, 3]
    let vector = Tensor::from_slice([1.0f32; 3]);
    let matrix = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    let result = vector.try_add(&matrix).unwrap();
    let shape = result.shape().unwrap();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0].as_const().unwrap(), 2);
    assert_eq!(shape[1].as_const().unwrap(), 3);
}

#[test]
fn test_broadcast_higher_rank() {
    // [2, 1, 3] + [1, 4, 3] -> [2, 4, 3]
    let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 1, 3]).unwrap();
    let b = Tensor::from_slice([1.0f32; 12]).try_reshape(&[1, 4, 3]).unwrap();
    let result = a.try_add(&b).unwrap();
    let shape = result.shape().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0].as_const().unwrap(), 2);
    assert_eq!(shape[1].as_const().unwrap(), 4);
    assert_eq!(shape[2].as_const().unwrap(), 3);
}

#[test]
fn test_broadcast_error_incompatible() {
    // Incompatible shapes: [3] + [4] should fail
    let a = Tensor::from_slice([1.0f32; 3]);
    let b = Tensor::from_slice([1.0f32; 4]);
    assert!(a.try_add(&b).is_err());
}

#[test]
fn test_broadcast_error_incompatible_dims() {
    // Incompatible dimensions: [2, 3] + [2, 4] should fail
    let a = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();
    let b = Tensor::from_slice([1.0f32; 8]).try_reshape(&[2, 4]).unwrap();
    assert!(a.try_add(&b).is_err());
}

#[test]
fn test_broadcast_comparison_ops() {
    // Test broadcasting with comparison operations
    let matrix = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let scalar = Tensor::from_slice([2.5f32]);
    let result = matrix.try_gt(&scalar).unwrap();
    let shape = result.shape().unwrap();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0].as_const().unwrap(), 2);
    assert_eq!(shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_broadcast_mul_different_ranks() {
    // [5] * [3, 4, 5] -> [3, 4, 5]
    let vector = Tensor::from_slice([1.0f32; 5]);
    let tensor_3d = Tensor::from_slice([1.0f32; 60]).try_reshape(&[3, 4, 5]).unwrap();
    let result = vector.try_mul(&tensor_3d).unwrap();
    let shape = result.shape().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0].as_const().unwrap(), 3);
    assert_eq!(shape[1].as_const().unwrap(), 4);
    assert_eq!(shape[2].as_const().unwrap(), 5);
}

#[test]
fn test_broadcast_commutative() {
    // Broadcasting should work in both directions
    let a = Tensor::from_slice([1.0f32; 3]);
    let b = Tensor::from_slice([1.0f32; 6]).try_reshape(&[2, 3]).unwrap();

    let result1 = a.try_add(&b).unwrap();
    let result2 = b.try_add(&a).unwrap();

    let shape1 = result1.shape().unwrap();
    let shape2 = result2.shape().unwrap();

    assert_eq!(shape1, shape2);
    assert_eq!(shape1.len(), 2);
    assert_eq!(shape1[0].as_const().unwrap(), 2);
    assert_eq!(shape1[1].as_const().unwrap(), 3);
}
