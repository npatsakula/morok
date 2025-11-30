use crate::*;
use morok_ir::Op;

// Helper to get concrete shape as Vec<usize>
fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop.shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

// =========================================================================
// Reshape Tests
// =========================================================================

#[test]
fn test_reshape_basic() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    assert_eq!(get_shape(&reshaped), vec![2, 3]);
    if let Op::Reshape { .. } = reshaped.uop.op() {
        // Correct operation type
    } else {
        panic!("Expected Reshape operation");
    }
}

#[test]
fn test_reshape_with_inference() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // -1 should infer dimension: 6 elements / 2 = 3
    let reshaped = t.try_reshape(&[-1, 2]).unwrap();
    assert_eq!(get_shape(&reshaped), vec![3, 2]);

    // -1 at different position
    let reshaped2 = t.try_reshape(&[3, -1]).unwrap();
    assert_eq!(get_shape(&reshaped2), vec![3, 2]);
}

#[test]
fn test_reshape_flatten() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();
    let flattened = reshaped.try_reshape(&[6]).unwrap();

    assert_eq!(get_shape(&flattened), vec![6]);
}

#[test]
fn test_reshape_identity() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[3]).unwrap();

    assert_eq!(get_shape(&reshaped), vec![3]);
}

#[test]
fn test_reshape_error_size_mismatch() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // 6 elements cannot be reshaped to [2, 4] = 8 elements
    let result = t.try_reshape(&[2, 4]);
    assert!(result.is_err());
}

#[test]
fn test_reshape_error_multiple_inference() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Multiple -1 dimensions not allowed
    let result = t.try_reshape(&[-1, -1]);
    assert!(result.is_err());
}

#[test]
fn test_reshape_error_invalid_negative() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    // Negative dimension other than -1 not allowed
    let result = t.try_reshape(&[-2, 3]);
    assert!(result.is_err());
}

// =========================================================================
// Permute Tests
// =========================================================================

#[test]
fn test_permute_basic() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    // Swap dimensions [2, 3] -> [3, 2]
    let permuted = reshaped.try_permute(&[1, 0]).unwrap();
    assert_eq!(get_shape(&permuted), vec![3, 2]);
}

#[test]
fn test_permute_3d() {
    let t = Tensor::from_slice([1.0f32; 24]);
    let reshaped = t.try_reshape(&[2, 3, 4]).unwrap();

    // Permute [2, 3, 4] -> [4, 2, 3]
    let permuted = reshaped.try_permute(&[2, 0, 1]).unwrap();
    assert_eq!(get_shape(&permuted), vec![4, 2, 3]);
}

#[test]
fn test_permute_identity() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    // Identity permutation
    let permuted = reshaped.try_permute(&[0, 1]).unwrap();
    assert_eq!(get_shape(&permuted), vec![2, 3]);
}

#[test]
fn test_permute_negative_indices() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    // Negative indices: -1 = last axis, -2 = second to last
    let permuted = reshaped.try_permute(&[-1, -2]).unwrap();
    assert_eq!(get_shape(&permuted), vec![3, 2]);
}

#[test]
fn test_permute_error_invalid_permutation() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    // Duplicate axis
    let result = reshaped.try_permute(&[0, 0]);
    assert!(result.is_err());

    // Missing axis
    let result2 = reshaped.try_permute(&[0, 2]);
    assert!(result2.is_err());
}

#[test]
fn test_permute_error_wrong_length() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    // Wrong number of axes
    let result = reshaped.try_permute(&[0, 1, 2]);
    assert!(result.is_err());
}

// =========================================================================
// Transpose Tests
// =========================================================================

#[test]
fn test_transpose_basic() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    let transposed = reshaped.try_transpose(0, 1).unwrap();
    assert_eq!(get_shape(&transposed), vec![3, 2]);
}

#[test]
fn test_transpose_3d() {
    let t = Tensor::from_slice([1.0f32; 24]);
    let reshaped = t.try_reshape(&[2, 3, 4]).unwrap();

    // Swap first and last dimensions
    let transposed = reshaped.try_transpose(0, 2).unwrap();
    assert_eq!(get_shape(&transposed), vec![4, 3, 2]);
}

#[test]
fn test_transpose_negative_indices() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    // -1 = last axis (1), -2 = second to last (0)
    let transposed = reshaped.try_transpose(-1, -2).unwrap();
    assert_eq!(get_shape(&transposed), vec![3, 2]);
}

#[test]
fn test_transpose_same_dimension() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    // Transpose dimension with itself = identity
    let transposed = reshaped.try_transpose(0, 0).unwrap();
    assert_eq!(get_shape(&transposed), vec![2, 3]);
}

// =========================================================================
// Expand Tests
// =========================================================================

#[test]
fn test_expand_basic() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[1, 3]).unwrap();

    // Expand first dimension from 1 to 4
    let expanded = reshaped.try_expand(&[4, -1]).unwrap();
    assert_eq!(get_shape(&expanded), vec![4, 3]);
}

#[test]
fn test_expand_keep_dimension() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[1, 3]).unwrap();

    // -1 keeps current dimension
    let expanded = reshaped.try_expand(&[-1, -1]).unwrap();
    assert_eq!(get_shape(&expanded), vec![1, 3]);
}

#[test]
fn test_expand_multiple_dims() {
    let t = Tensor::from_slice([1.0f32]);
    let reshaped = t.try_reshape(&[1, 1, 1]).unwrap();

    // Expand all dimensions
    let expanded = reshaped.try_expand(&[4, 5, 6]).unwrap();
    assert_eq!(get_shape(&expanded), vec![4, 5, 6]);
}

#[test]
fn test_expand_error_non_one_dimension() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[1, 3]).unwrap();

    // Cannot expand dimension of size 3 to 5
    let result = reshaped.try_expand(&[1, 5]);
    assert!(result.is_err());
}

#[test]
fn test_expand_error_dimension_mismatch() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[1, 3]).unwrap();

    // Wrong number of dimensions
    let result = reshaped.try_expand(&[4, 5, 6]);
    assert!(result.is_err());
}

// =========================================================================
// Squeeze Tests
// =========================================================================

#[test]
fn test_squeeze_all() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[1, 3, 1]).unwrap();

    // Remove all dimensions of size 1
    let squeezed = reshaped.try_squeeze(None).unwrap();
    assert_eq!(get_shape(&squeezed), vec![3]);
}

#[test]
fn test_squeeze_specific_dim() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[1, 3, 1]).unwrap();

    // Remove only first dimension
    let squeezed = reshaped.try_squeeze(Some(0)).unwrap();
    assert_eq!(get_shape(&squeezed), vec![3, 1]);

    // Remove last dimension
    let squeezed2 = reshaped.try_squeeze(Some(-1)).unwrap();
    assert_eq!(get_shape(&squeezed2), vec![1, 3]);
}

#[test]
fn test_squeeze_no_effect() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[3]).unwrap();

    // No dimensions of size 1 to squeeze
    let squeezed = reshaped.try_squeeze(None).unwrap();
    assert_eq!(get_shape(&squeezed), vec![3]);
}

#[test]
fn test_squeeze_error_not_size_one() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let reshaped = t.try_reshape(&[1, 3]).unwrap();

    // Cannot squeeze dimension of size 3
    let result = reshaped.try_squeeze(Some(1));
    assert!(result.is_err());
}

// =========================================================================
// Unsqueeze Tests
// =========================================================================

#[test]
fn test_unsqueeze_at_start() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    let unsqueezed = t.try_unsqueeze(0).unwrap();
    assert_eq!(get_shape(&unsqueezed), vec![1, 3]);
}

#[test]
fn test_unsqueeze_at_end() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    let unsqueezed = t.try_unsqueeze(1).unwrap();
    assert_eq!(get_shape(&unsqueezed), vec![3, 1]);
}

#[test]
fn test_unsqueeze_negative_index() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    // -1 means after last dimension
    let unsqueezed = t.try_unsqueeze(-1).unwrap();
    assert_eq!(get_shape(&unsqueezed), vec![3, 1]);
}

#[test]
fn test_unsqueeze_middle() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();

    let unsqueezed = reshaped.try_unsqueeze(1).unwrap();
    assert_eq!(get_shape(&unsqueezed), vec![2, 1, 3]);
}

#[test]
fn test_unsqueeze_multiple() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    let unsqueezed1 = t.try_unsqueeze(0).unwrap();
    let unsqueezed2 = unsqueezed1.try_unsqueeze(0).unwrap();
    assert_eq!(get_shape(&unsqueezed2), vec![1, 1, 3]);
}

// =========================================================================
// Combined Operations Tests
// =========================================================================

#[test]
fn test_reshape_then_transpose() {
    let t = Tensor::from_slice([1.0f32; 6]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();
    let transposed = reshaped.try_transpose(0, 1).unwrap();

    assert_eq!(get_shape(&transposed), vec![3, 2]);
}

#[test]
fn test_unsqueeze_then_expand() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let unsqueezed = t.try_unsqueeze(0).unwrap();
    let expanded = unsqueezed.try_expand(&[4, -1]).unwrap();

    assert_eq!(get_shape(&expanded), vec![4, 3]);
}

#[test]
fn test_expand_then_squeeze() {
    let t = Tensor::from_slice([1.0f32]);
    let reshaped = t.try_reshape(&[1, 1]).unwrap();
    let expanded = reshaped.try_expand(&[4, -1]).unwrap();
    let squeezed = expanded.try_squeeze(Some(1)).unwrap();

    assert_eq!(get_shape(&squeezed), vec![4]);
}

#[test]
fn test_lazy_evaluation() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reshaped = t.try_reshape(&[2, 3]).unwrap();
    let permuted = reshaped.try_permute(&[1, 0]).unwrap();
    let unsqueezed = reshaped.try_unsqueeze(0).unwrap();

    // Movement operations share the same underlying buffer via .base()
    // They all point to the same buffer (t's buffer)
    assert!(reshaped.buffer().is_some());
    assert!(permuted.buffer().is_some());
    assert!(unsqueezed.buffer().is_some());

    // All should share the same buffer ID (same base)
    assert_eq!(reshaped.uop.base().id, t.uop.base().id);
    assert_eq!(permuted.uop.base().id, t.uop.base().id);
    assert_eq!(unsqueezed.uop.base().id, t.uop.base().id);
}

#[test]
fn test_dtype_preservation() {
    let t_f32 = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let t_i32 = Tensor::from_slice([1i32, 2, 3]);

    let reshaped_f32 = t_f32.try_reshape(&[3, 1]).unwrap();
    let reshaped_i32 = t_i32.try_reshape(&[3, 1]).unwrap();

    assert_eq!(reshaped_f32.uop.dtype(), morok_dtype::DType::Float32);
    assert_eq!(reshaped_i32.uop.dtype(), morok_dtype::DType::Int32);
}

// =========================================================================
// Symbolic Shape Tests
// =========================================================================

#[test]
fn test_symbolic_shape_support() {
    use morok_ir::{ConstValue, DType};

    // Create a tensor with a symbolic dimension using DefineVar
    let batch_var = UOp::define_var("batch".to_string(), 1, 128);
    let batch_dim = morok_ir::SInt::Symbolic(batch_var);

    // Create shape: [batch, 3, 4] where batch is symbolic
    let symbolic_shape: morok_ir::shape::Shape =
        vec![batch_dim.clone(), morok_ir::SInt::from(3), morok_ir::SInt::from(4)].into();

    // Create a tensor with this symbolic shape using a const value
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let tensor_with_symbolic_shape = const_val.try_reshape(&symbolic_shape).unwrap();
    let tensor = Tensor::new(tensor_with_symbolic_shape);

    // Test 1: dim() returns SInt (can be symbolic or concrete)
    let dim0 = tensor.dim(0).unwrap();
    let dim1 = tensor.dim(1).unwrap();
    let dim2 = tensor.dim(2).unwrap();

    // First dimension is symbolic
    assert!(dim0.as_const().is_none()); // Symbolic, no concrete value
    assert_eq!(dim1.as_const(), Some(3)); // Concrete
    assert_eq!(dim2.as_const(), Some(4)); // Concrete

    // Test 2: ndim() works with symbolic shapes
    assert_eq!(tensor.ndim().unwrap(), 3);

    // Test 3: Reshape preserving symbolic dimension
    let new_shape: morok_ir::shape::Shape = vec![batch_dim.clone(), morok_ir::SInt::from(12)].into();
    let reshaped = tensor.uop.try_reshape(&new_shape).map(Tensor::new).unwrap();
    assert_eq!(reshaped.ndim().unwrap(), 2);

    // Test 4: Permute works with symbolic shapes
    let permuted = tensor.try_permute(&[1, 0, 2]).unwrap();
    let perm_shape = permuted.shape().unwrap();
    assert_eq!(perm_shape[0].as_const(), Some(3)); // Was dim 1
    assert!(perm_shape[1].as_const().is_none()); // Was dim 0 (symbolic)
    assert_eq!(perm_shape[2].as_const(), Some(4)); // Was dim 2
}

#[test]
fn test_symbolic_shape_broadcast() {
    use morok_ir::{ConstValue, DType};

    // Create symbolic batch dimension
    let batch_var = UOp::define_var("N".to_string(), 1, 1024);
    let batch_dim = morok_ir::SInt::Symbolic(batch_var);

    // Create tensor with shape [N, 4]
    let symbolic_shape: morok_ir::shape::Shape = vec![batch_dim.clone(), morok_ir::SInt::from(4)].into();

    let const_val = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let tensor_symbolic = const_val.try_reshape(&symbolic_shape).unwrap();
    let tensor = Tensor::new(tensor_symbolic);

    // Create a concrete tensor to broadcast against: [1, 4]
    let concrete = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 4]).unwrap();

    // Broadcasting should work with symbolic shapes
    let (broadcasted_symbolic, _broadcasted_concrete) = tensor.broadcast_for_binop(&concrete).unwrap();

    // Both should have shape [N, 4]
    let result_shape = broadcasted_symbolic.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert!(result_shape[0].as_const().is_none()); // Symbolic N
    assert_eq!(result_shape[1].as_const(), Some(4)); // Concrete 4
}

#[test]
fn test_symbolic_shape_binary_ops() {
    use morok_ir::{ConstValue, DType};

    // Create symbolic dimensions
    let dim_var = UOp::define_var("D".to_string(), 1, 512);
    let dim_sym = morok_ir::SInt::Symbolic(dim_var);

    // Create two tensors with symbolic shape [D]
    let shape: morok_ir::shape::Shape = vec![dim_sym.clone()].into();

    let const1 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let tensor1_uop = const1.try_reshape(&shape).unwrap();
    let tensor1 = Tensor::new(tensor1_uop);

    let const2 = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let tensor2_uop = const2.try_reshape(&shape).unwrap();
    let tensor2 = Tensor::new(tensor2_uop);

    // Binary operations should work with matching symbolic shapes
    let sum = tensor1.try_add(&tensor2).unwrap();
    let product = tensor1.try_mul(&tensor2).unwrap();

    // Results should preserve symbolic shape
    let sum_shape = sum.shape().unwrap();
    let product_shape = product.shape().unwrap();

    assert_eq!(sum_shape.len(), 1);
    assert!(sum_shape[0].as_const().is_none()); // Still symbolic

    assert_eq!(product_shape.len(), 1);
    assert!(product_shape[0].as_const().is_none()); // Still symbolic
}
