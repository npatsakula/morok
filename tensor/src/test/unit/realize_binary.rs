//! End-to-end realize() tests for binary operations.
//!
//! These tests verify that binary operations produce correct numerical results
//! after going through the full pipeline: realize() -> schedule -> codegen -> execute.

use crate::{
    Tensor,
    test::{helpers::*, reference::ops::*},
};

// ============================================================================
// Addition Tests
// ============================================================================

#[test]
fn test_add_f32_simple() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = &a + &b;

    let result_tensor = c.realize().unwrap();

    // Print kernels for debugging
    eprintln!("\n=== KERNELS ===");
    for (i, kernel) in result_tensor.kernels().iter().enumerate() {
        eprintln!("\n--- Kernel {} ---", i);
        eprintln!("Name: {}", kernel.name);
        eprintln!("Backend: {}", kernel.backend);
        eprintln!("Entry point: {}", kernel.entry_point);
        eprintln!("Code:\n{}", kernel.code);
    }

    let result = result_tensor.to_ndarray::<f32>().unwrap();
    let expected = add_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_add_f32_negative() {
    let _guard = test_setup();
    let a = Tensor::from_slice([-1.0f32, -2.0, -3.0]);
    let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let c = &a + &b;

    let result = realize_f32(c);
    let expected = add_f32(&[-1.0, -2.0, -3.0], &[1.0, 2.0, 3.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_add_f32_zero() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([0.0f32, 0.0, 0.0]);
    let c = &a + &b;

    let result = realize_f32(c);
    let expected = add_f32(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_add_f32_large_values() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1e6f32, 2e6, 3e6]);
    let b = Tensor::from_slice([4e6f32, 5e6, 6e6]);
    let c = &a + &b;

    let result = realize_f32(c);
    let expected = add_f32(&[1e6, 2e6, 3e6], &[4e6, 5e6, 6e6]);

    assert_close_f32(&result, &expected, 1e-1); // Relaxed tolerance for large values
}

// ============================================================================
// Subtraction Tests
// ============================================================================

#[test]
fn test_sub_f32_simple() {
    let _guard = test_setup();
    let a = Tensor::from_slice([5.0f32, 6.0, 7.0]);
    let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let c = &a - &b;

    let result = realize_f32(c);
    let expected = sub_f32(&[5.0, 6.0, 7.0], &[1.0, 2.0, 3.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_sub_f32_negative_result() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0]);
    let c = &a - &b;

    let result = realize_f32(c);
    let expected = sub_f32(&[1.0, 2.0, 3.0], &[5.0, 6.0, 7.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_sub_f32_zero() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let c = &a - &b;

    let result = realize_f32(c);
    let expected = sub_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

// ============================================================================
// Multiplication Tests
// ============================================================================

#[test]
fn test_mul_f32_simple() {
    let _guard = test_setup();
    let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0]);
    let c = &a * &b;

    let result = realize_f32(c);
    let expected = mul_f32(&[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_mul_f32_by_zero() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([0.0f32, 0.0, 0.0]);
    let c = &a * &b;

    let result = realize_f32(c);
    let expected = mul_f32(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_mul_f32_negative() {
    let _guard = test_setup();
    let a = Tensor::from_slice([2.0f32, -3.0, 4.0]);
    let b = Tensor::from_slice([-5.0f32, 6.0, -7.0]);
    let c = &a * &b;

    let result = realize_f32(c);
    let expected = mul_f32(&[2.0, -3.0, 4.0], &[-5.0, 6.0, -7.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_mul_f32_fractional() {
    let _guard = test_setup();
    let a = Tensor::from_slice([0.5f32, 0.25, 0.125]);
    let b = Tensor::from_slice([2.0f32, 4.0, 8.0]);
    let c = &a * &b;

    let result = realize_f32(c);
    let expected = mul_f32(&[0.5, 0.25, 0.125], &[2.0, 4.0, 8.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

// ============================================================================
// Division Tests
// ============================================================================

#[test]
fn test_div_f32_simple() {
    let _guard = test_setup();
    let a = Tensor::from_slice([10.0f32, 20.0, 30.0]);
    let b = Tensor::from_slice([2.0f32, 4.0, 5.0]);
    let c = &a / &b;

    let result = realize_f32(c);
    let expected = div_f32(&[10.0, 20.0, 30.0], &[2.0, 4.0, 5.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_div_f32_by_one() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let c = &a / &b;

    let result = realize_f32(c);
    let expected = div_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_div_f32_fractional() {
    let _guard = test_setup();
    let a = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let b = Tensor::from_slice([2.0f32, 4.0, 8.0]);
    let c = &a / &b;

    let result = realize_f32(c);
    let expected = div_f32(&[1.0, 1.0, 1.0], &[2.0, 4.0, 8.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_div_f32_negative() {
    let _guard = test_setup();
    let a = Tensor::from_slice([10.0f32, -20.0, 30.0]);
    let b = Tensor::from_slice([-2.0f32, 4.0, -5.0]);
    let c = &a / &b;

    let result = realize_f32(c);
    let expected = div_f32(&[10.0, -20.0, 30.0], &[-2.0, 4.0, -5.0]);

    assert_close_f32(&result, &expected, 1e-6);
}

// ============================================================================
// Power Tests
// ============================================================================

#[test]
fn test_pow_f32_simple() {
    let _guard = test_setup();
    let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
    let b = Tensor::from_slice([2.0f32, 2.0, 2.0]);
    let c = a.try_pow(&b).unwrap();

    let result = realize_f32(c);
    // 2^2=4, 3^2=9, 4^2=16
    let expected = vec![4.0, 9.0, 16.0];

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_pow_f32_zero_exponent() {
    let _guard = test_setup();
    let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
    let b = Tensor::from_slice([0.0f32, 0.0, 0.0]);
    let c = a.try_pow(&b).unwrap();

    let result = realize_f32(c);
    // x^0 = 1
    let expected = vec![1.0, 1.0, 1.0];

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_pow_f32_one_exponent() {
    let _guard = test_setup();
    let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
    let b = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let c = a.try_pow(&b).unwrap();

    let result = realize_f32(c);
    // x^1 = x
    let expected = vec![2.0, 3.0, 4.0];

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_pow_f32_fractional_exponent() {
    let _guard = test_setup();
    let a = Tensor::from_slice([4.0f32, 9.0, 16.0]);
    let b = Tensor::from_slice([0.5f32, 0.5, 0.5]);
    let c = a.try_pow(&b).unwrap();

    let result = realize_f32(c);
    // x^0.5 = sqrt(x): 4^0.5=2, 9^0.5=3, 16^0.5=4
    let expected = vec![2.0, 3.0, 4.0];

    assert_close_f32(&result, &expected, 1e-5);
}

#[test]
fn test_pow_f32_negative_base() {
    let _guard = test_setup();
    // Note: (-x)^y is complex for non-integer y, but for integer y it works
    let a = Tensor::from_slice([-2.0f32, -3.0, -4.0]);
    let b = Tensor::from_slice([2.0f32, 2.0, 2.0]);
    let c = a.try_pow(&b).unwrap();

    let result = realize_f32(c);
    // (-2)^2=4, (-3)^2=9, (-4)^2=16
    let expected = vec![4.0, 9.0, 16.0];

    assert_close_f32(&result, &expected, 1e-6);
}

// ============================================================================
// Chained Operations Tests
// ============================================================================

#[test]
fn test_chained_add_mul() {
    let _guard = test_setup();
    // (a + b) * c
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = Tensor::from_slice([2.0f32, 2.0, 2.0]);

    let result_tensor = (&a + &b) * &c;
    let result = realize_f32(result_tensor);

    // (1+4)*2=10, (2+5)*2=14, (3+6)*2=18
    let expected = vec![10.0, 14.0, 18.0];

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_chained_mul_add() {
    let _guard = test_setup();
    // a * b + c
    let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0]);
    let c = Tensor::from_slice([1.0f32, 1.0, 1.0]);

    let result_tensor = &a * &b + &c;
    let result = realize_f32(result_tensor);

    // 2*5+1=11, 3*6+1=19, 4*7+1=29
    let expected = vec![11.0, 19.0, 29.0];

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_chained_sub_div() {
    let _guard = test_setup();
    // (a - b) / c
    let a = Tensor::from_slice([10.0f32, 20.0, 30.0]);
    let b = Tensor::from_slice([2.0f32, 4.0, 6.0]);
    let c = Tensor::from_slice([2.0f32, 2.0, 2.0]);

    let result_tensor = (&a - &b) / &c;
    let result = realize_f32(result_tensor);

    // (10-2)/2=4, (20-4)/2=8, (30-6)/2=12
    let expected = vec![4.0, 8.0, 12.0];

    assert_close_f32(&result, &expected, 1e-6);
}

#[test]
fn test_complex_expression() {
    let _guard = test_setup();
    // ((a + b) * c - d) / e
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let b = Tensor::from_slice([3.0f32, 4.0]);
    let c = Tensor::from_slice([2.0f32, 2.0]);
    let d = Tensor::from_slice([1.0f32, 2.0]);
    let e = Tensor::from_slice([3.0f32, 4.0]);

    let result_tensor = ((&a + &b) * &c - &d) / &e;
    let result = realize_f32(result_tensor);

    // ((1+3)*2-1)/3 = (8-1)/3 = 7/3 ≈ 2.333
    // ((2+4)*2-2)/4 = (12-2)/4 = 10/4 = 2.5
    let expected = vec![7.0 / 3.0, 2.5];

    assert_close_f32(&result, &expected, 1e-6);
}
