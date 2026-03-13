//! End-to-end realize() tests for binary operations.
//!
//! These tests verify that binary operations produce correct numerical results
//! after going through the full pipeline: realize() -> schedule -> codegen -> execute.

use crate::{
    Tensor,
    test::{helpers::*, reference::ops::*},
};

crate::codegen_tests! {
    // ========================================================================
    // Addition Tests
    // ========================================================================

    fn test_add_f32_simple(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = &a + &b;

        let result_tensor = c.realize_with(&config).unwrap();
        let expected = add_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert_close_f32(&result_tensor.to_vec::<f32>().unwrap(), &expected, 1e-6);
    }

    fn test_add_f32_negative(config) {
        test_setup();
        let a = Tensor::from_slice([-1.0f32, -2.0, -3.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = &a + &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = add_f32(&[-1.0, -2.0, -3.0], &[1.0, 2.0, 3.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_add_f32_zero(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let c = &a + &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = add_f32(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_add_f32_large_values(config) {
        test_setup();
        let a = Tensor::from_slice([1e6f32, 2e6, 3e6]);
        let b = Tensor::from_slice([4e6f32, 5e6, 6e6]);
        let c = &a + &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = add_f32(&[1e6, 2e6, 3e6], &[4e6, 5e6, 6e6]);
        assert_close_f32(&result, &expected, 1e-1);
    }

    // ========================================================================
    // Subtraction Tests
    // ========================================================================

    fn test_sub_f32_simple(config) {
        test_setup();
        let a = Tensor::from_slice([5.0f32, 6.0, 7.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = &a - &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = sub_f32(&[5.0, 6.0, 7.0], &[1.0, 2.0, 3.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_sub_f32_negative_result(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0]);
        let c = &a - &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = sub_f32(&[1.0, 2.0, 3.0], &[5.0, 6.0, 7.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_sub_f32_zero(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let c = &a - &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = sub_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    // ========================================================================
    // Multiplication Tests
    // ========================================================================

    fn test_mul_f32_simple(config) {
        test_setup();
        let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0]);
        let c = &a * &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = mul_f32(&[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_mul_f32_by_zero(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let c = &a * &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = mul_f32(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_mul_f32_negative(config) {
        test_setup();
        let a = Tensor::from_slice([2.0f32, -3.0, 4.0]);
        let b = Tensor::from_slice([-5.0f32, 6.0, -7.0]);
        let c = &a * &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = mul_f32(&[2.0, -3.0, 4.0], &[-5.0, 6.0, -7.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_mul_f32_fractional(config) {
        test_setup();
        let a = Tensor::from_slice([0.5f32, 0.25, 0.125]);
        let b = Tensor::from_slice([2.0f32, 4.0, 8.0]);
        let c = &a * &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = mul_f32(&[0.5, 0.25, 0.125], &[2.0, 4.0, 8.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    // ========================================================================
    // Division Tests
    // ========================================================================

    fn test_div_f32_simple(config) {
        test_setup();
        let a = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let b = Tensor::from_slice([2.0f32, 4.0, 5.0]);
        let c = &a / &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = div_f32(&[10.0, 20.0, 30.0], &[2.0, 4.0, 5.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_div_f32_by_one(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([1.0f32, 1.0, 1.0]);
        let c = &a / &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = div_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_div_f32_fractional(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 1.0, 1.0]);
        let b = Tensor::from_slice([2.0f32, 4.0, 8.0]);
        let c = &a / &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = div_f32(&[1.0, 1.0, 1.0], &[2.0, 4.0, 8.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_div_f32_negative(config) {
        test_setup();
        let a = Tensor::from_slice([10.0f32, -20.0, 30.0]);
        let b = Tensor::from_slice([-2.0f32, 4.0, -5.0]);
        let c = &a / &b;

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = div_f32(&[10.0, -20.0, 30.0], &[-2.0, 4.0, -5.0]);
        assert_close_f32(&result, &expected, 1e-6);
    }

    // ========================================================================
    // Power Tests
    // ========================================================================

    fn test_pow_f32_simple(config) {
        test_setup();
        let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let b = Tensor::from_slice([2.0f32, 2.0, 2.0]);
        let c = a.try_pow(&b).unwrap();

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![4.0, 9.0, 16.0];
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_pow_f32_zero_exponent(config) {
        test_setup();
        let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let b = Tensor::from_slice([0.0f32, 0.0, 0.0]);
        let c = a.try_pow(&b).unwrap();

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![1.0, 1.0, 1.0];
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_pow_f32_one_exponent(config) {
        test_setup();
        let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let b = Tensor::from_slice([1.0f32, 1.0, 1.0]);
        let c = a.try_pow(&b).unwrap();

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![2.0, 3.0, 4.0];
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_pow_f32_fractional_exponent(config) {
        test_setup();
        let a = Tensor::from_slice([4.0f32, 9.0, 16.0]);
        let b = Tensor::from_slice([0.5f32, 0.5, 0.5]);
        let c = a.try_pow(&b).unwrap();

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![2.0, 3.0, 4.0];
        assert_close_f32(&result, &expected, 1e-5);
    }

    fn test_pow_f32_negative_base(config) {
        test_setup();
        let a = Tensor::from_slice([-2.0f32, -3.0, -4.0]);
        let b = Tensor::from_slice([2.0f32, 2.0, 2.0]);
        let c = a.try_pow(&b).unwrap();

        let result = c.realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![4.0, 9.0, 16.0];
        assert_close_f32(&result, &expected, 1e-6);
    }

    // ========================================================================
    // Chained Operations Tests
    // ========================================================================

    fn test_chained_add_mul(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
        let c = Tensor::from_slice([2.0f32, 2.0, 2.0]);

        let result = ((&a + &b) * &c).realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![10.0, 14.0, 18.0];
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_chained_mul_add(config) {
        test_setup();
        let a = Tensor::from_slice([2.0f32, 3.0, 4.0]);
        let b = Tensor::from_slice([5.0f32, 6.0, 7.0]);
        let c = Tensor::from_slice([1.0f32, 1.0, 1.0]);

        let result = (&a * &b + &c).realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![11.0, 19.0, 29.0];
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_chained_sub_div(config) {
        test_setup();
        let a = Tensor::from_slice([10.0f32, 20.0, 30.0]);
        let b = Tensor::from_slice([2.0f32, 4.0, 6.0]);
        let c = Tensor::from_slice([2.0f32, 2.0, 2.0]);

        let result = ((&a - &b) / &c).realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![4.0, 8.0, 12.0];
        assert_close_f32(&result, &expected, 1e-6);
    }

    fn test_complex_expression(config) {
        test_setup();
        let a = Tensor::from_slice([1.0f32, 2.0]);
        let b = Tensor::from_slice([3.0f32, 4.0]);
        let c = Tensor::from_slice([2.0f32, 2.0]);
        let d = Tensor::from_slice([1.0f32, 2.0]);
        let e = Tensor::from_slice([3.0f32, 4.0]);

        let result = ((&a + &b) * &c - &d).try_div(&e).unwrap()
            .realize_with(&config).unwrap().to_vec::<f32>().unwrap();
        let expected = vec![7.0 / 3.0, 2.5];
        assert_close_f32(&result, &expected, 1e-6);
    }
}
