use svod_dtype::DType;
use svod_ir::{ConstValue, SInt};

use crate::Tensor;
use crate::test::helpers::*;

crate::codegen_tests! {
    fn test_batch_two_independent(config) {
        test_setup();
        let a = Tensor::full(&[4], 2.0f32, DType::Float32).unwrap();
        let b = Tensor::full(&[3], 3.0f32, DType::Float32).unwrap();
        Tensor::realize_batch_with([&a, &b], &config).unwrap();
        assert_eq!(a.as_vec::<f32>().unwrap(), vec![2.0; 4]);
        assert_eq!(b.as_vec::<f32>().unwrap(), vec![3.0; 3]);
    }

    fn test_batch_shared_input(config) {
        test_setup();
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let ten = Tensor::full(&[3], 10.0f32, DType::Float32).unwrap();
        let two = Tensor::full(&[3], 2.0f32, DType::Float32).unwrap();
        let a = &x + &ten;
        let b = &x * &two;
        Tensor::realize_batch_with([&a, &b], &config).unwrap();
        assert_close_f32(&a.as_vec::<f32>().unwrap(), &[11.0, 12.0, 13.0], 1e-6);
        assert_close_f32(&b.as_vec::<f32>().unwrap(), &[2.0, 4.0, 6.0], 1e-6);
    }

    fn test_batch_single(config) {
        test_setup();
        let t = Tensor::full(&[5], 7.0f32, DType::Float32).unwrap();
        Tensor::realize_batch_with([&t], &config).unwrap();
        assert_eq!(t.as_vec::<f32>().unwrap(), vec![7.0; 5]);
    }

    fn test_batch_mixed_realized(config) {
        test_setup();
        let a = Tensor::full(&[3], 1.0f32, DType::Float32).unwrap();
        a.realize_with(&config).unwrap();
        let b = Tensor::full(&[3], 2.0f32, DType::Float32).unwrap();
        Tensor::realize_batch_with([&a, &b], &config).unwrap();
        assert_eq!(a.as_vec::<f32>().unwrap(), vec![1.0; 3]);
        assert_eq!(b.as_vec::<f32>().unwrap(), vec![2.0; 3]);
    }

    fn test_batch_mixed_dtypes(config) {
        test_setup();
        let a = Tensor::full(&[3], 5.0f32, DType::Float32).unwrap();
        let b = Tensor::full_dynamic(&[SInt::from(4)], ConstValue::Int(42), DType::Int32).unwrap();
        Tensor::realize_batch_with([&a, &b], &config).unwrap();
        assert_eq!(a.as_vec::<f32>().unwrap(), vec![5.0; 3]);
        assert_eq!(b.as_vec::<i32>().unwrap(), vec![42; 4]);
    }

    fn test_batch_three_outputs(config) {
        test_setup();
        let a = Tensor::full(&[2], 1.0f32, DType::Float32).unwrap();
        let b = Tensor::full(&[3], 2.0f32, DType::Float32).unwrap();
        let c = Tensor::full(&[4], 3.0f32, DType::Float32).unwrap();
        Tensor::realize_batch_with([&a, &b, &c], &config).unwrap();
        assert_eq!(a.as_vec::<f32>().unwrap(), vec![1.0; 2]);
        assert_eq!(b.as_vec::<f32>().unwrap(), vec![2.0; 3]);
        assert_eq!(c.as_vec::<f32>().unwrap(), vec![3.0; 4]);
    }

    fn test_batch_shared_with_reduce(config) {
        test_setup();
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        let two = Tensor::full(&[4], 2.0f32, DType::Float32).unwrap();
        let doubled = &x * &two;
        let sum = doubled.sum(()).unwrap();
        Tensor::realize_batch_with([&doubled, &sum], &config).unwrap();
        assert_close_f32(&doubled.as_vec::<f32>().unwrap(), &[2.0, 4.0, 6.0, 8.0], 1e-6);
        assert_close_f32(&sum.as_vec::<f32>().unwrap(), &[20.0], 1e-5);
    }

    // Diamond pattern: output A is consumed by kernel producing output B.
    // Both A and B must be in the output set.
    fn test_batch_diamond_output(config) {
        test_setup();
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0]);
        let ten = Tensor::full(&[3], 10.0f32, DType::Float32).unwrap();
        let a = &x + &ten;       // output 1
        let b = &a * &a;         // output 2 (consumes a)
        Tensor::realize_batch_with([&a, &b], &config).unwrap();
        assert_close_f32(&a.as_vec::<f32>().unwrap(), &[11.0, 12.0, 13.0], 1e-6);
        assert_close_f32(&b.as_vec::<f32>().unwrap(), &[121.0, 144.0, 169.0], 1e-6);
    }

    /// `&self` realize means a shared clone (or a `&Vec<Tensor>` of them) sees
    /// the result too: realization lives in the registry entry, not the handle.
    fn realize_batch_takes_shared_refs(config) {
        test_setup();
        let outputs: Vec<Tensor> =
            [2.0f32, 3.0].iter().map(|&v| &Tensor::from_slice([1.0f32, 2.0]) * &Tensor::full(&[2], v, DType::Float32).unwrap()).collect();
        let aliases = outputs.clone();

        Tensor::realize_batch_with(&outputs, &config).unwrap();

        assert_close_f32(&aliases[0].as_vec::<f32>().unwrap(), &[2.0, 4.0], 1e-6);
        assert_close_f32(&aliases[1].as_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-6);
    }

    fn test_prepare_batch_execute(config) {
        test_setup();
        let a = &Tensor::from_slice([1.0f32, 2.0]) + &Tensor::from_slice([3.0f32, 4.0]);
        let b = &Tensor::from_slice([10.0f32, 20.0]) * &Tensor::from_slice([2.0f32, 3.0]);
        let plan = Tensor::prepare_batch_with([&a, &b], &config).unwrap();
        plan.execute().unwrap();
        assert_eq!(plan.num_outputs(), 2);
        assert_close_f32(&a.as_vec::<f32>().unwrap(), &[4.0, 6.0], 1e-6);
        assert_close_f32(&b.as_vec::<f32>().unwrap(), &[20.0, 60.0], 1e-6);
    }
}

#[test]
fn test_batch_empty() {
    Tensor::realize_batch(std::iter::empty::<&Tensor>()).unwrap();
}

#[test]
fn test_batch_output_count() {
    let a = Tensor::full(&[2], 1.0f32, DType::Float32).unwrap();
    let b = Tensor::full(&[3], 2.0f32, DType::Float32).unwrap();
    let c = Tensor::full(&[4], 3.0f32, DType::Float32).unwrap();
    Tensor::realize_batch([&a, &b, &c]).unwrap();
}
