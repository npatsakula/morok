use morok_dtype::{DType, DeviceSpec};
use morok_ir::{CallInfo, ConstValue, Op, SInt, UOp, shape::Shape};
use smallvec::smallvec;

use crate::{CpuBackend, PrepareConfig, Tensor, test::helpers::*};

#[test]
fn test_tensor_custom_kernel_builds_after_call_outputs() {
    let a = Tensor::empty(&[4], DType::Float32);
    let b = Tensor::empty(&[4], DType::Float32);

    let outputs = a
        .custom_kernel(&[&b], |placeholders| {
            assert_eq!(placeholders.len(), 2);
            UOp::sink(vec![placeholders[0].clone(), placeholders[1].clone()])
        })
        .expect("custom kernel should build");

    assert_eq!(outputs.len(), 2);
    for out in outputs {
        match out.uop().op() {
            Op::After { passthrough, deps } => {
                assert!(passthrough.has_buffer_identity());
                assert_eq!(deps.len(), 1);
                match deps[0].op() {
                    Op::Call { body, args, info } => {
                        assert!(matches!(body.op(), Op::Sink { .. }));
                        assert_eq!(args.len(), 2);
                        assert_eq!(*info, CallInfo::default());
                    }
                    op => panic!("expected CALL dep, got {op:?}"),
                }
            }
            op => panic!("expected AFTER output, got {op:?}"),
        }
    }
}

#[test]
fn test_tensor_custom_kernel_with_call_info() {
    let a = Tensor::empty(&[4], DType::Float32);
    let info =
        CallInfo { grad_tag: Some("grad_tag".to_string()), metadata: vec!["meta".to_string()], ..CallInfo::default() };

    let outputs = a
        .custom_kernel_with(&[], info.clone(), |placeholders| UOp::sink(vec![placeholders[0].clone()]))
        .expect("custom kernel should build");

    assert_eq!(outputs.len(), 1);
    match outputs[0].uop().op() {
        Op::After { deps, .. } => {
            assert_eq!(deps.len(), 1);
            match deps[0].op() {
                Op::Call { info: call_info, .. } => assert_eq!(*call_info, info),
                op => panic!("expected CALL dep, got {op:?}"),
            }
        }
        op => panic!("expected AFTER output, got {op:?}"),
    }
}

#[test]
fn test_tensor_custom_kernel_symbolic_placeholder_error() {
    // `placeholder_like` rejects symbolic-shaped inputs at construction
    // time (tinygrad parity).
    let n = UOp::define_var("N".to_string(), 1, 8);
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let shaped = buf.try_reshape(&Shape::from_iter([SInt::from(n)])).unwrap();
    let symbolic = Tensor::from_lazy(shaped);

    let err = match symbolic.custom_kernel(&[], |placeholders| UOp::sink(vec![placeholders[0].clone()])) {
        Ok(_) => panic!("symbolic placeholder_like should fail"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("symbolic shape is not supported"));
}

#[test]
fn test_tensor_custom_kernel_placeholder_like_multi_shape() {
    let shard = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(3)]))
        .unwrap();
    let multi = Tensor::from_lazy(UOp::multi(shard, 0));

    let outputs = multi
        .custom_kernel(&[], |placeholders| {
            assert_eq!(placeholders.len(), 1);
            let shape = placeholders[0].shape().unwrap().cloned().expect("placeholder shape");
            assert_eq!(shape.iter().map(|d| d.as_const()).collect::<Vec<_>>(), vec![Some(2), Some(3)]);
            UOp::sink(vec![placeholders[0].clone()])
        })
        .expect("custom kernel should build for multi-device shard shape");

    assert_eq!(outputs.len(), 1);
}

crate::codegen_tests! {
    fn test_tensor_custom_kernel_numerical_results(config) {
        test_setup();

        let src = Tensor::from_slice([3.5f32]);
        let dst = Tensor::empty(&[1], DType::Float32);

        let mut outputs = dst
            .custom_kernel(&[&src], |placeholders| {
                assert_eq!(placeholders.len(), 2);
                let out_buf = placeholders[0].clone();
                let in_buf = placeholders[1].clone();

                let idx = UOp::index_const(0);
                let load_idx = UOp::index().buffer(in_buf.clone()).indices(vec![idx.clone()]).ptr(true).call().unwrap();
                let store_idx = UOp::index().buffer(out_buf.clone()).indices(vec![idx]).ptr(true).call().unwrap();

                let loaded = UOp::load().buffer(in_buf).index(load_idx).call();
                let two = UOp::const_(DType::Float32, ConstValue::Float(2.0));
                let doubled = loaded.try_mul(&two).unwrap();
                let store = store_idx.store(doubled);
                UOp::sink(vec![store])
            })
            .expect("custom kernel should build");

        let mut out = outputs.remove(0);
        out.realize_with(&config).unwrap();

        let result = out.as_vec::<f32>().unwrap();
        assert_close_f32(&result, &[7.0], 1e-6);
    }
}

fn run_custom_op_numerical_test(backend: CpuBackend, mul_tpl: &str, add_tpl: &str) {
    test_setup();

    let src = Tensor::from_slice([3.5f32]);
    let dst = Tensor::empty(&[1], DType::Float32);

    let mut outputs = dst
        .custom_kernel(&[&src], |placeholders| {
            assert_eq!(placeholders.len(), 2);
            let out_buf = placeholders[0].clone();
            let in_buf = placeholders[1].clone();

            let idx = UOp::index_const(0);
            let load_idx = UOp::index().buffer(in_buf.clone()).indices(vec![idx.clone()]).ptr(true).call().unwrap();
            let store_idx = UOp::index().buffer(out_buf.clone()).indices(vec![idx]).ptr(true).call().unwrap();

            let loaded = UOp::load().buffer(in_buf).index(load_idx).call();
            let scaled = UOp::custom(smallvec![loaded], mul_tpl.to_string(), DType::Float32);
            let shifted = UOp::custom(smallvec![scaled], add_tpl.to_string(), DType::Float32);
            let store = store_idx.store(shifted);
            UOp::sink(vec![store])
        })
        .expect("custom kernel should build");

    let mut out = outputs.remove(0);
    let config = PrepareConfig::for_cpu_backend(backend);
    out.realize_with(&config).unwrap();

    let result = out.as_vec::<f32>().unwrap();
    assert_close_f32(&result, &[8.0], 1e-6);
}

#[test]
fn test_tensor_custom_op_numerical_clang_backend() {
    // C backend template strings.
    run_custom_op_numerical_test(CpuBackend::Clang, "({0} * 2.0f)", "({0} + 1.0f)");
}

#[test]
fn test_tensor_custom_op_llvm_backend_is_explicitly_unsupported() {
    test_setup();

    let src = Tensor::from_slice([3.5f32]);
    let dst = Tensor::empty(&[1], DType::Float32);

    let mut outputs = dst
        .custom_kernel(&[&src], |placeholders| {
            let out_buf = placeholders[0].clone();
            let in_buf = placeholders[1].clone();

            let idx = UOp::index_const(0);
            let load_idx = UOp::index().buffer(in_buf.clone()).indices(vec![idx.clone()]).ptr(true).call().unwrap();
            let store_idx = UOp::index().buffer(out_buf.clone()).indices(vec![idx]).ptr(true).call().unwrap();

            let loaded = UOp::load().buffer(in_buf).index(load_idx).call();
            let custom = UOp::custom(smallvec![loaded], "fmul float {0}, 2.0".to_string(), DType::Float32);
            let store = store_idx.store(custom);
            UOp::sink(vec![store])
        })
        .expect("custom kernel should build");

    let mut out = outputs.remove(0);
    let config = PrepareConfig::for_cpu_backend(CpuBackend::Llvm);
    let err = out.realize_with(&config).expect_err("LLVM backend should reject CUSTOM/CUSTOMI templates");
    assert!(format!("{err}").contains("does not support CUSTOM/CUSTOMI"), "unexpected error: {err}");
}
