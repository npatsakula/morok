extern crate self as svod_model;

use svod_macros::jit_wrapper;
use svod_tensor::{PrepareConfig, Tensor};

use crate::jit::InputSpec;

struct AddModel;

impl AddModel {
    fn forward(&self, x: &Tensor, y: &Tensor) -> crate::jit::Result<Tensor> {
        x.try_add(y).map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })
    }
}

jit_wrapper! {
    AddJit(AddModel) {
        x: Tensor,
        y: Tensor,

        build(x, y) {
            model.forward(x, y)
        }
    }
}

struct SplitModel;

impl SplitModel {
    fn forward(&self, x: &Tensor, y: &Tensor) -> crate::jit::Result<(Tensor, Tensor)> {
        let sum = x.try_add(y).map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        let diff = x.try_sub(y).map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        Ok((sum, diff))
    }
}

// Two same-dtype outputs from one plan.
jit_wrapper! {
    SplitJit(SplitModel) {
        x: Tensor,
        y: Tensor,

        outputs { sum, diff },

        build(x, y) {
            model.forward(x, y)
        }
    }
}

struct MixedModel;

impl MixedModel {
    /// One f32 output (`x + y`) and one int32 output (`argmax(x)`) — exercises
    /// per-output dtype: each output buffer carries its own dtype.
    fn forward(&self, x: &Tensor, y: &Tensor) -> crate::jit::Result<(Tensor, Tensor)> {
        let sum = x.try_add(y).map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        let arg = x.argmax(None).map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        Ok((sum, arg))
    }
}

jit_wrapper! {
    MixedJit(MixedModel) {
        x: Tensor,
        y: Tensor,

        outputs { sum, arg },

        build(x, y) {
            model.forward(x, y)
        }
    }
}

#[test]
fn test_jit_single_input_prepare_and_execute() {
    let mut jit = AddJit::new(AddModel);

    let cfg = PrepareConfig::default();
    jit.prepare_with_config(InputSpec::f32(&[3]), InputSpec::f32(&[3]), &cfg).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());

    jit.execute().unwrap();

    let output = jit.output().unwrap();
    let mut result = vec![0.0f32; 3];
    output.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_jit_replay() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([100.0f32, 200.0, 300.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    let output = jit.output().unwrap();
    let mut result = vec![0.0f32; 3];
    output.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
    assert_eq!(result, vec![101.0, 202.0, 303.0]);
}

#[test]
fn test_jit_multiple_replays() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    for i in 0..5 {
        copy_tensor_to_buffer(&Tensor::from_slice([i as f32; 3]), jit.x_mut().unwrap());
        copy_tensor_to_buffer(&Tensor::from_slice([(i + 1) as f32; 3]), jit.y_mut().unwrap());

        jit.execute().unwrap();

        let output = jit.output().unwrap();
        let mut result = vec![0.0f32; 3];
        output.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
        assert_eq!(result, vec![(i + i + 1) as f32; 3]);
    }
}

#[test]
fn test_jit_profiled_execution_apis() {
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());

    let profiles = jit.execute_profiled().unwrap();
    assert!(!profiles.is_empty(), "expected at least one kernel profile");

    let profiles_with_vars = jit.execute_with_vars_profiled(&[]).unwrap();
    assert!(!profiles_with_vars.is_empty(), "expected at least one kernel profile with vars");

    let output = jit.output().unwrap();
    let mut result = vec![0.0f32; 3];
    output.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_jit_multi_output_same_dtype() {
    let mut jit = SplitJit::new(SplitModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();

    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    let mut sum = vec![0.0f32; 3];
    jit.sum().unwrap().copyout(unsafe { std::slice::from_raw_parts_mut(sum.as_mut_ptr() as *mut u8, 12) }).unwrap();
    assert_eq!(sum, vec![11.0, 22.0, 33.0]);

    let mut diff = vec![0.0f32; 3];
    jit.diff().unwrap().copyout(unsafe { std::slice::from_raw_parts_mut(diff.as_mut_ptr() as *mut u8, 12) }).unwrap();
    assert_eq!(diff, vec![9.0, 18.0, 27.0]);
}

#[test]
fn test_jit_multi_output_mixed_dtype() {
    let mut jit = MixedJit::new(MixedModel);
    jit.prepare(InputSpec::f32(&[4]), InputSpec::f32(&[4])).unwrap();

    // argmax(x) is index 2 (value 9.0); x + y is elementwise.
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 5.0, 9.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 10.0, 10.0, 10.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    let mut sum = vec![0.0f32; 4];
    jit.sum().unwrap().copyout(unsafe { std::slice::from_raw_parts_mut(sum.as_mut_ptr() as *mut u8, 16) }).unwrap();
    assert_eq!(sum, vec![11.0, 15.0, 19.0, 13.0]);

    // The int32 output reads back with its own dtype.
    let mut arg = vec![0i32; 1];
    jit.arg().unwrap().copyout(unsafe { std::slice::from_raw_parts_mut(arg.as_mut_ptr() as *mut u8, 4) }).unwrap();
    assert_eq!(arg, vec![2]);
}

fn copy_tensor_to_buffer(tensor: &Tensor, dst: &mut svod_device::Buffer) {
    let src_buf = tensor.buffer().unwrap();
    let mut data = vec![0u8; src_buf.size()];
    src_buf.copyout(&mut data).unwrap();
    dst.copyin(&data).unwrap();
}

#[test]
fn test_jit_replicate_executes_independently() {
    let read = |buffer: &svod_device::Buffer| {
        let mut result = vec![0.0f32; 3];
        buffer.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
        result
    };

    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 2.0, 3.0]), jit.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([10.0f32, 20.0, 30.0]), jit.y_mut().unwrap());
    jit.execute().unwrap();

    // The replica snapshots the original's inputs, then diverges.
    let mut replica = jit.replicate().unwrap();
    copy_tensor_to_buffer(&Tensor::from_slice([100.0f32, 200.0, 300.0]), replica.x_mut().unwrap());
    copy_tensor_to_buffer(&Tensor::from_slice([1.0f32, 1.0, 1.0]), replica.y_mut().unwrap());
    replica.execute().unwrap();
    assert_eq!(read(replica.output().unwrap()), vec![101.0, 201.0, 301.0]);

    // The original's inputs and output are untouched by the replica.
    jit.execute().unwrap();
    assert_eq!(read(jit.output().unwrap()), vec![11.0, 22.0, 33.0]);

    // Replicas mint further replicas, snapshotting the replica's state.
    let mut second = replica.replicate().unwrap();
    second.execute().unwrap();
    assert_eq!(read(second.output().unwrap()), vec![101.0, 201.0, 301.0]);
}

/// Phase-2 acceptance: multiple models preparing AND executing concurrently
/// from several threads must not cross-talk (per-tensor input identities,
/// winner-computes caches, per-plan queues).
#[test]
fn test_concurrent_multi_model_prepare_and_execute() {
    let read3 = |buffer: &svod_device::Buffer| {
        let mut result = vec![0.0f32; 3];
        buffer.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
        result
    };
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        for worker in 0..4u32 {
            let barrier = &barrier;
            let read3 = &read3;
            scope.spawn(move || {
                for round in 0..8u32 {
                    barrier.wait();
                    let x = [(worker * 100 + round) as f32; 3];
                    let y = [(worker + round * 10) as f32 + 0.5; 3];
                    if worker % 2 == 0 {
                        let mut jit = AddJit::new(AddModel);
                        jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();
                        copy_tensor_to_buffer(&Tensor::from_slice(x), jit.x_mut().unwrap());
                        copy_tensor_to_buffer(&Tensor::from_slice(y), jit.y_mut().unwrap());
                        jit.execute().unwrap();
                        assert_eq!(read3(jit.output().unwrap()), vec![x[0] + y[0]; 3]);
                    } else {
                        let mut jit = SplitJit::new(SplitModel);
                        jit.prepare(InputSpec::f32(&[3]), InputSpec::f32(&[3])).unwrap();
                        copy_tensor_to_buffer(&Tensor::from_slice(x), jit.x_mut().unwrap());
                        copy_tensor_to_buffer(&Tensor::from_slice(y), jit.y_mut().unwrap());
                        jit.execute().unwrap();
                        assert_eq!(read3(jit.sum().unwrap()), vec![x[0] + y[0]; 3]);
                        assert_eq!(read3(jit.diff().unwrap()), vec![x[0] - y[0]; 3]);
                    }
                }
            });
        }
    });
}

/// Phase-4 acceptance benchmark: two replicas doing per-step
/// copyin+execute+copyout must overlap in wall-clock time (scoped host sync —
/// one plan's readback no longer drains the other's lanes). Run manually:
/// `SVOD_DEVICE=AMD cargo test -p svod-model --lib two_replica_throughput -- --ignored --nocapture`
#[test]
#[ignore = "throughput benchmark; run manually with --ignored --nocapture (AMD for the real measurement)"]
fn two_replica_throughput_overlaps() {
    const STEPS: usize = 200;
    let mut jit = AddJit::new(AddModel);
    jit.prepare(InputSpec::f32(&[1024]), InputSpec::f32(&[1024])).unwrap();
    let mut replica = jit.replicate().unwrap();

    let bytes = vec![1u8; 4096];
    let run = |jit: &mut AddJit, steps: usize| {
        let mut out = vec![0u8; 4096];
        for _ in 0..steps {
            jit.x_mut().unwrap().copyin(&bytes).unwrap();
            jit.y_mut().unwrap().copyin(&bytes).unwrap();
            jit.execute().unwrap();
            jit.output().unwrap().copyout(&mut out).unwrap();
        }
    };

    run(&mut jit, 5);
    run(&mut replica, 5);

    let start = std::time::Instant::now();
    run(&mut jit, STEPS);
    run(&mut replica, STEPS);
    let serial = start.elapsed();

    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        let (first, second) = (&mut jit, &mut replica);
        let run = &run;
        scope.spawn(move || run(first, STEPS));
        scope.spawn(move || run(second, STEPS));
    });
    let concurrent = start.elapsed();

    eprintln!(
        "serial={serial:?} concurrent={concurrent:?} speedup={:.2}x",
        serial.as_secs_f64() / concurrent.as_secs_f64()
    );
    assert!(concurrent < serial, "concurrent replicas must overlap: serial={serial:?} concurrent={concurrent:?}");
}
