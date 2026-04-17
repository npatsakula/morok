extern crate self as morok_model;

use morok_macros::jit_wrapper;
use morok_tensor::Tensor;

struct AddModel;

impl AddModel {
    fn forward(&self, x: &Tensor, y: &Tensor) -> morok_tensor::error::Result<Tensor> {
        x.try_add(y)
    }
}

jit_wrapper! {
    AddJit(AddModel) {
        x: Tensor,
        y: Tensor,

        build(x, y) {
            model.forward(&x, &y)
        }
    }
}

#[test]
fn test_jit_single_input_prepare_and_execute() {
    let model = AddModel;
    let mut jit = AddJit::new(model);

    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let y = Tensor::from_slice(&[10.0f32, 20.0, 30.0]);

    jit.prepare(&x, &y).unwrap();
    jit.execute().unwrap();

    let output = jit.output().unwrap();
    let mut result = vec![0.0f32; 3];
    output.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_jit_replay() {
    let model = AddModel;
    let mut jit = AddJit::new(model);

    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let y = Tensor::from_slice(&[10.0f32, 20.0, 30.0]);

    jit.prepare(&x, &y).unwrap();
    jit.execute().unwrap();

    let x2 = Tensor::from_slice(&[100.0f32, 200.0, 300.0]);
    let y2 = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);

    let x_buf = jit.x_mut().unwrap();
    let x2_buf = x2.buffer().unwrap();
    let x2_bytes = x2_buf.size();
    let mut x2_data = vec![0u8; x2_bytes];
    x2_buf.copyout(&mut x2_data).unwrap();
    x_buf.copyin(&x2_data).unwrap();

    let y_buf = jit.y_mut().unwrap();
    let y2_buf = y2.buffer().unwrap();
    let y2_bytes = y2_buf.size();
    let mut y2_data = vec![0u8; y2_bytes];
    y2_buf.copyout(&mut y2_data).unwrap();
    y_buf.copyin(&y2_data).unwrap();

    jit.execute().unwrap();

    let output = jit.output().unwrap();
    let mut result = vec![0.0f32; 3];
    output.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
    assert_eq!(result, vec![101.0, 202.0, 303.0]);
}

#[test]
fn test_jit_multiple_replays() {
    let model = AddModel;
    let mut jit = AddJit::new(model);

    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let y = Tensor::from_slice(&[10.0f32, 20.0, 30.0]);

    jit.prepare(&x, &y).unwrap();

    for i in 0..5 {
        let xi = Tensor::from_slice(&[i as f32; 3]);
        let yi = Tensor::from_slice(&[(i + 1) as f32; 3]);

        copy_tensor_to_buffer(&xi, jit.x_mut().unwrap());
        copy_tensor_to_buffer(&yi, jit.y_mut().unwrap());

        jit.execute().unwrap();

        let output = jit.output().unwrap();
        let mut result = vec![0.0f32; 3];
        output.copyout(unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, 12) }).unwrap();
        assert_eq!(result, vec![(i + i + 1) as f32; 3]);
    }
}

fn copy_tensor_to_buffer(tensor: &Tensor, dst: &mut morok_device::Buffer) {
    let src_buf = tensor.buffer().unwrap();
    let mut data = vec![0u8; src_buf.size()];
    src_buf.copyout(&mut data).unwrap();
    dst.copyin(&data).unwrap();
}
