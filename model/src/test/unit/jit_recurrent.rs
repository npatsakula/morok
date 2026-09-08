extern crate self as svod_model;

use svod_dtype::DType;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;

use crate::jit::{InputSpec, JitError, JitRecurrent, LstmState, RecurrentJit, Result};

const HIDDEN: usize = 2;

struct RecurrentTestModel;

impl RecurrentTestModel {
    /// `head = x + 1`, `h_next = h + 1`, `c_next = c + 2`.
    /// Output is `[head | h_next | c_next]` along axis 0 (single 1-D tensor of
    /// length `1 + HIDDEN + HIDDEN`).
    fn forward(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<Tensor> {
        let one_scalar = Tensor::ones(&[1], DType::Float32);
        let one_vec = Tensor::ones(&[HIDDEN], DType::Float32);
        let two_vec = Tensor::full(&[HIDDEN], 2.0f32, DType::Float32);
        let head = x.try_add(&one_scalar)?;
        let h_next = h.try_add(&one_vec)?;
        let c_next = c.try_add(&two_vec)?;
        Ok(Tensor::cat(&[&head, &h_next, &c_next], 0)?)
    }
}

jit_wrapper! {
    RecurrentTestJit(RecurrentTestModel) {
        x: Tensor,
        h: Tensor,
        c: Tensor,

        build(x, h, c) {
            model.forward(x, h, c)
        }
    }
}

impl RecurrentJit for RecurrentTestJit {
    fn pack_state(&mut self, s: &LstmState) -> Result<()> {
        {
            let buf = self.h_mut()?;
            let mut view = buf.as_array_mut::<f32>()?;
            view.as_slice_mut().expect("contiguous h").copy_from_slice(&s.h);
        }
        {
            let buf = self.c_mut()?;
            let mut view = buf.as_array_mut::<f32>()?;
            view.as_slice_mut().expect("contiguous c").copy_from_slice(&s.c);
        }
        Ok(())
    }

    fn execute_step(&mut self) -> Result<()> {
        self.execute()
    }

    fn output_buffer(&self) -> Result<&svod_device::Buffer> {
        self.output()
    }
}

fn prepare_jit() -> RecurrentTestJit {
    let mut jit = RecurrentTestJit::new(RecurrentTestModel);
    jit.prepare(InputSpec::f32(&[1]), InputSpec::f32(&[HIDDEN]), InputSpec::f32(&[HIDDEN])).unwrap();
    jit
}

fn write_x(jit: &mut RecurrentTestJit, value: f32) -> Result<()> {
    let buf = jit.x_mut()?;
    let mut view = buf.as_array_mut::<f32>()?;
    view.as_slice_mut().expect("contiguous x")[0] = value;
    Ok(())
}

#[test]
fn test_jit_recurrent_step_advances_state_and_returns_head() {
    let mut rec = JitRecurrent::new(prepare_jit(), LstmState::zeros(HIDDEN), 1).unwrap();

    let head = rec.step(|jit| write_x(jit, 10.0)).unwrap().to_vec();
    assert_eq!(head, vec![11.0]);
    assert_eq!(rec.state().h, vec![1.0, 1.0]);
    assert_eq!(rec.state().c, vec![2.0, 2.0]);

    let head = rec.step(|jit| write_x(jit, 5.0)).unwrap().to_vec();
    assert_eq!(head, vec![6.0]);
    assert_eq!(rec.state().h, vec![2.0, 2.0]);
    assert_eq!(rec.state().c, vec![4.0, 4.0]);
}

#[test]
fn test_jit_recurrent_reset_zeros_state_without_reallocating() {
    let mut rec = JitRecurrent::new(prepare_jit(), LstmState::zeros(HIDDEN), 1).unwrap();
    rec.step(|jit| write_x(jit, 7.0)).unwrap();
    let h_ptr_before = rec.state().h.as_ptr();
    let c_ptr_before = rec.state().c.as_ptr();

    rec.reset();
    assert_eq!(rec.state().h, vec![0.0, 0.0]);
    assert_eq!(rec.state().c, vec![0.0, 0.0]);
    assert_eq!(rec.state().h.as_ptr(), h_ptr_before);
    assert_eq!(rec.state().c.as_ptr(), c_ptr_before);

    let head = rec.step(|jit| write_x(jit, 3.0)).unwrap().to_vec();
    assert_eq!(head, vec![4.0]);
    assert_eq!(rec.state().h, vec![1.0, 1.0]);
    assert_eq!(rec.state().c, vec![2.0, 2.0]);
}

#[test]
fn test_jit_recurrent_rejects_output_layout_mismatch() {
    // Toy JIT's output is `[head(1) | h(HIDDEN) | c(HIDDEN)]`. Declaring a
    // head length that disagrees with the actual layout must surface as a
    // typed error at construction, not silent mis-splitting at step time.
    let Err(err) = JitRecurrent::new(prepare_jit(), LstmState::zeros(HIDDEN), 99) else {
        panic!("expected OutputLayoutMismatch, got Ok");
    };
    match err {
        JitError::OutputLayoutMismatch { declared_head, declared_state, actual } => {
            assert_eq!(declared_head, 99);
            assert_eq!(declared_state, 2 * HIDDEN);
            assert_eq!(actual, 1 + 2 * HIDDEN);
        }
        other => panic!("expected OutputLayoutMismatch, got {other:?}"),
    }
}
