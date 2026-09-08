use std::time::{Duration, Instant};

use snafu::ensure;
use svod_device::Buffer;

use crate::jit::{OutputLayoutMismatchSnafu, Result};

/// Flat host-side LSTM state. `h` and `c` are `f32` vectors of equal length;
/// for a single-layer cell `h.len() = hidden_size`, for a multi-layer stack
/// `h.len() = num_layers * hidden_size` (layer-major flat layout matching
/// `Tensor::stack([..], 0).reshape([1, 1, L*P])`).
pub struct LstmState {
    pub h: Vec<f32>,
    pub c: Vec<f32>,
}

impl LstmState {
    pub fn zeros(total: usize) -> Self {
        Self { h: vec![0.0f32; total], c: vec![0.0f32; total] }
    }

    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}

/// Per-step timing for a [`JitRecurrent::step`] call. `pack` = `pack_inputs` +
/// [`RecurrentJit::pack_state`]; `exec` = JIT execute; `read` = output copy +
/// state update.
#[derive(Default, Clone, Debug)]
pub struct StepTiming {
    pub pack: Duration,
    pub exec: Duration,
    pub read: Duration,
}

/// A `jit_wrapper!`-generated JIT that participates in a recurrent loop.
///
/// Expected output layout: `[non_state_head | h_flat | c_flat]` concatenated
/// along the last axis, where `h_flat` and `c_flat` each have length
/// `state.h.len()` floats. [`JitRecurrent::step`] copies the head into a
/// wrapper-owned buffer and writes the tail back into the host-side state.
pub trait RecurrentJit {
    /// Copy the active state into the JIT's typed state input buffers. Called
    /// once per step before [`execute_step`](Self::execute_step).
    fn pack_state(&mut self, state: &LstmState) -> Result<()>;

    /// Run the prepared JIT.
    fn execute_step(&mut self) -> Result<()>;

    /// Borrow the JIT's output buffer (`[head | h_tail | c_tail]`).
    fn output_buffer(&self) -> Result<&Buffer>;
}

/// Wraps a recurrent JIT and its host-side LSTM state. Owns the head read
/// buffer so callers can borrow the non-state output between steps.
pub struct JitRecurrent<J: RecurrentJit> {
    pub jit: J,
    state: LstmState,
    head_buf: Vec<f32>,
    head_len: usize,
    pub last_timing: StepTiming,
}

impl<J: RecurrentJit> JitRecurrent<J> {
    /// Construct from a prepared JIT, the initial state, and the declared
    /// head length in `f32` elements. The JIT's output buffer is read once
    /// and its size is checked against `head_len + |h| + |c|` to catch
    /// `build`-closure layout drift at construction time instead of letting
    /// silently mis-split output corrupt downstream values.
    pub fn new(jit: J, state: LstmState, head_len: usize) -> Result<Self> {
        let declared_state = state.h.len() + state.c.len();
        let actual = jit.output_buffer()?.size() / std::mem::size_of::<f32>();
        ensure!(
            actual == head_len + declared_state,
            OutputLayoutMismatchSnafu { declared_head: head_len, declared_state, actual }
        );
        Ok(Self { jit, state, head_buf: vec![0.0; head_len], head_len, last_timing: StepTiming::default() })
    }

    pub fn state(&self) -> &LstmState {
        &self.state
    }
    pub fn state_mut(&mut self) -> &mut LstmState {
        &mut self.state
    }

    pub fn head_len(&self) -> usize {
        self.head_len
    }

    /// One recurrent step. `pack_inputs` writes per-step non-state inputs
    /// (audio chunk, token id, encoder frame, …) into the JIT. The wrapper
    /// then packs state, executes, splits the output into
    /// `[head | h_tail | c_tail]`, and updates host state. Returns the head
    /// slice borrowed from the wrapper-owned buffer.
    pub fn step<F>(&mut self, pack_inputs: F) -> Result<&[f32]>
    where
        F: FnOnce(&mut J) -> Result<()>,
    {
        let t0 = Instant::now();
        pack_inputs(&mut self.jit)?;
        self.jit.pack_state(&self.state)?;
        let t1 = Instant::now();
        self.jit.execute_step()?;
        let t2 = Instant::now();
        {
            let out = self.jit.output_buffer()?;
            let arr = out.as_array::<f32>()?;
            let flat = arr.as_slice().expect("contiguous JIT output");
            let head_len = self.head_len;
            let h_len = self.state.h.len();
            self.head_buf.copy_from_slice(&flat[..head_len]);
            self.state.h.copy_from_slice(&flat[head_len..head_len + h_len]);
            self.state.c.copy_from_slice(&flat[head_len + h_len..]);
        }
        let t3 = Instant::now();
        self.last_timing = StepTiming { pack: t1 - t0, exec: t2 - t1, read: t3 - t2 };
        Ok(&self.head_buf)
    }

    /// Zero the host-side state. The next [`step`](Self::step) writes the
    /// fresh state into the JIT.
    pub fn reset(&mut self) {
        self.state.reset();
    }
}
