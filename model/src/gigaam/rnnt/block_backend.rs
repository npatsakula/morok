//! Device-block RNN-T backend ([`svod_arch::rnnt::BatchBlockStep`]):
//! [`super::block::forward_block`] unrolled to a single graph-captured plan. The
//! five carried states are `state { .. }` slots, so they recycle in place inside
//! `execute()`; per block the host only reads back three small tapes + one flag.

use snafu::ResultExt;
use svod_arch::rnnt::{BatchBlockStep, BlockTapes};

use crate::jit::{BuildSnafu, InputSpec, JitError};

use super::block::BLOCK_STEPS;
use super::jit::{RnntBlockJit, RnntEncProjJit};
use crate::gigaam::model::GigaAm;

pub struct RnntBlockBackend {
    jit: RnntBlockJit,
    /// Per-wave encoder projection `[B, T, E] -> [B, T, J]` — one MFMA matmul
    /// replaces the per-step row projection inside the block.
    proj: RnntEncProjJit,
    lanes: usize,
    max_t: usize,
    enc_hidden: usize,
    blank_id: usize,

    // Host-side tape staging (read once per block).
    tokens: Vec<i32>,
    emit: Vec<i32>,
    frames_tape: Vec<i32>,

    pub stats: BlockStats,
}

#[derive(Default, Clone, Debug)]
pub struct BlockStats {
    pub n_blocks: u64,
    /// Real (non-blank) emissions counted over the full `lanes * BLOCK_STEPS`
    /// tape, across all blocks — i.e. total emitted tokens (window-invariant).
    /// Compare against `n_blocks * lanes * BLOCK_STEPS` (tape slots) for the
    /// useful fraction; the per-lane step count `n_blocks * BLOCK_STEPS` is the
    /// separate lever a wider window cuts.
    pub steps_emitted: u64,
    /// Time inside `execute()` (graph submit; the block runs async).
    pub t_exec: std::time::Duration,
    /// Time reading back the tapes — includes the wait for the block's compute,
    /// since `execute()` does not block.
    pub t_read: std::time::Duration,
}

impl RnntBlockBackend {
    /// `max_t` is the encoder-frame capacity (`max_t_sub`); the `enc` input is
    /// `[lanes, max_t, d_model]` and stays device-local across the wave.
    pub fn from_model(model: GigaAm, lanes: usize, max_t: usize) -> crate::jit::Result<Self> {
        let (head, _) = model.head.expect_rnnt("RnntBlockBackend").boxed().context(BuildSnafu)?;
        let (layers, p) = (head.pred_rnn_layers, head.pred_hidden);
        let joint_hidden = head.joint_hidden;
        let enc_hidden = model.config.d_model;
        let blank_id = head.predictor.blank_id;

        let config = svod_tensor::PrepareConfig::device_local();
        let mut proj = RnntEncProjJit::new(model.clone());
        proj.prepare_with_config(InputSpec::f32(&[lanes, max_t, enc_hidden]).device_local(), &config)?;

        // Inputs first, then the `state { .. }` slots (always device-local).
        let mut jit = RnntBlockJit::new(model);
        jit.prepare_with_config(
            InputSpec::f32(&[lanes, max_t, joint_hidden]).device_local(),
            InputSpec::i32(&[lanes, 1]),
            InputSpec::i64(&[lanes, 1]),
            InputSpec::i64(&[lanes, 1]),
            InputSpec::i32(&[lanes, 1]),
            InputSpec::f32(&[layers, lanes, p]),
            InputSpec::f32(&[layers, lanes, p]),
            &config,
        )?;

        Ok(Self {
            jit,
            proj,
            lanes,
            max_t,
            enc_hidden,
            blank_id,
            tokens: vec![0; lanes * BLOCK_STEPS],
            emit: vec![0; lanes * BLOCK_STEPS],
            frames_tape: vec![0; lanes * BLOCK_STEPS],
            stats: BlockStats::default(),
        })
    }

    /// Stage the wave's encoder rows + valid frame counts. `frames[i]` is the
    /// tight `[valid[i], enc_hidden]` block; unused rows stay stale (clamped
    /// gather + emit mask keep them inert).
    pub fn bind_batch(&mut self, frames: &[Vec<f32>], valid: &[usize]) -> crate::jit::Result<()> {
        let row = self.max_t * self.enc_hidden;
        let mut staged = vec![0f32; self.lanes * row];
        for (i, f) in frames.iter().enumerate() {
            staged[i * row..i * row + f.len()].copy_from_slice(f);
        }
        self.proj.enc_mut()?.copyin(bytemuck::cast_slice(&staged))?;
        self.proj.execute()?;
        // Projected rows -> block input, device->device (drains the proj exec).
        let proj_out = self.proj.output()?;
        let bytes = proj_out.size();
        self.jit.enc_mut()?.copy_region_from(0, proj_out, 0, bytes)?;

        let mut view = self.jit.valid_view_mut::<i32>()?;
        let slice = view.as_slice_mut().expect("contiguous valid");
        slice.fill(0);
        for (dst, &n) in slice.iter_mut().zip(valid) {
            *dst = n as i32;
        }
        Ok(())
    }
}

impl BatchBlockStep for RnntBlockBackend {
    type Error = JitError;

    fn batch(&self) -> usize {
        self.lanes
    }

    fn block_steps(&self) -> usize {
        BLOCK_STEPS
    }

    fn run_block(&mut self) -> Result<BlockTapes<'_>, Self::Error> {
        let t0 = std::time::Instant::now();
        // `execute()` recycles the `time/prev/symbols/h/c` state slots in place,
        // so the next block reads updated state with no host-issued copy.
        self.jit.execute()?;
        let t1 = std::time::Instant::now();

        self.jit.tape()?.copyout_prefix(bytemuck::cast_slice_mut(&mut self.tokens))?;
        self.jit.emit()?.copyout_prefix(bytemuck::cast_slice_mut(&mut self.emit))?;
        self.jit.frame()?.copyout_prefix(bytemuck::cast_slice_mut(&mut self.frames_tape))?;
        let active_any = self.jit.active_any_to_vec::<i32>()?[0] != 0;
        let t2 = std::time::Instant::now();

        self.stats.n_blocks += 1;
        self.stats.steps_emitted += self.emit.iter().filter(|&&e| e != 0).count() as u64;
        self.stats.t_exec += t1 - t0;
        self.stats.t_read += t2 - t1;
        Ok(BlockTapes { tokens: &self.tokens, emit: &self.emit, frames: &self.frames_tape, active_any })
    }

    /// Cold start: zero every carried state, then seed `prev` with the blank id
    /// (the predictor's start-of-sequence token — the only slot whose reset
    /// value is not zero).
    fn reset(&mut self) -> Result<(), Self::Error> {
        self.jit.reset()?;
        let blanks: Vec<i64> = vec![self.blank_id as i64; self.lanes];
        self.jit.prev_mut()?.copyin(bytemuck::cast_slice(&blanks))?;
        Ok(())
    }
}
