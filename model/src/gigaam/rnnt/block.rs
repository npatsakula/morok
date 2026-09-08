//! K-step device-resident RNN-T decode block (NeMo FULL_GRAPH analog) with
//! WIND windowed non-blank detection.
//!
//! [`forward_block`] traces [`BLOCK_STEPS`] decode steps with all state
//! on-device and masked `where` everywhere (the predictor runs unconditionally;
//! non-emitting lanes keep their committed state). The host reads one token tape
//! per block. No runtime vars — everything concrete, so the plan graph-captures
//! into one submit.
//!
//! Each step evaluates the joint over a window of `W` frames (the
//! [`forward_block`] const generic) against the FIXED predictor state and jumps
//! to the first non-blank (WIND, arXiv 2505.13765). This is an exact
//! reformulation of greedy decoding — for a fixed predictor state every frame
//! before the first non-blank is provably blank, so skipping them matches
//! frame-by-frame decoding — and it subsumes label-looping for this
//! architecture: by collapsing a blank run into one step it cuts the
//! predictor/joint evaluations and host syncs that a per-frame loop spends on
//! blanks. `W == 1` is the per-frame baseline; the output is byte-identical
//! across windows (verified against `decode_batch_labels` in the arch tests),
//! so `W` is a pure performance knob whose optimum shifts with the GPU.
//!
//! Per step (identical greedy semantics to `decode_batch_labels` for any window):
//! toks = joint(enc[time .. time+W], g) → first usable (in-bounds, non-blank)
//! offset → emit one token there, prev/state where-commit, symbols run length +
//! cap → time jumps past the leading blanks (or the whole blank window);
//! tapes record (token, emit, emission-frame).
//!
//! On-device control flow the runtime lacks (a data-dependent while-loop, e.g.
//! CUDA-graph conditional nodes) would remove the per-block host readback
//! entirely; that is out of scope here — the host drives the block loop on the
//! `active_any` flag.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::gigaam::Result;

use crate::gigaam::model::GigaAm;
use crate::state::scoped;

/// Decode steps per block execute. Amortizes the per-block readback; bounds
/// the unrolled plan (~40 kernels/step).
pub(crate) const BLOCK_STEPS: usize = 16;

/// `RnntBlockJit`'s build tuple: the four read-back tapes/flag followed by the
/// five carried-state values, in the order its `outputs { .. }` and
/// `state { .. }` blocks declare them.
pub(crate) type BlockOutputs = (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);

/// `enc_proj [B, T, J]` (pre-projected encoder, [`super::joint::RnntJoint::project_encoder`]), `time/prev [B,1] i64`, `symbols/valid [B,1] i32`,
/// `h/c [L, B, P]` → `(tape, emit, frame [B,K] i32, active_any [1,1] i32,
/// time, prev, symbols, h, c)`. The last five are the `state { .. }` slot
/// values: the wrapper stores each back into its own input buffer, so the
/// carried state recycles without a device→device copy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_block<const W: usize>(
    model: &GigaAm,
    enc_proj: &Tensor,
    time: &Tensor,
    prev: &Tensor,
    symbols: &Tensor,
    valid: &Tensor,
    h: &Tensor,
    c: &Tensor,
) -> Result<BlockOutputs> {
    const { assert!(W >= 1, "WIND window W must be >= 1") };
    let (head, runtime) = model.head.expect_rnnt("RnntBlockJit")?;
    let max_symbols = runtime.max_symbols_per_step.max(1);
    let blank = head.predictor.blank_id as i64;
    let (l, p) = (head.pred_rnn_layers as isize, head.pred_hidden as isize);
    let (b, j) = (enc_proj.dim_const(0)? as isize, enc_proj.dim_const(2)? as isize);
    let valid64 = valid.cast(DType::Int64);
    let last = valid64.try_sub(1i64)?;
    // Window offsets `[0, W)` as a row vector, broadcast onto each lane's frame
    // cursor. At `W == 1` the window collapses to the single current frame and
    // every step below reduces to the per-frame greedy update.
    let w = W as isize;
    let arange_w = {
        let a = Tensor::arange(0, Some(W as i64), Some(1))?;
        a.cast(DType::Int64).try_reshape([1, w])?
    };

    let (mut time, mut prev, mut symbols, mut h, mut c) =
        (time.clone(), prev.clone(), symbols.clone(), h.clone(), c.clone());
    let (mut tapes, mut emits, mut frames) = (Vec::new(), Vec::new(), Vec::new());

    for _ in 0..BLOCK_STEPS {
        let in_bounds = time.try_lt(&valid64)?; // [B,1] bool
        // Clamp the gather index for finished lanes (mask restores correctness).
        let safe_t = time.where_(&in_bounds, &last)?; // [B,1] i64
        // Joint over a window of W frames against the FIXED predictor state:
        // build `[safe_t, safe_t+W)`, clamp each row into a legal gather index
        // (off-window / finished rows are masked out via `off_valid` below).
        let win_t = safe_t.try_add(&arange_w)?; // [B,W]
        let win_clamp = win_t.minimum(&last)?; // [B,W]
        let idx = win_clamp.try_reshape([b, w, 1])?.try_expand([b, w, j])?;
        let enc_window = enc_proj.gather(1, &idx)?; // [B,W,J]

        let (g, new_h, new_c) = scoped("head", || scoped("predictor", || head.predictor.forward_parts(&prev, &h, &c)))?;
        // `argmax_preproj` broadcasts the `[B,1,J]` predictor projection over the
        // window axis, so the same call serves W=1 and W>1.
        let toks = scoped("head", || scoped("joint", || head.joint.argmax_preproj(&enc_window, &g)))?; // [B,W] i32
        let tok64 = toks.cast(DType::Int64);

        let is_blank = tok64.try_eq(blank)?; // [B,W]
        let not_blank = is_blank.logical_not()?;
        // A window offset is usable only if it maps to a real (in-bounds) frame.
        let off_valid = time.try_add(&arange_w)?.try_lt(&valid64)?; // [B,W]
        let usable_i32 = off_valid.try_bitand(&not_blank)?.cast(DType::Int32); // [B,W] {0,1}
        let any_nb = usable_i32.max_with().axes(1isize).keepdim(true).call()?; // [B,1]
        let emit = any_nb.try_ge(1i32)?; // [B,1] bool
        // argmax ties resolve to the first index → first usable (non-blank) offset
        // (0 when none, which `emit == false` masks out).
        let first_nb = usable_i32.argmax_with().axis(Some(1isize)).keepdim(true).call()?; // [B,1] i32
        let first_nb64 = first_nb.cast(DType::Int64);
        let tok_sel = tok64.gather(1, &first_nb64)?; // [B,1] i64 — token at the first non-blank

        // Commit the selected token + predictor state on emitting lanes.
        prev = tok_sel.where_(&emit, &prev)?;
        // Commit state for emitting lanes: [B,1,L*P] → [L,B,P], masked.
        let emit_lbp = emit.try_reshape([1, b, 1])?.try_expand([l, b, p])?;
        let to_lbp = |s: Tensor| -> Result<Tensor> {
            Ok(s.try_reshape([b, l, p])?.try_permute(&[1, 0, 2])?.try_reshape([l, b, p])?)
        };
        h = to_lbp(new_h)?.where_(&emit_lbp, &h)?;
        c = to_lbp(new_c)?.where_(&emit_lbp, &c)?;

        // Same-frame run length: a window jump lands on a fresh frame, so reset
        // the counter before this emission; the cap then forces a single-frame
        // advance exactly as the per-frame loop does.
        let jumped = emit.try_bitand(&first_nb64.try_ge(1i64)?)?;
        let sym_base = jumped.select(0i32, &symbols)?;
        let symbols1 = sym_base.try_add(1i32)?.where_(&emit, &sym_base)?;
        let cap = symbols1.try_ge(max_symbols as i32)?;
        // Advance time: jump past the leading blanks (emit at `first_nb`) or the
        // whole in-bounds window (all blank), clamped so time never overshoots
        // valid; the cap adds the extra single-frame step after a capped emit.
        let rem = valid64.try_sub(&time)?;
        let blank_run = rem.clamp().min(0i64).max(W as i64).call()?;
        let blank_skip = first_nb64.where_(&emit, &blank_run)?;
        let cap_adv = emit.try_bitand(&cap)?.cast(DType::Int64);
        time = time.try_add(&blank_skip)?.try_add(&cap_adv)?;
        let adv_frame = in_bounds.cast(DType::Bool).try_bitand(&emit.logical_not()?)?.try_bitor(&cap)?;
        symbols = adv_frame.select(0i32, &symbols1)?;

        // Emission frame = safe_t + first_nb on emit (the jumped frame); the
        // value is filtered out by `emit == 0` otherwise.
        let frame_tape = safe_t.try_add(&emit.select(&first_nb64, 0i64)?)?;
        tapes.push(tok_sel.cast(DType::Int32));
        emits.push(emit.cast(DType::Int32));
        frames.push(frame_tape.cast(DType::Int32));
    }

    let cat = |v: &[Tensor]| -> Result<Tensor> { Ok(Tensor::cat(&v.iter().collect::<Vec<_>>(), 1)?) }; // [B,K]
    let active = time.try_lt(&valid64)?;
    let active_any = active.cast(DType::Int32).sum_with().axes(0isize).keepdim(true).call()?.try_reshape([1, 1])?;

    // The carried state is returned as-is: the `state { .. }` slots store each
    // value back into the input buffer it was read from (read-before-write is
    // safe — the inputs are read only at step 0), so `execute()` recycles state
    // in place and the host never copies output→input between blocks.
    Ok((cat(&tapes)?, cat(&emits)?, cat(&frames)?, active_any, time, prev, symbols, h, c))
}
