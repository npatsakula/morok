//! Recurrent layers: RNN (Elman), GRU and LSTM.
//!
//! One core drives everything: a [`RecurrentCell`] hoists its input projection
//! out of the time loop (`x @ W_ih^T + b_ih` for the whole sequence in a single
//! matmul) and then contributes one graph per step. [`Tensor::rnn`],
//! [`Tensor::gru`] and [`Tensor::lstm`] unroll that loop over a *concrete* `T`;
//! the batch extent may stay symbolic. A symbolic `T` is out of scope for this
//! phase — the IR can express a runtime-trip `RANGE`, but the tensor scheduler
//! materializes step boundaries at `prepare()` time, so `T` must be a constant
//! (see the recurrence spike: no `Op::Scan`, no data-dependent trip count).
//!
//! # Two weight spellings
//!
//! * **ONNX**: `w` `[D, G*H, I]`, `r` `[D, G*H, H]`, `bias` `[D, 2*G*H]`
//!   (`Wb ++ Rb`), GRU gate order `z, r, h`, LSTM gate order `i, o, f, c`.
//! * **PyTorch**: `weight_ih` `[D*G*H, I]` (or `[D, G*H, I]`), `weight_hh`
//!   `[D*G*H, H]`, `bias_ih`/`bias_hh` `[D*G*H]`, GRU gate order `r, z, n`,
//!   LSTM gate order `i, f, g, o`.
//!
//! Exactly one spelling must be supplied. The PyTorch one accepts both the
//! 2-D concatenation (`cat([weight_ih_l0, weight_ih_l0_reverse], 0)`) and the
//! 3-D stack (`stack([...], 0)`) so a state dict loads with one call and no
//! axis remapping in either style.

use bon::bon;
use snafu::OptionExt;
use strum::{Display, EnumString};
use svod_dtype::DType;
use svod_ir::SInt;

use crate::error::{ExclusiveParamsSnafu, NdimExactSnafu, NonConstDimSnafu, ParamRangeSnafu};

use super::*;

// =========================================================================
// Configuration enums
// =========================================================================

/// Which axis of a recurrent layer's input comes first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum RnnLayout {
    /// `[seq, batch, input]` — ONNX `layout=0`, PyTorch `batch_first=False`.
    #[default]
    #[strum(serialize = "seq_first")]
    SeqFirst,
    /// `[batch, seq, input]` — ONNX `layout=1`, PyTorch `batch_first=True`.
    #[strum(serialize = "batch_first")]
    BatchFirst,
}

impl From<usize> for RnnLayout {
    fn from(layout: usize) -> Self {
        if layout == 0 { Self::SeqFirst } else { Self::BatchFirst }
    }
}

/// Time directions a recurrent layer runs in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, EnumString, Display)]
pub enum RnnDirection {
    #[default]
    #[strum(serialize = "forward")]
    Forward,
    /// Runs over the time-reversed sequence; outputs stay in input time order.
    #[strum(serialize = "reverse", serialize = "backward")]
    Backward,
    /// Forward and backward, concatenated on the feature axis.
    #[strum(serialize = "bidirectional")]
    Bidirectional,
}

/// [`RnnDirection`] under the name the GRU call sites use.
pub type GruDirection = RnnDirection;

impl RnnDirection {
    /// `num_directions` in ONNX terms: 2 for [`Self::Bidirectional`], else 1.
    pub fn num_directions(self) -> usize {
        match self {
            Self::Bidirectional => 2,
            _ => 1,
        }
    }

    /// Whether direction slot `d` walks the sequence backwards.
    fn is_reverse(self, d: usize) -> bool {
        matches!(self, Self::Backward) || (matches!(self, Self::Bidirectional) && d == 1)
    }
}

/// GRU reset placement. `true` is PyTorch's `nn.GRU`
/// (`n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`), `false` is the ONNX
/// default (`n = tanh(W_in x + b_in + W_hn (r * h) + b_hn)`).
///
/// A newtype rather than a bare `bool` so both `true` and ONNX's `0`/`1`
/// attribute integer are accepted by the same builder setter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearBeforeReset(pub bool);

impl From<bool> for LinearBeforeReset {
    fn from(v: bool) -> Self {
        Self(v)
    }
}

impl From<usize> for LinearBeforeReset {
    fn from(v: usize) -> Self {
        Self(v != 0)
    }
}

// =========================================================================
// Outputs
// =========================================================================

/// Output of an RNN forward pass.
pub struct RnnOutput {
    /// ONNX shape: `[seq, num_directions, batch, hidden]` (`SeqFirst`) or
    /// `[batch, seq, num_directions, hidden]` (`BatchFirst`).
    pub y: Tensor,
    /// ONNX shape: `[num_directions, batch, hidden]` (`SeqFirst`) or
    /// `[batch, num_directions, hidden]` (`BatchFirst`).
    pub y_h: Tensor,
    /// PyTorch shape: `[seq, batch, D*hidden]` or `[batch, seq, D*hidden]`.
    pub output: Tensor,
    /// PyTorch shape: `[num_directions, batch, hidden]`, both layouts.
    pub h_n: Tensor,
}

/// Output of a GRU forward pass. Fields as in [`RnnOutput`].
pub struct GruOutput {
    pub y: Tensor,
    pub y_h: Tensor,
    pub output: Tensor,
    pub h_n: Tensor,
}

/// Output of an LSTM forward pass. Fields as in [`RnnOutput`], plus the cell state.
pub struct LstmOutput {
    pub y: Tensor,
    pub y_h: Tensor,
    /// ONNX-shaped final cell state, laid out like [`Self::y_h`].
    pub y_c: Tensor,
    pub output: Tensor,
    pub h_n: Tensor,
    /// PyTorch shape: `[num_directions, batch, hidden]`, both layouts.
    pub c_n: Tensor,
}

// =========================================================================
// Cells
// =========================================================================

/// One time step of a recurrent layer over an owned state.
///
/// The input projection is separated from the recurrence so a sequence runner
/// can hoist it: `project_input` sees `[T, B, I]` once, `step_projected` sees
/// `[B, G*H]` per step.
pub trait RecurrentCell {
    /// State carried across steps: `Tensor` for RNN/GRU, `(h, c)` for LSTM.
    type State: Clone;

    fn hidden_size(&self) -> usize;

    /// `x @ W_ih^T + b_ih` over any leading axes: `[.., I] -> [.., G*H]`.
    fn project_input(&self, x: &Tensor) -> Result<Tensor>;

    /// One step from an already-projected row `[B, G*H]`, returning
    /// `(output, next_state)`.
    fn step_projected(&self, gx: &Tensor, state: &Self::State) -> Result<(Tensor, Self::State)>;

    /// One step from a raw input row `[B, I]`.
    fn step_state(&self, x: &Tensor, state: &Self::State) -> Result<(Tensor, Self::State)> {
        self.step_projected(&self.project_input(x)?, state)
    }
}

/// Elman RNN cell: `h' = tanh(W_ih x + b_ih + W_hh h + b_hh)`.
#[derive(Clone)]
pub struct RnnCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
    hidden_size: usize,
}

impl RnnCell {
    #[track_caller]
    pub fn new(weight_ih: Tensor, weight_hh: Tensor, bias_ih: Option<Tensor>, bias_hh: Option<Tensor>) -> Result<Self> {
        let hidden_size = weight_hh.dim_const(-1)?;
        Ok(Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// One step: `x [B, I]`, `h [B, H]` → `h' [B, H]`.
    #[track_caller]
    pub fn step(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        origin_call!("RnnCell::step");
        Ok(self.step_state(x, &h.clone())?.1)
    }
}

impl RecurrentCell for RnnCell {
    type State = Tensor;

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn project_input(&self, x: &Tensor) -> Result<Tensor> {
        x.linear().weight(&self.weight_ih).maybe_bias(self.bias_ih.as_ref()).call()
    }

    fn step_projected(&self, gx: &Tensor, h: &Self::State) -> Result<(Tensor, Self::State)> {
        let gh = h.linear().weight(&self.weight_hh).maybe_bias(self.bias_hh.as_ref()).call()?;
        let next = gx.try_add(&gh)?.tanh()?;
        Ok((next.clone(), next))
    }
}

/// GRU cell in PyTorch's `[r, z, n]` gate order.
///
/// `weight_ih` is `[3H, I]`, `weight_hh` is `[3H, H]`, biases are `[3H]`.
/// `linear_before_reset` selects between `nn.GRU` (`true`, the default) and the
/// ONNX default formulation (`false`).
#[derive(Clone)]
pub struct GruCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
    hidden_size: usize,
    linear_before_reset: bool,
    /// `weight_hh` rows for `r, z` — `[2H, H]`.
    w_hh_rz: Tensor,
    /// `weight_hh` rows for `n` — `[H, H]`.
    w_hh_n: Tensor,
    b_hh_rz: Option<Tensor>,
    b_hh_n: Option<Tensor>,
}

impl GruCell {
    /// Build a cell from PyTorch-ordered weights. Uses `nn.GRU` semantics
    /// (`linear_before_reset = true`).
    #[track_caller]
    pub fn new(weight_ih: Tensor, weight_hh: Tensor, bias_ih: Option<Tensor>, bias_hh: Option<Tensor>) -> Result<Self> {
        Self::with_reset_mode(weight_ih, weight_hh, bias_ih, bias_hh, true)
    }

    /// As [`Self::new`], choosing the reset placement explicitly.
    #[track_caller]
    pub fn with_reset_mode(
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Option<Tensor>,
        bias_hh: Option<Tensor>,
        linear_before_reset: bool,
    ) -> Result<Self> {
        let hidden_size = weight_hh.dim_const(-1)?;
        let w_hh_rz = weight_hh.narrow(0, 0usize, 2 * hidden_size)?;
        let w_hh_n = weight_hh.narrow(0, 2 * hidden_size, hidden_size)?;
        let b_hh_rz = bias_hh.as_ref().map(|b| b.narrow(0, 0usize, 2 * hidden_size)).transpose()?;
        let b_hh_n = bias_hh.as_ref().map(|b| b.narrow(0, 2 * hidden_size, hidden_size)).transpose()?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
            linear_before_reset,
            w_hh_rz,
            w_hh_n,
            b_hh_rz,
            b_hh_n,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// One step: `x [B, I]`, `h [B, H]` → `h' [B, H]`.
    #[track_caller]
    pub fn step(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        origin_call!("GruCell::step");
        Ok(self.step_state(x, &h.clone())?.1)
    }
}

impl RecurrentCell for GruCell {
    type State = Tensor;

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn project_input(&self, x: &Tensor) -> Result<Tensor> {
        x.linear().weight(&self.weight_ih).maybe_bias(self.bias_ih.as_ref()).call()
    }

    fn step_projected(&self, gx: &Tensor, h: &Self::State) -> Result<(Tensor, Self::State)> {
        let hs = self.hidden_size;
        let gh_rz = h.linear().weight(&self.w_hh_rz).maybe_bias(self.b_hh_rz.as_ref()).call()?;
        let r = gx.narrow(-1, 0usize, hs)?.try_add(&gh_rz.narrow(-1, 0usize, hs)?)?.sigmoid()?;
        let z = gx.narrow(-1, hs, hs)?.try_add(&gh_rz.narrow(-1, hs, hs)?)?.sigmoid()?;
        let gx_n = gx.narrow(-1, 2 * hs, hs)?;

        let n = if self.linear_before_reset {
            let gh_n = h.linear().weight(&self.w_hh_n).maybe_bias(self.b_hh_n.as_ref()).call()?;
            gx_n.try_add(&r.try_mul(&gh_n)?)?.tanh()?
        } else {
            let gh_n = r.try_mul(h)?.linear().weight(&self.w_hh_n).maybe_bias(self.b_hh_n.as_ref()).call()?;
            gx_n.try_add(&gh_n)?.tanh()?
        };

        // (1 - z) * n + z * h, written to reuse `n` once.
        let next = n.try_add(&z.try_mul(&h.try_sub(&n)?)?)?;
        Ok((next.clone(), next))
    }
}

/// A stack of recurrent cells applied layer-by-layer within one time step.
///
/// This is the RNN-T predictor pattern: each layer consumes the layer below's
/// new hidden state and carries its own state across calls.
#[derive(Clone)]
pub struct RnnStack<C> {
    pub cells: Vec<C>,
}

impl<C: RecurrentCell> RnnStack<C> {
    pub fn new(cells: Vec<C>) -> Self {
        Self { cells }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// One step through every layer. `states` must have one entry per cell;
    /// returns the top layer's output and the new per-layer states.
    #[track_caller]
    pub fn step(&self, x: &Tensor, states: &[C::State]) -> Result<(Tensor, Vec<C::State>)> {
        origin_call!("RnnStack::step");
        snafu::ensure!(
            states.len() == self.cells.len(),
            ParamRangeSnafu {
                op: "RnnStack::step",
                param: "states",
                value: states.len().to_string(),
                constraint: "one state per cell"
            }
        );
        let mut layer_in = x.clone();
        let mut next = Vec::with_capacity(self.cells.len());
        for (cell, state) in self.cells.iter().zip(states) {
            let (y, s) = cell.step_state(&layer_in, state)?;
            layer_in = y;
            next.push(s);
        }
        Ok((layer_in, next))
    }
}

// =========================================================================
// Sequence runner
// =========================================================================

/// Run one direction over a pre-projected sequence `gx [T, B, G*H]`.
///
/// Every step builds a structurally identical graph, differing only in the
/// constant time offset of its input slice.
fn run_direction<C: RecurrentCell>(
    cell: &C,
    gx: &Tensor,
    t_len: usize,
    init: C::State,
    reverse: bool,
) -> Result<(Tensor, C::State)> {
    let mut state = init;
    let mut outs = Vec::with_capacity(t_len);
    for i in 0..t_len {
        let t = if reverse { t_len - 1 - i } else { i };
        let gx_t = gx.narrow(0, t, 1usize)?.try_squeeze(Some(0))?;
        let (y, next) = cell.step_projected(&gx_t, &state)?;
        state = next;
        outs.push(y);
    }
    if reverse {
        outs.reverse();
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    Ok((Tensor::stack(&refs, 0)?, state))
}

/// Drive `cells` (one per direction) over `x [T, B, I]`, returning
/// `(y_onnx, output_torch, final_states)`.
fn run_sequence<C: RecurrentCell>(
    op: &'static str,
    x: &Tensor,
    out_layout: RnnLayout,
    direction: RnnDirection,
    cells: &[C],
    init: &[C::State],
) -> Result<(Tensor, Tensor, Vec<C::State>)> {
    let shape = x.shape()?;
    let t_len = shape[0].as_const().context(NonConstDimSnafu { axis: 0_isize, dim: shape[0].clone() })?;
    snafu::ensure!(t_len > 0, ParamRangeSnafu { op, param: "seq_length", value: t_len.to_string(), constraint: "> 0" });

    let mut seqs = Vec::with_capacity(cells.len());
    let mut finals = Vec::with_capacity(cells.len());
    for (d, cell) in cells.iter().enumerate() {
        let gx = cell.project_input(x)?;
        let (seq, state) = run_direction(cell, &gx, t_len, init[d].clone(), direction.is_reverse(d))?;
        seqs.push(seq);
        finals.push(state);
    }

    let refs: Vec<&Tensor> = seqs.iter().collect();
    let y = Tensor::stack(&refs, 1)?; // [T, D, B, H]
    let output = if refs.len() == 1 { seqs[0].clone() } else { Tensor::cat(&refs, -1)? }; // [T, B, D*H]

    Ok(match out_layout {
        RnnLayout::SeqFirst => (y, output, finals),
        RnnLayout::BatchFirst => (y.try_permute(&[2, 0, 1, 3])?, output.try_permute(&[1, 0, 2])?, finals),
    })
}

/// Stack per-direction `[B, H]` states into `[D, B, H]`, plus the ONNX view.
fn pack_states(states: &[Tensor], layout: RnnLayout) -> Result<(Tensor, Tensor)> {
    let refs: Vec<&Tensor> = states.iter().collect();
    let n = Tensor::stack(&refs, 0)?; // [D, B, H]
    let onnx = match layout {
        RnnLayout::SeqFirst => n.clone(),
        RnnLayout::BatchFirst => n.try_permute(&[1, 0, 2])?,
    };
    Ok((onnx, n))
}

// =========================================================================
// Weight resolution
// =========================================================================

/// Per-direction weights, always in PyTorch gate order.
struct RnnWeights {
    w_ih: Vec<Tensor>,
    w_hh: Vec<Tensor>,
    b_ih: Vec<Option<Tensor>>,
    b_hh: Vec<Option<Tensor>>,
    hidden: usize,
    dirs: usize,
    /// Whether the PyTorch spelling was used (selects the GRU reset default).
    torch: bool,
}

/// Reorder gate blocks along dim 0: `order[j]` is the source block for slot `j`.
fn reorder_gates(t: &Tensor, block: usize, order: &[usize]) -> Result<Tensor> {
    if order.iter().enumerate().all(|(j, &i)| j == i) {
        return Ok(t.clone());
    }
    let parts = order.iter().map(|&i| t.narrow(0, i * block, block)).collect::<Result<Vec<_>>>()?;
    Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)
}

/// Split a `[D, ...]`-normalized tensor into its per-direction slices.
fn per_direction(t: &Tensor, dirs: usize) -> Result<Vec<Tensor>> {
    (0..dirs).map(|d| t.narrow(0, d, 1usize)?.try_squeeze(Some(0))).collect()
}

#[allow(clippy::too_many_arguments)]
fn resolve_weights(
    op: &'static str,
    gates: usize,
    order: &[usize],
    hidden_size: Option<usize>,
    onnx: (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>),
    torch: (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>, Option<&Tensor>),
) -> Result<RnnWeights> {
    let (w, r, bias) = onnx;
    let (weight_ih, weight_hh, bias_ih, bias_hh) = torch;
    let use_onnx = w.is_some() && r.is_some();
    let use_torch = weight_ih.is_some() && weight_hh.is_some();
    snafu::ensure!(use_onnx != use_torch, ExclusiveParamsSnafu { op, options: "(w, r) or (weight_ih, weight_hh)" });

    if use_onnx {
        let (w, r) = (w.expect("checked"), r.expect("checked"));
        let dirs = w.dim_const(0)?;
        let hidden = match hidden_size {
            Some(h) => h,
            None => r.dim_const(-1)?,
        };
        let block = gates * hidden;
        let w_ih = per_direction(w, dirs)?.iter().map(|t| reorder_gates(t, hidden, order)).collect::<Result<_>>()?;
        let w_hh = per_direction(r, dirs)?.iter().map(|t| reorder_gates(t, hidden, order)).collect::<Result<_>>()?;
        let (b_ih, b_hh) = match bias {
            Some(b) => {
                let mut ih = Vec::with_capacity(dirs);
                let mut hh = Vec::with_capacity(dirs);
                for bd in per_direction(b, dirs)? {
                    ih.push(Some(reorder_gates(&bd.narrow(0, 0usize, block)?, hidden, order)?));
                    hh.push(Some(reorder_gates(&bd.narrow(0, block, block)?, hidden, order)?));
                }
                (ih, hh)
            }
            None => (vec![None; dirs], vec![None; dirs]),
        };
        return Ok(RnnWeights { w_ih, w_hh, b_ih, b_hh, hidden, dirs, torch: false });
    }

    let (weight_ih, weight_hh) = (weight_ih.expect("checked"), weight_hh.expect("checked"));
    let hidden = match hidden_size {
        Some(h) => h,
        None => weight_hh.dim_const(-1)?,
    };
    let block = gates * hidden;
    let dirs = if weight_ih.ndim()? == 3 { weight_ih.dim_const(0)? } else { weight_ih.dim_const(0)? / block };
    snafu::ensure!(
        dirs > 0,
        ParamRangeSnafu { op, param: "weight_ih", value: dirs.to_string(), constraint: "at least one direction" }
    );
    let norm2 = |t: &Tensor| -> Result<Tensor> {
        if t.ndim()? == 3 { Ok(t.clone()) } else { t.try_reshape([dirs as isize, block as isize, -1]) }
    };
    let norm1 = |t: &Tensor| -> Result<Tensor> {
        if t.ndim()? == 2 { Ok(t.clone()) } else { t.try_reshape([dirs as isize, block as isize]) }
    };
    let w_ih = per_direction(&norm2(weight_ih)?, dirs)?;
    let w_hh = per_direction(&norm2(weight_hh)?, dirs)?;
    let split_bias = |b: Option<&Tensor>| -> Result<Vec<Option<Tensor>>> {
        match b {
            Some(b) => Ok(per_direction(&norm1(b)?, dirs)?.into_iter().map(Some).collect()),
            None => Ok(vec![None; dirs]),
        }
    };
    Ok(RnnWeights { w_ih, w_hh, b_ih: split_bias(bias_ih)?, b_hh: split_bias(bias_hh)?, hidden, dirs, torch: true })
}

/// Reconcile an explicit `direction` with the direction count the weights carry.
fn resolve_direction(op: &'static str, direction: Option<RnnDirection>, dirs: usize) -> Result<RnnDirection> {
    let dir = direction.unwrap_or(if dirs == 2 { RnnDirection::Bidirectional } else { RnnDirection::Forward });
    snafu::ensure!(
        dir.num_directions() == dirs,
        ParamRangeSnafu {
            op,
            param: "direction",
            value: dir.to_string(),
            constraint: "num_directions must match the weight tensors"
        }
    );
    Ok(dir)
}

/// `[T, B, I]` view of the input plus its batch extent.
fn seq_first(op: &'static str, x: &Tensor, layout: RnnLayout) -> Result<(Tensor, SInt)> {
    let ndim = x.ndim()?;
    snafu::ensure!(ndim == 3, NdimExactSnafu { op, expected: 3_usize, actual: ndim });
    let x = match layout {
        RnnLayout::SeqFirst => x.clone(),
        RnnLayout::BatchFirst => x.try_permute(&[1, 0, 2])?,
    };
    let batch = x.shape()?[1].clone();
    Ok((x, batch))
}

/// Per-direction initial states, sliced out of a `[D, B, H]` tensor or zeroed.
fn initial_states(
    dirs: usize,
    hidden: usize,
    init: Option<&Tensor>,
    batch: &SInt,
    dtype: DType,
) -> Result<Vec<Tensor>> {
    match init {
        Some(h) => per_direction(h, dirs),
        None => {
            (0..dirs).map(|_| Tensor::zeros_dynamic(&[batch.clone(), SInt::Const(hidden)], dtype.clone())).collect()
        }
    }
}

#[bon]
impl Tensor {
    /// Simple RNN (Elman network): `h' = tanh(W_ih x + b_ih + W_hh h + b_hh)`.
    ///
    /// Supply either the ONNX weights (`w`, `r`, `bias`) or the PyTorch ones
    /// (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`); both use a single gate
    /// so no reordering happens either way. `initial_h` (alias `h0`) is
    /// `[num_directions, batch, hidden]` in both layouts.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::Array3;
    /// // seq=2, batch=1, input=3
    /// let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
    /// let w = Tensor::from_ndarray(&Array3::from_elem((1, 4, 3), 0.1f32)); // [1, hidden=4, input=3]
    /// let r = Tensor::from_ndarray(&Array3::from_elem((1, 4, 4), 0.1f32)); // [1, hidden=4, hidden=4]
    /// let out = x.rnn().w(&w).r(&r).hidden_size(4).call().unwrap();
    /// let y_shape: Vec<usize> = out.y.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(y_shape, vec![2, 1, 1, 4]); // [seq, num_directions, batch, hidden]
    /// let yh_shape: Vec<usize> = out.y_h.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(yh_shape, vec![1, 1, 4]); // [num_directions, batch, hidden]
    /// ```
    #[builder]
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn rnn(
        &self,
        w: Option<&Tensor>,
        r: Option<&Tensor>,
        bias: Option<&Tensor>,
        weight_ih: Option<&Tensor>,
        weight_hh: Option<&Tensor>,
        bias_ih: Option<&Tensor>,
        bias_hh: Option<&Tensor>,
        hidden_size: Option<usize>,
        initial_h: Option<&Tensor>,
        h0: Option<&Tensor>,
        #[builder(into)] direction: Option<RnnDirection>,
        #[builder(default, into)] layout: RnnLayout,
    ) -> Result<RnnOutput> {
        origin_call!("rnn");
        let weights =
            resolve_weights("rnn", 1, &[0], hidden_size, (w, r, bias), (weight_ih, weight_hh, bias_ih, bias_hh))?;
        let direction = resolve_direction("rnn", direction, weights.dirs)?;
        let cells = (0..weights.dirs)
            .map(|d| {
                RnnCell::new(
                    weights.w_ih[d].clone(),
                    weights.w_hh[d].clone(),
                    weights.b_ih[d].clone(),
                    weights.b_hh[d].clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let (x, batch) = seq_first("rnn", self, layout)?;
        let init = initial_states(weights.dirs, weights.hidden, h0.or(initial_h), &batch, self.dtype())?;
        let (y, output, finals) = run_sequence("rnn", &x, layout, direction, &cells, &init)?;
        let (y_h, h_n) = pack_states(&finals, layout)?;
        Ok(RnnOutput { y, y_h, output, h_n })
    }

    /// GRU (Gated Recurrent Unit).
    ///
    /// PyTorch gate order is `r, z, n`; ONNX's is `z, r, h`, mapped onto it by
    /// swapping the first two blocks. With ONNX weights `linear_before_reset`
    /// defaults to `false` (the ONNX default); with PyTorch weights it defaults
    /// to `true`, which is `nn.GRU`:
    /// `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`.
    ///
    /// `initial_h` (alias `h0`) is `[num_directions, batch, hidden]` in both
    /// layouts.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::Array3;
    /// // seq=2, batch=1, input=3, hidden=4
    /// let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
    /// let w = Tensor::from_ndarray(&Array3::from_elem((1, 12, 3), 0.1f32));
    /// let r = Tensor::from_ndarray(&Array3::from_elem((1, 12, 4), 0.1f32));
    /// let out = x.gru().w(&w).r_weights(&r).hidden_size(4).call().unwrap();
    /// let y_shape: Vec<usize> = out.y.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(y_shape, vec![2, 1, 1, 4]); // [seq, num_directions, batch, hidden]
    /// let out_shape: Vec<usize> = out.output.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(out_shape, vec![2, 1, 4]); // [seq, batch, D*hidden]
    /// ```
    #[builder]
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn gru(
        &self,
        w: Option<&Tensor>,
        r_weights: Option<&Tensor>,
        bias: Option<&Tensor>,
        weight_ih: Option<&Tensor>,
        weight_hh: Option<&Tensor>,
        bias_ih: Option<&Tensor>,
        bias_hh: Option<&Tensor>,
        hidden_size: Option<usize>,
        initial_h: Option<&Tensor>,
        h0: Option<&Tensor>,
        #[builder(into)] linear_before_reset: Option<LinearBeforeReset>,
        #[builder(into)] direction: Option<GruDirection>,
        #[builder(default, into)] layout: RnnLayout,
    ) -> Result<GruOutput> {
        origin_call!("gru");
        // ONNX GRU gates are z, r, h; PyTorch's are r, z, n.
        let weights = resolve_weights(
            "gru",
            3,
            &[1, 0, 2],
            hidden_size,
            (w, r_weights, bias),
            (weight_ih, weight_hh, bias_ih, bias_hh),
        )?;
        let direction = resolve_direction("gru", direction, weights.dirs)?;
        let lbr = linear_before_reset.map_or(weights.torch, |l| l.0);
        let cells = (0..weights.dirs)
            .map(|d| {
                GruCell::with_reset_mode(
                    weights.w_ih[d].clone(),
                    weights.w_hh[d].clone(),
                    weights.b_ih[d].clone(),
                    weights.b_hh[d].clone(),
                    lbr,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let (x, batch) = seq_first("gru", self, layout)?;
        let init = initial_states(weights.dirs, weights.hidden, h0.or(initial_h), &batch, self.dtype())?;
        let (y, output, finals) = run_sequence("gru", &x, layout, direction, &cells, &init)?;
        let (y_h, h_n) = pack_states(&finals, layout)?;
        Ok(GruOutput { y, y_h, output, h_n })
    }

    /// LSTM (Long Short-Term Memory).
    ///
    /// PyTorch gate order is `i, f, g, o`; ONNX's is `i, o, f, c`, mapped onto
    /// it block-wise. `peepholes` is the ONNX extension `[D, 3H]` (`p_i, p_o,
    /// p_f`); `nn.LSTM` has none. `initial_h`/`initial_c` (aliases `h0`/`c0`)
    /// are `[num_directions, batch, hidden]` in both layouts.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svod_tensor::Tensor;
    /// # use ndarray::Array3;
    /// // seq=2, batch=1, input=3, hidden=4
    /// let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
    /// let w = Tensor::from_ndarray(&Array3::from_elem((1, 16, 3), 0.1f32));
    /// let r = Tensor::from_ndarray(&Array3::from_elem((1, 16, 4), 0.1f32));
    /// let out = x.lstm().w(&w).r(&r).hidden_size(4).call().unwrap();
    /// let y_shape: Vec<usize> = out.y.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(y_shape, vec![2, 1, 1, 4]); // [seq, num_directions, batch, hidden]
    /// let yc_shape: Vec<usize> = out.y_c.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(yc_shape, vec![1, 1, 4]); // [num_directions, batch, hidden]
    /// ```
    #[builder]
    #[track_caller]
    #[allow(clippy::too_many_arguments)]
    pub fn lstm(
        &self,
        w: Option<&Tensor>,
        r: Option<&Tensor>,
        bias: Option<&Tensor>,
        weight_ih: Option<&Tensor>,
        weight_hh: Option<&Tensor>,
        bias_ih: Option<&Tensor>,
        bias_hh: Option<&Tensor>,
        hidden_size: Option<usize>,
        initial_h: Option<&Tensor>,
        initial_c: Option<&Tensor>,
        h0: Option<&Tensor>,
        c0: Option<&Tensor>,
        peepholes: Option<&Tensor>,
        #[builder(into)] direction: Option<RnnDirection>,
        #[builder(default, into)] layout: RnnLayout,
    ) -> Result<LstmOutput> {
        origin_call!("lstm");
        // ONNX LSTM gates are i, o, f, c; PyTorch's are i, f, g, o.
        let weights = resolve_weights(
            "lstm",
            4,
            &[0, 2, 3, 1],
            hidden_size,
            (w, r, bias),
            (weight_ih, weight_hh, bias_ih, bias_hh),
        )?;
        let direction = resolve_direction("lstm", direction, weights.dirs)?;
        let hs = weights.hidden;
        let dtype = self.dtype();
        // ONNX peepholes are p_i, p_o, p_f; the cell wants (p_i, p_f, p_o).
        let peep = match peepholes {
            Some(p) => per_direction(p, weights.dirs)?
                .iter()
                .map(|pd| Ok(Some((pd.narrow(0, 0usize, hs)?, pd.narrow(0, 2 * hs, hs)?, pd.narrow(0, hs, hs)?))))
                .collect::<Result<Vec<_>>>()?,
            None => vec![None; weights.dirs],
        };
        let cells = (0..weights.dirs)
            .map(|d| {
                let zero = || Tensor::full(&[4 * hs], 0.0f32, dtype.clone());
                let cell = LstmCell::new(
                    weights.w_ih[d].clone(),
                    weights.w_hh[d].clone(),
                    weights.b_ih[d].clone().unwrap_or_else(zero),
                    weights.b_hh[d].clone().unwrap_or_else(zero),
                );
                match peep[d].clone() {
                    Some(p) => cell.with_peepholes(p.0, p.1, p.2),
                    None => cell,
                }
            })
            .collect::<Vec<_>>();

        let (x, batch) = seq_first("lstm", self, layout)?;
        let h_init = initial_states(weights.dirs, hs, h0.or(initial_h), &batch, dtype.clone())?;
        let c_init = initial_states(weights.dirs, hs, c0.or(initial_c), &batch, dtype)?;
        let init: Vec<(Tensor, Tensor)> = h_init.into_iter().zip(c_init).collect();
        let (y, output, finals) = run_sequence("lstm", &x, layout, direction, &cells, &init)?;
        let hs_final: Vec<Tensor> = finals.iter().map(|(h, _)| h.clone()).collect();
        let cs_final: Vec<Tensor> = finals.iter().map(|(_, c)| c.clone()).collect();
        let (y_h, h_n) = pack_states(&hs_final, layout)?;
        let (y_c, c_n) = pack_states(&cs_final, layout)?;
        Ok(LstmOutput { y, y_h, y_c, output, h_n, c_n })
    }
}
