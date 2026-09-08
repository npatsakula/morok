//! RNN-T predictor: token embedding + multi-layer LSTM. Stateful per-utterance:
//! the search loop carries `(h, c)` across calls and resets to zeros at the
//! start of a new utterance.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{LstmCell, Module, RnnStack, StateDict, get_tensor, prefixed};

use crate::init::fan_in_uniform;
use crate::state::scoped;

use crate::gigaam::Result;

/// RNN-T predictor: token embedding + multi-layer LSTM.
///
/// The empty-prefix predictor call (Python `predict(None, None, batch_size)`)
/// is realized by passing `prev_token = blank_id` with zero `(h, c)`. PyTorch's
/// `nn.Embedding(padding_idx=blank_id)` keeps the blank row at zero through
/// training, so this is equivalent to "embedding of zero vector". We zero the
/// row at load time.
///
/// The LSTM stack reuses [`RnnStack<LstmCell>`] from `svod_tensor::nn`, which
/// applies PyTorch's `[i, f, g, o]` gate order — matching the reference exactly
/// so checkpoints load without gate-axis remapping.
#[derive(Clone)]
pub struct RnntPredictor {
    /// `[num_classes, pred_hidden]`. Row `blank_id` must be zeros.
    pub embed: Tensor,
    pub lstm: RnnStack<LstmCell>,
    pub pred_hidden: usize,
    pub num_classes: usize,
    pub blank_id: usize,
}

impl RnntPredictor {
    pub fn empty(pred_hidden: usize, num_layers: usize, num_classes: usize) -> Self {
        let blank_id = num_classes - 1;
        let h4 = 4 * pred_hidden;
        let gate = |shape: &[usize]| fan_in_uniform(shape, pred_hidden, DType::Float32);
        Self {
            embed: fan_in_uniform(&[num_classes, pred_hidden], num_classes, DType::Float32),
            lstm: RnnStack::new(
                (0..num_layers)
                    .map(|_| {
                        LstmCell::new(gate(&[h4, pred_hidden]), gate(&[h4, pred_hidden]), gate(&[h4]), gate(&[h4]))
                    })
                    .collect(),
            ),
            pred_hidden,
            num_classes,
            blank_id,
        }
    }

    /// Run one batched predictor step. `prev_token [B, 1]`, `h_in`/`c_in
    /// [L, B, P]` → `(g [B, 1, P], new_h_flat [B, 1, L*P], new_c_flat
    /// [B, 1, L*P])`, batch taken from the state shape. Kept as separate
    /// tensors so the fused step JIT can feed `g` straight into the joint
    /// on-device and expose the state as its own output.
    pub fn forward_parts(&self, prev_token: &Tensor, h_in: &Tensor, c_in: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let p = self.pred_hidden as isize;
        let l = self.lstm.len() as isize;
        let b = h_in.dim_const(1)? as isize;

        // Embed lookup: prev_token [B, 1] -> emb [B, 1, P].
        // Squeeze the seq-len axis to feed the LSTM cell shape [B, P].
        let emb = scoped("embed", || -> Result<Tensor> { Ok(self.embed.embedding(prev_token)?) })?;
        let layer_in = emb.try_squeeze(Some(1))?; // [B, P]

        let (top, new_h, new_c) =
            scoped("lstm", || -> Result<_> { Ok(self.lstm.step_stacked(&layer_in, h_in, c_in)?) })?;

        // g = last layer output [B, P] → [B, 1, P]; state [L, B, P] → batch-major
        // [B, 1, L * P].
        let flat = |s: Tensor| -> Result<Tensor> { Ok(s.try_permute(&[1, 0, 2])?.try_reshape([b, 1, l * p])?) };
        Ok((top.try_unsqueeze(1)?, flat(new_h)?, flat(new_c)?))
    }

    /// Zero the blank-id embedding row — matches Python's
    /// `predict(None, None, batch_size)` empty-prefix path without a
    /// separate fresh-step JIT. Load-bearing for checkpoints like
    /// `v3_e2e_rnnt` whose fine-tuned blank row is non-zero.
    pub(crate) fn prepare_for_inference(&mut self) -> Result<()> {
        let blank = Tensor::arange(self.num_classes as i64, None, None)?
            .try_eq(self.blank_id as i64)?
            .try_reshape([self.num_classes as isize, 1])?;
        self.embed = self.embed.masked_fill(&blank, 0.0)?;
        self.embed.realize()?;
        Ok(())
    }
}

/// Hand-written: the checkpoint keys the cells `lstm.{i}.{w_ih,w_hh,b_ih,b_hh}`,
/// while [`LstmCell`]'s own fields are PyTorch's `weight_ih`/`bias_hh`/… — and
/// the cell carries no `Module` impl to rename them from.
impl Module for RnntPredictor {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        out.insert(prefixed(prefix, "embed"), self.embed.clone());
        for (i, cell) in self.lstm.cells.iter().enumerate() {
            for (name, tensor) in cell_params(cell) {
                out.insert(prefixed(prefix, &format!("lstm.{i}.{name}")), tensor.clone());
            }
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> svod_tensor::error::Result<()> {
        self.embed = get_tensor(sd, &prefixed(prefix, "embed"))?;
        for (i, cell) in self.lstm.cells.iter_mut().enumerate() {
            let mut slot = [&mut cell.weight_ih, &mut cell.weight_hh, &mut cell.bias_ih, &mut cell.bias_hh];
            for (field, name) in slot.iter_mut().zip(PARAM_NAMES) {
                **field = get_tensor(sd, &prefixed(prefix, &format!("lstm.{i}.{name}")))?;
            }
        }
        Ok(())
    }
}

const PARAM_NAMES: [&str; 4] = ["w_ih", "w_hh", "b_ih", "b_hh"];

fn cell_params(cell: &LstmCell) -> [(&'static str, &Tensor); 4] {
    [
        (PARAM_NAMES[0], &cell.weight_ih),
        (PARAM_NAMES[1], &cell.weight_hh),
        (PARAM_NAMES[2], &cell.bias_ih),
        (PARAM_NAMES[3], &cell.bias_hh),
    ]
}
