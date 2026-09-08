//! RNN-T predictor: token embedding + multi-layer LSTM. Stateful per-utterance:
//! the search loop carries `(h, c)` across calls and resets to zeros at the
//! start of a new utterance.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::LSTMCell;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed, scoped, scoped_index};

use crate::gigaam::Result;

/// RNN-T predictor: token embedding + multi-layer LSTM.
///
/// The empty-prefix predictor call (Python `predict(None, None, batch_size)`)
/// is realized by passing `prev_token = blank_id` with zero `(h, c)`. PyTorch's
/// `nn.Embedding(padding_idx=blank_id)` keeps the blank row at zero through
/// training, so this is equivalent to "embedding of zero vector". We assert
/// the row is in fact zero at load time.
///
/// The LSTM stack reuses [`LSTMCell`] from `svod_tensor::nn`, which applies
/// PyTorch's `[i, f, g, o]` gate order — matching the reference exactly so
/// checkpoints load without gate-axis remapping.
#[derive(Clone)]
pub struct RnntPredictor {
    /// `[num_classes, pred_hidden]`. Row `blank_id` must be zeros.
    pub embed: Tensor,
    pub layers: Vec<LSTMCell>,
    pub pred_hidden: usize,
    pub num_classes: usize,
    pub blank_id: usize,
}

impl RnntPredictor {
    pub fn empty(pred_hidden: usize, num_layers: usize, num_classes: usize) -> Self {
        let blank_id = num_classes - 1;
        let h4 = 4 * pred_hidden;
        Self {
            embed: fan_in_uniform(&[num_classes, pred_hidden], num_classes, DType::Float32),
            layers: (0..num_layers)
                .map(|_| {
                    LSTMCell::new(
                        fan_in_uniform(&[h4, pred_hidden], pred_hidden, DType::Float32),
                        fan_in_uniform(&[h4, pred_hidden], pred_hidden, DType::Float32),
                        fan_in_uniform(&[h4], pred_hidden, DType::Float32),
                        fan_in_uniform(&[h4], pred_hidden, DType::Float32),
                    )
                })
                .collect(),
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
        let l = self.layers.len() as isize;
        let b = h_in.dim_const(1)? as isize;

        // Embed lookup: prev_token [B, 1] -> emb [B, 1, P].
        // Squeeze the seq-len axis to feed the LSTM cell shape [B, P].
        let emb = scoped("embed", || -> Result<Tensor> { Ok(self.embed.embedding(prev_token)?) })?;
        let mut layer_in = emb.try_squeeze(Some(1))?; // [B, P]

        let mut new_hs: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        let mut new_cs: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        for (i, cell) in self.layers.iter().enumerate() {
            let i_i = i as isize;
            // Slice layer i's h, c → [1, B, P], squeeze leading axis → [B, P].
            let h_i = h_in.try_shrink([(i_i, i_i + 1), (0, b), (0, p)])?.try_squeeze(Some(0))?;
            let c_i = c_in.try_shrink([(i_i, i_i + 1), (0, b), (0, p)])?.try_squeeze(Some(0))?;
            let (new_h, new_c) =
                scoped_index("lstm", i, || -> Result<(Tensor, Tensor)> { Ok(cell.step(&layer_in, &h_i, &c_i)?) })?;
            new_hs.push(new_h.clone());
            new_cs.push(new_c.clone());
            layer_in = new_h;
        }

        // g = last layer output [B, P] → [B, 1, P].
        let g = layer_in.try_unsqueeze(1)?;

        // Stack per-layer h, c → [L, B, P]; batch-major → [B, 1, L * P].
        let new_h_stacked = Tensor::stack(&new_hs.iter().collect::<Vec<_>>(), 0)?;
        let new_c_stacked = Tensor::stack(&new_cs.iter().collect::<Vec<_>>(), 0)?;
        let new_h_flat = new_h_stacked.try_permute(&[1, 0, 2])?.try_reshape([b, 1, l * p])?;
        let new_c_flat = new_c_stacked.try_permute(&[1, 0, 2])?.try_reshape([b, 1, l * p])?;

        Ok((g, new_h_flat, new_c_flat))
    }

    /// Zero the blank-id embedding row in place — matches Python's
    /// `predict(None, None, batch_size)` empty-prefix path without a
    /// separate fresh-step JIT. Load-bearing for checkpoints like
    /// `v3_e2e_rnnt` whose fine-tuned blank row is non-zero.
    pub(crate) fn prepare_for_inference(&mut self) -> Result<()> {
        let mut mask_data = vec![1.0_f32; self.num_classes];
        mask_data[self.blank_id] = 0.0;
        let embed_dtype = self.embed.dtype();
        let mask = Tensor::from_slice(&mask_data).try_reshape([self.num_classes, 1])?.cast(embed_dtype)?;
        self.embed = self.embed.try_mul(&mask)?;
        self.embed.realize()?;
        Ok(())
    }
}

impl HasStateDict for RnntPredictor {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "embed"), self.embed.clone());
        for (i, cell) in self.layers.iter().enumerate() {
            let p = prefixed(prefix, &format!("lstm.{i}"));
            sd.insert(prefixed(&p, "w_ih"), cell.weight_ih.clone());
            sd.insert(prefixed(&p, "w_hh"), cell.weight_hh.clone());
            sd.insert(prefixed(&p, "b_ih"), cell.bias_ih.clone());
            sd.insert(prefixed(&p, "b_hh"), cell.bias_hh.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.embed = get_tensor(sd, &prefixed(prefix, "embed"))?;
        for (i, cell) in self.layers.iter_mut().enumerate() {
            let p = prefixed(prefix, &format!("lstm.{i}"));
            cell.weight_ih = get_tensor(sd, &prefixed(&p, "w_ih"))?;
            cell.weight_hh = get_tensor(sd, &prefixed(&p, "w_hh"))?;
            cell.bias_ih = get_tensor(sd, &prefixed(&p, "b_ih"))?;
            cell.bias_hh = get_tensor(sd, &prefixed(&p, "b_hh"))?;
        }
        Ok(())
    }
}
