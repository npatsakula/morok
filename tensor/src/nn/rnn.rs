//! Recurrent neural network layers (RNN, GRU, LSTM).

use bon::bon;

use crate::error::{NdimExactSnafu, ParamRangeSnafu};

use super::*;

/// Output of an RNN forward pass.
pub struct RnnOutput {
    /// All hidden states: `[seq_length, num_directions, batch, hidden_size]`
    pub y: Tensor,
    /// Final hidden state: `[num_directions, batch, hidden_size]`
    pub y_h: Tensor,
}

/// Output of a GRU forward pass.
pub struct GruOutput {
    /// All hidden states: `[seq_length, num_directions, batch, hidden_size]`
    pub y: Tensor,
    /// Final hidden state: `[num_directions, batch, hidden_size]`
    pub y_h: Tensor,
}

/// Output of an LSTM forward pass.
pub struct LstmOutput {
    /// All hidden states: `[seq_length, num_directions, batch, hidden_size]`
    pub y: Tensor,
    /// Final hidden state: `[num_directions, batch, hidden_size]`
    pub y_h: Tensor,
    /// Final cell state: `[num_directions, batch, hidden_size]`
    pub y_c: Tensor,
}

#[bon]
impl Tensor {
    /// Simple RNN (Elman network).
    ///
    /// `H_t = tanh(X_t @ W^T + H_{t-1} @ R^T + Wb + Rb)`
    ///
    /// - `x`: input `[seq_length, batch_size, input_size]` (layout=0) or
    ///         `[batch_size, seq_length, input_size]` (layout=1)
    /// - `w`: input weights `[num_directions, hidden_size, input_size]`
    /// - `r`: recurrence weights `[num_directions, hidden_size, hidden_size]`
    /// - `bias`: optional bias `[num_directions, 2 * hidden_size]` (Wb ++ Rb)
    /// - `initial_h`: optional initial hidden state `[num_directions, batch_size, hidden_size]`
    /// - `layout`: 0 = seq-first (default), 1 = batch-first
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::{array, Array3};
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
    pub fn rnn(
        &self,
        w: &Tensor,
        r: &Tensor,
        hidden_size: usize,
        bias: Option<&Tensor>,
        initial_h: Option<&Tensor>,
        #[builder(default = 0)] layout: usize,
    ) -> Result<RnnOutput> {
        let ndim = self.ndim()?;
        snafu::ensure!(ndim == 3, NdimExactSnafu { op: "rnn", expected: 3_usize, actual: ndim });
        snafu::ensure!(
            hidden_size > 0,
            ParamRangeSnafu { op: "rnn", param: "hidden_size", value: hidden_size.to_string(), constraint: "> 0" }
        );
        let x = if layout != 0 { self.try_permute(&[1, 0, 2])? } else { self.clone() };
        let x_shape = x.shape()?;
        let seq_length = x_shape[0].as_const().expect("static seq_length");
        let batch_size = x_shape[1].as_const().expect("static batch_size");
        let input_size = x_shape[2].as_const().expect("static input_size");
        let num_directions = w.shape()?[0].as_const().expect("static num_directions");
        let dtype = x.uop().dtype();

        snafu::ensure!(
            num_directions == 1,
            ParamRangeSnafu {
                op: "rnn",
                param: "num_directions",
                value: num_directions.to_string(),
                constraint: "== 1"
            }
        );

        let w0 = w.try_squeeze(Some(0))?; // [hidden, input]
        let r0 = r.try_squeeze(Some(0))?; // [hidden, hidden]
        let wt = w0.try_permute(&[1, 0])?; // [input, hidden]
        let rt = r0.try_permute(&[1, 0])?; // [hidden, hidden]

        let combined_bias = if let Some(b) = bias {
            let b0 = b.try_squeeze(Some(0))?; // [2*hidden]
            let parts = b0.split(&[hidden_size, hidden_size], 0)?;
            Some(parts[0].try_add(&parts[1])?) // [hidden]
        } else {
            None
        };

        let mut h_t = if let Some(h0) = initial_h {
            h0.try_squeeze(Some(0))? // [batch, hidden]
        } else {
            Tensor::full(&[batch_size, hidden_size], 0.0f32, dtype)?
        };

        let mut h_list = Vec::with_capacity(seq_length);
        for t in 0..seq_length {
            let x_t =
                x.try_shrink([(t as isize, t as isize + 1), (0, batch_size as isize), (0, input_size as isize)])?;
            let x_t = x_t.try_squeeze(Some(0))?; // [batch, input]

            let mut gate = x_t.matmul(&wt)?.try_add(&h_t.matmul(&rt)?)?;
            if let Some(ref b) = combined_bias {
                gate = gate.try_add(b)?;
            }
            h_t = gate.tanh()?;
            h_list.push(h_t.clone());
        }

        let h_refs: Vec<&Tensor> = h_list.iter().collect();
        let y_seq = Tensor::stack(&h_refs, 0)?; // [seq, batch, hidden]
        let y = y_seq.try_unsqueeze(1)?; // [seq, 1, batch, hidden]

        let y = if layout != 0 {
            y.try_permute(&[2, 0, 1, 3])? // [batch, seq, 1, hidden]
        } else {
            y
        };

        let y_h = if layout != 0 {
            h_t.try_unsqueeze(1)? // [batch, 1, hidden]
        } else {
            h_t.try_unsqueeze(0)? // [1, batch, hidden]
        };

        Ok(RnnOutput { y, y_h })
    }

    /// GRU (Gated Recurrent Unit).
    ///
    /// Gate order: `[z, r, h]` (update, reset, hidden).
    ///
    /// Equations (default, `linear_before_reset=0`):
    /// - `z = sigmoid(X @ W_z^T + H @ R_z^T + w_bz + r_bz)`
    /// - `r = sigmoid(X @ W_r^T + H @ R_r^T + w_br + r_br)`
    /// - `h = tanh(X @ W_h^T + (r * H) @ R_h^T + w_bh + r_bh)`
    /// - `H_new = (1 - z) * h + z * H_prev`
    ///
    /// When `linear_before_reset=1`:
    /// - `h = tanh(X @ W_h^T + r * (H @ R_h^T + r_bh) + w_bh)`
    ///
    /// - `x`: input `[seq_length, batch_size, input_size]` (layout=0) or
    ///         `[batch_size, seq_length, input_size]` (layout=1)
    /// - `w`: input weights `[num_directions, 3*hidden_size, input_size]`
    /// - `r_weights`: recurrence weights `[num_directions, 3*hidden_size, hidden_size]`
    /// - `bias`: optional `[num_directions, 6*hidden_size]` (Wb ++ Rb)
    /// - `initial_h`: optional `[num_directions, batch_size, hidden_size]`
    /// - `linear_before_reset`: 0 (default) or 1
    /// - `layout`: 0 = seq-first (default), 1 = batch-first
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::{array, Array3};
    /// // seq=2, batch=1, input=3, hidden=4
    /// let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
    /// // GRU: w is [num_directions, 3*hidden_size, input_size]
    /// let w = Tensor::from_ndarray(&Array3::from_elem((1, 12, 3), 0.1f32));
    /// // GRU: r is [num_directions, 3*hidden_size, hidden_size]
    /// let r = Tensor::from_ndarray(&Array3::from_elem((1, 12, 4), 0.1f32));
    /// let out = x.gru().w(&w).r_weights(&r).hidden_size(4).call().unwrap();
    /// let y_shape: Vec<usize> = out.y.shape().unwrap().iter()
    ///     .map(|d| d.as_const().unwrap()).collect();
    /// assert_eq!(y_shape, vec![2, 1, 1, 4]); // [seq, num_directions, batch, hidden]
    /// ```
    #[builder]
    pub fn gru(
        &self,
        w: &Tensor,
        r_weights: &Tensor,
        hidden_size: usize,
        bias: Option<&Tensor>,
        initial_h: Option<&Tensor>,
        #[builder(default = 0)] linear_before_reset: usize,
        #[builder(default = 0)] layout: usize,
    ) -> Result<GruOutput> {
        let ndim = self.ndim()?;
        snafu::ensure!(ndim == 3, NdimExactSnafu { op: "gru", expected: 3_usize, actual: ndim });
        snafu::ensure!(
            hidden_size > 0,
            ParamRangeSnafu { op: "gru", param: "hidden_size", value: hidden_size.to_string(), constraint: "> 0" }
        );
        let x = if layout != 0 { self.try_permute(&[1, 0, 2])? } else { self.clone() };
        let x_shape = x.shape()?;
        let seq_length = x_shape[0].as_const().expect("static seq_length");
        let batch_size = x_shape[1].as_const().expect("static batch_size");
        let input_size = x_shape[2].as_const().expect("static input_size");
        let num_directions = w.shape()?[0].as_const().expect("static num_directions");
        let dtype = x.uop().dtype();

        snafu::ensure!(
            num_directions == 1,
            ParamRangeSnafu {
                op: "gru",
                param: "num_directions",
                value: num_directions.to_string(),
                constraint: "== 1"
            }
        );

        let w0 = w.try_squeeze(Some(0))?; // [3*hidden, input]
        let r0 = r_weights.try_squeeze(Some(0))?; // [3*hidden, hidden]

        // Split W into [W_z, W_r, W_h] and R into [R_z, R_r, R_h]
        let w_parts = w0.split(&[hidden_size; 3], 0)?;
        let r_parts = r0.split(&[hidden_size; 3], 0)?;

        // Combine z,r weights for joint computation: gates_w = [W_z; W_r]^T
        let gates_w = Tensor::cat(&[&w_parts[0], &w_parts[1]], 0)?.try_permute(&[1, 0])?;
        let gates_r = Tensor::cat(&[&r_parts[0], &r_parts[1]], 0)?.try_permute(&[1, 0])?;

        // W_h and R_h kept separate (reset gate interacts differently)
        let w_h_t = w_parts[2].try_permute(&[1, 0])?; // [input, hidden]
        let r_h_t = r_parts[2].try_permute(&[1, 0])?; // [hidden, hidden]

        // Bias: [6*hidden] → [w_bz, w_br, w_bh, r_bz, r_br, r_bh]
        let (gates_b, w_bh, r_bh) = if let Some(b) = bias {
            let b0 = b.try_squeeze(Some(0))?;
            let parts = b0.split(&[hidden_size; 6], 0)?;
            // gates_b = (w_bz + r_bz) ++ (w_br + r_br)
            let gates_b = Tensor::cat(&[&parts[0].try_add(&parts[3])?, &parts[1].try_add(&parts[4])?], 0)?;
            (Some(gates_b), Some(parts[2].clone()), Some(parts[5].clone()))
        } else {
            (None, None, None)
        };

        let mut h_t = if let Some(h0) = initial_h {
            h0.try_squeeze(Some(0))?
        } else {
            Tensor::full(&[batch_size, hidden_size], 0.0f32, dtype)?
        };

        let mut h_list = Vec::with_capacity(seq_length);
        for t in 0..seq_length {
            let x_t =
                x.try_shrink([(t as isize, t as isize + 1), (0, batch_size as isize), (0, input_size as isize)])?;
            let x_t = x_t.try_squeeze(Some(0))?; // [batch, input]

            // z, r gates: combined matmul
            let mut gates = x_t.matmul(&gates_w)?.try_add(&h_t.matmul(&gates_r)?)?;
            if let Some(ref gb) = gates_b {
                gates = gates.try_add(gb)?;
            }
            let zr = gates.split(&[hidden_size; 2], -1)?;
            let z = zr[0].sigmoid()?;
            let r = zr[1].sigmoid()?;

            // Hidden candidate
            let h_candidate = if linear_before_reset != 0 {
                // h = tanh(x @ W_h^T + r * (H @ R_h^T + r_bh) + w_bh)
                let mut rh = h_t.matmul(&r_h_t)?;
                if let Some(ref rb) = r_bh {
                    rh = rh.try_add(rb)?;
                }
                let mut h = x_t.matmul(&w_h_t)?.try_add(&r.try_mul(&rh)?)?;
                if let Some(ref wb) = w_bh {
                    h = h.try_add(wb)?;
                }
                h.tanh()?
            } else {
                // h = tanh(x @ W_h^T + (r * H) @ R_h^T + w_bh + r_bh)
                let mut h = x_t.matmul(&w_h_t)?.try_add(&r.try_mul(&h_t)?.matmul(&r_h_t)?)?;
                if let Some(ref wb) = w_bh {
                    h = h.try_add(wb)?;
                }
                if let Some(ref rb) = r_bh {
                    h = h.try_add(rb)?;
                }
                h.tanh()?
            };

            // H = (1 - z) * h_candidate + z * H_prev
            let one = Tensor::full(&[1], 1.0f32, z.uop().dtype())?;
            h_t = one.try_sub(&z)?.try_mul(&h_candidate)?.try_add(&z.try_mul(&h_t)?)?;
            h_list.push(h_t.clone());
        }

        let h_refs: Vec<&Tensor> = h_list.iter().collect();
        let y_seq = Tensor::stack(&h_refs, 0)?; // [seq, batch, hidden]
        let y = y_seq.try_unsqueeze(1)?; // [seq, 1, batch, hidden]

        let y = if layout != 0 {
            y.try_permute(&[2, 0, 1, 3])? // [batch, seq, 1, hidden]
        } else {
            y
        };

        let y_h = if layout != 0 {
            h_t.try_unsqueeze(1)? // [batch, 1, hidden]
        } else {
            h_t.try_unsqueeze(0)? // [1, batch, hidden]
        };

        Ok(GruOutput { y, y_h })
    }

    /// LSTM (Long Short-Term Memory).
    ///
    /// Gate order: `[i, o, f, c]` (input, output, forget, cell).
    ///
    /// - `x`: input `[seq_length, batch_size, input_size]` (layout=0) or
    ///         `[batch_size, seq_length, input_size]` (layout=1)
    /// - `w`: input weights `[num_directions, 4*hidden_size, input_size]`
    /// - `r`: recurrence weights `[num_directions, 4*hidden_size, hidden_size]`
    /// - `bias`: optional `[num_directions, 8*hidden_size]` (Wb ++ Rb)
    /// - `initial_h`: optional `[num_directions, batch_size, hidden_size]`
    /// - `initial_c`: optional `[num_directions, batch_size, hidden_size]`
    /// - `peepholes`: optional `[num_directions, 3*hidden_size]` (p_i, p_o, p_f)
    /// - `layout`: 0 = seq-first (default), 1 = batch-first
    ///
    /// # Examples
    ///
    /// ```
    /// # use morok_tensor::Tensor;
    /// # use ndarray::Array3;
    /// // seq=2, batch=1, input=3, hidden=4
    /// let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
    /// // LSTM: w is [num_directions, 4*hidden_size, input_size]
    /// let w = Tensor::from_ndarray(&Array3::from_elem((1, 16, 3), 0.1f32));
    /// // LSTM: r is [num_directions, 4*hidden_size, hidden_size]
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
    pub fn lstm(
        &self,
        w: &Tensor,
        r: &Tensor,
        hidden_size: usize,
        bias: Option<&Tensor>,
        initial_h: Option<&Tensor>,
        initial_c: Option<&Tensor>,
        peepholes: Option<&Tensor>,
        #[builder(default = 0)] layout: usize,
    ) -> Result<LstmOutput> {
        let ndim = self.ndim()?;
        snafu::ensure!(ndim == 3, NdimExactSnafu { op: "lstm", expected: 3_usize, actual: ndim });
        snafu::ensure!(
            hidden_size > 0,
            ParamRangeSnafu { op: "lstm", param: "hidden_size", value: hidden_size.to_string(), constraint: "> 0" }
        );
        let x = if layout != 0 {
            self.try_permute(&[1, 0, 2])? // batch-first → seq-first
        } else {
            self.clone()
        };
        let x_shape = x.shape()?;
        let seq_length = x_shape[0].as_const().expect("static seq_length");
        let batch_size = x_shape[1].as_const().expect("static batch_size");
        let input_size = x_shape[2].as_const().expect("static input_size");
        let num_directions = w.shape()?[0].as_const().expect("static num_directions");
        let dtype = x.uop().dtype();

        snafu::ensure!(
            num_directions == 1,
            ParamRangeSnafu {
                op: "lstm",
                param: "num_directions",
                value: num_directions.to_string(),
                constraint: "== 1"
            }
        );

        let w0 = w.try_squeeze(Some(0))?; // [4*hidden, input]
        let r0 = r.try_squeeze(Some(0))?; // [4*hidden, hidden]
        let wt = w0.try_permute(&[1, 0])?; // [input, 4*hidden]
        let rt = r0.try_permute(&[1, 0])?; // [hidden, 4*hidden]

        // Bias: [8*hidden] → split into Wb [4*hidden] and Rb [4*hidden], add together
        let combined_bias = if let Some(b) = bias {
            let b0 = b.try_squeeze(Some(0))?;
            let hs4 = 4 * hidden_size;
            let parts = b0.split(&[hs4, hs4], 0)?;
            Some(parts[0].try_add(&parts[1])?)
        } else {
            None
        };

        // Peepholes: [3*hidden] → [p_i, p_o, p_f]
        let (p_i, p_o, p_f) = if let Some(p) = peepholes {
            let p0 = p.try_squeeze(Some(0))?;
            let parts = p0.split(&[hidden_size, hidden_size, hidden_size], 0)?;
            (Some(parts[0].clone()), Some(parts[1].clone()), Some(parts[2].clone()))
        } else {
            (None, None, None)
        };

        let mut h_t = if let Some(h0) = initial_h {
            h0.try_squeeze(Some(0))?
        } else {
            Tensor::full(&[batch_size, hidden_size], 0.0f32, dtype.clone())?
        };
        let mut c_t = if let Some(c0) = initial_c {
            c0.try_squeeze(Some(0))?
        } else {
            Tensor::full(&[batch_size, hidden_size], 0.0f32, dtype)?
        };

        let mut h_list = Vec::with_capacity(seq_length);
        for t in 0..seq_length {
            let x_t =
                x.try_shrink([(t as isize, t as isize + 1), (0, batch_size as isize), (0, input_size as isize)])?;
            let x_t = x_t.try_squeeze(Some(0))?; // [batch, input]

            // gates = X_t @ W^T + H_{t-1} @ R^T + bias
            let mut gates = x_t.matmul(&wt)?.try_add(&h_t.matmul(&rt)?)?;
            if let Some(ref b) = combined_bias {
                gates = gates.try_add(b)?;
            }

            // Split into [i, o, f, c] — each [batch, hidden]
            let gate_parts = gates.split(&[hidden_size; 4], -1)?;
            let (mut gi, mut go, mut gf, gc) =
                (gate_parts[0].clone(), gate_parts[1].clone(), gate_parts[2].clone(), gate_parts[3].clone());

            // Peephole connections: i and f use previous cell state
            if let Some(ref pi) = p_i {
                gi = gi.try_add(&c_t.try_mul(pi)?)?;
            }
            if let Some(ref pf) = p_f {
                gf = gf.try_add(&c_t.try_mul(pf)?)?;
            }

            let i = gi.sigmoid()?;
            let f = gf.sigmoid()?;
            let c = gc.tanh()?;

            // C = f * C_prev + i * c
            c_t = f.try_mul(&c_t)?.try_add(&i.try_mul(&c)?)?;

            // Peephole: o uses NEW cell state
            if let Some(ref po) = p_o {
                go = go.try_add(&c_t.try_mul(po)?)?;
            }
            let o = go.sigmoid()?;

            // H = o * tanh(C)
            h_t = o.try_mul(&c_t.tanh()?)?;
            h_list.push(h_t.clone());
        }

        let h_refs: Vec<&Tensor> = h_list.iter().collect();
        let y_seq = Tensor::stack(&h_refs, 0)?; // [seq, batch, hidden]
        let y = y_seq.try_unsqueeze(1)?; // [seq, 1, batch, hidden]

        // Apply layout transform to output
        let y = if layout != 0 {
            y.try_permute(&[2, 0, 1, 3])? // [batch, seq, 1, hidden]
        } else {
            y
        };

        let (y_h, y_c) = if layout != 0 {
            // layout=1: Y_h/Y_c are [batch, num_directions, hidden]
            (h_t.try_unsqueeze(1)?, c_t.try_unsqueeze(1)?)
        } else {
            // layout=0: Y_h/Y_c are [num_directions, batch, hidden]
            (h_t.try_unsqueeze(0)?, c_t.try_unsqueeze(0)?)
        };

        Ok(LstmOutput { y, y_h, y_c })
    }
}
