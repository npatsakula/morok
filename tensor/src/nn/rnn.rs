use super::*;

/// Output of an RNN forward pass.
pub struct RnnOutput {
    /// All hidden states: `[seq_length, num_directions, batch, hidden_size]`
    pub y: Tensor,
    /// Final hidden state: `[num_directions, batch, hidden_size]`
    pub y_h: Tensor,
}

impl Tensor {
    /// Simple RNN (Elman network).
    ///
    /// `H_t = tanh(X_t @ W^T + H_{t-1} @ R^T + Wb + Rb)`
    ///
    /// - `x`: input sequence `[seq_length, batch_size, input_size]`
    /// - `w`: input weights `[num_directions, hidden_size, input_size]`
    /// - `r`: recurrence weights `[num_directions, hidden_size, hidden_size]`
    /// - `bias`: optional bias `[num_directions, 2 * hidden_size]` (Wb ++ Rb)
    /// - `initial_h`: optional initial hidden state `[num_directions, batch_size, hidden_size]`
    pub fn rnn(
        &self,
        w: &Tensor,
        r: &Tensor,
        bias: Option<&Tensor>,
        initial_h: Option<&Tensor>,
        hidden_size: usize,
    ) -> Result<RnnOutput> {
        let x = self;
        let x_shape = x.shape()?;
        let seq_length = x_shape[0].as_const().expect("static seq_length");
        let batch_size = x_shape[1].as_const().expect("static batch_size");
        let num_directions = w.shape()?[0].as_const().expect("static num_directions");
        let dtype = x.uop().dtype();

        assert_eq!(num_directions, 1, "RNN: only forward direction supported");

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

        let input_size = x_shape[2].as_const().expect("static input_size");
        let mut h_list = Vec::with_capacity(seq_length);
        for t in 0..seq_length {
            let x_t =
                x.try_shrink(&[(t as isize, t as isize + 1), (0, batch_size as isize), (0, input_size as isize)])?;
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
        let y_h = h_t.try_unsqueeze(0)?; // [1, batch, hidden]

        Ok(RnnOutput { y, y_h })
    }
}
