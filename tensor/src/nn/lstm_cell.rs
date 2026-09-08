use svod_dtype::DType;

use crate::Tensor;

type Result<T> = crate::Result<T>;

/// LSTM cell with PyTorch's `[i, f, g, o]` gate order.
///
/// `weight_ih` shape: `[4*hidden, input]`; `weight_hh` shape: `[4*hidden, hidden]`.
/// `bias_ih` and `bias_hh` both `[4*hidden]` — summed in [`Self::step`] to match
/// `nn.LSTM`'s packing, so PyTorch checkpoints load without remapping.
///
/// Not a [`Layer`](crate::nn::Layer) — cells take `(x, h, c)`, not a single tensor.
#[derive(Clone)]
pub struct LSTMCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Tensor,
    pub bias_hh: Tensor,
    hidden_size: usize,
}

impl LSTMCell {
    /// Create an LSTM cell from existing weight/bias tensors. `hidden_size` is
    /// derived from `weight_ih.shape()[0] / 4`.
    pub fn new(weight_ih: Tensor, weight_hh: Tensor, bias_ih: Tensor, bias_hh: Tensor) -> Self {
        let shape = weight_ih.shape().expect("lstm_cell: weight_ih shape");
        let four_hidden = shape[0].as_const().expect("lstm_cell: 4*hidden must be concrete");
        Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size: four_hidden / 4 }
    }

    /// Create an LSTM cell with deterministic `sin()` initialization, zero biases.
    #[track_caller]
    pub fn with_dims(input_size: usize, hidden_size: usize, dtype: DType) -> Self {
        origin_call!("LSTMCell::with_dims");
        let four_hidden = 4 * hidden_size;
        let w_ih_data: Vec<f32> = (0..four_hidden * input_size).map(|i| ((i as f32) * 0.1).sin() * 0.1).collect();
        let weight_ih = Tensor::from_slice(&w_ih_data)
            .try_reshape([four_hidden as isize, input_size as isize])
            .expect("lstm_cell weight_ih reshape failed");
        let w_hh_data: Vec<f32> = (0..four_hidden * hidden_size).map(|i| ((i as f32) * 0.1).sin() * 0.1).collect();
        let weight_hh = Tensor::from_slice(&w_hh_data)
            .try_reshape([four_hidden as isize, hidden_size as isize])
            .expect("lstm_cell weight_hh reshape failed");
        let bias_ih = Tensor::full(&[four_hidden], 0.0, dtype.clone());
        let bias_hh = Tensor::full(&[four_hidden], 0.0, dtype);
        Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// One LSTM step. Returns `(h_next, c_next)`.
    ///
    /// Shapes: `x: [B, input]`, `h, c: [B, hidden]`.
    #[track_caller]
    pub fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        origin_call!("LSTMCell::step");
        let gates_x = x.linear().weight(&self.weight_ih).bias(&self.bias_ih).call()?;
        let gates_h = h.linear().weight(&self.weight_hh).bias(&self.bias_hh).call()?;
        let gates = gates_x.try_add(&gates_h)?;

        let h_sz = self.hidden_size;
        let parts = gates.split(&[h_sz, h_sz, h_sz, h_sz], 1)?;
        let i = parts[0].sigmoid()?;
        let f = parts[1].sigmoid()?;
        let g = parts[2].tanh()?;
        let o = parts[3].sigmoid()?;

        let new_c = f.try_mul(c)?.try_add(&i.try_mul(&g)?)?;
        let new_h = o.try_mul(&new_c.tanh()?)?;
        Ok((new_h, new_c))
    }
}
