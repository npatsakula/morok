use svod_dtype::DType;

use crate::Tensor;
use crate::nn::rnn::RecurrentCell;

type Result<T> = crate::Result<T>;

/// LSTM cell with PyTorch's `[i, f, g, o]` gate order.
///
/// `weight_ih` shape: `[4*hidden, input]`; `weight_hh` shape: `[4*hidden, hidden]`.
/// `bias_ih` and `bias_hh` both `[4*hidden]` — summed in [`Self::step`] to match
/// `nn.LSTM`'s packing, so PyTorch checkpoints load without remapping.
///
/// Optionally carries ONNX peephole vectors, which `nn.LSTM` never has.
///
/// Not a [`Layer`](crate::nn::Layer) — cells take `(x, h, c)`, not a single tensor.
#[derive(Clone)]
pub struct LstmCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Tensor,
    pub bias_hh: Tensor,
    hidden_size: usize,
    /// ONNX peepholes in cell order `(p_i, p_f, p_o)`, each `[hidden]`.
    peepholes: Option<(Tensor, Tensor, Tensor)>,
}

/// The pre-`RnnStack` spelling of [`LstmCell`].
pub type LSTMCell = LstmCell;

impl LstmCell {
    /// Create an LSTM cell from existing weight/bias tensors. `hidden_size` is
    /// derived from `weight_ih.shape()[0] / 4`.
    pub fn new(weight_ih: Tensor, weight_hh: Tensor, bias_ih: Tensor, bias_hh: Tensor) -> Self {
        let shape = weight_ih.shape().expect("lstm_cell: weight_ih shape");
        let four_hidden = shape[0].as_const().expect("lstm_cell: 4*hidden must be concrete");
        Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size: four_hidden / 4, peepholes: None }
    }

    /// Attach ONNX peephole vectors `(p_i, p_f, p_o)`, each `[hidden]`.
    pub fn with_peepholes(mut self, p_i: Tensor, p_f: Tensor, p_o: Tensor) -> Self {
        self.peepholes = Some((p_i, p_f, p_o));
        self
    }

    /// Create an LSTM cell with deterministic `sin()` initialization, zero biases.
    #[track_caller]
    pub fn with_dims(input_size: usize, hidden_size: usize, dtype: DType) -> Self {
        origin_call!("LstmCell::with_dims");
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
        Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size, peepholes: None }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// One LSTM step. Returns `(h_next, c_next)`.
    ///
    /// Shapes: `x: [B, input]`, `h, c: [B, hidden]`.
    #[track_caller]
    pub fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        origin_call!("LstmCell::step");
        Ok(self.step_state(x, &(h.clone(), c.clone()))?.1)
    }
}

impl RecurrentCell for LstmCell {
    type State = (Tensor, Tensor);

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn project_input(&self, x: &Tensor) -> Result<Tensor> {
        x.linear().weight(&self.weight_ih).bias(&self.bias_ih).call()
    }

    fn step_projected(&self, gx: &Tensor, (h, c): &Self::State) -> Result<(Tensor, Self::State)> {
        let gates = gx.try_add(&h.linear().weight(&self.weight_hh).bias(&self.bias_hh).call()?)?;

        let hs = self.hidden_size;
        let parts = gates.split(&[hs, hs, hs, hs], -1)?;
        let (mut gi, mut gf, gg, mut go) = (parts[0].clone(), parts[1].clone(), parts[2].clone(), parts[3].clone());
        if let Some((p_i, p_f, _)) = &self.peepholes {
            gi = gi.try_add(&c.try_mul(p_i)?)?;
            gf = gf.try_add(&c.try_mul(p_f)?)?;
        }

        let i = gi.sigmoid()?;
        let f = gf.sigmoid()?;
        let g = gg.tanh()?;
        let new_c = f.try_mul(c)?.try_add(&i.try_mul(&g)?)?;

        if let Some((_, _, p_o)) = &self.peepholes {
            go = go.try_add(&new_c.try_mul(p_o)?)?;
        }
        let new_h = go.sigmoid()?.try_mul(&new_c.tanh()?)?;
        Ok((new_h.clone(), (new_h, new_c)))
    }
}

impl crate::nn::rnn::RnnStack<LstmCell> {
    /// One step of a stacked LSTM whose state is packed as `[L, B, H]`.
    ///
    /// `x` is `[B, input]`; returns `(y [B, H], h [L, B, H], c [L, B, H])`.
    #[track_caller]
    pub fn step_stacked(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        origin_call!("RnnStack::step_stacked");
        let states = (0..self.cells.len())
            .map(|l| Ok((h.narrow(0, l, 1usize)?.try_squeeze(Some(0))?, c.narrow(0, l, 1usize)?.try_squeeze(Some(0))?)))
            .collect::<Result<Vec<_>>>()?;
        let (y, next) = self.step(x, &states)?;
        let hs: Vec<&Tensor> = next.iter().map(|(h, _)| h).collect();
        let cs: Vec<&Tensor> = next.iter().map(|(_, c)| c).collect();
        Ok((y, Tensor::stack(&hs, 0)?, Tensor::stack(&cs, 0)?))
    }
}
