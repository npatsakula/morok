//! Tests: rnn_v2 — the shared RNN/GRU/LSTM core.
//!
//! Every numeric check compares against a host reference written in plain Rust
//! over `Vec<f32>`, transcribed from `torch.nn.GRU` / `torch.nn.LSTM`.

use svod_dtype::DType;
use svod_ir::{Op, SInt, UOp};
use test_case::test_case;

use crate::error::ErrorKind;
use crate::nn::{GruCell, LstmCell, RnnDirection, RnnLayout, RnnStack};
use crate::{Tensor, Variable};

const TOL: f32 = 2e-5;

/// Deterministic values in `[-0.5, 0.5]`, distinct per `seed`.
fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `w [rows, cols] · x + b`.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    (0..rows).map(|r| (0..cols).fold(b.map_or(0.0, |b| b[r]), |acc, c| acc + w[r * cols + c] * x[c])).collect()
}

#[track_caller]
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {} != {}", got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < TOL, "{what}[{i}]: {g} != {w} (Δ {})", (g - w).abs());
    }
}

/// Per-step outputs, `[t][b][h]`.
type Seq3 = Vec<Vec<Vec<f32>>>;
/// A recurrent state, `[b][h]`.
type Seq2 = Vec<Vec<f32>>;

fn is_reverse(direction: RnnDirection, d: usize) -> bool {
    direction == RnnDirection::Backward || (direction == RnnDirection::Bidirectional && d == 1)
}

// =========================================================================
// Host references (PyTorch semantics)
// =========================================================================

/// `x[t][b][i]` → `(y[t][b][h] in input time order, final h[b][h])`.
/// `nn.GRU`: gates `r, z, n`, `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`.
#[allow(clippy::too_many_arguments)]
fn gru_ref(
    x: &[Vec<Vec<f32>>],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: Option<&[f32]>,
    b_hh: Option<&[f32]>,
    hs: usize,
    reverse: bool,
) -> (Seq3, Seq2) {
    let (t_len, batch, is) = (x.len(), x[0].len(), x[0][0].len());
    let mut h = vec![vec![0.0f32; hs]; batch];
    let mut ys = vec![vec![vec![0.0f32; hs]; batch]; t_len];
    for step in 0..t_len {
        let t = if reverse { t_len - 1 - step } else { step };
        for b in 0..batch {
            let gi = matvec(w_ih, 3 * hs, is, &x[t][b], b_ih);
            let gh = matvec(w_hh, 3 * hs, hs, &h[b], b_hh);
            let next: Vec<f32> = (0..hs)
                .map(|k| {
                    let r = sigmoid(gi[k] + gh[k]);
                    let z = sigmoid(gi[hs + k] + gh[hs + k]);
                    let n = (gi[2 * hs + k] + r * gh[2 * hs + k]).tanh();
                    (1.0 - z) * n + z * h[b][k]
                })
                .collect();
            ys[t][b].clone_from(&next);
            h[b] = next;
        }
    }
    (ys, h)
}

/// `nn.LSTM`: gates `i, f, g, o`.
#[allow(clippy::too_many_arguments)]
fn lstm_ref(
    x: &[Vec<Vec<f32>>],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: Option<&[f32]>,
    b_hh: Option<&[f32]>,
    hs: usize,
    reverse: bool,
) -> (Seq3, Seq2, Seq2) {
    let (t_len, batch, is) = (x.len(), x[0].len(), x[0][0].len());
    let mut h = vec![vec![0.0f32; hs]; batch];
    let mut c = vec![vec![0.0f32; hs]; batch];
    let mut ys = vec![vec![vec![0.0f32; hs]; batch]; t_len];
    for step in 0..t_len {
        let t = if reverse { t_len - 1 - step } else { step };
        for b in 0..batch {
            let gi = matvec(w_ih, 4 * hs, is, &x[t][b], b_ih);
            let gh = matvec(w_hh, 4 * hs, hs, &h[b], b_hh);
            let g = |k: usize| gi[k] + gh[k];
            let (mut nh, mut nc) = (vec![0.0f32; hs], vec![0.0f32; hs]);
            for k in 0..hs {
                let (i, f) = (sigmoid(g(k)), sigmoid(g(hs + k)));
                let (gg, o) = (g(2 * hs + k).tanh(), sigmoid(g(3 * hs + k)));
                nc[k] = f * c[b][k] + i * gg;
                nh[k] = o * nc[k].tanh();
            }
            ys[t][b].clone_from(&nh);
            h[b] = nh;
            c[b] = nc;
        }
    }
    (ys, h, c)
}

// =========================================================================
// Fixture
// =========================================================================

const T: usize = 4;
const B: usize = 2;
const I: usize = 3;
const H: usize = 5;

/// `[T][B][I]` inputs plus their flat forms in both layouts.
struct Input {
    host: Vec<Vec<Vec<f32>>>,
    seq_first: Vec<f32>,
    batch_first: Vec<f32>,
}

fn input() -> Input {
    let flat = seq(T * B * I, 0.31);
    let host: Vec<Vec<Vec<f32>>> =
        (0..T).map(|t| (0..B).map(|b| (0..I).map(|i| flat[(t * B + b) * I + i]).collect()).collect()).collect();
    let mut batch_first = Vec::with_capacity(T * B * I);
    for b in 0..B {
        for t in 0..T {
            batch_first.extend_from_slice(&flat[(t * B + b) * I..(t * B + b + 1) * I]);
        }
    }
    Input { host, seq_first: flat, batch_first }
}

impl Input {
    fn tensor(&self, layout: RnnLayout) -> Tensor {
        match layout {
            RnnLayout::SeqFirst => {
                Tensor::from_slice(&self.seq_first).try_reshape([T as isize, B as isize, I as isize]).unwrap()
            }
            RnnLayout::BatchFirst => {
                Tensor::from_slice(&self.batch_first).try_reshape([B as isize, T as isize, I as isize]).unwrap()
            }
        }
    }
}

fn realized(t: &Tensor) -> Vec<f32> {
    let t = t.contiguous();
    t.realize().unwrap();
    t.as_vec::<f32>().unwrap()
}

/// Flatten `ys[d][t][b][h]` into the layout `output` uses: `[T, B, D*H]` or `[B, T, D*H]`.
fn flat_output(ys: &[Seq3], layout: RnnLayout) -> Vec<f32> {
    let dirs = ys.len();
    visit_tb(ys, layout).flat_map(|(t, b)| (0..dirs).flat_map(move |d| ys[d][t][b].iter().copied())).collect()
}

/// Flatten into the ONNX `y` layout: `[T, D, B, H]` or `[B, T, D, H]` — the
/// direction axis sits *inside* the time axis, so the emission order differs
/// from [`flat_output`] only for `BatchFirst`.
fn flat_y(ys: &[Seq3], layout: RnnLayout) -> Vec<f32> {
    let (t_len, batch, dirs) = (ys[0].len(), ys[0][0].len(), ys.len());
    let order: Vec<(usize, usize, usize)> = match layout {
        RnnLayout::SeqFirst => {
            (0..t_len).flat_map(|t| (0..dirs).flat_map(move |d| (0..batch).map(move |b| (d, t, b)))).collect()
        }
        RnnLayout::BatchFirst => {
            (0..batch).flat_map(|b| (0..t_len).flat_map(move |t| (0..dirs).map(move |d| (d, t, b)))).collect()
        }
    };
    order.into_iter().flat_map(|(d, t, b)| ys[d][t][b].iter().copied()).collect()
}

/// `(t, b)` pairs in the order the requested layout stores them.
fn visit_tb(ys: &[Seq3], layout: RnnLayout) -> Box<dyn Iterator<Item = (usize, usize)>> {
    let (t_len, batch) = (ys[0].len(), ys[0][0].len());
    match layout {
        RnnLayout::SeqFirst => Box::new((0..t_len).flat_map(move |t| (0..batch).map(move |b| (t, b)))),
        RnnLayout::BatchFirst => Box::new((0..batch).flat_map(move |b| (0..t_len).map(move |t| (t, b)))),
    }
}

/// `[D, B, H]` state stack.
fn flat_state(states: &[Seq2]) -> Vec<f32> {
    states.iter().flat_map(|s| s.iter().flat_map(|v| v.iter().copied())).collect()
}

fn dims(t: &Tensor) -> Vec<usize> {
    t.shape().unwrap().iter().map(|d| d.as_const().unwrap()).collect()
}

// =========================================================================
// GRU / LSTM against the host reference
// =========================================================================

#[test_case(RnnLayout::SeqFirst, RnnDirection::Forward, true; "seq fwd bias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Forward, false; "seq fwd nobias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Backward, true; "seq bwd bias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Backward, false; "seq bwd nobias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Bidirectional, true; "seq bidir bias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Bidirectional, false; "seq bidir nobias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Forward, true; "batch fwd bias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Forward, false; "batch fwd nobias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Backward, true; "batch bwd bias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Backward, false; "batch bwd nobias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Bidirectional, true; "batch bidir bias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Bidirectional, false; "batch bidir nobias")]
fn gru_matches_pytorch(layout: RnnLayout, direction: RnnDirection, bias: bool) {
    let d = direction.num_directions();
    let inp = input();
    let (w_ih_f, w_hh_f) = (seq(d * 3 * H * I, 0.17), seq(d * 3 * H * H, 0.23));
    let (b_ih_f, b_hh_f) = (seq(d * 3 * H, 0.41), seq(d * 3 * H, 0.53));

    let w_ih = Tensor::from_slice(&w_ih_f).try_reshape([(d * 3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(&w_hh_f).try_reshape([(d * 3 * H) as isize, H as isize]).unwrap();
    let b_ih = Tensor::from_slice(&b_ih_f);
    let b_hh = Tensor::from_slice(&b_hh_f);

    let out = inp
        .tensor(layout)
        .gru()
        .weight_ih(&w_ih)
        .weight_hh(&w_hh)
        .maybe_bias_ih(bias.then_some(&b_ih))
        .maybe_bias_hh(bias.then_some(&b_hh))
        .direction(direction)
        .layout(layout)
        .call()
        .unwrap();

    let mut ys = Vec::with_capacity(d);
    let mut hs = Vec::with_capacity(d);
    for dir in 0..d {
        let (y, h) = gru_ref(
            &inp.host,
            &w_ih_f[dir * 3 * H * I..(dir + 1) * 3 * H * I],
            &w_hh_f[dir * 3 * H * H..(dir + 1) * 3 * H * H],
            bias.then(|| &b_ih_f[dir * 3 * H..(dir + 1) * 3 * H]),
            bias.then(|| &b_hh_f[dir * 3 * H..(dir + 1) * 3 * H]),
            H,
            is_reverse(direction, dir),
        );
        ys.push(y);
        hs.push(h);
    }

    assert_close(&realized(&out.output), &flat_output(&ys, layout), "output");
    assert_close(&realized(&out.h_n), &flat_state(&hs), "h_n");
    assert_close(&realized(&out.y), &flat_y(&ys, layout), "y");
    let expect_out = match layout {
        RnnLayout::SeqFirst => vec![T, B, d * H],
        RnnLayout::BatchFirst => vec![B, T, d * H],
    };
    assert_eq!(dims(&out.output), expect_out);
    assert_eq!(dims(&out.h_n), vec![d, B, H]);
    assert_eq!(
        dims(&out.y),
        match layout {
            RnnLayout::SeqFirst => vec![T, d, B, H],
            RnnLayout::BatchFirst => vec![B, T, d, H],
        }
    );
}

#[test_case(RnnLayout::SeqFirst, RnnDirection::Forward, true; "seq fwd bias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Forward, false; "seq fwd nobias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Backward, true; "seq bwd bias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Bidirectional, true; "seq bidir bias")]
#[test_case(RnnLayout::SeqFirst, RnnDirection::Bidirectional, false; "seq bidir nobias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Forward, true; "batch fwd bias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Backward, false; "batch bwd nobias")]
#[test_case(RnnLayout::BatchFirst, RnnDirection::Bidirectional, true; "batch bidir bias")]
fn lstm_matches_pytorch(layout: RnnLayout, direction: RnnDirection, bias: bool) {
    let d = direction.num_directions();
    let inp = input();
    let (w_ih_f, w_hh_f) = (seq(d * 4 * H * I, 0.19), seq(d * 4 * H * H, 0.29));
    let (b_ih_f, b_hh_f) = (seq(d * 4 * H, 0.43), seq(d * 4 * H, 0.59));

    let w_ih = Tensor::from_slice(&w_ih_f).try_reshape([(d * 4 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(&w_hh_f).try_reshape([(d * 4 * H) as isize, H as isize]).unwrap();
    let b_ih = Tensor::from_slice(&b_ih_f);
    let b_hh = Tensor::from_slice(&b_hh_f);

    let out = inp
        .tensor(layout)
        .lstm()
        .weight_ih(&w_ih)
        .weight_hh(&w_hh)
        .maybe_bias_ih(bias.then_some(&b_ih))
        .maybe_bias_hh(bias.then_some(&b_hh))
        .direction(direction)
        .layout(layout)
        .call()
        .unwrap();

    let (mut ys, mut hs, mut cs) = (Vec::new(), Vec::new(), Vec::new());
    for dir in 0..d {
        let (y, h, c) = lstm_ref(
            &inp.host,
            &w_ih_f[dir * 4 * H * I..(dir + 1) * 4 * H * I],
            &w_hh_f[dir * 4 * H * H..(dir + 1) * 4 * H * H],
            bias.then(|| &b_ih_f[dir * 4 * H..(dir + 1) * 4 * H]),
            bias.then(|| &b_hh_f[dir * 4 * H..(dir + 1) * 4 * H]),
            H,
            is_reverse(direction, dir),
        );
        ys.push(y);
        hs.push(h);
        cs.push(c);
    }

    assert_close(&realized(&out.output), &flat_output(&ys, layout), "output");
    assert_close(&realized(&out.h_n), &flat_state(&hs), "h_n");
    assert_close(&realized(&out.c_n), &flat_state(&cs), "c_n");
    assert_eq!(dims(&out.c_n), vec![d, B, H]);
}

/// A 3-D `[D, G*H, X]` weight stack must load exactly like the 2-D
/// concatenation — both are one call away from a PyTorch state dict.
#[test]
fn stacked_and_concatenated_weight_layouts_agree() {
    let inp = input();
    let (w_ih_f, w_hh_f) = (seq(2 * 3 * H * I, 0.17), seq(2 * 3 * H * H, 0.23));
    let flat_ih = Tensor::from_slice(&w_ih_f).try_reshape([(2 * 3 * H) as isize, I as isize]).unwrap();
    let flat_hh = Tensor::from_slice(&w_hh_f).try_reshape([(2 * 3 * H) as isize, H as isize]).unwrap();
    let stacked_ih = flat_ih.try_reshape([2isize, (3 * H) as isize, I as isize]).unwrap();
    let stacked_hh = flat_hh.try_reshape([2isize, (3 * H) as isize, H as isize]).unwrap();

    let run = |wi: &Tensor, wh: &Tensor| {
        realized(
            &inp.tensor(RnnLayout::SeqFirst)
                .gru()
                .weight_ih(wi)
                .weight_hh(wh)
                .direction(RnnDirection::Bidirectional)
                .call()
                .unwrap()
                .output,
        )
    };
    assert_close(&run(&flat_ih, &flat_hh), &run(&stacked_ih, &stacked_hh), "stacked vs concatenated");
}

// =========================================================================
// ONNX gate-order mapping
// =========================================================================

/// Reorder gate blocks of a `[G*H, X]` host buffer.
fn reorder(src: &[f32], hs: usize, cols: usize, order: &[usize]) -> Vec<f32> {
    order.iter().flat_map(|&g| src[g * hs * cols..(g + 1) * hs * cols].to_vec()).collect()
}

/// ONNX GRU gates are `z, r, h`; PyTorch's are `r, z, n` — block order `[1, 0, 2]`.
/// With `linear_before_reset=1` the two formulations coincide, so the same
/// weights under either spelling must produce the same sequence.
#[test]
fn onnx_gru_gate_order_maps_onto_pytorch() {
    let inp = input();
    let (w_f, r_f) = (seq(3 * H * I, 0.17), seq(3 * H * H, 0.23));
    let bias_f = seq(6 * H, 0.41);

    let w = Tensor::from_slice(&w_f).try_reshape([1isize, (3 * H) as isize, I as isize]).unwrap();
    let r = Tensor::from_slice(&r_f).try_reshape([1isize, (3 * H) as isize, H as isize]).unwrap();
    let bias = Tensor::from_slice(&bias_f).try_reshape([1isize, (6 * H) as isize]).unwrap();
    let onnx = inp
        .tensor(RnnLayout::SeqFirst)
        .gru()
        .w(&w)
        .r_weights(&r)
        .bias(&bias)
        .hidden_size(H)
        .linear_before_reset(1usize)
        .call()
        .unwrap();

    let order = [1usize, 0, 2];
    let w_ih_f = reorder(&w_f, H, I, &order);
    let w_hh_f = reorder(&r_f, H, H, &order);
    let b_ih_f = reorder(&bias_f[..3 * H], H, 1, &order);
    let b_hh_f = reorder(&bias_f[3 * H..], H, 1, &order);
    let w_ih = Tensor::from_slice(&w_ih_f).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(&w_hh_f).try_reshape([(3 * H) as isize, H as isize]).unwrap();
    let (b_ih, b_hh) = (Tensor::from_slice(&b_ih_f), Tensor::from_slice(&b_hh_f));
    let torch = inp
        .tensor(RnnLayout::SeqFirst)
        .gru()
        .weight_ih(&w_ih)
        .weight_hh(&w_hh)
        .bias_ih(&b_ih)
        .bias_hh(&b_hh)
        .call()
        .unwrap();

    assert_close(&realized(&onnx.output), &realized(&torch.output), "onnx vs torch gru");
    let (ys, hs) = gru_ref(&inp.host, &w_ih_f, &w_hh_f, Some(&b_ih_f), Some(&b_hh_f), H, false);
    assert_close(&realized(&onnx.output), &flat_output(&[ys], RnnLayout::SeqFirst), "onnx gru vs host");
    assert_close(&realized(&onnx.y_h), &flat_state(&[hs]), "onnx gru y_h");
}

/// ONNX LSTM gates are `i, o, f, c`; PyTorch's are `i, f, g, o` — block order
/// `[0, 2, 3, 1]`. The formulations are identical, so both must agree exactly.
#[test]
fn onnx_lstm_gate_order_maps_onto_pytorch() {
    let inp = input();
    let (w_f, r_f) = (seq(4 * H * I, 0.19), seq(4 * H * H, 0.29));
    let bias_f = seq(8 * H, 0.43);

    let w = Tensor::from_slice(&w_f).try_reshape([1isize, (4 * H) as isize, I as isize]).unwrap();
    let r = Tensor::from_slice(&r_f).try_reshape([1isize, (4 * H) as isize, H as isize]).unwrap();
    let bias = Tensor::from_slice(&bias_f).try_reshape([1isize, (8 * H) as isize]).unwrap();
    let onnx = inp.tensor(RnnLayout::SeqFirst).lstm().w(&w).r(&r).bias(&bias).hidden_size(H).call().unwrap();

    let order = [0usize, 2, 3, 1];
    let w_ih_f = reorder(&w_f, H, I, &order);
    let w_hh_f = reorder(&r_f, H, H, &order);
    let b_ih_f = reorder(&bias_f[..4 * H], H, 1, &order);
    let b_hh_f = reorder(&bias_f[4 * H..], H, 1, &order);
    let (ys, hs, cs) = lstm_ref(&inp.host, &w_ih_f, &w_hh_f, Some(&b_ih_f), Some(&b_hh_f), H, false);

    assert_close(&realized(&onnx.output), &flat_output(&[ys], RnnLayout::SeqFirst), "onnx lstm vs host");
    assert_close(&realized(&onnx.y_h), &flat_state(&[hs]), "onnx lstm y_h");
    assert_close(&realized(&onnx.y_c), &flat_state(&[cs]), "onnx lstm y_c");
}

// =========================================================================
// Cells vs the full-sequence builders
// =========================================================================

#[test]
fn gru_cell_loop_matches_the_sequence_builder() {
    let inp = input();
    let (w_ih_f, w_hh_f) = (seq(3 * H * I, 0.17), seq(3 * H * H, 0.23));
    let (b_ih_f, b_hh_f) = (seq(3 * H, 0.41), seq(3 * H, 0.53));
    let w_ih = Tensor::from_slice(&w_ih_f).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(&w_hh_f).try_reshape([(3 * H) as isize, H as isize]).unwrap();
    let (b_ih, b_hh) = (Tensor::from_slice(&b_ih_f), Tensor::from_slice(&b_hh_f));

    let x = inp.tensor(RnnLayout::SeqFirst);
    let built = x.gru().weight_ih(&w_ih).weight_hh(&w_hh).bias_ih(&b_ih).bias_hh(&b_hh).call().unwrap();

    let cell = GruCell::new(w_ih, w_hh, Some(b_ih), Some(b_hh)).unwrap();
    assert_eq!(cell.hidden_size(), H);
    let mut h = Tensor::full(&[B, H], 0.0f32, DType::Float32);
    let mut outs = Vec::new();
    for t in 0..T {
        let x_t = x.narrow(0, t, 1usize).unwrap().try_squeeze(Some(0)).unwrap();
        h = cell.step(&x_t, &h).unwrap();
        outs.push(h.clone());
    }
    let stacked = Tensor::stack(&outs.iter().collect::<Vec<_>>(), 0).unwrap();
    assert_close(&realized(&built.output), &realized(&stacked), "gru cell loop");
    assert_close(&realized(&built.h_n), &realized(&h.try_unsqueeze(0).unwrap()), "gru cell final h");
}

#[test]
fn lstm_cell_loop_matches_the_sequence_builder() {
    let inp = input();
    let (w_ih_f, w_hh_f) = (seq(4 * H * I, 0.19), seq(4 * H * H, 0.29));
    let (b_ih_f, b_hh_f) = (seq(4 * H, 0.43), seq(4 * H, 0.59));
    let w_ih = Tensor::from_slice(&w_ih_f).try_reshape([(4 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(&w_hh_f).try_reshape([(4 * H) as isize, H as isize]).unwrap();
    let (b_ih, b_hh) = (Tensor::from_slice(&b_ih_f), Tensor::from_slice(&b_hh_f));

    let x = inp.tensor(RnnLayout::SeqFirst);
    let built = x.lstm().weight_ih(&w_ih).weight_hh(&w_hh).bias_ih(&b_ih).bias_hh(&b_hh).call().unwrap();

    let cell = LstmCell::new(w_ih, w_hh, b_ih, b_hh);
    let mut h = Tensor::full(&[B, H], 0.0f32, DType::Float32);
    let mut c = Tensor::full(&[B, H], 0.0f32, DType::Float32);
    let mut outs = Vec::new();
    for t in 0..T {
        let x_t = x.narrow(0, t, 1usize).unwrap().try_squeeze(Some(0)).unwrap();
        let (nh, nc) = cell.step(&x_t, &h, &c).unwrap();
        h = nh;
        c = nc;
        outs.push(h.clone());
    }
    let stacked = Tensor::stack(&outs.iter().collect::<Vec<_>>(), 0).unwrap();
    assert_close(&realized(&built.output), &realized(&stacked), "lstm cell loop");
    assert_close(&realized(&built.c_n), &realized(&c.try_unsqueeze(0).unwrap()), "lstm cell final c");
}

/// `RnnStack::step_stacked` must reproduce the hand-written RNN-T predictor
/// loop: slice `[L, B, P]` state, step each layer on the one below's output.
#[test]
fn rnn_stack_matches_the_manual_predictor_loop() {
    const L: usize = 3;
    let cells: Vec<LstmCell> = (0..L)
        .map(|l| {
            let s = 0.11 + l as f32 * 0.07;
            let w_ih = Tensor::from_slice(seq(4 * H * H, s)).try_reshape([(4 * H) as isize, H as isize]).unwrap();
            let w_hh =
                Tensor::from_slice(seq(4 * H * H, s + 0.03)).try_reshape([(4 * H) as isize, H as isize]).unwrap();
            LstmCell::new(w_ih, w_hh, Tensor::from_slice(seq(4 * H, s + 0.05)), Tensor::from_slice(seq(4 * H, s)))
        })
        .collect();

    let x = Tensor::from_slice(seq(B * H, 0.37)).try_reshape([B as isize, H as isize]).unwrap();
    let h0 = Tensor::from_slice(seq(L * B * H, 0.61)).try_reshape([L as isize, B as isize, H as isize]).unwrap();
    let c0 = Tensor::from_slice(seq(L * B * H, 0.67)).try_reshape([L as isize, B as isize, H as isize]).unwrap();

    // Manual loop, as `RnntPredictor::forward_parts` writes it.
    let mut layer_in = x.clone();
    let (mut hs, mut cs) = (Vec::new(), Vec::new());
    for (l, cell) in cells.iter().enumerate() {
        let h_l = h0.narrow(0, l, 1usize).unwrap().try_squeeze(Some(0)).unwrap();
        let c_l = c0.narrow(0, l, 1usize).unwrap().try_squeeze(Some(0)).unwrap();
        let (nh, nc) = cell.step(&layer_in, &h_l, &c_l).unwrap();
        layer_in = nh.clone();
        hs.push(nh);
        cs.push(nc);
    }
    let want_h = Tensor::stack(&hs.iter().collect::<Vec<_>>(), 0).unwrap();
    let want_c = Tensor::stack(&cs.iter().collect::<Vec<_>>(), 0).unwrap();

    let stack = RnnStack::new(cells);
    assert_eq!(stack.len(), L);
    assert!(!stack.is_empty());
    let (y, h, c) = stack.step_stacked(&x, &h0, &c0).unwrap();
    assert_close(&realized(&y), &realized(&layer_in), "stack y");
    assert_close(&realized(&h), &realized(&want_h), "stack h");
    assert_close(&realized(&c), &realized(&want_c), "stack c");
}

/// An explicit `h0`/`c0` must be threaded in per direction.
#[test]
fn initial_state_is_honoured() {
    let inp = input();
    let w_ih = Tensor::from_slice(seq(3 * H * I, 0.17)).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(seq(3 * H * H, 0.23)).try_reshape([(3 * H) as isize, H as isize]).unwrap();
    let zeros = Tensor::full(&[1, B, H], 0.0f32, DType::Float32);
    let x = inp.tensor(RnnLayout::SeqFirst);

    let bare = x.gru().weight_ih(&w_ih).weight_hh(&w_hh).call().unwrap();
    let with_zero = x.gru().weight_ih(&w_ih).weight_hh(&w_hh).h0(&zeros).call().unwrap();
    assert_close(&realized(&bare.output), &realized(&with_zero.output), "zero h0 == default");

    let nonzero = Tensor::from_slice(seq(B * H, 0.71)).try_reshape([1isize, B as isize, H as isize]).unwrap();
    let with_state = x.gru().weight_ih(&w_ih).weight_hh(&w_hh).initial_h(&nonzero).call().unwrap();
    let (got, want) = (realized(&with_state.output), realized(&bare.output));
    assert!(got.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-3), "a non-zero h0 must change the output");
}

// =========================================================================
// Symbolic shapes
// =========================================================================

/// The batch axis may stay symbolic; only `T` has to be concrete.
#[test]
fn symbolic_batch_flows_through() {
    let batch = Variable::new("b", 1, 8).bind(4).unwrap().as_sint();
    let x = Tensor::empty_dynamic(&[SInt::Const(T), batch.clone(), SInt::Const(I)], DType::Float32);
    let w_ih = Tensor::from_slice(seq(2 * 3 * H * I, 0.17)).try_reshape([(2 * 3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(seq(2 * 3 * H * H, 0.23)).try_reshape([(2 * 3 * H) as isize, H as isize]).unwrap();

    let out = x.gru().weight_ih(&w_ih).weight_hh(&w_hh).direction(RnnDirection::Bidirectional).call().unwrap();
    let shape = out.output.shape().unwrap();
    assert_eq!(shape[0], SInt::Const(T));
    assert_eq!(shape[1], batch);
    assert_eq!(shape[2], SInt::Const(2 * H));
    let h_shape = out.h_n.shape().unwrap();
    assert_eq!(h_shape[0], SInt::Const(2));
    assert_eq!(h_shape[1], batch);
}

/// A symbolic sequence length is out of scope for this phase and must be
/// rejected rather than silently mis-unrolled.
#[test]
fn symbolic_seq_length_is_rejected() {
    let t = Variable::new("t", 1, 8).bind(4).unwrap().as_sint();
    let x = Tensor::empty_dynamic(&[t, SInt::Const(B), SInt::Const(I)], DType::Float32);
    let w_ih = Tensor::from_slice(seq(3 * H * I, 0.17)).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(seq(3 * H * H, 0.23)).try_reshape([(3 * H) as isize, H as isize]).unwrap();

    let err = x.gru().weight_ih(&w_ih).weight_hh(&w_hh).call().err().unwrap();
    assert!(matches!(err.kind(), ErrorKind::NonConstDim { axis: 0, .. }), "expected NonConstDim, got {err}");
}

/// Exactly one weight spelling: neither both nor none.
#[test]
fn weight_spellings_are_exclusive() {
    let inp = input();
    let x = inp.tensor(RnnLayout::SeqFirst);
    let w_ih = Tensor::from_slice(seq(3 * H * I, 0.17)).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(seq(3 * H * H, 0.23)).try_reshape([(3 * H) as isize, H as isize]).unwrap();
    let w = w_ih.try_unsqueeze(0).unwrap();
    let r = w_hh.try_unsqueeze(0).unwrap();

    for err in [
        x.gru().call().err().unwrap(),
        x.gru().w(&w).r_weights(&r).weight_ih(&w_ih).weight_hh(&w_hh).call().err().unwrap(),
    ] {
        assert!(matches!(err.kind(), ErrorKind::ExclusiveParams { .. }), "expected ExclusiveParams, got {err}");
    }
}

/// `direction` must agree with the direction count the weights carry.
#[test]
fn direction_must_match_the_weight_stack() {
    let inp = input();
    let w_ih = Tensor::from_slice(seq(3 * H * I, 0.17)).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(seq(3 * H * H, 0.23)).try_reshape([(3 * H) as isize, H as isize]).unwrap();
    let err = inp
        .tensor(RnnLayout::SeqFirst)
        .gru()
        .weight_ih(&w_ih)
        .weight_hh(&w_hh)
        .direction(RnnDirection::Bidirectional)
        .call()
        .err()
        .unwrap();
    assert!(matches!(err.kind(), ErrorKind::ParamRange { param: "direction", .. }), "got {err}");
}

// =========================================================================
// Kernel count
// =========================================================================

/// `(launches, distinct kernel bodies)` for the rangeified graph — the second
/// number is what the `(ast_id, device)` kernel cache actually compiles.
fn count_kernels(t: &Tensor) -> (usize, usize) {
    let sink = UOp::sink(vec![t.uop().contiguous()]);
    let rangeified = svod_schedule::rangeify_with_map(sink).expect("rangeify");
    let (kernels, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("kernel graph");
    let bodies: Vec<_> = kernels
        .toposort_call_aware(false)
        .iter()
        .filter_map(|n| match n.op() {
            Op::Call(call) => Some(call.body.clone()),
            _ => None,
        })
        .collect();
    let distinct: std::collections::HashSet<_> = bodies.iter().map(|b| format!("{b:?}")).collect();
    (bodies.len(), distinct.len())
}

/// Host-unrolling a `T`-step GRU costs `1 + 2*T` kernels: one hoisted input
/// projection over the whole sequence, then two per step (the `r`/`z` matmul
/// and the `n` matmul, each fused with its elementwise tail).
///
/// The bodies are *not* deduped: every step's input slice carries a distinct
/// constant time offset, so the `(ast_id, device)` kernel cache sees `2*T`
/// different ASTs. Sharing one compiled step needs the schedule-level
/// `END(CALL, [RANGE])` loop from the recurrence spike, which is out of scope
/// for this phase. Pinned so a regression that stops hoisting the projection
/// (or that starts emitting a third matmul per step) is visible.
#[test]
fn t8_gru_kernel_count() {
    let x = Tensor::empty(&[8, 2, I], DType::Float32);
    let w_ih = Tensor::empty(&[3 * H, I], DType::Float32);
    let w_hh = Tensor::empty(&[3 * H, H], DType::Float32);
    let out = x.gru().weight_ih(&w_ih).weight_hh(&w_hh).call().unwrap();
    let (launches, distinct) = count_kernels(&out.output);
    assert_eq!((launches, distinct), T8_GRU_KERNELS, "T=8 GRU kernel count moved: {launches}/{distinct}");
}

/// Measured `(launches, distinct bodies)`; see [`t8_gru_kernel_count`].
const T8_GRU_KERNELS: (usize, usize) = (17, 17);

// =========================================================================
// Downstream call shapes
// =========================================================================

/// The GTCRN call shape: ONNX-packed `[D, 3H, *]` weights and a `[D, 2*3H]`
/// bias, driven with the `RnnLayout`/`GruDirection` enums and
/// `linear_before_reset(true)`. Guards the exact chain `model/src/gtcrn`
/// writes, including its post-processing of `y`/`y_h`.
#[test]
fn gtcrn_gru_call_shape() {
    let inp = input();
    let x = inp.tensor(RnnLayout::BatchFirst);
    let pack = |seed: f32, cols: usize| {
        Tensor::from_slice(seq(3 * H * cols, seed)).try_reshape([(3 * H) as isize, cols as isize]).unwrap()
    };
    let (w0, r0) = (pack(0.17, I), pack(0.23, H));
    let (b_ih, b_hh) = (Tensor::from_slice(seq(3 * H, 0.41)), Tensor::from_slice(seq(3 * H, 0.53)));

    // Unidirectional, explicit state — `GruWeights::forward_with_state`.
    let w = w0.try_unsqueeze(0).unwrap();
    let r = r0.try_unsqueeze(0).unwrap();
    let bias = Tensor::cat(&[&b_ih, &b_hh], 0).unwrap().try_unsqueeze(0).unwrap();
    let h0 = Tensor::from_slice(seq(B * H, 0.71)).try_reshape([B as isize, H as isize]).unwrap();
    let out = x
        .gru()
        .w(&w)
        .r_weights(&r)
        .hidden_size(H)
        .bias(&bias)
        .initial_h(&h0.try_unsqueeze(0).unwrap())
        .direction(RnnDirection::Forward)
        .linear_before_reset(true)
        .layout(RnnLayout::BatchFirst)
        .call()
        .unwrap();
    assert_eq!(dims(&out.y.try_squeeze(Some(2)).unwrap()), vec![B, T, H]);
    assert_eq!(dims(&out.y_h.try_squeeze(Some(1)).unwrap()), vec![B, H]);

    // Bidirectional — `bidir_gru` stacks the two directions' weights.
    let w2 = Tensor::stack(&[&w0, &pack(0.19, I)], 0).unwrap();
    let r2 = Tensor::stack(&[&r0, &pack(0.29, H)], 0).unwrap();
    let bias2 =
        Tensor::cat(&[&Tensor::stack(&[&b_ih, &b_ih], 0).unwrap(), &Tensor::stack(&[&b_hh, &b_hh], 0).unwrap()], 1)
            .unwrap();
    let bi = x
        .gru()
        .w(&w2)
        .r_weights(&r2)
        .hidden_size(H)
        .bias(&bias2)
        .direction(RnnDirection::Bidirectional)
        .linear_before_reset(true)
        .layout(RnnLayout::BatchFirst)
        .call()
        .unwrap();
    let merged = bi.y.try_reshape([B as isize, T as isize, (2 * H) as isize]).unwrap();
    assert_eq!(dims(&merged), vec![B, T, 2 * H]);
    assert_close(&realized(&merged), &realized(&bi.output), "gtcrn bidir reshape == output");
}
