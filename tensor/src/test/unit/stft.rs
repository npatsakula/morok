//! Tests: stft.
//!
//! The graph STFT/iSTFT is checked against a naive host DFT written here, so
//! the reference shares no code with the implementation under test.

use std::f64::consts::TAU;

use svod_dtype::DType;
use svod_ir::SInt;
use test_case::test_case;

use crate::error::ErrorKind;
use crate::nn::{MelLog, MelNorm, MelScale, Window};
use crate::{Tensor, Variable};

// =========================================================================
// Host reference
// =========================================================================

/// Deterministic, spread-out test signal of `n` samples.
fn signal(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            0.6 * (0.7 * t + seed).sin() + 0.4 * (0.13 * t + 2.0 * seed).cos() + 0.1 * (1.9 * t).sin()
        })
        .collect()
}

/// `(a0, a1, p)` of the cosine-sum window `(a0 - a1·cos(2πk/d))^p`.
fn cosine_sum(kind: &Window) -> (f64, f64, f64) {
    match kind {
        Window::Rectangular => (1.0, 0.0, 1.0),
        Window::Hann => (0.5, 0.5, 1.0),
        Window::Hamming => (0.54, 0.46, 1.0),
        Window::Povey => (0.5, 0.5, 0.85),
        Window::Custom(_) => unreachable!("custom windows are supplied directly"),
    }
}

/// Cosine-sum window, periodic form (denominator `n`).
fn host_window(kind: &Window, n: usize) -> Vec<f64> {
    let (a0, a1, p) = cosine_sum(kind);
    (0..n).map(|k| (a0 - a1 * (TAU * k as f64 / n as f64).cos()).powf(p)).collect()
}

/// `torch.nn.functional.pad(mode="reflect")`: mirror without repeating edges.
fn reflect_pad(x: &[f64], pad: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((0..pad).rev().map(|j| x[j + 1]));
    out.extend_from_slice(x);
    out.extend((0..pad).map(|j| x[n - 2 - j]));
    out
}

/// Naive `torch.stft`, returned flat in the `[F, T, 2]` layout.
fn ref_stft(
    x: &[f64],
    n_fft: usize,
    hop: usize,
    win: &[f64],
    center: bool,
    onesided: bool,
    normalized: bool,
) -> Vec<f64> {
    let sig = if center { reflect_pad(x, n_fft / 2) } else { x.to_vec() };
    let frames = (sig.len() - n_fft) / hop + 1;
    let bins = if onesided { n_fft / 2 + 1 } else { n_fft };
    let scale = if normalized { 1.0 / (n_fft as f64).sqrt() } else { 1.0 };
    let mut out = vec![0.0; bins * frames * 2];
    for k in 0..bins {
        for t in 0..frames {
            let (mut re, mut im) = (0.0, 0.0);
            for n in 0..n_fft {
                let angle = TAU * ((k * n) % n_fft) as f64 / n_fft as f64;
                let v = sig[t * hop + n] * win[n];
                re += v * angle.cos();
                im -= v * angle.sin();
            }
            out[(k * frames + t) * 2] = re * scale;
            out[(k * frames + t) * 2 + 1] = im * scale;
        }
    }
    out
}

fn assert_close(got: &[f32], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!((*g as f64 - e).abs() <= tol, "element {i}: got {g}, expected {e} (tol {tol})");
    }
}

// =========================================================================
// Window
// =========================================================================

#[test_case(Window::Hann, 8, true; "hann periodic")]
#[test_case(Window::Hann, 8, false; "hann symmetric")]
#[test_case(Window::Hamming, 7, true; "hamming periodic")]
#[test_case(Window::Hamming, 7, false; "hamming symmetric")]
#[test_case(Window::Povey, 9, true; "povey periodic")]
#[test_case(Window::Povey, 9, false; "povey symmetric (kaldi)")]
#[test_case(Window::Rectangular, 5, true; "rectangular")]
fn window_matches_torch(kind: Window, n: usize, periodic: bool) {
    let (a0, a1, p) = cosine_sum(&kind);
    let denom = if periodic || n == 1 { n } else { n - 1 };
    let expected: Vec<f64> = (0..n).map(|k| (a0 - a1 * (TAU * k as f64 / denom as f64).cos()).powf(p)).collect();
    let got = Tensor::window(&kind, n, periodic, DType::Float32).unwrap().to_vec::<f32>().unwrap();
    assert_close(&got, &expected, 1e-6);
}

#[test]
fn window_custom_is_used_verbatim_and_length_checked() {
    let w = Tensor::from_slice([0.25f32, 0.5, 0.75, 1.0]);
    let got = Tensor::window(&Window::Custom(w.clone()), 4, true, DType::Float32).unwrap();
    assert_eq!(got.to_vec::<f32>().unwrap(), vec![0.25, 0.5, 0.75, 1.0]);

    let err = Tensor::window(&Window::Custom(w), 5, true, DType::Float32).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ShapeMismatch { .. }), "got {err}");
}

#[test]
fn window_rejects_zero_length() {
    let err = Tensor::window(&Window::Hann, 0, true, DType::Float32).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ParamRange { .. }), "got {err}");
}

// =========================================================================
// stft against the naive DFT
// =========================================================================

#[test_case(16, 4, Window::Hann, true, true, false; "hann 16/4 center onesided")]
#[test_case(16, 8, Window::Hann, false, true, false; "hann 16/8 no center")]
#[test_case(32, 8, Window::Hamming, true, true, false; "hamming 32/8 center")]
#[test_case(32, 16, Window::Rectangular, true, true, false; "rect 32/16 center")]
#[test_case(16, 4, Window::Hann, true, false, false; "hann 16/4 two-sided")]
#[test_case(16, 4, Window::Hann, true, true, true; "hann 16/4 normalized")]
#[test_case(64, 16, Window::Hann, true, true, false; "hann 64/16 center")]
#[test_case(32, 8, Window::Povey, false, true, false; "povey 32/8 no center")]
#[test_case(15, 5, Window::Hann, true, true, false; "odd n_fft 15/5")]
fn stft_matches_naive_dft(n_fft: usize, hop: usize, kind: Window, center: bool, onesided: bool, normalized: bool) {
    let len = 96;
    let x = signal(len, 0.3);
    let win = host_window(&kind, n_fft);
    let expected = ref_stft(&x, n_fft, hop, &win, center, onesided, normalized);

    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let spec = input
        .stft()
        .n_fft(n_fft)
        .hop(hop)
        .window(kind)
        .center(center)
        .onesided(onesided)
        .normalized(normalized)
        .call()
        .unwrap();

    let bins = if onesided { n_fft / 2 + 1 } else { n_fft };
    let padded = if center { len + n_fft } else { len };
    let frames = (padded - n_fft) / hop + 1;
    assert_eq!(spec.dims().unwrap(), vec![bins, frames, 2]);
    assert_close(&spec.to_vec::<f32>().unwrap(), &expected, 2e-4);
}

#[test]
fn stft_win_length_is_zero_padded_symmetrically() {
    let (n_fft, win_length, hop, len) = (32usize, 20usize, 8usize, 96usize);
    let x = signal(len, 1.1);
    // torch centers a short window inside the n_fft frame.
    let short = host_window(&Window::Hann, win_length);
    let left = (n_fft - win_length) / 2;
    let mut win = vec![0.0; n_fft];
    win[left..left + win_length].copy_from_slice(&short);
    let expected = ref_stft(&x, n_fft, hop, &win, true, true, false);

    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let spec = input.stft().n_fft(n_fft).hop(hop).win_length(win_length).call().unwrap();
    assert_close(&spec.to_vec::<f32>().unwrap(), &expected, 2e-4);
}

#[test]
fn stft_defaults_match_torch_hop_and_window() {
    let (n_fft, len) = (32usize, 96usize);
    let x = signal(len, 0.9);
    let expected = ref_stft(&x, n_fft, n_fft / 4, &host_window(&Window::Hann, n_fft), true, true, false);
    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let spec = input.stft().n_fft(n_fft).call().unwrap();
    assert_close(&spec.to_vec::<f32>().unwrap(), &expected, 2e-4);
}

#[test]
fn stft_custom_window_matches_gtcrn_sqrt_hann() {
    // GTCRN analyses with `hann_window(n_fft).sqrt()`.
    let (n_fft, hop, len) = (32usize, 16usize, 96usize);
    let win: Vec<f64> = host_window(&Window::Hann, n_fft).iter().map(|w| w.sqrt()).collect();
    let x = signal(len, 2.4);
    let expected = ref_stft(&x, n_fft, hop, &win, true, true, false);

    let custom = Tensor::from_slice(win.iter().map(|&w| w as f32).collect::<Vec<_>>());
    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let spec = input.stft().n_fft(n_fft).hop(hop).window(Window::Custom(custom)).call().unwrap();
    assert_close(&spec.to_vec::<f32>().unwrap(), &expected, 2e-4);
}

/// A `Custom` window that is still a graph (a `hann.sqrt()` built in-graph,
/// as GTCRN's could be) cannot be tabulated on the host, so the DFT kernels
/// are built in-graph — one launch each for the analysis and the synthesis
/// kernel — and must agree with the host tables to f32 rounding.
#[test]
fn lazy_custom_window_falls_back_to_in_graph_kernels() {
    let (n_fft, hop, len) = (32usize, 16usize, 96usize);
    let input = Tensor::from_slice(signal(len, 2.4).iter().map(|&v| v as f32).collect::<Vec<_>>());
    let sqrt_hann = || Tensor::window(&Window::Hann, n_fft, true, DType::Float32).unwrap().try_sqrt().unwrap();
    let host = Tensor::from_slice(sqrt_hann().to_vec::<f32>().unwrap());
    let lazy = sqrt_hann();
    assert!(lazy.buffer().is_none(), "the window must reach stft unrealized");

    let stft = |w: Tensor| input.stft().n_fft(n_fft).hop(hop).window(Window::Custom(w)).call().unwrap();
    let (spec_host, spec_lazy) = (stft(host.clone()), stft(lazy.clone()));
    assert_eq!(count_kernels(&spec_lazy), count_kernels(&spec_host) + 1);
    let expected: Vec<f64> = spec_host.to_vec::<f32>().unwrap().into_iter().map(f64::from).collect();
    assert_close(&spec_lazy.to_vec::<f32>().unwrap(), &expected, 1e-5);

    let istft = |spec: &Tensor, w: Tensor| {
        spec.istft().n_fft(n_fft).hop(hop).window(Window::Custom(w)).length(len).call().unwrap()
    };
    let (back_host, back_lazy) = (istft(&spec_host, host), istft(&spec_host, lazy));
    assert_eq!(count_kernels(&back_lazy), count_kernels(&back_host) + 1);
    let expected: Vec<f64> = back_host.to_vec::<f32>().unwrap().into_iter().map(f64::from).collect();
    assert_close(&back_lazy.to_vec::<f32>().unwrap(), &expected, 1e-5);
}

/// The untrimmed form: bins and frames rounded up to multiples of 8, the
/// true extents alongside, the prefix equal to `stft` and the surplus zero.
#[test_case(16, 4, 96; "9 bins -> 16, 25 frames -> 32")]
#[test_case(32, 8, 120; "17 bins -> 24, 16 frames stay")]
fn stft_padded_keeps_the_tileable_extents(n_fft: usize, hop: usize, len: usize) {
    let x = Tensor::from_slice(signal(len, 0.8).iter().map(|&v| v as f32).collect::<Vec<_>>());
    let (spec, bins, frames) = x.stft_padded().n_fft(n_fft).hop(hop).call().unwrap();
    let trimmed = x.stft().n_fft(n_fft).hop(hop).call().unwrap();
    assert_eq!((bins, frames), (n_fft / 2 + 1, len / hop + 1));
    let dims = spec.dims().unwrap();
    assert_eq!(dims, vec![1, bins.next_multiple_of(8), frames.next_multiple_of(8), 2]);

    let (padded_bins, padded_frames) = (dims[1], dims[2]);
    let got = spec.to_vec::<f32>().unwrap();
    let want = trimmed.to_vec::<f32>().unwrap();
    for k in 0..padded_bins {
        for t in 0..padded_frames {
            let pair = &got[(k * padded_frames + t) * 2..][..2];
            if k < bins && t < frames {
                assert_eq!(pair, &want[(k * frames + t) * 2..][..2], "bin {k} frame {t}");
            } else {
                assert_eq!(pair, &[0.0, 0.0], "surplus bin {k} frame {t} must be zero");
            }
        }
    }
}

// =========================================================================
// Batch and rank handling
// =========================================================================

#[test]
fn stft_batches_rows_independently() {
    let (n_fft, hop, len) = (16usize, 4usize, 48usize);
    let rows: Vec<Vec<f64>> = (0..3).map(|b| signal(len, b as f64 * 1.7)).collect();
    let win = host_window(&Window::Hann, n_fft);
    let expected: Vec<f64> = rows.iter().flat_map(|r| ref_stft(r, n_fft, hop, &win, true, true, false)).collect();

    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    let input = Tensor::from_slice(flat).try_reshape([3, len as isize]).unwrap();
    let spec = input.stft().n_fft(n_fft).hop(hop).call().unwrap();
    assert_eq!(spec.dims().unwrap(), vec![3, n_fft / 2 + 1, (len + n_fft - n_fft) / hop + 1, 2]);
    assert_close(&spec.to_vec::<f32>().unwrap(), &expected, 2e-4);
}

#[test]
fn stft_1d_equals_the_single_row_of_the_2d_form() {
    let (n_fft, hop, len) = (16usize, 4usize, 48usize);
    let x: Vec<f32> = signal(len, 0.5).iter().map(|&v| v as f32).collect();
    let one_d = Tensor::from_slice(x.clone()).stft().n_fft(n_fft).hop(hop).call().unwrap();
    let two_d =
        Tensor::from_slice(x).try_reshape([1, len as isize]).unwrap().stft().n_fft(n_fft).hop(hop).call().unwrap();
    assert_eq!(one_d.ndim().unwrap(), 3);
    assert_eq!(two_d.ndim().unwrap(), 4);
    assert_eq!(one_d.to_vec::<f32>().unwrap(), two_d.to_vec::<f32>().unwrap());
}

#[test]
fn stft_keeps_a_symbolic_batch_dimension() {
    let (n_fft, hop, len) = (16usize, 4usize, 48usize);
    let batch = Variable::new("B", 1, 8);
    let bound = batch.bind(2).unwrap();
    let x = Tensor::empty_dynamic(&[bound.as_sint(), SInt::from(len)], DType::Float32);
    let spec = x.stft().n_fft(n_fft).hop(hop).call().unwrap();

    let shape = spec.shape().unwrap();
    assert!(shape[0].is_symbolic(), "batch axis must stay symbolic, got {:?}", shape[0]);
    assert_eq!(shape[1], SInt::from(n_fft / 2 + 1));
    assert_eq!(shape[2], SInt::from(len / hop + 1));
    assert_eq!(shape[3], SInt::from(2usize));

    // And the same graph reconstructs: istft must accept the symbolic batch too.
    let back = spec.istft().n_fft(n_fft).hop(hop).length(len).call().unwrap();
    let back_shape = back.shape().unwrap();
    assert!(back_shape[0].is_symbolic());
    assert_eq!(back_shape[1], SInt::from(len));
}

// =========================================================================
// istft round trip
// =========================================================================

#[test_case(64, 16, 256; "hann 64/16 75pct overlap")]
#[test_case(32, 8, 192; "hann 32/8 75pct overlap")]
#[test_case(16, 4, 128; "hann 16/4 75pct overlap")]
fn istft_reconstructs_the_signal(n_fft: usize, hop: usize, len: usize) {
    let x = signal(len, 0.42);
    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let spec = input.stft().n_fft(n_fft).hop(hop).call().unwrap();
    let back = spec.istft().n_fft(n_fft).hop(hop).length(len).call().unwrap();
    assert_eq!(back.dims().unwrap(), vec![len]);
    assert_close(&back.to_vec::<f32>().unwrap(), &x, 1e-4);
}

#[test_case(true, true; "center onesided")]
#[test_case(false, true; "no center onesided")]
#[test_case(true, false; "center two-sided")]
#[test_case(false, false; "no center two-sided")]
fn istft_round_trips_across_center_and_sidedness(center: bool, onesided: bool) {
    let (n_fft, hop, len) = (32usize, 8usize, 192usize);
    let x = signal(len, 1.25);
    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let spec = input.stft().n_fft(n_fft).hop(hop).center(center).onesided(onesided).call().unwrap();
    let back = spec.istft().n_fft(n_fft).hop(hop).center(center).onesided(onesided).call().unwrap();

    // Without `center` the analysis drops the tail no full frame covers, and
    // the leading half-window sits outside the NOLA-safe region.
    let got = back.to_vec::<f32>().unwrap();
    let compare = got.len().min(x.len());
    assert!(compare >= len / 2, "reconstruction too short: {}", got.len());
    let start = if center { 0 } else { n_fft };
    assert_close(&got[start..compare], &x[start..compare], 1e-4);
}

#[test]
fn istft_round_trips_a_batch_and_normalized_spectrogram() {
    let (n_fft, hop, len) = (32usize, 8usize, 128usize);
    let rows: Vec<Vec<f64>> = (0..2).map(|b| signal(len, 0.6 + b as f64)).collect();
    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    let expected: Vec<f64> = rows.iter().flatten().copied().collect();

    let input = Tensor::from_slice(flat).try_reshape([2, len as isize]).unwrap();
    let spec = input.stft().n_fft(n_fft).hop(hop).normalized(true).call().unwrap();
    let back = spec.istft().n_fft(n_fft).hop(hop).normalized(true).length(len).call().unwrap();
    assert_eq!(back.dims().unwrap(), vec![2, len]);
    assert_close(&back.to_vec::<f32>().unwrap(), &expected, 1e-4);
}

#[test]
fn istft_length_trims_and_zero_pads() {
    let (n_fft, hop, len) = (32usize, 8usize, 128usize);
    let x = signal(len, 0.11);
    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let spec = input.stft().n_fft(n_fft).hop(hop).call().unwrap();

    let short = spec.istft().n_fft(n_fft).hop(hop).length(len - 10).call().unwrap();
    assert_eq!(short.dims().unwrap(), vec![len - 10]);
    assert_close(&short.to_vec::<f32>().unwrap(), &x[..len - 10], 1e-4);

    // A `length` past the analysed signal keeps reading the reflect padding the
    // forward transform saw, exactly as `torch.istft` does.
    let long = spec.istft().n_fft(n_fft).hop(hop).length(len + 5).call().unwrap().to_vec::<f32>().unwrap();
    assert_eq!(long.len(), len + 5);
    assert_close(&long[..len], &x, 1e-4);
    let padded = reflect_pad(&x, n_fft / 2);
    assert_close(&long[len..], &padded[n_fft / 2 + len..n_fft / 2 + len + 5], 1e-4);

    // Past the overlap-add buffer there is nothing left to reconstruct: zeros.
    let frames = spec.dim_const(1).unwrap();
    let reach = (frames - 1) * hop + n_fft - n_fft / 2;
    let far = spec.istft().n_fft(n_fft).hop(hop).length(reach + 7).call().unwrap().to_vec::<f32>().unwrap();
    assert_eq!(far.len(), reach + 7);
    assert!(far[reach..].iter().all(|&v| v == 0.0), "beyond the buffer must be zero: {:?}", &far[reach..]);
}

#[test]
fn istft_of_a_3d_spectrogram_drops_the_batch_axis() {
    let (n_fft, hop, len) = (16usize, 4usize, 64usize);
    let x = signal(len, 3.3);
    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let back =
        input.stft().n_fft(n_fft).hop(hop).call().unwrap().istft().n_fft(n_fft).hop(hop).length(len).call().unwrap();
    assert_eq!(back.dims().unwrap(), vec![len]);
}

// =========================================================================
// Complex helpers
// =========================================================================

/// `[2, 3, 2]` complex tensor plus its host `(re, im)` pairs.
fn complex_pair(seed: f64) -> (Tensor, Vec<(f64, f64)>) {
    let raw = signal(12, seed);
    let host: Vec<(f64, f64)> = raw.chunks(2).map(|c| (c[0], c[1])).collect();
    let t = Tensor::from_slice(raw.iter().map(|&v| v as f32).collect::<Vec<_>>()).try_reshape([2, 3, 2]).unwrap();
    (t, host)
}

#[test]
fn power_and_magnitude_match_the_host() {
    let (t, host) = complex_pair(0.8);
    let power: Vec<f64> = host.iter().map(|(r, i)| r * r + i * i).collect();
    assert_eq!(t.power().unwrap().dims().unwrap(), vec![2, 3]);
    assert_close(&t.power().unwrap().to_vec::<f32>().unwrap(), &power, 1e-6);

    let eps = 1e-12;
    let mag: Vec<f64> = power.iter().map(|p| (p + eps).sqrt()).collect();
    assert_close(&t.magnitude(eps).unwrap().to_vec::<f32>().unwrap(), &mag, 1e-6);

    let abs: Vec<f64> = power.iter().map(|p| p.sqrt()).collect();
    assert_close(&t.complex_abs().unwrap().to_vec::<f32>().unwrap(), &abs, 1e-6);
}

#[test]
fn complex_parts_split_the_trailing_axis() {
    let (t, host) = complex_pair(1.6);
    let re: Vec<f64> = host.iter().map(|p| p.0).collect();
    let im: Vec<f64> = host.iter().map(|p| p.1).collect();
    assert_close(&t.complex_real().unwrap().to_vec::<f32>().unwrap(), &re, 1e-6);
    assert_close(&t.complex_imag().unwrap().to_vec::<f32>().unwrap(), &im, 1e-6);
}

#[test]
fn complex_mul_matches_the_host_product() {
    let (a, ha) = complex_pair(0.2);
    let (b, hb) = complex_pair(2.9);
    let expected: Vec<f64> =
        ha.iter().zip(&hb).flat_map(|(&(ar, ai), &(br, bi))| [ar * br - ai * bi, ar * bi + ai * br]).collect();
    let got = a.complex_mul(&b).unwrap();
    assert_eq!(got.dims().unwrap(), vec![2, 3, 2]);
    assert_close(&got.to_vec::<f32>().unwrap(), &expected, 1e-6);
}

#[test]
fn complex_from_polar_inverts_magnitude_and_phase() {
    let mag = Tensor::from_slice([1.0f32, 2.0, 0.5]);
    let phase = Tensor::from_slice([0.0f32, 1.0, -2.0]);
    let z = Tensor::complex_from_polar(&mag, &phase).unwrap();
    assert_eq!(z.dims().unwrap(), vec![3, 2]);
    let expected: Vec<f64> =
        [(1.0f64, 0.0f64), (2.0, 1.0), (0.5, -2.0)].iter().flat_map(|&(m, p)| [m * p.cos(), m * p.sin()]).collect();
    assert_close(&z.to_vec::<f32>().unwrap(), &expected, 1e-6);
    assert_close(&z.complex_abs().unwrap().to_vec::<f32>().unwrap(), &[1.0, 2.0, 0.5], 1e-6);
}

#[test]
fn complex_helpers_reject_a_non_pair_trailing_axis() {
    let t = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    for err in [t.power().unwrap_err(), t.complex_real().unwrap_err(), t.complex_mul(&t).unwrap_err()] {
        assert!(matches!(err.kind(), ErrorKind::ShapeMismatch { .. }), "got {err}");
    }
}

/// The GTCRN mask branch: magnitude features and a complex ratio mask, both
/// expressed on the `[B, F, T, 2]` layout `stft` produces.
#[test]
fn complex_helpers_reproduce_the_gtcrn_mask_branch() {
    let (n_fft, hop, len) = (32usize, 16usize, 96usize);
    let x = signal(len, 0.77);
    let spec = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>())
        .stft()
        .n_fft(n_fft)
        .hop(hop)
        .call()
        .unwrap();

    let re = spec.complex_real().unwrap();
    let im = spec.complex_imag().unwrap();
    let manual = re.square().try_add(im.square()).unwrap().try_add(1e-12f32).unwrap().try_sqrt().unwrap();
    assert_close(
        &spec.magnitude(1e-12).unwrap().to_vec::<f32>().unwrap(),
        &manual.to_vec::<f32>().unwrap().iter().map(|&v| v as f64).collect::<Vec<_>>(),
        1e-6,
    );

    // A unit mask must leave the spectrogram untouched.
    let dims = spec.dims().unwrap();
    let mut unit = vec![0.0f32; dims.iter().product::<usize>()];
    for (i, v) in unit.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { 0.0 };
    }
    let mask = Tensor::from_slice(unit).try_reshape(dims.iter().map(|&d| d as isize).collect::<Vec<_>>()).unwrap();
    let masked = spec.complex_mul(&mask).unwrap();
    assert_close(
        &masked.to_vec::<f32>().unwrap(),
        &spec.to_vec::<f32>().unwrap().iter().map(|&v| v as f64).collect::<Vec<_>>(),
        1e-6,
    );
}

/// The Silero VAD front-end: the magnitude of the one-sided spectrum.
#[test]
fn stft_magnitude_reproduces_the_silero_front_end() {
    let (n_fft, hop, len) = (32usize, 16usize, 96usize);
    let x = signal(len, 1.9);
    let win = host_window(&Window::Hann, n_fft);
    let flat = ref_stft(&x, n_fft, hop, &win, true, true, false);
    let expected: Vec<f64> = flat.chunks(2).map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt()).collect();

    let mag = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>())
        .stft()
        .n_fft(n_fft)
        .hop(hop)
        .call()
        .unwrap()
        .complex_abs()
        .unwrap();
    assert_eq!(mag.dims().unwrap(), vec![n_fft / 2 + 1, len / hop + 1]);
    assert_close(&mag.to_vec::<f32>().unwrap(), &expected, 2e-4);
}

// =========================================================================
// Parameter validation
// =========================================================================

#[test]
fn stft_rejects_bad_parameters() {
    let x = Tensor::from_slice(vec![0.1f32; 64]);
    assert!(matches!(x.stft().n_fft(0).call().unwrap_err().kind(), ErrorKind::ParamRange { .. }));
    assert!(matches!(x.stft().n_fft(16).hop(0).call().unwrap_err().kind(), ErrorKind::ParamRange { .. }));
    assert!(matches!(x.stft().n_fft(16).win_length(17).call().unwrap_err().kind(), ErrorKind::ParamRange { .. }));
    // Signal shorter than one frame.
    let short = Tensor::from_slice(vec![0.1f32; 8]);
    assert!(matches!(short.stft().n_fft(32).center(false).call().unwrap_err().kind(), ErrorKind::ParamRange { .. }));
    // Rank outside 1-D / 2-D.
    let cube = Tensor::from_slice(vec![0.1f32; 64]).try_reshape([2, 4, 8]).unwrap();
    assert!(matches!(cube.stft().n_fft(8).call().unwrap_err().kind(), ErrorKind::NdimExact { .. }));
    // Integer input.
    let ints = Tensor::from_slice(vec![1i32; 64]);
    assert!(matches!(ints.stft().n_fft(16).call().unwrap_err().kind(), ErrorKind::FloatDTypeRequired { .. }));
}

#[test]
fn istft_rejects_parameters_that_disagree_with_the_analysis() {
    let x = Tensor::from_slice(vec![0.1f32; 128]);
    let spec = x.stft().n_fft(32).hop(8).call().unwrap();

    // n_fft implies a different bin count.
    let err = spec.istft().n_fft(16).hop(8).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ShapeMismatch { .. }), "got {err}");

    // A one-sided spectrogram read as two-sided.
    let err = spec.istft().n_fft(32).hop(8).onesided(false).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ShapeMismatch { .. }), "got {err}");

    // Rank outside 3-D / 4-D.
    let err = x.istft().n_fft(32).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::NdimExact { .. }), "got {err}");

    // Trailing axis that does not hold a complex pair.
    let odd = Tensor::from_slice(vec![0.1f32; 17 * 5 * 3]).try_reshape([17, 5, 3]).unwrap();
    let err = odd.istft().n_fft(32).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ShapeMismatch { .. }), "got {err}");
}

// =========================================================================
// Kernel count
// =========================================================================

/// Kernels the rangeified graph would launch — one `Op::Call` each.
fn count_kernels(t: &Tensor) -> usize {
    use svod_ir::{Op, UOp};
    let sink = UOp::sink(vec![t.uop().contiguous()]);
    let rangeified = svod_schedule::rangeify_with_map(sink).expect("rangeify");
    let (kernels, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("kernel graph");
    kernels.toposort_call_aware(false).iter().filter(|n| matches!(n.op(), Op::Call(..))).count()
}

/// The conv formulation: one launch pads the framed signal to a tileable
/// frame count, one convolves it at the padded extents with the host-built
/// `[2F', 1, n_fft]` DFT kernel (an input buffer, not a launch), and one
/// trims the result for a consumer that wants it materialized — trimming the
/// convolution lazily would hand the natural extents back to the reduce. The
/// inverse reads the trim as a view, so it adds its own convolution plus one
/// launch for the window-square overlap-add divisor on top of the first two.
#[test]
fn stft_is_three_kernels_and_istft_is_five() {
    let x = Tensor::empty(&[4, 16000], DType::Float32);
    let spec = x.stft().n_fft(512).hop(256).call().unwrap();
    let back = spec.istft().n_fft(512).hop(256).call().unwrap();
    assert_eq!((count_kernels(&spec), count_kernels(&back)), (3, 5));
}

// =========================================================================
// Mel filterbank and mel spectrogram
// =========================================================================

/// `librosa.hz_to_mel` for both scales, written out independently of the op.
fn ref_hz_to_mel(hz: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_mel = 1000.0 / f_sp;
            let logstep = 6.4f64.ln() / 27.0;
            if hz >= 1000.0 { min_log_mel + (hz / 1000.0).ln() / logstep } else { hz / f_sp }
        }
    }
}

fn ref_mel_to_hz(mel: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10f64.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_mel = 1000.0 / f_sp;
            let logstep = 6.4f64.ln() / 27.0;
            if mel >= min_log_mel { 1000.0 * ((mel - min_log_mel) * logstep).exp() } else { mel * f_sp }
        }
    }
}

/// `librosa.filters.mel` / `torchaudio.functional.melscale_fbanks`, flat
/// `[n_mels, n_fft / 2 + 1]`, using the ramp intersection rather than the
/// op's `min(up, down)` form.
fn ref_mel_filterbank(
    sr: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
    scale: MelScale,
    norm: Option<MelNorm>,
) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    let (m_lo, m_hi) = (ref_hz_to_mel(f_min, scale), ref_hz_to_mel(f_max, scale));
    let hz: Vec<f64> =
        (0..n_mels + 2).map(|i| ref_mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (n_mels + 1) as f64, scale)).collect();
    let mut fb = vec![0.0; n_mels * n_bins];
    for m in 0..n_mels {
        let (lo, mid, hi) = (hz[m], hz[m + 1], hz[m + 2]);
        let enorm = match norm {
            Some(MelNorm::Slaney) => 2.0 / (hi - lo),
            None => 1.0,
        };
        for k in 0..n_bins {
            let f = k as f64 * sr as f64 / n_fft as f64;
            let w = if f >= lo && f <= mid {
                (f - lo) / (mid - lo)
            } else if f > mid && f <= hi {
                (hi - f) / (hi - mid)
            } else {
                0.0
            };
            fb[m * n_bins + k] = w * enorm;
        }
    }
    fb
}

#[test_case(16000, 400, 80, 0.0, 8000.0, MelScale::Slaney, Some(MelNorm::Slaney); "whisper slaney normalized")]
#[test_case(16000, 400, 64, 0.0, 8000.0, MelScale::Htk, None; "gigaam htk")]
#[test_case(16000, 512, 80, 20.0, 7600.0, MelScale::Htk, Some(MelNorm::Slaney); "htk normalized band-limited")]
#[test_case(8000, 64, 10, 50.0, 3500.0, MelScale::Slaney, None; "small slaney unnormalized")]
fn mel_filterbank_matches_host_reference(
    sr: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f64,
    f_max: f64,
    scale: MelScale,
    norm: Option<MelNorm>,
) {
    let expected = ref_mel_filterbank(sr, n_fft, n_mels, f_min, f_max, scale, norm);
    let fb = Tensor::mel_filterbank(sr, n_fft, n_mels, f_min, f_max, scale, norm, DType::Float32).unwrap();
    assert_eq!(fb.dims().unwrap(), vec![n_mels, n_fft / 2 + 1]);
    let got = fb.to_vec::<f32>().unwrap();
    assert_close(&got, &expected, 1e-6);
    // Every filter has support, and adjacent filters overlap: the ramps tile the band.
    let n_bins = n_fft / 2 + 1;
    for row in got.chunks(n_bins) {
        assert!(row.iter().any(|&w| w > 0.0), "empty filter");
    }
    // Unnormalized triangles peak at one; normalized ones integrate to two.
    if norm.is_none() {
        assert!(got.iter().cloned().fold(0.0f32, f32::max) <= 1.0 + 1e-6);
    }
}

#[test]
fn mel_filterbank_honours_dtype_and_rejects_bad_parameters() {
    let fb = Tensor::mel_filterbank(16000, 64, 8, 0.0, 8000.0, MelScale::Htk, None, DType::Float16).unwrap();
    assert_eq!(fb.dtype(), DType::Float16);
    let bad = [
        Tensor::mel_filterbank(0, 64, 8, 0.0, 8000.0, MelScale::Htk, None, DType::Float32),
        Tensor::mel_filterbank(16000, 0, 8, 0.0, 8000.0, MelScale::Htk, None, DType::Float32),
        Tensor::mel_filterbank(16000, 64, 0, 0.0, 8000.0, MelScale::Htk, None, DType::Float32),
        Tensor::mel_filterbank(16000, 64, 8, -1.0, 8000.0, MelScale::Htk, None, DType::Float32),
        Tensor::mel_filterbank(16000, 64, 8, 4000.0, 4000.0, MelScale::Htk, None, DType::Float32),
    ];
    for err in bad {
        assert!(matches!(err.unwrap_err().kind(), ErrorKind::ParamRange { .. }));
    }
    let err = Tensor::mel_filterbank(16000, 64, 8, 0.0, 8000.0, MelScale::Htk, None, DType::Int32).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::FloatDTypeRequired { .. }), "got {err}");
}

/// Host log-mel pipeline: naive STFT → `|X|^power` → filterbank → log.
#[allow(clippy::too_many_arguments)]
fn ref_mel_spectrogram(
    x: &[f64],
    sr: usize,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    scale: MelScale,
    norm: Option<MelNorm>,
    power: f64,
    log: Option<MelLog>,
) -> Vec<f64> {
    let win = host_window(&Window::Hann, n_fft);
    let spec = ref_stft(x, n_fft, hop, &win, true, true, false);
    let bins = n_fft / 2 + 1;
    let frames = spec.len() / (bins * 2);
    let energy: Vec<f64> = spec.chunks(2).map(|c| (c[0] * c[0] + c[1] * c[1]).powf(power / 2.0)).collect();
    let fb = ref_mel_filterbank(sr, n_fft, n_mels, 0.0, sr as f64 / 2.0, scale, norm);
    let mut mel = vec![0.0; n_mels * frames];
    for m in 0..n_mels {
        for t in 0..frames {
            mel[m * frames + t] = (0..bins).map(|k| fb[m * bins + k] * energy[k * frames + t]).sum();
        }
    }
    match log {
        None => mel,
        Some(MelLog::Ln { min, max }) => mel.iter().map(|v| v.clamp(min, max).ln()).collect(),
        Some(MelLog::Whisper) => {
            let logged: Vec<f64> = mel.iter().map(|v| v.max(1e-10).log10()).collect();
            let floor = logged.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - 8.0;
            logged.iter().map(|v| (v.max(floor) + 4.0) / 4.0).collect()
        }
    }
}

#[test_case(MelScale::Htk, None, 2.0, None; "torchaudio power")]
#[test_case(MelScale::Htk, None, 1.0, None; "torchaudio magnitude")]
#[test_case(MelScale::Htk, None, 2.0, Some(MelLog::Ln { min: 1e-9, max: 1e9 }); "gigaam log")]
#[test_case(MelScale::Slaney, Some(MelNorm::Slaney), 2.0, Some(MelLog::Whisper); "whisper log")]
#[test_case(MelScale::Slaney, Some(MelNorm::Slaney), 3.0, None; "fractional power")]
fn mel_spectrogram_matches_naive_host_pipeline(
    scale: MelScale,
    norm: Option<MelNorm>,
    power: f64,
    log: Option<MelLog>,
) {
    let (sr, n_fft, hop, n_mels, len) = (8000usize, 64usize, 16usize, 12usize, 256usize);
    let x = signal(len, 0.35);
    let expected = ref_mel_spectrogram(&x, sr, n_fft, hop, n_mels, scale, norm, power, log);

    let input = Tensor::from_slice(x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let mel = input
        .mel_spectrogram()
        .sample_rate(sr)
        .n_fft(n_fft)
        .hop(hop)
        .n_mels(n_mels)
        .mel_scale(scale)
        .maybe_norm(norm)
        .power(power)
        .maybe_log(log)
        .call()
        .unwrap();
    assert_eq!(mel.dims().unwrap(), vec![n_mels, len / hop + 1]);
    let tol = if log.is_some() { 1e-3 } else { 2e-3 };
    assert_close(&mel.to_vec::<f32>().unwrap(), &expected, tol);
}

/// The Whisper floor is `max - 8` of each signal, not of the batch: a quiet
/// row next to a loud one keeps its own dynamic range.
#[test]
fn mel_spectrogram_whisper_log_floors_each_signal_separately() {
    let (sr, n_fft, hop, n_mels, len) = (8000usize, 64usize, 16usize, 12usize, 256usize);
    let loud = signal(len, 0.35);
    let quiet: Vec<f64> = loud.iter().map(|v| v * 1e-3).collect();
    let rows = [loud.clone(), quiet.clone()];
    let expected: Vec<f64> = rows
        .iter()
        .flat_map(|r| {
            ref_mel_spectrogram(
                r,
                sr,
                n_fft,
                hop,
                n_mels,
                MelScale::Slaney,
                Some(MelNorm::Slaney),
                2.0,
                Some(MelLog::Whisper),
            )
        })
        .collect();

    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    let input = Tensor::from_slice(flat).try_reshape([2, len as isize]).unwrap();
    let mel = input
        .mel_spectrogram()
        .sample_rate(sr)
        .n_fft(n_fft)
        .hop(hop)
        .n_mels(n_mels)
        .mel_scale(MelScale::Slaney)
        .norm(MelNorm::Slaney)
        .log(MelLog::Whisper)
        .call()
        .unwrap();
    assert_eq!(mel.dims().unwrap(), vec![2, n_mels, len / hop + 1]);
    let got = mel.to_vec::<f32>().unwrap();
    assert_close(&got, &expected, 1e-3);
    // In normalized units the floor is `row_max - 2` (`(x - 8 + 4) / 4`).
    // Each row honours its own; a batch-wide floor would have lifted the
    // quiet row above the loud row's.
    let per_row = n_mels * (len / hop + 1);
    let (loud_row, quiet_row) = got.split_at(per_row);
    let bounds = |row: &[f32]| {
        (row.iter().cloned().fold(f32::INFINITY, f32::min), row.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
    };
    let (loud_min, loud_max) = bounds(loud_row);
    let (quiet_min, quiet_max) = bounds(quiet_row);
    assert!(loud_min >= loud_max - 2.0 - 1e-5 && quiet_min >= quiet_max - 2.0 - 1e-5);
    assert!(
        (loud_max - quiet_max - 1.5).abs() < 1e-3,
        "1e-3 in amplitude is 1e-6 in power, -1.5 normalized: {loud_max} vs {quiet_max}"
    );
    assert!(quiet_min < loud_max - 2.0, "quiet row floored against the loud row's maximum");
}

#[test]
fn mel_spectrogram_rejects_bad_parameters() {
    let x = Tensor::from_slice(vec![0.1f32; 256]);
    let err = x.mel_spectrogram().sample_rate(8000).n_fft(64).n_mels(8).power(0.0).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ParamRange { .. }), "got {err}");
    let err = x.mel_spectrogram().sample_rate(8000).n_fft(64).n_mels(0).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ParamRange { .. }), "got {err}");
    let err = x.mel_spectrogram().sample_rate(8000).n_fft(64).n_mels(8).f_min(5000.0).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ParamRange { .. }), "got {err}");
    let ints = Tensor::from_slice(vec![1i32; 256]);
    let err = ints.mel_spectrogram().sample_rate(8000).n_fft(64).n_mels(8).call().unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::FloatDTypeRequired { .. }), "got {err}");
    let err = Tensor::from_slice(vec![0.1f32; 8]).mel_log(MelLog::Whisper).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::NdimMinimum { .. }), "got {err}");
}

/// A Whisper-sized front-end: the STFT's signal pad and conv (the DFT kernel
/// and the filterbank are host tables), then the filterbank contraction,
/// which absorbs the trailing frame trim (its output is the trimmed
/// `[B, n_mels, T]`), the `Ln` log and the three-pass Whisper log (reduce,
/// floor, normalize).
#[test]
fn mel_spectrogram_kernel_count_stays_small() {
    let x = Tensor::empty(&[4, 16000 * 30], DType::Float32);
    let power = x.mel_spectrogram().sample_rate(16000).n_fft(400).hop(160).n_mels(80).call().unwrap();
    let whisper = power.mel_log(MelLog::Whisper).unwrap();
    let ln = power.mel_log(MelLog::Ln { min: 1e-9, max: 1e9 }).unwrap();
    assert_eq!((count_kernels(&power), count_kernels(&ln), count_kernels(&whisper)), (3, 3, 6));
}
