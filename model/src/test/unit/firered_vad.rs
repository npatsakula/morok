//! FireRedVAD tests.
//!
//! Cheap default tier: forward-graph shape on random weights (no compile),
//! fbank parity against an embedded `kaldi-native-fbank` golden, and the
//! smoothing function against numpy reference values. The `#[ignore]` tier
//! compiles and executes the JIT on the local CPU device: window/halo stitch
//! exactness on random weights, and — when the converted checkpoint is
//! present (`scripts/convert_firered_vad.py`) — probability parity against
//! the PyTorch reference golden.

use std::path::{Path, PathBuf};

use svod_dtype::DType;
use svod_tensor::Tensor;

use svod_arch::pipelines::audio::Splitter;

use crate::audio::EncoderBounds;
use crate::firered_vad::{
    CORE, FRAME_LENGTH, FRAME_SHIFT, FireRedFbank, FireRedVad, FireRedVadInference, FireRedVadSplitter, N_MELS,
    smooth_trailing,
};

// ---------------------------------------------------------------------------
// Cheap default tests.
// ---------------------------------------------------------------------------

/// Full DFSMN forward on `[2, 64, 80]` zeros: catches axis/padding bugs in
/// the CMVN → linears → 8 FSMN layers → head chain without any compile.
#[test]
fn forward_zero_input_shape() {
    let model = FireRedVad::with_random_weights();
    let feat = Tensor::zeros(&[2, 64, N_MELS], DType::Float32).unwrap();
    let out = model.forward(&feat, None).unwrap();
    let shape = out.dims().unwrap();
    assert_eq!(shape, vec![2, 64]);
}

/// Deterministic ~U(-1,1) stream (LCG); the test needs reproducibility, not
/// statistical quality.
pub(super) fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Two-tone synthetic signal in int16-scale integers (exactly representable
/// in f32), divided down to the `[-1, 1]` waveform scale the pipeline uses.
/// Must match the generator in the golden block below.
pub(super) fn synthetic_waveform(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / 16_000.0;
            let s = 8000.0 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 4000.0 * (2.0 * std::f64::consts::PI * 1337.0 * t + 0.5).sin();
            (s.round() / 32_768.0) as f32
        })
        .collect()
}

/// `kaldi-native-fbank` output (3 frames x 80 bins) for
/// [`synthetic_waveform`]`(720)` with the FireRedVAD options (16 kHz,
/// 25 ms / 10 ms, snip_edges, dither 0, 80 bins). Regenerate by feeding the
/// int16-scale [`synthetic_waveform`] signal (multiply by 32768) into
/// `knf.OnlineFbank` configured exactly as `golden()` in
/// `scripts/convert_firered_vad.py`.
#[rustfmt::skip]
const KNF_GOLDEN: [f32; 240] = [
    8.76748, 8.86774, 7.987, 6.88474, 8.95847, 10.2703, 11.1921, 11.6446,
    10.6674, 11.7287, 14.7883, 16.7222, 19.7846, 23.018, 23.7681, 22.7523,
    19.2142, 16.3669, 13.1138, 12.5074, 12.0309, 10.3394, 9.34566, 9.81421,
    9.26127, 8.65936, 9.87204, 10.9262, 10.7244, 13.3577, 14.1095, 19.0623,
    23.7878, 24.8127, 22.3086, 16.5985, 12.983, 12.2672, 10.1706, 9.87137,
    8.42635, 8.05031, 7.29084, 6.48446, 6.27015, 5.42928, 5.21114, 5.10249,
    4.88863, 4.6797, 4.2891, 3.99888, 4.77948, 4.68546, 3.44245, 4.2068,
    4.33012, 4.40705, 4.98153, 5.45435, 5.17412, 5.37495, 3.73236, 4.05775,
    4.59792, 4.79237, 5.44654, 5.55449, 4.42755, 5.32081, 5.58989, 5.68844,
    5.53729, 7.02982, 6.24119, 6.32627, 6.32612, 5.49748, 6.0237, 6.00042,
    8.83358, 9.46813, 9.01153, 7.65047, 8.96579, 10.3414, 11.3166, 11.775,
    10.8372, 11.7011, 14.7957, 16.7281, 19.7852, 23.0179, 23.7681, 22.7525,
    19.2132, 16.3653, 13.1091, 12.484, 12.024, 10.3717, 9.23362, 9.85687,
    9.38589, 8.58461, 9.95869, 10.9627, 10.7426, 13.368, 14.1113, 19.0626,
    23.7878, 24.8128, 22.3087, 16.5987, 12.9808, 12.2688, 10.1599, 9.87413,
    8.41198, 8.06972, 7.23964, 6.49947, 6.30301, 5.56274, 5.48823, 4.90279,
    4.7577, 4.47446, 3.8039, 3.71815, 3.99427, 4.46599, 3.83783, 3.82598,
    3.50366, 3.39719, 4.37434, 5.29055, 4.88626, 5.04198, 5.21705, 5.34855,
    5.31901, 4.97399, 5.14239, 5.98186, 4.9886, 4.76554, 4.47148, 4.9417,
    5.9914, 6.23969, 5.1716, 5.77916, 6.23199, 6.86786, 5.73129, 5.53833,
    8.75796, 9.66771, 9.32461, 7.84195, 8.89712, 10.3496, 11.3691, 11.8403,
    10.9385, 11.6639, 14.7974, 16.7311, 19.7855, 23.0178, 23.7681, 22.7525,
    19.2125, 16.3639, 13.1058, 12.4706, 12.017, 10.3913, 9.16499, 9.87541,
    9.42908, 8.5617, 9.99962, 10.973, 10.7505, 13.3703, 14.1118, 19.0627,
    23.7878, 24.8128, 22.3086, 16.5987, 12.9847, 12.2661, 10.1461, 9.87408,
    8.40856, 8.04596, 7.26322, 6.4143, 6.34618, 5.41838, 5.42221, 4.83895,
    4.37814, 4.54379, 4.31374, 4.13929, 4.03519, 3.60523, 3.86561, 3.34763,
    2.4535, 4.45431, 4.20054, 3.87932, 4.37844, 4.57245, 5.31544, 5.65941,
    5.45759, 5.02761, 4.96965, 5.18873, 5.76042, 5.21652, 4.62031, 4.33738,
    4.47017, 5.31772, 5.50067, 4.95214, 6.44142, 6.36777, 5.63803, 4.85101,
];

/// Rust fbank vs `kaldi-native-fbank` on the synthetic two-tone signal.
/// Pins the full per-frame chain: int16 scaling, DC removal, per-frame
/// pre-emphasis, Povey window, power spectrum, Kaldi mel banks, log floor.
#[test]
fn fbank_matches_kaldi_native_fbank() {
    let waveform = synthetic_waveform(720);
    let fbank = FireRedFbank::new();
    assert_eq!(fbank.num_frames(waveform.len()), 3);
    assert_eq!(fbank.num_frames(FRAME_LENGTH - 1), 0);
    assert_eq!(fbank.num_frames(FRAME_LENGTH + FRAME_SHIFT), 2);

    let feat = fbank.forward(&waveform);
    assert_eq!(feat.len(), KNF_GOLDEN.len());
    let max_abs = feat.iter().zip(&KNF_GOLDEN).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-3, "fbank drifted from kaldi-native-fbank: max |delta| = {max_abs}");
}

/// `smooth_trailing` vs numpy reference
/// (`np.convolve(p, ones(w)/w, 'full')[:n]` + cumulative-mean ramp), the
/// exact semantics of `VadPostprocessor._smooth_prob`.
#[test]
fn smooth_trailing_matches_numpy() {
    let probs = [0.1, 0.9, 0.2, 0.8, 0.5, 0.3, 0.95, 0.05];
    let want = [0.1, 0.5, 0.4, 0.5, 0.5, 0.54, 0.55, 0.52];
    let got = smooth_trailing(&probs, 5);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-6, "smoothing mismatch: {got:?} vs {want:?}");
    }
    // Shorter than the window: pure cumulative means.
    assert_eq!(smooth_trailing(&probs[..3], 5), vec![0.1, 0.5, 0.4]);
    // Degenerate windows pass through.
    assert_eq!(smooth_trailing(&probs, 1), probs.to_vec());
    assert!(smooth_trailing(&[], 5).is_empty());
}

// ---------------------------------------------------------------------------
// Heavy tests (compile + execute on the local CPU device).
// ---------------------------------------------------------------------------

/// Stitched windowed inference == direct full-length forward. The input spans
/// two windows (`CORE + 37` frames), so the second window's left halo carries
/// real neighbour context — exactly the receptive-field argument the
/// window/halo constants encode. Tolerance covers float reassociation between
/// the `[BATCH, CHUNK_T]` and `[1, T]` graphs.
#[test]
#[ignore = "heavy: FireRedVad JIT compile + execute on random weights"]
fn stitched_windows_match_full_forward() {
    let model = FireRedVad::with_random_weights();
    let n_frames = CORE + 37;
    let mut seed = 0x5eed;
    let feat: Vec<f32> = (0..n_frames * N_MELS).map(|_| lcg(&mut seed)).collect();

    let feat_t =
        Tensor::from_slice(feat.clone()).try_reshape([1isize, n_frames as isize, N_MELS as isize]).expect("reshape");
    let full = model.forward(&feat_t, None).expect("forward");
    full.realize().expect("realize");
    let want = full.as_vec::<f32>().expect("readout");

    let mut inf = FireRedVadInference::new(model).expect("prepare");
    let got = inf.probs(&feat, n_frames).expect("probs");

    assert_eq!(got.len(), want.len());
    let max_abs = got.iter().zip(&want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-4, "stitched probs drifted from full forward: max |delta| = {max_abs}");
}

/// Resolve `firered_vad.safetensors` / `golden.safetensors` for the
/// real-checkpoint tests: `SVOD_FIRERED_VAD` dir override → local
/// `data/firered_vad/` (output of `scripts/convert_firered_vad.py`) → HF Hub
/// download from [`crate::firered_vad::HUB_REPO`].
pub(super) fn real_file(name: &str) -> PathBuf {
    let dir = std::env::var_os("SVOD_FIRERED_VAD")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/firered_vad"));
    let local = dir.join(name);
    if local.exists() { local } else { crate::firered_vad::hub_file(name).expect("download from HF Hub") }
}

pub(super) fn load_golden_vec(sd: &crate::state::StateDict, key: &str) -> Vec<f32> {
    let t = sd.get(key).unwrap_or_else(|| panic!("golden key {key}")).clone();
    t.realize().expect("realize golden");
    t.as_vec::<f32>().expect("golden readout")
}

/// Probability parity against the PyTorch reference: fbank vs the
/// `kaldi-native-fbank` golden on real speech, the device DFSMN vs the
/// reference `DetectModel` probs (golden feat in — isolates the model), and
/// the combined fbank → model pipeline.
#[test]
#[ignore = "heavy: real-weights JIT vs PyTorch golden (local or HF Hub download)"]
fn real_weights_match_pytorch_golden() {
    let weights = real_file("firered_vad.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");
    let samples = load_golden_vec(&golden, "samples");
    let feat_want = load_golden_vec(&golden, "feat");
    let probs_want = load_golden_vec(&golden, "probs");
    let n_frames = probs_want.len();

    let fbank = FireRedFbank::new();
    let feat_got = fbank.forward(&samples);
    assert_eq!(feat_got.len(), feat_want.len(), "frame count mismatch vs knf");
    let fbank_delta = feat_got.iter().zip(&feat_want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(fbank_delta < 1e-3, "fbank drifted from knf golden: max |delta| = {fbank_delta}");

    let model = FireRedVad::from_safetensors(&weights).expect("load weights");
    let mut inf = FireRedVadInference::new(model).expect("prepare");

    // Golden feat in: isolates the device DFSMN against the PyTorch model.
    let probs_got = inf.probs(&feat_want, n_frames).expect("probs");
    let model_delta = probs_got.iter().zip(&probs_want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(model_delta < 1e-3, "DFSMN drifted from PyTorch golden: max |delta| = {model_delta}");

    // Full pipeline: our fbank feeding the model.
    let probs_e2e = inf.probs(&feat_got, n_frames).expect("probs e2e");
    let e2e_delta = probs_e2e.iter().zip(&probs_want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(e2e_delta < 5e-3, "end-to-end probs drifted from golden: max |delta| = {e2e_delta}");

    eprintln!("parity: fbank {fbank_delta:.2e}, model {model_delta:.2e}, e2e {e2e_delta:.2e}");
}

/// Hub round-trip: `from_hub` must download the published weights and
/// reproduce the PyTorch golden probs — verifies the uploaded artifact, not
/// just the local conversion output.
#[test]
#[ignore = "heavy: downloads weights from HF Hub + JIT compile"]
fn from_hub_matches_pytorch_golden() {
    let model = FireRedVad::from_hub().expect("hub weights");
    let golden_path = crate::firered_vad::hub_file("golden.safetensors").expect("hub golden");
    let golden = crate::state::load_safetensors(&golden_path).expect("golden");
    let feat = load_golden_vec(&golden, "feat");
    let probs_want = load_golden_vec(&golden, "probs");

    let mut inf = FireRedVadInference::new(model).expect("prepare");
    let probs_got = inf.probs(&feat, probs_want.len()).expect("probs");
    let max_abs = probs_got.iter().zip(&probs_want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-3, "hub weights drifted from PyTorch golden: max |delta| = {max_abs}");
    eprintln!("hub parity: max |delta| = {max_abs:.2e}");
}

/// Splitter end-to-end on the golden speech sample: the detected chunks must
/// be well-formed and cover the utterance's speech region (~0.44 - 1.82 s in
/// `hello_zh.wav`).
#[test]
#[ignore = "heavy: real-weights splitter end-to-end (local or HF Hub download)"]
fn splitter_detects_speech_on_golden_audio() {
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");
    let samples = load_golden_vec(&golden, "samples");

    let bounds = EncoderBounds {
        sample_rate: 16_000,
        hop_length: 160,
        subsampling_factor: 4,
        max_mel_frames: 3000,
        recommended_target_secs: None,
    };
    let mut splitter =
        FireRedVadSplitter::from_safetensors(&real_file("firered_vad.safetensors"), &bounds).expect("splitter");
    let chunks = splitter.split(&samples).expect("split");

    assert!(!chunks.is_empty(), "no speech detected in a known-speech sample");
    for c in &chunks {
        assert!(c.start_sample < c.end_sample, "inverted chunk {c:?}");
        assert!(c.end_sample <= samples.len(), "chunk past waveform end {c:?}");
    }
    // The upstream postprocessor places speech at ~0.44-1.82 s; our chunker's
    // boundary rules differ by design (no onset back-fill / FSM tail), so
    // require coverage of the conservative speech core only.
    let (speech_lo, speech_hi) = (9_600usize, 25_600usize); // 0.6 - 1.6 s
    assert!(
        chunks.iter().any(|c| c.start_sample <= speech_lo && c.end_sample >= speech_hi),
        "no chunk covers the speech core: {chunks:?}",
    );
}
